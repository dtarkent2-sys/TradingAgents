"""FastAPI SSE backend for TradingAgents."""

import os
import time
import uuid
import asyncio
import json
import traceback as _tb
from datetime import date

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# If using Groq (or other OpenAI-compatible), set OPENAI_API_KEY for langchain
if not os.environ.get("OPENAI_API_KEY"):
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        os.environ["OPENAI_API_KEY"] = groq_key

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.stats_handler import StatsCallbackHandler
from cli.main import (
    MessageBuffer,
    classify_message_type,
    update_analyst_statuses,
)

app = FastAPI(title="TradingAgents API")

# --- CORS ---
_cors_env = os.getenv("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth dependency ---
_API_KEY = os.getenv("AGENTS_API_KEY", "")


async def verify_api_key(request: Request):
    if not _API_KEY:
        return  # dev mode — no auth
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {_API_KEY}":
        raise HTTPException(401, "Invalid or missing API key")


# --- Concurrency ---
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_ANALYSES", "3"))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Active analysis state: id -> {queue, events (replay buffer), done, created_at}
analyses: dict[str, dict] = {}


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str | None = None


def build_config():
    """Build TradingAgents config — uses Groq (OpenAI-compatible) by default."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = os.getenv("LLM_PROVIDER", "openai")
    config["deep_think_llm"] = os.getenv("DEEP_THINK_MODEL", "llama-3.3-70b-versatile")
    config["quick_think_llm"] = os.getenv("QUICK_THINK_MODEL", "llama-3.3-70b-versatile")
    config["backend_url"] = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    config["parallel_analysts"] = True
    print(f"[CONFIG] provider={config['llm_provider']}, deep={config['deep_think_llm']}, quick={config['quick_think_llm']}, url={config['backend_url']}", flush=True)
    return config


def get_stats_dict(stats_handler, buf, start_time):
    """Build stats dict for SSE events."""
    s = stats_handler.get_stats()
    agents_done = sum(1 for v in buf.agent_status.values() if v == "completed")
    elapsed = time.time() - start_time
    return {
        "agents_done": agents_done,
        "agents_total": len(buf.agent_status),
        "llm_calls": s["llm_calls"],
        "tool_calls": s["tool_calls"],
        "tokens_in": s["tokens_in"],
        "tokens_out": s["tokens_out"],
        "reports_done": buf.get_completed_reports_count(),
        "reports_total": len(buf.report_sections),
        "elapsed": round(elapsed, 1),
    }


def _agent_stage(agent_name):
    """Map agent name to pipeline stage."""
    if agent_name in ("Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"):
        return "analysts"
    if agent_name in ("Bull Researcher", "Bear Researcher", "Research Manager"):
        return "research"
    if agent_name == "Trader":
        return "trading"
    if agent_name in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
        return "risk"
    if agent_name == "Portfolio Manager":
        return "decision"
    return "unknown"


async def _run_analysis_inner(analysis_id: str, ticker: str, trade_date: str):
    """Core analysis logic."""
    state = analyses[analysis_id]
    q = state["queue"]
    config = build_config()
    stats_handler = StatsCallbackHandler()
    selected_analysts = ["market", "social", "news", "fundamentals"]

    try:
        graph = TradingAgentsGraph(
            selected_analysts=selected_analysts,
            debug=False,
            config=config,
            callbacks=[stats_handler],
        )
        print(f"[ANALYSIS] LLM types: deep={type(graph.deep_thinking_llm).__name__}, quick={type(graph.quick_thinking_llm).__name__}", flush=True)
    except Exception as e:
        print(f"[ANALYSIS] Init failed: {e}\n{_tb.format_exc()}", flush=True)
        await q.put({"type": "error", "message": f"Init failed: {e}"})
        await q.put(None)
        return

    buf = MessageBuffer()
    buf.init_for_analysis(selected_analysts)
    init_state = graph.propagator.create_initial_state(ticker, trade_date)
    args = graph.propagator.get_graph_args(callbacks=[stats_handler])
    start_time = time.time()

    emitted_reports = set()
    research_emitted = False
    trader_emitted = False
    risk_emitted = False
    final_state = None
    prev_statuses = {}

    # Emit all analysts as "in_progress" immediately (they run in parallel)
    analyst_name_map = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }
    for analyst_type in selected_analysts:
        agent_name = analyst_name_map[analyst_type]
        buf.update_agent_status(agent_name, "in_progress")
        st = get_stats_dict(stats_handler, buf, start_time)
        evt = {
            "type": "agent_update",
            "agent": agent_name,
            "stage": "analysts",
            "status": "in_progress",
            "stats": st,
        }
        state["events"].append(evt)
        await q.put(evt)
        prev_statuses[agent_name] = "in_progress"

    try:
        async for chunk in graph.graph.astream(init_state, **args):
            final_state = chunk

            # Process messages (same logic as Chainlit app)
            if chunk.get("messages") and len(chunk["messages"]) > 0:
                last_msg = chunk["messages"][-1]
                msg_id = getattr(last_msg, "id", None)
                if msg_id != buf._last_message_id:
                    buf._last_message_id = msg_id
                    msg_type, content = classify_message_type(last_msg)
                    if content and content.strip():
                        buf.add_message(msg_type, content)
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            if isinstance(tc, dict):
                                buf.add_tool_call(tc["name"], tc["args"])
                            else:
                                buf.add_tool_call(tc.name, tc.args)

            update_analyst_statuses(buf, chunk)
            st = get_stats_dict(stats_handler, buf, start_time)

            # Emit agent status changes only (avoid flooding)
            for agent, status in buf.agent_status.items():
                if prev_statuses.get(agent) != status:
                    prev_statuses[agent] = status
                    evt = {
                        "type": "agent_update",
                        "agent": agent,
                        "stage": _agent_stage(agent),
                        "status": status,
                        "stats": st,
                    }
                    state["events"].append(evt)
                    await q.put(evt)

            # Analyst reports
            report_map = {
                "market_report": ("Market Analyst", "analysts"),
                "sentiment_report": ("Social Analyst", "analysts"),
                "news_report": ("News Analyst", "analysts"),
                "fundamentals_report": ("Fundamentals Analyst", "analysts"),
            }
            for field, (agent_name, stage) in report_map.items():
                if field not in emitted_reports and chunk.get(field):
                    emitted_reports.add(field)
                    evt = {
                        "type": "report",
                        "agent": agent_name,
                        "stage": stage,
                        "field": field,
                        "report": chunk[field],
                        "stats": st,
                    }
                    state["events"].append(evt)
                    await q.put(evt)

            # Research debate
            if chunk.get("investment_debate_state") and not research_emitted:
                debate = chunk["investment_debate_state"]
                bull = debate.get("bull_history", "").strip()
                bear = debate.get("bear_history", "").strip()
                judge = debate.get("judge_decision", "").strip()

                if bull or bear:
                    for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                        buf.update_agent_status(a, "in_progress")

                if judge:
                    research_emitted = True
                    for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                        buf.update_agent_status(a, "completed")
                    buf.update_agent_status("Trader", "in_progress")
                    buf.update_report_section("investment_plan", judge)
                    evt = {
                        "type": "debate",
                        "stage": "research",
                        "bull": bull,
                        "bear": bear,
                        "judge": judge,
                        "stats": get_stats_dict(stats_handler, buf, start_time),
                    }
                    state["events"].append(evt)
                    await q.put(evt)

            # Trader plan
            if chunk.get("trader_investment_plan") and not trader_emitted:
                trader_emitted = True
                buf.update_agent_status("Trader", "completed")
                buf.update_agent_status("Aggressive Analyst", "in_progress")
                buf.update_agent_status("Conservative Analyst", "in_progress")
                buf.update_agent_status("Neutral Analyst", "in_progress")
                buf.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                evt = {
                    "type": "trader",
                    "stage": "trading",
                    "plan": chunk["trader_investment_plan"],
                    "stats": get_stats_dict(stats_handler, buf, start_time),
                }
                state["events"].append(evt)
                await q.put(evt)

            # Risk debate
            if chunk.get("risk_debate_state") and not risk_emitted:
                risk = chunk["risk_debate_state"]
                agg = risk.get("aggressive_history", "").strip()
                con = risk.get("conservative_history", "").strip()
                neu = risk.get("neutral_history", "").strip()
                judge = risk.get("judge_decision", "").strip()

                if agg:
                    buf.update_agent_status("Aggressive Analyst", "in_progress")
                if con:
                    buf.update_agent_status("Conservative Analyst", "in_progress")
                if neu:
                    buf.update_agent_status("Neutral Analyst", "in_progress")

                if judge:
                    risk_emitted = True
                    buf.update_agent_status("Aggressive Analyst", "completed")
                    buf.update_agent_status("Conservative Analyst", "completed")
                    buf.update_agent_status("Neutral Analyst", "completed")
                    buf.update_agent_status("Portfolio Manager", "completed")
                    evt = {
                        "type": "risk",
                        "stage": "risk",
                        "aggressive": agg,
                        "conservative": con,
                        "neutral": neu,
                        "judge": judge,
                        "stats": get_stats_dict(stats_handler, buf, start_time),
                    }
                    state["events"].append(evt)
                    await q.put(evt)

    except Exception as e:
        print(f"[ANALYSIS] Stream error: {e}\n{_tb.format_exc()}", flush=True)
        evt = {"type": "error", "message": str(e)}
        state["events"].append(evt)
        await q.put(evt)
        state["done"] = True
        await q.put(None)
        return

    # Final decision
    if final_state:
        decision_text = final_state.get("final_trade_decision", "No decision reached.")
        signal = graph.process_signal(decision_text)
        buf.update_report_section("final_trade_decision", decision_text)
        for agent in buf.agent_status:
            buf.update_agent_status(agent, "completed")
        st = get_stats_dict(stats_handler, buf, start_time)
        for agent, status in buf.agent_status.items():
            if prev_statuses.get(agent) != "completed":
                prev_statuses[agent] = "completed"
                evt = {
                    "type": "agent_update",
                    "agent": agent,
                    "stage": _agent_stage(agent),
                    "status": "completed",
                    "stats": st,
                }
                state["events"].append(evt)
                await q.put(evt)
        evt = {
            "type": "decision",
            "stage": "decision",
            "signal": signal,
            "decision_text": decision_text,
            "stats": st,
        }
        state["events"].append(evt)
        await q.put(evt)

    state["done"] = True
    await q.put(None)  # sentinel — stream done


async def run_analysis(analysis_id: str, ticker: str, trade_date: str):
    """Background task: acquires semaphore, runs analysis with timeout."""
    state = analyses[analysis_id]
    q = state["queue"]
    async with _semaphore:
        try:
            await asyncio.wait_for(
                _run_analysis_inner(analysis_id, ticker, trade_date),
                timeout=600,  # 10 minutes
            )
        except asyncio.TimeoutError:
            print(f"[ANALYSIS] Timeout for {analysis_id}", flush=True)
            evt = {"type": "error", "message": "Analysis timed out after 10 minutes"}
            state["events"].append(evt)
            await q.put(evt)
            state["done"] = True
            await q.put(None)


# --- Memory cleanup background task ---
async def _cleanup_loop():
    """Remove analyses older than 30 minutes every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        expired = [aid for aid, s in analyses.items() if now - s["created_at"] > 1800]
        for aid in expired:
            analyses.pop(aid, None)
        if expired:
            print(f"[CLEANUP] Removed {len(expired)} expired analyses", flush=True)


@app.on_event("startup")
async def _start_cleanup():
    asyncio.create_task(_cleanup_loop())


# --- Routes ---

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def start_analysis(req: AnalyzeRequest):
    ticker = req.ticker.upper().strip()
    if not ticker or len(ticker) > 5 or not ticker.isalpha():
        raise HTTPException(400, "Invalid ticker")
    trade_date = req.date or str(date.today())
    analysis_id = str(uuid.uuid4())
    analyses[analysis_id] = {
        "queue": asyncio.Queue(),
        "events": [],
        "done": False,
        "created_at": time.time(),
    }
    asyncio.create_task(run_analysis(analysis_id, ticker, trade_date))
    return {"id": analysis_id, "ticker": ticker, "date": trade_date}


@app.get("/analyze/{analysis_id}/stream", dependencies=[Depends(verify_api_key)])
async def stream_analysis(analysis_id: str, last_event: int = 0):
    """Stream SSE events. Supports reconnection via ?last_event=N to replay missed events."""
    if analysis_id not in analyses:
        raise HTTPException(404, "Analysis not found")
    state = analyses[analysis_id]

    async def event_generator():
        idx = last_event
        # Replay any events the client missed
        while idx < len(state["events"]):
            evt = state["events"][idx]
            idx += 1
            yield {"id": str(idx), "data": json.dumps(evt)}
        # If analysis already done after replay, stop
        if state["done"]:
            return
        # Stream new events from queue
        q = state["queue"]
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "data": ""}
                continue
            if event is None:
                break
            idx += 1
            yield {"id": str(idx), "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health():
    return {"status": "ok"}
