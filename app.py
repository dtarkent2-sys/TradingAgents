"""FastAPI SSE backend for TradingAgents."""

import os
import time
import uuid
import asyncio
import json
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.stats_handler import StatsCallbackHandler
from cli.main import (
    MessageBuffer,
    classify_message_type,
    update_analyst_statuses,
)

app = FastAPI(title="TradingAgents API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active analysis state: id -> {queue, events (replay buffer), done}
analyses: dict[str, dict] = {}


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str | None = None


def build_config():
    """Build TradingAgents config for Anthropic/Claude."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = os.getenv("DEEP_THINK_MODEL", "claude-sonnet-4-6")
    config["quick_think_llm"] = os.getenv("QUICK_THINK_MODEL", "claude-haiku-4-5-20251001")
    config["backend_url"] = None
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
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


async def run_analysis(analysis_id: str, ticker: str, trade_date: str):
    """Background task that runs the TradingAgents pipeline and pushes SSE events."""
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
    except Exception as e:
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
            if chunk.get("investment_debate_state"):
                debate = chunk["investment_debate_state"]
                bull = debate.get("bull_history", "").strip()
                bear = debate.get("bear_history", "").strip()
                judge = debate.get("judge_decision", "").strip()

                if bull or bear:
                    for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                        buf.update_agent_status(a, "in_progress")

                if judge and not research_emitted:
                    research_emitted = True
                    for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                        buf.update_agent_status(a, "completed")
                    buf.update_agent_status("Trader", "in_progress")
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
                evt = {
                    "type": "trader",
                    "stage": "trading",
                    "plan": chunk["trader_investment_plan"],
                    "stats": get_stats_dict(stats_handler, buf, start_time),
                }
                state["events"].append(evt)
                await q.put(evt)

            # Risk debate
            if chunk.get("risk_debate_state"):
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

                if judge and not risk_emitted:
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
        for agent in buf.agent_status:
            buf.update_agent_status(agent, "completed")
        st = get_stats_dict(stats_handler, buf, start_time)
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


@app.post("/analyze")
async def start_analysis(req: AnalyzeRequest):
    ticker = req.ticker.upper().strip()
    if not ticker or len(ticker) > 5:
        raise HTTPException(400, "Invalid ticker")
    trade_date = req.date or str(date.today())
    analysis_id = str(uuid.uuid4())
    analyses[analysis_id] = {"queue": asyncio.Queue(), "events": [], "done": False}
    asyncio.create_task(run_analysis(analysis_id, ticker, trade_date))
    return {"id": analysis_id, "ticker": ticker, "date": trade_date}


@app.get("/analyze/{analysis_id}/stream")
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
