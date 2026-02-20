"""Chainlit web UI for TradingAgents — mirrors the CLI experience."""

import os
import re
import time
import datetime
from datetime import date

import chainlit as cl

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.stats_handler import StatsCallbackHandler
from cli.main import (
    MessageBuffer,
    classify_message_type,
    update_analyst_statuses,
    update_research_team_status,
    ANALYST_ORDER,
)


def parse_ticker_date(text: str):
    """Extract ticker symbol and optional date from user message."""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    trade_date = date_match.group(1) if date_match else str(date.today())

    candidates = re.findall(r"\b([A-Z]{1,5})\b", text)
    skip = {"I", "A", "THE", "AND", "OR", "FOR", "TO", "IN", "ON", "AT", "IS",
            "IT", "OF", "BY", "AS", "AN", "BE", "IF", "SO", "DO", "MY", "UP",
            "NO", "NOT", "ALL", "BUT", "HOW", "GET", "HAS", "HAD", "CAN",
            "WHAT", "ABOUT", "BUY", "SELL", "HOLD"}
    tickers = [c for c in candidates if c not in skip]
    return tickers[0] if tickers else None, trade_date


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


def format_agent_status_table(buf):
    """Build a markdown table showing agent status (like the CLI progress panel)."""
    teams = {
        "Analyst Team": ["Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst"],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    icons = {"pending": "\u23f3", "in_progress": "\u26a1", "completed": "\u2705", "error": "\u274c"}
    lines = ["| Team | Agent | Status |", "|---|---|---|"]

    for team, agents in teams.items():
        active = [a for a in agents if a in buf.agent_status]
        for i, agent in enumerate(active):
            status = buf.agent_status.get(agent, "pending")
            icon = icons.get(status, "")
            team_col = team if i == 0 else ""
            lines.append(f"| {team_col} | {agent} | {icon} {status} |")

    return "\n".join(lines)


def format_stats(stats_handler, buf, start_time):
    """Format footer stats like the CLI."""
    s = stats_handler.get_stats()
    agents_done = sum(1 for v in buf.agent_status.values() if v == "completed")
    agents_total = len(buf.agent_status)
    reports_done = buf.get_completed_reports_count()
    reports_total = len(buf.report_sections)
    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

    return (
        f"Agents: {agents_done}/{agents_total} | "
        f"LLM: {s['llm_calls']} | Tools: {s['tool_calls']} | "
        f"Tokens: {s['tokens_in']:,}\u2191 {s['tokens_out']:,}\u2193 | "
        f"Reports: {reports_done}/{reports_total} | "
        f"\u23f1 {elapsed_str}"
    )


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "**TradingAgents** \u2014 Multi-Agent LLM Trading Analysis\n\n"
            "Send a ticker symbol to analyze. Examples:\n"
            "- `NVDA`\n"
            "- `Analyze AAPL 2024-12-01`\n"
            "- `What's the outlook for TSLA?`\n\n"
            "I'll run a team of AI analysts, researchers, traders, and risk managers "
            "to produce a trading decision."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    ticker, trade_date = parse_ticker_date(message.content)

    if not ticker:
        await cl.Message(
            content="I couldn't find a ticker symbol. Try something like `NVDA` or `Analyze AAPL 2024-12-01`."
        ).send()
        return

    # --- Build graph ---
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
        await cl.Message(content=f"Failed to initialize agents: {e}").send()
        return

    # --- Initialize message buffer (same as CLI) ---
    buf = MessageBuffer()
    buf.init_for_analysis(selected_analysts)

    # --- Status message (will be updated as agents progress) ---
    status_msg = cl.Message(content=f"**Analyzing {ticker} for {trade_date}...**\n\n{format_agent_status_table(buf)}")
    await status_msg.send()

    # --- Stream the graph ---
    init_state = graph.propagator.create_initial_state(ticker, trade_date)
    args = graph.propagator.get_graph_args(callbacks=[stats_handler])
    start_time = time.time()

    # Steps we'll create as agents complete
    analyst_steps = {}  # field -> Step
    research_step = None
    trader_step = None
    risk_step = None
    last_status_update = 0
    final_state = None

    try:
        async for chunk in graph.graph.astream(init_state, **args):
            final_state = chunk

            # --- Process messages (same as CLI lines 1024-1044) ---
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

            # --- Update analyst statuses (same as CLI line 1047) ---
            update_analyst_statuses(buf, chunk)

            # --- Emit analyst report Steps as they complete ---
            report_names = {
                "market_report": "Market Analyst",
                "sentiment_report": "Sentiment Analyst",
                "news_report": "News Analyst",
                "fundamentals_report": "Fundamentals Analyst",
            }
            for field, name in report_names.items():
                if field not in analyst_steps and chunk.get(field):
                    analyst_steps[field] = True
                    async with cl.Step(name=f"\u2705 {name} Report", type="tool") as step:
                        report = chunk[field]
                        step.output = report[:4000] if len(report) > 4000 else report

            # --- Research debate (same as CLI lines 1050-1072) ---
            if chunk.get("investment_debate_state"):
                debate = chunk["investment_debate_state"]
                bull = debate.get("bull_history", "").strip()
                bear = debate.get("bear_history", "").strip()
                judge = debate.get("judge_decision", "").strip()

                if bull or bear:
                    update_research_team_status("in_progress")
                    buf.update_report_section("investment_plan",
                        (f"### Bull Researcher\n{bull}\n\n### Bear Researcher\n{bear}") if bear else f"### Bull Researcher\n{bull}")

                if judge and not research_step:
                    research_step = True
                    buf.update_report_section("investment_plan", f"### Research Manager Decision\n{judge}")
                    update_research_team_status("completed")
                    buf.update_agent_status("Trader", "in_progress")
                    async with cl.Step(name="\u2705 Research Debate", type="tool") as step:
                        step.output = f"**Bull Case:**\n{bull}\n\n---\n\n**Bear Case:**\n{bear}\n\n---\n\n**Research Manager Decision:**\n{judge}"

            # --- Trader plan (same as CLI lines 1075-1081) ---
            if chunk.get("trader_investment_plan") and not trader_step:
                trader_step = True
                buf.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                buf.update_agent_status("Trader", "completed")
                buf.update_agent_status("Aggressive Analyst", "in_progress")
                async with cl.Step(name="\u2705 Trader Plan", type="tool") as step:
                    plan = chunk["trader_investment_plan"]
                    step.output = plan[:4000] if len(plan) > 4000 else plan

            # --- Risk debate (same as CLI lines 1084-1118) ---
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

                if judge and not risk_step:
                    risk_step = True
                    buf.update_agent_status("Aggressive Analyst", "completed")
                    buf.update_agent_status("Conservative Analyst", "completed")
                    buf.update_agent_status("Neutral Analyst", "completed")
                    buf.update_agent_status("Portfolio Manager", "completed")
                    buf.update_report_section("final_trade_decision", f"### Portfolio Manager Decision\n{judge}")

                    async with cl.Step(name="\u2705 Risk Assessment", type="tool") as step:
                        parts = []
                        if agg:
                            parts.append(f"**Aggressive Analyst:**\n{agg}")
                        if con:
                            parts.append(f"**Conservative Analyst:**\n{con}")
                        if neu:
                            parts.append(f"**Neutral Analyst:**\n{neu}")
                        parts.append(f"**Portfolio Manager Decision:**\n{judge}")
                        step.output = "\n\n---\n\n".join(parts)

            # --- Update status message periodically ---
            now = time.time()
            if now - last_status_update > 5:
                last_status_update = now
                status_msg.content = (
                    f"**Analyzing {ticker} for {trade_date}...**\n\n"
                    f"{format_agent_status_table(buf)}\n\n"
                    f"*{format_stats(stats_handler, buf, start_time)}*"
                )
                await status_msg.update()

    except Exception as e:
        await cl.Message(content=f"Error during analysis: {e}").send()
        return

    if not final_state:
        await cl.Message(content="Analysis produced no results.").send()
        return

    # --- Final decision ---
    decision_text = final_state.get("final_trade_decision", "No decision reached.")
    signal = graph.process_signal(decision_text)

    # Mark all agents completed
    for agent in buf.agent_status:
        buf.update_agent_status(agent, "completed")

    # Final status update
    status_msg.content = (
        f"**Analysis complete for {ticker} ({trade_date})**\n\n"
        f"{format_agent_status_table(buf)}\n\n"
        f"*{format_stats(stats_handler, buf, start_time)}*"
    )
    await status_msg.update()

    # Send the final decision
    await cl.Message(
        content=(
            f"## {ticker} \u2014 Trading Decision\n\n"
            f"### Signal: {signal}\n\n"
            f"---\n\n"
            f"{decision_text}"
        )
    ).send()
