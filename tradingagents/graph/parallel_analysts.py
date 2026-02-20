"""Parallel analyst execution for TradingAgents.

Runs all analyst agents (Market, Social, News, Fundamentals) concurrently
instead of sequentially, cutting the analyst phase from ~8-9 min to ~2-3 min.
"""

import asyncio
from langchain_core.messages import HumanMessage, RemoveMessage


def create_parallel_analyst_node(analyst_fns, tool_nodes, selected_analysts):
    """Create a single LangGraph node that runs all analysts in parallel.

    Each analyst gets its own isolated message state and runs its complete
    tool-calling loop independently. Results are merged at the end.

    Args:
        analyst_fns: dict mapping analyst type (e.g. "market") to node function
        tool_nodes: dict mapping analyst type to ToolNode instance
        selected_analysts: list of analyst types to run
    """

    async def parallel_analysts_node(state):
        """Run all analysts concurrently and merge their reports."""

        async def run_single(analyst_type):
            """Run one analyst through its complete tool-calling loop."""
            fn = analyst_fns[analyst_type]
            tn = tool_nodes[analyst_type]

            # Each analyst gets its own isolated message state
            local_state = {
                "messages": list(state["messages"]),
                "trade_date": state["trade_date"],
                "company_of_interest": state["company_of_interest"],
            }

            result = {}
            for _ in range(10):  # safety limit on tool rounds
                result = await asyncio.to_thread(fn, local_state)
                ai_msg = result["messages"][0]
                local_state["messages"] = local_state["messages"] + [ai_msg]

                if not ai_msg.tool_calls:
                    break

                # Process tool calls
                tool_result = await asyncio.to_thread(tn.invoke, local_state)
                local_state["messages"] = (
                    local_state["messages"] + tool_result["messages"]
                )

            # Return only report fields (not messages)
            return {k: v for k, v in result.items() if k != "messages"}

        # Run all analysts concurrently
        tasks = [run_single(at) for at in selected_analysts if at in analyst_fns]
        results = await asyncio.gather(*tasks)

        # Merge all report fields
        merged = {}
        for r in results:
            merged.update(r)

        # Clear messages and add placeholder (same as Msg Clear nodes)
        messages = state.get("messages", [])
        removal_ops = [
            RemoveMessage(id=m.id)
            for m in messages
            if hasattr(m, "id") and m.id
        ]
        merged["messages"] = removal_ops + [HumanMessage(content="Continue")]

        return merged

    return parallel_analysts_node
