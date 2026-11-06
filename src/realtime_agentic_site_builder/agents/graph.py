"""Wires the agents into a LangGraph state machine and reports their transitions live."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from realtime_agentic_site_builder.agents.nodes import (
    backend_node,
    coder_node,
    critic_node,
    planner_node,
    styler_node,
)
from realtime_agentic_site_builder.agents.provider import LLMProvider
from realtime_agentic_site_builder.agents.runtime import NodeContext, SwarmRuntime
from realtime_agentic_site_builder.agents.state import AgentWorkflowState
from realtime_agentic_site_builder.core.hub import SwarmSessionHub
from realtime_agentic_site_builder.domain.errors import SessionNotFoundError
from realtime_agentic_site_builder.domain.swarm import AgentAction, AgentRole

PLANNER: Final = "planner"
CODER: Final = "coder"
BACKEND: Final = "backend"
STYLER: Final = "styler"
CRITIC: Final = "critic"
_STEPS_PER_ITERATION: Final = 3

Node = Callable[[AgentWorkflowState, NodeContext], Awaitable[dict[str, Any]]]


class GraphNode(Protocol):
    """LangGraph's signature for a node, whose parameter must be named ``state``."""

    async def __call__(self, state: AgentWorkflowState) -> dict[str, Any]: ...


SwarmGraph = CompiledStateGraph[AgentWorkflowState, None, AgentWorkflowState, AgentWorkflowState]


def _instrument(
    runtime: SwarmRuntime, name: str, role: AgentRole, node: Node, *, parallel: bool = False
) -> GraphNode:
    """Report entering and leaving a node, whatever way it ends, and merge what it gathered.

    Parallel nodes leave ``current_bundle`` and ``active_locks`` alone. Those have no reducer, so
    two writers in one step would be an error, and the next sequential node refreshes them anyway.
    """

    async def run(state: AgentWorkflowState) -> dict[str, Any]:
        context = NodeContext(runtime, state, role)
        await context.emit(
            AgentAction.NAVIGATING,
            f"{context.identity.name} started {name}.",
            tool_name="graph.enter",
            tool_args={"node": name, "iteration": state["iteration"]},
        )
        try:
            extra = await node(state, context)
        finally:
            # A session closed under the run must not replace the error that ended it.
            with contextlib.suppress(SessionNotFoundError):
                await context.emit(
                    AgentAction.IDLE,
                    f"{context.identity.name} finished {name}.",
                    tool_name="graph.exit",
                    tool_args={"node": name},
                )
        update = {**context.update(), **extra}
        if not parallel:
            session_id = state["session_id"]
            update["current_bundle"] = context.bundle()
            update["active_locks"] = runtime.hub.lock_manager(session_id).active_locks()
        return update

    return run


def route_after_critic(state: AgentWorkflowState) -> Sequence[str] | str:
    """Send each kind of rejected element to the agent that owns it, or finish."""
    targets = []
    if state["retry_frontend"]:
        targets.append(CODER)
    if state["retry_backend"]:
        targets.append(BACKEND)
    return targets or END


def create_swarm_graph(
    hub: SwarmSessionHub,
    llm_provider: LLMProvider,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SwarmGraph:
    """Build the planner -> (coder || backend) -> styler -> critic loop.

    Coder and backend share one step, and both have a plain edge to the styler, which therefore runs
    once whether one or both of them were scheduled. A join edge would wait forever on a retry that
    only needs one. The session must already exist on the hub.
    """
    runtime = SwarmRuntime(hub, llm_provider, sleep=sleep)
    builder = StateGraph(AgentWorkflowState)
    builder.add_node(PLANNER, _instrument(runtime, PLANNER, AgentRole.ARCHITECT, planner_node))
    builder.add_node(
        CODER, _instrument(runtime, CODER, AgentRole.FRONTEND_CODER, coder_node, parallel=True)
    )
    builder.add_node(
        BACKEND,
        _instrument(runtime, BACKEND, AgentRole.BACKEND_ENGINEER, backend_node, parallel=True),
    )
    builder.add_node(STYLER, _instrument(runtime, STYLER, AgentRole.STYLER, styler_node))
    builder.add_node(CRITIC, _instrument(runtime, CRITIC, AgentRole.CRITIC_LINTER, critic_node))
    builder.add_edge(START, PLANNER)
    builder.add_edge(PLANNER, CODER)
    builder.add_edge(PLANNER, BACKEND)
    builder.add_edge(CODER, STYLER)
    builder.add_edge(BACKEND, STYLER)
    builder.add_edge(STYLER, CRITIC)
    builder.add_conditional_edges(CRITIC, route_after_critic, [CODER, BACKEND, END])
    graph = builder.compile()
    limit = 2 + _STEPS_PER_ITERATION * hub.settings.agent_max_iterations
    return graph.with_config(recursion_limit=limit)
