"""The state LangGraph threads through the swarm."""

from operator import add
from typing import Annotated, TypedDict

from realtime_agentic_site_builder.agents.schemas import SwarmPlan
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.swarm import (
    AgentThoughtStream,
    ElementLock,
    StreamDelta,
)


class AgentWorkflowState(TypedDict):
    """Coder and backend run in the same step, so what both write needs a reducer.

    ``current_bundle`` and ``active_locks`` are only written by sequential nodes. A last-writer
    merge of two parallel views would keep whichever task LangGraph happened to order last, so
    nodes read live state from the hub and this is a view refreshed between steps.

    ``pending_deltas`` holds what the run has emitted. They are already committed, because a delta
    must land while its lock is held, and they arrive in task order rather than commit order.
    """

    session_id: str
    user_prompt: str
    current_bundle: FullstackProjectBundle
    active_locks: tuple[ElementLock, ...]
    pending_deltas: Annotated[list[StreamDelta], add]
    agent_thoughts: Annotated[list[AgentThoughtStream], add]
    critic_feedback: Annotated[list[str], add]
    iteration: int
    completed: bool
    plan: SwarmPlan
    retry_frontend: tuple[str, ...]
    retry_backend: tuple[str, ...]


def initial_state(
    session_id: str, user_prompt: str, bundle: FullstackProjectBundle
) -> AgentWorkflowState:
    return AgentWorkflowState(
        session_id=session_id,
        user_prompt=user_prompt,
        current_bundle=bundle,
        active_locks=(),
        pending_deltas=[],
        agent_thoughts=[],
        critic_feedback=[],
        iteration=0,
        completed=False,
        plan=SwarmPlan(),
        retry_frontend=(),
        retry_backend=(),
    )
