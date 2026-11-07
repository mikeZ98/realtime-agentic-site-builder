"""Who the swarm's agents are. Kept apart so that using this does not load the model clients."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from realtime_agentic_site_builder.domain.swarm import AgentRole


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    role: AgentRole
    agent_id: str
    name: str
    color: str
    priority: int


# A later role's work builds on an earlier one's, so the earlier role wins a deadlock.
_IDENTITY_ROWS: Final = (
    (AgentRole.ARCHITECT, "agent-architect", "Architect", "#8B5CF6", 90),
    (AgentRole.BACKEND_ENGINEER, "agent-backend", "Backend Engineer", "#10B981", 70),
    (AgentRole.FRONTEND_CODER, "agent-coder", "Frontend Coder", "#3B82F6", 60),
    (AgentRole.COPYWRITER, "agent-copywriter", "Copywriter", "#06B6D4", 50),
    (AgentRole.STYLER, "agent-styler", "Styler", "#EC4899", 40),
    (AgentRole.CRITIC_LINTER, "agent-critic", "Critic", "#F59E0B", 30),
)

IDENTITIES: Final[Mapping[AgentRole, AgentIdentity]] = {
    row[0]: AgentIdentity(*row) for row in _IDENTITY_ROWS
}
