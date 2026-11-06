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
    (AgentRole.ARCHITECT, "agent-architect", "Architect", "#7c3aed", 90),
    (AgentRole.BACKEND_ENGINEER, "agent-backend", "Backend Engineer", "#059669", 70),
    (AgentRole.FRONTEND_CODER, "agent-coder", "Frontend Coder", "#2563eb", 60),
    (AgentRole.COPYWRITER, "agent-copywriter", "Copywriter", "#0891b2", 50),
    (AgentRole.STYLER, "agent-styler", "Styler", "#db2777", 40),
    (AgentRole.CRITIC_LINTER, "agent-critic", "Critic", "#d97706", 30),
)

IDENTITIES: Final[Mapping[AgentRole, AgentIdentity]] = {
    row[0]: AgentIdentity(*row) for row in _IDENTITY_ROWS
}
