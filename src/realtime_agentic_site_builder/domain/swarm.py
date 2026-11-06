"""Swarm telemetry, the element lock engine and the event-sourced replay protocol."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    AfterValidator,
    AliasChoices,
    Field,
    StringConstraints,
    model_serializer,
    model_validator,
)

from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.base import (
    DomainModel,
    FrozenJson,
    FrozenJsonObject,
    Identifier,
    NonNegativeInt,
    SanitizedText,
    SanitizedTitle,
    Sha256Hex,
    UnixMillis,
    require_unique,
    thaw_json,
)
from realtime_agentic_site_builder.domain.errors import (
    ElementLockConflictError,
    LockAcquisitionTimeoutError,
    ReplayStateDivergenceError,
    element_resource_uri,
)

VIRTUAL_VIEWPORT_WIDTH: Final = 3840
VIRTUAL_VIEWPORT_HEIGHT: Final = 65536
DEFAULT_LOCK_TIMEOUT_MS: Final = 5000
MAX_LOCK_TIMEOUT_MS: Final = 30000


class AgentRole(StrEnum):
    ARCHITECT = "ARCHITECT"
    FRONTEND_CODER = "FRONTEND_CODER"
    STYLER = "STYLER"
    COPYWRITER = "COPYWRITER"
    CRITIC_LINTER = "CRITIC_LINTER"
    OPTIMIZER = "OPTIMIZER"
    BACKEND_ENGINEER = "BACKEND_ENGINEER"
    DATABASE_ARCHITECT = "DATABASE_ARCHITECT"


class AgentAction(StrEnum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    THINKING = "THINKING"
    ACQUIRING_LOCK = "ACQUIRING_LOCK"
    EDITING = "EDITING"
    BLOCKED_WAITING = "BLOCKED_WAITING"
    PATCHING_AST = "PATCHING_AST"
    COMPILING_BACKEND = "COMPILING_BACKEND"
    RESOLVING_CONFLICT = "RESOLVING_CONFLICT"


class LockState(StrEnum):
    UNLOCKED = "UNLOCKED"
    ACQUIRED = "ACQUIRED"
    WAITING = "WAITING"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"
    TIMEOUT_RELEASED = "TIMEOUT_RELEASED"


class JsonPatchOp(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"


_VALUE_OPS: Final = frozenset({JsonPatchOp.ADD, JsonPatchOp.REPLACE, JsonPatchOp.TEST})
_FROM_OPS: Final = frozenset({JsonPatchOp.MOVE, JsonPatchOp.COPY})


def _clamp_x(value: int) -> int:
    return min(max(value, 0), VIRTUAL_VIEWPORT_WIDTH)


def _clamp_y(value: int) -> int:
    return min(max(value, 0), VIRTUAL_VIEWPORT_HEIGHT)


# Whitespace is significant in RFC 6901 pointers, so the model-wide stripping is switched off.
JsonPointer = Annotated[
    str, StringConstraints(strip_whitespace=False, pattern=r"^(?:/(?:[^~/]|~[01])*)*$")
]
ViewportX = Annotated[int, Field(strict=True), AfterValidator(_clamp_x)]
ViewportY = Annotated[int, Field(strict=True), AfterValidator(_clamp_y)]


class AgentThoughtStream(DomainModel):
    agent_id: Identifier
    role: AgentRole
    thought: SanitizedText
    tool_name: (
        Annotated[str, Field(max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")] | None
    ) = None
    tool_args: FrozenJsonObject | None = None
    tokens_used: NonNegativeInt = 0
    execution_latency_ms: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
    timestamp_ms: UnixMillis

    @model_validator(mode="after")
    def _require_tool_for_args(self) -> Self:
        if self.tool_args is not None and self.tool_name is None:
            raise ValueError("tool_args requires a tool_name")
        return self


class ElementLock(DomainModel):
    """One agent's hold on, or queued request for, an element.

    For a ``WAITING`` lock, ``acquired_at_ms`` is when the request was made, so the same deadline
    bounds both how long a lock may be held and how long a request may wait.
    """

    target_id: Identifier
    locked_by_agent_id: Identifier
    state: LockState
    priority: Annotated[int, Field(strict=True, ge=1, le=100)]
    acquired_at_ms: UnixMillis
    timeout_ms: Annotated[int, Field(strict=True, ge=1, le=MAX_LOCK_TIMEOUT_MS)] = (
        DEFAULT_LOCK_TIMEOUT_MS
    )
    lamport_ts: NonNegativeInt = 0

    @property
    def expires_at_ms(self) -> int:
        return self.acquired_at_ms + self.timeout_ms

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_at_ms

    @property
    def rank(self) -> tuple[int, int, str]:
        """Total order for arbitration: higher priority, then older Lamport time, then agent id.

        The Lamport timestamp breaks ties rather than wall time, which skews between workers. The
        agent id makes the order total, so every replica picks the same winner.
        """
        return (self.priority, -self.lamport_ts, self.locked_by_agent_id)

    def outranks(self, other: "ElementLock") -> bool:
        return self.rank > other.rank


class LockArbitration(DomainModel):
    held: ElementLock
    request: ElementLock


def arbitrate_lock(held: ElementLock, request: ElementLock, *, now_ms: int) -> LockArbitration:
    """Decide a request for an element against its current lock record.

    Priority alone never preempts a live holder. It only matters once a deadlock forms, which
    :func:`resolve_deadlock` handles.
    """
    if held.target_id != request.target_id:
        raise ValueError("arbitration requires both locks to target the same element")
    if held.locked_by_agent_id == request.locked_by_agent_id:
        raise ValueError("an agent cannot contend with its own lock")
    if request.state is not LockState.WAITING:
        raise ValueError("a lock request must be in the WAITING state")
    if request.is_expired(now_ms):
        raise LockAcquisitionTimeoutError(
            agent_id=request.locked_by_agent_id,
            target_id=request.target_id,
            timeout_ms=request.timeout_ms,
        )
    if held.state is LockState.ACQUIRED:
        if not held.is_expired(now_ms):
            return LockArbitration(held=held, request=request)
        held = held.model_copy(update={"state": LockState.TIMEOUT_RELEASED})
    return LockArbitration(held=held, request=_grant(request, now_ms))


def _grant(lock: ElementLock, now_ms: int) -> ElementLock:
    return lock.model_copy(update={"state": LockState.ACQUIRED, "acquired_at_ms": now_ms})


def _holders_by_target(locks: Iterable[ElementLock]) -> dict[str, ElementLock]:
    holders: dict[str, ElementLock] = {}
    for lock in locks:
        if lock.state is not LockState.ACQUIRED:
            continue
        rival = holders.get(lock.target_id)
        if rival is not None:
            raise ElementLockConflictError(
                competing_agent_ids=(rival.locked_by_agent_id, lock.locked_by_agent_id),
                resource_uri=element_resource_uri(lock.target_id),
            )
        holders[lock.target_id] = lock
    return holders


def _cycle_from(
    start: str, waits_for: Mapping[str, set[str]], cleared: set[str]
) -> tuple[str, ...]:
    """Depth-first search, iterative so that a long wait chain cannot exhaust the call stack."""
    path = {start: None}  # A dict is an insertion-ordered set: fast membership, ordered cycle.
    branches = [iter(sorted(waits_for.get(start, ())))]
    while branches:
        next_agent = next(branches[-1], None)
        if next_agent is None:
            branches.pop()
            cleared.add(path.popitem()[0])
        elif next_agent in path:
            agents = list(path)
            return tuple(agents[agents.index(next_agent) :])
        elif next_agent not in cleared:
            path[next_agent] = None
            branches.append(iter(sorted(waits_for.get(next_agent, ()))))
    return ()


def find_wait_cycle(locks: Iterable[ElementLock]) -> tuple[str, ...]:
    """Return the agents of one deadlock in the wait-for graph, or an empty tuple if none exists."""
    locks = tuple(locks)
    holders = _holders_by_target(locks)
    waits_for: defaultdict[str, set[str]] = defaultdict(set)
    for lock in locks:
        holder = holders.get(lock.target_id)
        if (
            lock.state is LockState.WAITING
            and holder is not None
            and holder.locked_by_agent_id != lock.locked_by_agent_id
        ):
            waits_for[lock.locked_by_agent_id].add(holder.locked_by_agent_id)
    cleared: set[str] = set()
    for agent in sorted(waits_for):
        cycle = _cycle_from(agent, waits_for, cleared)
        if cycle:
            return cycle
    return ()


class DeadlockResolution(DomainModel):
    cycle: tuple[Identifier, ...]
    victim_agent_id: Identifier
    released_target_ids: tuple[Identifier, ...]
    locks: tuple[ElementLock, ...]

    def to_error(self) -> ElementLockConflictError:
        """Build the error that tells the victim to drop its work and retry."""
        return ElementLockConflictError(
            competing_agent_ids=self.cycle,
            resource_uri=element_resource_uri(self.released_target_ids[0]),
        )


def resolve_deadlock(locks: Sequence[ElementLock], *, now_ms: int) -> DeadlockResolution | None:
    """Break one deadlock by preempting its lowest-ranked agent, or return ``None`` if none exists.

    The victim gives up everything it holds or waits for, as a two-phase locking abort requires, and
    each released element passes to its highest-ranked waiter. Call again until this returns
    ``None`` to clear several independent deadlocks.
    """
    cycle = find_wait_cycle(locks)
    if not cycle:
        return None

    active = (LockState.ACQUIRED, LockState.WAITING)

    def agent_rank(agent: str) -> tuple[int, int, str]:
        return max(
            lock.rank for lock in locks if lock.locked_by_agent_id == agent and lock.state in active
        )

    victim = min(cycle, key=agent_rank)
    released = sorted(
        lock.target_id
        for lock in locks
        if lock.locked_by_agent_id == victim and lock.state is LockState.ACQUIRED
    )

    grants: list[ElementLock] = []
    for target_id in released:
        waiters = [
            lock
            for lock in locks
            if lock.target_id == target_id
            and lock.state is LockState.WAITING
            and lock.locked_by_agent_id != victim
        ]
        if waiters:
            grants.append(max(waiters, key=lambda lock: lock.rank))

    def after_resolution(lock: ElementLock) -> ElementLock:
        if lock.locked_by_agent_id == victim and lock.state in active:
            return lock.model_copy(update={"state": LockState.CONFLICT_RESOLVED})
        return _grant(lock, now_ms) if lock in grants else lock

    return DeadlockResolution(
        cycle=cycle,
        victim_agent_id=victim,
        released_target_ids=tuple(released),
        locks=tuple(after_resolution(lock) for lock in locks),
    )


class AgentCursorFrame(DomainModel):
    agent_id: Identifier
    name: SanitizedTitle
    role: AgentRole
    color: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
    x: ViewportX
    y: ViewportY
    current_action: AgentAction = AgentAction.IDLE
    current_thought: AgentThoughtStream | None = None
    active_lock: ElementLock | None = None

    @model_validator(mode="after")
    def _require_own_state(self) -> Self:
        thought = self.current_thought
        if thought is not None and (thought.agent_id != self.agent_id or thought.role != self.role):
            raise ValueError(f"current_thought does not belong to agent {self.agent_id}")
        lock = self.active_lock
        if lock is not None and lock.locked_by_agent_id != self.agent_id:
            raise ValueError(f"active_lock does not belong to agent {self.agent_id}")
        return self


class PatchOperation(DomainModel):
    """One RFC 6902 operation.

    Beyond ``op``, ``path`` and ``value``, RFC 6902 requires a ``from`` member for ``move`` and
    ``copy``. ``value`` is tracked by presence because ``null`` is a legal value, distinct from
    the member being absent.
    """

    op: JsonPatchOp
    path: JsonPointer
    value: FrozenJson = None
    from_: JsonPointer | None = Field(default=None, validation_alias=AliasChoices("from", "from_"))

    @model_validator(mode="after")
    def _require_rfc6902_members(self) -> Self:
        requires_value = self.op in _VALUE_OPS
        if ("value" in self.model_fields_set) != requires_value:
            rule = "required" if requires_value else "not allowed"
            raise ValueError(f"'value' is {rule} on '{self.op.value}'")
        requires_from = self.op in _FROM_OPS
        if (self.from_ is not None) != requires_from:
            rule = "required" if requires_from else "not allowed"
            raise ValueError(f"'from' is {rule} on '{self.op.value}'")
        if (
            self.op is JsonPatchOp.MOVE
            and self.from_ is not None
            and self.path.startswith(f"{self.from_}/")
        ):
            raise ValueError("'move' cannot relocate a location into one of its own children")
        return self

    @model_serializer
    def _to_rfc6902(self) -> dict[str, Any]:
        document: dict[str, Any] = {"op": self.op.value, "path": self.path}
        if self.op in _VALUE_OPS:
            document["value"] = thaw_json(self.value)
        if self.op in _FROM_OPS:
            document["from"] = self.from_
        return document


class StreamDelta(DomainModel):
    session_id: Identifier
    agent_id: Identifier
    delta_seq: NonNegativeInt
    target_domain: Literal["frontend", "backend"]
    target_id: Identifier
    patches: Annotated[tuple[PatchOperation, ...], Field(min_length=1)]
    emitted_at_ms: UnixMillis


class SwarmSnapshot(DomainModel):
    frame_seq: NonNegativeInt
    step_name: Annotated[str, Field(max_length=128, pattern=r"^[A-Z][A-Z0-9_]*$")]
    timestamp_ms: UnixMillis
    active_agents: tuple[AgentCursorFrame, ...] = ()
    active_locks: tuple[ElementLock, ...] = ()
    bundle_state: FullstackProjectBundle
    bundle_hash: Sha256Hex

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        self.verify_replayed_state(self.bundle_state)
        require_unique((agent.agent_id for agent in self.active_agents), "agent ids in a frame")
        _holders_by_target(self.active_locks)
        for agent in self.active_agents:
            if agent.active_lock is not None and agent.active_lock not in self.active_locks:
                raise ValueError(f"lock of agent {agent.agent_id} is missing from active_locks")
        return self

    def verify_replayed_state(self, replayed: FullstackProjectBundle) -> None:
        """Compare state rebuilt during a replay scrub against the hash recorded for this frame."""
        computed = replayed.state_hash()
        if computed != self.bundle_hash:
            raise ReplayStateDivergenceError(
                frame_seq=self.frame_seq, expected_hash=self.bundle_hash, computed_hash=computed
            )

    @classmethod
    def capture(
        cls,
        *,
        frame_seq: int,
        step_name: str,
        timestamp_ms: int,
        bundle_state: FullstackProjectBundle,
        active_agents: tuple[AgentCursorFrame, ...] = (),
        active_locks: tuple[ElementLock, ...] = (),
    ) -> Self:
        """Record a frame, deriving its hash from the given bundle."""
        return cls(
            frame_seq=frame_seq,
            step_name=step_name,
            timestamp_ms=timestamp_ms,
            active_agents=active_agents,
            active_locks=active_locks,
            bundle_state=bundle_state,
            bundle_hash=bundle_state.state_hash(),
        )


class ReplayTimeline(DomainModel):
    """A session's event log. Individual records cannot enforce ordering between each other."""

    session_id: Identifier
    deltas: tuple[StreamDelta, ...] = ()
    snapshots: tuple[SwarmSnapshot, ...] = ()

    @model_validator(mode="after")
    def _require_monotonic_order(self) -> Self:
        for delta in self.deltas:
            if delta.session_id != self.session_id:
                raise ValueError(f"delta {delta.delta_seq} belongs to another session")
        for earlier, later in pairwise(self.deltas):
            if later.delta_seq <= earlier.delta_seq:
                raise ValueError(f"delta_seq {later.delta_seq} is not monotonic")
        for frame, next_frame in pairwise(self.snapshots):
            if next_frame.frame_seq <= frame.frame_seq:
                raise ValueError(f"frame_seq {next_frame.frame_seq} is not monotonic")
            if next_frame.timestamp_ms < frame.timestamp_ms:
                raise ValueError(f"frame {next_frame.frame_seq} moves backwards in time")
        return self
