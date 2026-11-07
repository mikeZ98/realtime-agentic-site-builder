"""In-memory two-phase element lock arbitration over the domain's lock and deadlock rules."""

from collections.abc import Callable
from enum import StrEnum

from realtime_agentic_site_builder.core.clock import Clock, system_now_ms
from realtime_agentic_site_builder.domain.base import DomainModel
from realtime_agentic_site_builder.domain.errors import (
    ElementLockConflictError,
    LockAcquisitionTimeoutError,
    ProblemDetails,
    element_resource_uri,
)
from realtime_agentic_site_builder.domain.swarm import (
    DEFAULT_LOCK_TIMEOUT_MS,
    DeadlockResolution,
    ElementLock,
    LockState,
    arbitrate_lock,
    resolve_deadlock,
)


class LockEventKind(StrEnum):
    PROMOTED = "PROMOTED"
    PREEMPTED = "PREEMPTED"
    TIMED_OUT = "TIMED_OUT"
    # A person took an element from a lower-priority holder, which priority alone never does.
    OVERRIDDEN = "OVERRIDDEN"


class LockEvent(DomainModel):
    """A lock change its agent did not ask for. ``problem`` tells the agent why it lost the lock."""

    kind: LockEventKind
    lock: ElementLock
    problem: ProblemDetails | None = None


class AsyncLockManager:
    """Holds one lease per element, plus a queue of contenders for each contested element.

    Public methods never suspend, so each one is atomic on the event loop without a mutex. Granting
    a lock can complete a wait cycle just as a new request can, because waiters on that element now
    wait for the new holder. Every state change therefore ends by clearing deadlocks.
    """

    def __init__(
        self,
        *,
        lease_ms: int = DEFAULT_LOCK_TIMEOUT_MS,
        clock: Clock = system_now_ms,
        on_event: Callable[[LockEvent], None] = lambda _event: None,
    ) -> None:
        self._lease_ms = lease_ms
        self._clock = clock
        self._on_event = on_event
        self._holders: dict[str, ElementLock] = {}
        self._waiters: dict[str, dict[str, ElementLock]] = {}
        self._lamport = 0

    async def acquire(
        self, target_id: str, agent_id: str, priority: int, now_ms: int
    ) -> ElementLock:
        """Return the agent's lock record for the element once arbitration has settled.

        The state is ``ACQUIRED`` when held (a holder's repeat call renews its lease), ``WAITING``
        when queued, or ``CONFLICT_RESOLVED`` when this request made the agent a deadlock victim.
        """
        self._expire_waiters(target_id, now_ms)
        holder = self._holders.get(target_id)
        if holder is not None and holder.locked_by_agent_id == agent_id:
            if not holder.is_expired(now_ms):
                renewed = holder.model_copy(update={"acquired_at_ms": now_ms})
                self._holders[target_id] = renewed
                return renewed
            self._reclaim_holder(holder, now_ms)
            holder = self._holders.get(target_id)

        queue = self._waiters.get(target_id, {})
        request = queue.get(agent_id) or self._new_lock(target_id, agent_id, priority, now_ms)
        if holder is None:
            # A released element passes straight to its best waiter, so no holder means no queue.
            granted = request.model_copy(update={"state": LockState.ACQUIRED})
            self._holders[target_id] = granted
            return granted

        arbitration = arbitrate_lock(holder, request, now_ms=now_ms)
        self._waiters.setdefault(target_id, {})[agent_id] = request
        if arbitration.request.state is LockState.ACQUIRED:
            # The lease lapsed before a sweep noticed. The queue decides who is next, not arrival.
            self._reclaim_holder(holder, now_ms)
        preempted = self._settle(now_ms)
        return self._record_of(target_id, agent_id) or preempted[target_id, agent_id]

    async def override(
        self, target_id: str, agent_id: str, priority: int, now_ms: int
    ) -> ElementLock:
        """Take an element now, preempting a live holder of lower priority.

        This is the one exception to the rule that priority never takes a lock from a live holder,
        and it exists for a person to stop an agent. The holder loses only this element, and is
        told with a ``PREEMPTED`` event. Everyone else learns of it from an ``OVERRIDDEN`` one.
        Against a holder of equal or higher priority, or no holder, this is a plain acquire.
        """
        holder = self._holders.get(target_id)
        if (
            holder is None
            or holder.locked_by_agent_id == agent_id
            or holder.is_expired(now_ms)
            or holder.priority >= priority
        ):
            return await self.acquire(target_id, agent_id, priority, now_ms)
        del self._holders[target_id]
        self._withdraw(target_id, agent_id)
        error = ElementLockConflictError(
            competing_agent_ids=(holder.locked_by_agent_id, agent_id),
            resource_uri=element_resource_uri(target_id),
        )
        revoked = holder.model_copy(update={"state": LockState.CONFLICT_RESOLVED})
        self._on_event(
            LockEvent(kind=LockEventKind.PREEMPTED, lock=revoked, problem=error.to_problem())
        )
        granted = self._new_lock(target_id, agent_id, priority, now_ms).model_copy(
            update={"state": LockState.ACQUIRED}
        )
        self._holders[target_id] = granted
        self._on_event(LockEvent(kind=LockEventKind.OVERRIDDEN, lock=granted))
        # Whatever settling then decides is the answer, exactly as for an ordinary acquire.
        preempted = self._settle(now_ms)
        return self._record_of(target_id, agent_id) or preempted[target_id, agent_id]

    async def release(self, target_id: str, agent_id: str) -> ElementLock | None:
        """Free the agent's hold and return the element's new holder, if there is one.

        Only the holder can free a lock. A queued agent that calls this withdraws its request, and
        any other caller, such as a holder whose lease was already reclaimed, changes nothing.
        """
        holder = self._holders.get(target_id)
        if holder is None or holder.locked_by_agent_id != agent_id:
            self._withdraw(target_id, agent_id)
            return None
        del self._holders[target_id]
        now_ms = self._clock()
        self._promote(target_id, now_ms)
        self._settle(now_ms)
        return self._holders.get(target_id)

    async def sweep_expired(self, now_ms: int) -> tuple[ElementLock, ...]:
        """Reclaim lapsed leases and requests, and return them as ``TIMEOUT_RELEASED``.

        Each one is reported as a ``TIMED_OUT`` event carrying its ``LockAcquisitionTimeoutError``
        problem. Raising the error itself would abandon the rest of the sweep.
        """
        reclaimed = self._sweep(now_ms)
        self._break_deadlocks(now_ms)
        return reclaimed

    def live_holder(self, target_id: str, now_ms: int) -> str | None:
        """Return who holds the element. A lapsed lease counts as nobody, even before a sweep."""
        holder = self._holders.get(target_id)
        if holder is None or holder.is_expired(now_ms):
            return None
        return holder.locked_by_agent_id

    def active_locks(self) -> tuple[ElementLock, ...]:
        """Return every hold and queued request, in an order that is stable across replicas."""
        queued = (lock for queue in self._waiters.values() for lock in queue.values())
        return tuple(
            sorted(
                (*self._holders.values(), *queued),
                key=lambda lock: (lock.target_id, lock.state.value, lock.locked_by_agent_id),
            )
        )

    def _new_lock(self, target_id: str, agent_id: str, priority: int, now_ms: int) -> ElementLock:
        self._lamport += 1
        return ElementLock(
            target_id=target_id,
            locked_by_agent_id=agent_id,
            state=LockState.WAITING,
            priority=priority,
            acquired_at_ms=now_ms,
            timeout_ms=self._lease_ms,
            lamport_ts=self._lamport,
        )

    def _record_of(self, target_id: str, agent_id: str) -> ElementLock | None:
        holder = self._holders.get(target_id)
        if holder is not None and holder.locked_by_agent_id == agent_id:
            return holder
        return self._waiters.get(target_id, {}).get(agent_id)

    def _withdraw(self, target_id: str, agent_id: str) -> None:
        queue = self._waiters.get(target_id, {})
        queue.pop(agent_id, None)
        if not queue:
            self._waiters.pop(target_id, None)

    def _report_timeout(self, lock: ElementLock) -> ElementLock:
        released = lock.model_copy(update={"state": LockState.TIMEOUT_RELEASED})
        error = LockAcquisitionTimeoutError(
            agent_id=lock.locked_by_agent_id, target_id=lock.target_id, timeout_ms=lock.timeout_ms
        )
        self._on_event(
            LockEvent(kind=LockEventKind.TIMED_OUT, lock=released, problem=error.to_problem())
        )
        return released

    def _expire_waiters(self, target_id: str, now_ms: int) -> list[ElementLock]:
        queue = self._waiters.get(target_id, {})
        expired = [lock for lock in queue.values() if lock.is_expired(now_ms)]
        for lock in expired:
            self._withdraw(target_id, lock.locked_by_agent_id)
        return [self._report_timeout(lock) for lock in expired]

    def _promote(self, target_id: str, now_ms: int) -> None:
        self._expire_waiters(target_id, now_ms)
        queue = self._waiters.get(target_id)
        if queue:
            winner = max(queue.values(), key=lambda lock: lock.rank)
            self._withdraw(target_id, winner.locked_by_agent_id)
            granted = winner.model_copy(
                update={"state": LockState.ACQUIRED, "acquired_at_ms": now_ms}
            )
            self._holders[target_id] = granted
            self._on_event(LockEvent(kind=LockEventKind.PROMOTED, lock=granted))

    def _reclaim_holder(self, holder: ElementLock, now_ms: int) -> ElementLock:
        del self._holders[holder.target_id]
        released = self._report_timeout(holder)
        self._promote(holder.target_id, now_ms)
        return released

    def _sweep(self, now_ms: int) -> tuple[ElementLock, ...]:
        reclaimed: list[ElementLock] = []
        for target_id in sorted(self._waiters):
            reclaimed.extend(self._expire_waiters(target_id, now_ms))
        for holder in sorted(self._holders.values(), key=lambda lock: lock.target_id):
            if holder.is_expired(now_ms):
                reclaimed.append(self._reclaim_holder(holder, now_ms))
        return tuple(reclaimed)

    def _settle(self, now_ms: int) -> dict[tuple[str, str], ElementLock]:
        # A lapsed lease is about to dissolve its wait edges, so no agent should be evicted over it.
        self._sweep(now_ms)
        return self._break_deadlocks(now_ms)

    def _break_deadlocks(self, now_ms: int) -> dict[tuple[str, str], ElementLock]:
        preempted: dict[tuple[str, str], ElementLock] = {}
        while (resolution := resolve_deadlock(self.active_locks(), now_ms=now_ms)) is not None:
            preempted.update(self._apply_resolution(resolution))
        return preempted

    def _apply_resolution(
        self, resolution: DeadlockResolution
    ) -> dict[tuple[str, str], ElementLock]:
        previous_holders = self._holders
        self._holders = {}
        self._waiters = {}
        promoted: list[ElementLock] = []
        victims: dict[tuple[str, str], ElementLock] = {}
        for lock in resolution.locks:
            if lock.state is LockState.ACQUIRED:
                self._holders[lock.target_id] = lock
                if previous_holders.get(lock.target_id) != lock:
                    promoted.append(lock)
            elif lock.state is LockState.WAITING:
                self._waiters.setdefault(lock.target_id, {})[lock.locked_by_agent_id] = lock
            else:
                victims[lock.target_id, lock.locked_by_agent_id] = lock

        problem = resolution.to_error().to_problem()
        for lock in victims.values():
            self._on_event(LockEvent(kind=LockEventKind.PREEMPTED, lock=lock, problem=problem))
        for lock in promoted:
            self._on_event(LockEvent(kind=LockEventKind.PROMOTED, lock=lock))
        return victims
