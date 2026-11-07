"""An exhaustive model check of the lock protocol, run on the real lock manager.

Random workloads sample the protocol. This explores all of it, for a small world: two agents and
a person, two elements, and every interleaving of acquire, release, override, a clock tick and a
sweep. It runs the production ``AsyncLockManager`` itself, cloned at every state, so there is no
model to drift from the code.

The world is finite once two things are abstracted, both soundly:

- Time. The clock starts at zero and only ever advances by ``TICK_MS``, so every timestamp is a
  multiple of it and a lock's age decides everything about it. A lock is expired once its age
  reaches ``LEASE_MS``, and stays expired, so ages are capped at the first expired multiple.
- Lamport stamps. They only ever break ties by comparison, so only their relative order matters.

Two states with the same abstraction behave the same under every future, so the search runs to a
fixpoint rather than to a depth bound. Every safety property is checked in every reachable state
and across every transition.
"""

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from realtime_agentic_site_builder.core.lock_manager import (
    AsyncLockManager,
    LockEvent,
    LockEventKind,
)
from realtime_agentic_site_builder.domain.swarm import ElementLock, LockState, resolve_deadlock
from realtime_agentic_site_builder.gateway.connection import HUMAN_OVERRIDE_PRIORITY

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

LEASE_MS: Final = 5_000
TICK_MS: Final = 3_000
AGE_CAP_MS: Final = 2 * TICK_MS  # the first multiple of TICK_MS at which a lease has lapsed
TARGETS: Final = ("x", "y")
AGENTS: Final = {"agent-a": 50, "agent-b": 70}
HUMAN: Final = "user:h"

Key = tuple[tuple[object, ...], ...]


@dataclass
class World:
    manager: AsyncLockManager
    now_ms: int


@dataclass(frozen=True)
class Action:
    name: str
    target_id: str = ""
    agent_id: str = ""


def actions() -> list[Action]:
    found = [Action("tick"), Action("sweep")]
    for target in TARGETS:
        found += [Action("acquire", target, agent) for agent in AGENTS]
        found += [Action("release", target, agent) for agent in (*AGENTS, HUMAN)]
        found.append(Action("override", target, HUMAN))
    return found


def key_of(world: World) -> Key:
    """The abstraction: what each lock is, its age capped, and the order of its Lamport stamp."""
    locks = world.manager.active_locks()
    order = {stamp: index for index, stamp in enumerate(sorted(lock.lamport_ts for lock in locks))}
    return tuple(
        (
            lock.target_id,
            lock.locked_by_agent_id,
            lock.state.value,
            lock.priority,
            min(world.now_ms - lock.acquired_at_ms, AGE_CAP_MS),
            order[lock.lamport_ts],
        )
        for lock in locks
    )


def clone(manager: AsyncLockManager) -> AsyncLockManager:
    """Copy a manager's tables. Its locks are frozen, so they are shared rather than copied."""
    copied = copy.copy(manager)
    copied._holders = dict(manager._holders)
    copied._waiters = {target: dict(queue) for target, queue in manager._waiters.items()}
    return copied


async def step(world: World, action: Action) -> tuple[World, list[LockEvent], object]:
    """Apply one action to a copy of the world, and return it with what the manager reported."""
    nxt = World(clone(world.manager), world.now_ms)
    events: list[LockEvent] = []
    nxt.manager._on_event = events.append
    nxt.manager._clock = lambda: nxt.now_ms
    manager, now = nxt.manager, nxt.now_ms
    calls: dict[str, Callable[[], Awaitable[object]]] = {
        "acquire": lambda: manager.acquire(
            action.target_id, action.agent_id, AGENTS.get(action.agent_id, 0), now
        ),
        "override": lambda: manager.override(
            action.target_id, action.agent_id, HUMAN_OVERRIDE_PRIORITY, now
        ),
        "release": lambda: manager.release(action.target_id, action.agent_id),
        "sweep": lambda: manager.sweep_expired(now),
    }
    if action.name == "tick":
        nxt.now_ms += TICK_MS
        return nxt, events, None
    return nxt, events, await calls[action.name]()


def holders(world: World) -> dict[str, ElementLock]:
    return {
        lock.target_id: lock
        for lock in world.manager.active_locks()
        if lock.state is LockState.ACQUIRED
    }


def check_state(world: World) -> None:
    locks = world.manager.active_locks()
    held = [lock.target_id for lock in locks if lock.state is LockState.ACQUIRED]
    assert len(held) == len(set(held)), f"two holders of one element: {locks}"
    assert all(lock.state in {LockState.ACQUIRED, LockState.WAITING} for lock in locks)
    for lock in locks:
        if lock.state is LockState.WAITING:
            holder = holders(world).get(lock.target_id)
            assert holder is not None, f"a queue with nobody holding the element: {lock}"
            assert holder.locked_by_agent_id != lock.locked_by_agent_id, "waiting on itself"


def check_transition(
    before: World, action: Action, after: World, events: list[LockEvent], result: object
) -> None:
    # A public call always ends with every wait cycle broken.
    if action.name != "tick":
        assert resolve_deadlock(after.manager.active_locks(), now_ms=after.now_ms) is None

    # Priority never takes a live hold. Only its holder's release, a lapse, an override or being
    # chosen to break a deadlock ends one, and the last two always tell the holder why.
    preempted = {
        (event.lock.target_id, event.lock.locked_by_agent_id)
        for event in events
        if event.kind is LockEventKind.PREEMPTED
    }
    for target_id, lock in holders(before).items():
        if lock.is_expired(after.now_ms):
            continue
        current = holders(after).get(target_id)
        if current is not None and current.locked_by_agent_id == lock.locked_by_agent_id:
            continue
        released = action.name == "release" and action.agent_id == lock.locked_by_agent_id
        assert released or (target_id, lock.locked_by_agent_id) in preempted, (
            f"{action} took {target_id} from {lock.locked_by_agent_id} unannounced"
        )
        if action.name == "override" and action.target_id == target_id:
            assert lock.priority < HUMAN_OVERRIDE_PRIORITY

    # A freed element goes to the best-ranked request that was waiting for it. A request made
    # during the same call, such as a lapsed holder asking again, joins the queue behind it.
    queued_before = {
        (lock.target_id, lock.locked_by_agent_id, lock.lamport_ts)
        for lock in before.manager.active_locks()
        if lock.state is LockState.WAITING
    }
    for event in events:
        if event.kind is LockEventKind.PROMOTED:
            lock = event.lock
            rivals = [
                waiting.rank
                for waiting in after.manager.active_locks()
                if waiting.target_id == lock.target_id
                and waiting.state is LockState.WAITING
                and (waiting.target_id, waiting.locked_by_agent_id, waiting.lamport_ts)
                in queued_before
            ]
            assert all(lock.rank > rank for rank in rivals), f"{lock} promoted over a better rival"

    if action.name == "acquire" or action.name == "override":
        assert isinstance(result, ElementLock)
        record = after.manager._record_of(action.target_id, action.agent_id)
        assert result == record or result.state is LockState.CONFLICT_RESOLVED
    if action.name == "override":
        assert isinstance(result, ElementLock)
        overridden = [event for event in events if event.kind is LockEventKind.OVERRIDDEN]
        live = holders(before).get(action.target_id)
        lower = (
            live is not None
            and not live.is_expired(before.now_ms)
            and live.locked_by_agent_id != HUMAN
            and live.priority < HUMAN_OVERRIDE_PRIORITY
        )
        # Against a live, lower holder the person takes the element at once, and everyone hears.
        assert bool(overridden) == lower
        if lower:
            assert result.state in {LockState.ACQUIRED, LockState.CONFLICT_RESOLVED}
            assert overridden[0].lock.locked_by_agent_id == HUMAN


async def test_the_lock_protocol_is_safe_in_every_reachable_state() -> None:
    start = World(AsyncLockManager(lease_ms=LEASE_MS, clock=lambda: 0), 0)
    seen: dict[Key, World] = {key_of(start): start}
    frontier = deque([start])
    transitions = 0
    kinds: set[LockEventKind] = set()
    deadlocks_broken = 0
    moves = actions()
    while frontier:
        world = frontier.popleft()
        check_state(world)
        for action in moves:
            after, events, result = await step(world, action)
            transitions += 1
            check_transition(world, action, after, events, result)
            kinds |= {event.kind for event in events}
            # Outside an override, only breaking a wait cycle preempts anyone.
            deadlocks_broken += action.name != "override" and any(
                event.kind is LockEventKind.PREEMPTED for event in events
            )
            if (key := key_of(after)) not in seen:
                seen[key] = after
                frontier.append(after)

    # The search must have reached every interesting corner, or it proves little.
    assert kinds == set(LockEventKind)
    assert deadlocks_broken > 0, "no state ever had a wait cycle to break"
    assert len(seen) > 500
    assert transitions == len(seen) * len(moves)
