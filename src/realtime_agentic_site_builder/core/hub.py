"""Session orchestration, listener fan-out and the event bus that links workers together.

Each worker owns its sessions in memory. Local listeners are served directly, so they keep working
through a bus outage. The bus only carries events to other workers, which adopt them as replicas.
Pub/Sub delivery is at-most-once and does not arbitrate writers, so a session needs a single
writing worker at a time.
"""

import asyncio
import contextlib
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Final, Literal, Protocol, Self, cast

from pydantic import Field, StrictBool, TypeAdapter, ValidationError, model_validator
from redis.asyncio import Redis
from redis.exceptions import RedisError

from realtime_agentic_site_builder.config import Settings, get_settings
from realtime_agentic_site_builder.core.clock import Clock, system_now_ms
from realtime_agentic_site_builder.core.lock_manager import AsyncLockManager, LockEvent
from realtime_agentic_site_builder.core.patch_engine import PatchEngine, changed_elements
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.base import DomainModel, Identifier, NonNegativeInt
from realtime_agentic_site_builder.domain.errors import (
    DeltaRejectedError,
    ElementLockConflictError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SiteBuilderError,
    element_resource_uri,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentCursorFrame,
    AgentThoughtStream,
    ReplayTimeline,
    StreamDelta,
    SwarmSnapshot,
    ViewportX,
    ViewportY,
)

logger = logging.getLogger(__name__)

DEFAULT_LISTENER_CAPACITY: Final = 256
REDIS_TIMEOUT_S: Final = 2.0
REDIS_READ_POLL_S: Final = 1.0
REDIS_RETRY_DELAY_S: Final = 1.0
REDIS_HEALTH_CHECK_INTERVAL_S: Final = 30

_SESSION_CREATED_STEP: Final = "SESSION_CREATED"
_DELTA_APPLIED_STEP: Final = "DELTA_APPLIED"


class CursorEvent(DomainModel):
    kind: Literal["cursor"] = "cursor"
    origin: Identifier
    session_id: Identifier
    frame: AgentCursorFrame


class StateEvent(DomainModel):
    """A committed delta and the frame it produced, which carries the full resulting state."""

    kind: Literal["state"] = "state"
    origin: Identifier
    session_id: Identifier
    delta: StreamDelta
    snapshot: SwarmSnapshot

    @model_validator(mode="after")
    def _require_own_delta(self) -> Self:
        if self.delta.session_id != self.session_id:
            raise ValueError(f"delta {self.delta.delta_seq} belongs to another session")
        return self


class LockNotice(DomainModel):
    kind: Literal["lock"] = "lock"
    origin: Identifier
    session_id: Identifier
    event: LockEvent


class ThoughtEvent(DomainModel):
    """One agent thought or tool call, which unlike a cursor frame is never conflated away.

    A local listener receives every one. Pub/Sub is at-most-once, so between workers ``seq`` is what
    lets a receiver notice a gap. It counts per session, and a replica discards what it already saw.
    """

    kind: Literal["thought"] = "thought"
    origin: Identifier
    session_id: Identifier
    seq: NonNegativeInt
    thought: AgentThoughtStream


class PresenceEvent(DomainModel):
    """Where a person is and what they have selected. People are not agents: they have no role and
    do not belong in a snapshot's ``active_agents``. Conflatable, like an agent's cursor.
    """

    kind: Literal["presence"] = "presence"
    origin: Identifier
    session_id: Identifier
    user_id: Identifier
    x: ViewportX = 0
    y: ViewportY = 0
    selected_id: Identifier | None = None
    active: StrictBool = True


SwarmEvent = CursorEvent | StateEvent | LockNotice | ThoughtEvent | PresenceEvent
BusHandler = Callable[[SwarmEvent], None]

_EVENT_ADAPTER: Final[TypeAdapter[SwarmEvent]] = TypeAdapter(
    Annotated[SwarmEvent, Field(discriminator="kind")]
)
_SESSION_ID: Final = TypeAdapter(Identifier)


@dataclass(frozen=True, slots=True)
class ThoughtBacklog:
    """Retained thoughts for a reader. ``gap`` means some it asked for are gone for good."""

    events: tuple[ThoughtEvent, ...]
    gap: bool
    latest_seq: int | None


def session_channel(session_id: str) -> str:
    return f"rasb:session:{session_id}"


class EventBus(Protocol):
    """Worker-to-worker transport. Handlers must not block.

    ``publish`` is best effort and does not raise on transport failure, because the state it
    announces is already committed and a caller retrying would be rejected as a duplicate.
    """

    async def publish(self, channel: str, event: SwarmEvent) -> None: ...

    async def subscribe(self, channel: str, handler: BusHandler) -> None: ...

    async def unsubscribe(self, channel: str, handler: BusHandler) -> None: ...

    async def close(self) -> None: ...


class InMemoryEventBus:
    """Delivers synchronously within the process, passing the immutable events by reference."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[BusHandler]] = {}

    async def publish(self, channel: str, event: SwarmEvent) -> None:
        for handler in tuple(self._handlers.get(channel, ())):
            handler(event)

    async def subscribe(self, channel: str, handler: BusHandler) -> None:
        self._handlers.setdefault(channel, []).append(handler)

    async def unsubscribe(self, channel: str, handler: BusHandler) -> None:
        handlers = self._handlers[channel]
        handlers.remove(handler)
        if not handlers:
            del self._handlers[channel]

    async def close(self) -> None:
        self._handlers.clear()


class RedisPubSubClient(Protocol):
    """The part of ``redis.asyncio.client.PubSub`` this bus relies on."""

    async def subscribe(self, *channels: str) -> None: ...

    def unsubscribe(self, *channels: str) -> Awaitable[object]: ...

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0
    ) -> Mapping[str, Any] | None: ...

    async def aclose(self) -> None: ...


class RedisClient(Protocol):
    """The part of ``redis.asyncio.Redis`` this bus relies on."""

    def publish(self, channel: str, message: str) -> Awaitable[int]: ...

    def pubsub(self) -> RedisPubSubClient: ...

    async def aclose(self) -> None: ...


class RedisEventBus:
    """Multiplexes all channels over one Pub/Sub connection, so sessions do not cost connections.

    The client reads raw bytes rather than decoding responses, so that a foreign publisher's invalid
    UTF-8 fails event validation instead of killing the reader task.
    """

    def __init__(self, client: RedisClient, *, retry_delay_s: float = REDIS_RETRY_DELAY_S) -> None:
        self._client = client
        self._pubsub = client.pubsub()
        self._retry_delay_s = retry_delay_s
        self._handlers: dict[str, list[BusHandler]] = {}
        self._reader: asyncio.Task[None] | None = None

    @classmethod
    def from_url(cls, url: str) -> Self:
        """Build a bus. The client connects lazily, on first use."""
        client = Redis.from_url(
            url,
            socket_connect_timeout=REDIS_TIMEOUT_S,
            socket_timeout=REDIS_TIMEOUT_S,
            health_check_interval=REDIS_HEALTH_CHECK_INTERVAL_S,
        )
        return cls(cast("RedisClient", client))

    async def publish(self, channel: str, event: SwarmEvent) -> None:
        try:
            await self._client.publish(channel, event.model_dump_json())
        except (RedisError, OSError) as error:
            logger.warning("dropping %s event on %s: %r", event.kind, channel, error)

    async def subscribe(self, channel: str, handler: BusHandler) -> None:
        """Register a handler, raising if Redis cannot take the subscription right now."""
        if channel not in self._handlers:
            await self._pubsub.subscribe(channel)
        self._handlers.setdefault(channel, []).append(handler)
        if self._reader is None:
            self._reader = asyncio.create_task(self._read_loop())

    async def unsubscribe(self, channel: str, handler: BusHandler) -> None:
        handlers = self._handlers[channel]
        handlers.remove(handler)
        if handlers:
            return
        del self._handlers[channel]
        try:
            await self._pubsub.unsubscribe(channel)
        except (RedisError, OSError) as error:
            logger.warning("could not unsubscribe from %s: %r", channel, error)

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        self._handlers.clear()
        await self._pubsub.aclose()
        await self._client.aclose()

    async def _read_loop(self) -> None:
        while True:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=REDIS_READ_POLL_S
                )
            except (RedisError, OSError) as error:
                # The client restores its subscriptions when the next read reconnects.
                logger.warning("event bus read failed: %r", error)
                await asyncio.sleep(self._retry_delay_s)
                continue
            if message is not None:
                self._deliver(message)

    def _deliver(self, message: Mapping[str, Any]) -> None:
        channel = message["channel"].decode(errors="replace")
        try:
            event = _EVENT_ADAPTER.validate_json(message["data"])
        except (ValidationError, SiteBuilderError) as error:
            logger.warning("discarding invalid event on %s: %s", channel, type(error).__name__)
            return
        for handler in tuple(self._handlers.get(channel, ())):
            handler(event)


def create_event_bus(settings: Settings) -> EventBus:
    if settings.redis_url is None:
        return InMemoryEventBus()
    return RedisEventBus.from_url(str(settings.redis_url))


class SessionListener:
    """One consumer's buffer. Offering never blocks, so a slow consumer only hurts itself.

    Cursor frames are ephemeral, so only the latest per agent is kept. State and lock events wait in
    a bounded queue that drops the oldest on overflow. That is safe for state, because every state
    event carries the full bundle. ``dropped`` counts what those two lanes lost.

    Thoughts have no such substitute, so their lane has no bound. A stalled listener therefore costs
    memory instead of data, until it closes or its session does. The lane only holds references to
    events every listener shares. Delivery order is state and locks, then thoughts, then cursors.
    """

    def __init__(self, capacity: int | None, on_close: Callable[["SessionListener"], None]) -> None:
        # No capacity means nothing is dropped, for a consumer that evicts its slow readers instead.
        self._events: deque[StateEvent | LockNotice] = deque(maxlen=capacity)
        self._thoughts: deque[ThoughtEvent] = deque()
        self._cursors: dict[str, CursorEvent | PresenceEvent] = {}
        self._wakeup = asyncio.Event()
        self._on_close = on_close
        self._closed = False
        self.dropped = 0

    def offer(self, event: SwarmEvent) -> None:
        if isinstance(event, CursorEvent | PresenceEvent):
            key = (
                event.frame.agent_id if isinstance(event, CursorEvent) else f"user:{event.user_id}"
            )
            self.dropped += key in self._cursors
            self._cursors[key] = event
        elif isinstance(event, ThoughtEvent):
            self._thoughts.append(event)
        else:
            self.dropped += len(self._events) == self._events.maxlen
            self._events.append(event)
        self._wakeup.set()

    def close(self) -> None:
        """Stop receiving. Buffered events can still be read before iteration ends."""
        self._closed = True
        self._on_close(self)
        self._wakeup.set()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> SwarmEvent:
        while (event := self.poll()) is None:
            if self._closed:
                raise StopAsyncIteration
            await self.wait()
        return event

    def poll(self) -> SwarmEvent | None:
        """Take the next buffered event without waiting, for consumers merging other sources."""
        if self._events:
            return self._events.popleft()
        if self._thoughts:
            return self._thoughts.popleft()
        if self._cursors:
            return self._cursors.pop(next(iter(self._cursors)))
        return None

    async def wait(self) -> None:
        """Sleep until something is offered, the listener closes, or ``wake`` is called."""
        self._wakeup.clear()
        await self._wakeup.wait()

    def wake(self) -> None:
        self._wakeup.set()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def lossless_pending(self) -> int:
        """Count buffered events that must reach the reader, which excludes conflatable cursors."""
        return len(self._events) + len(self._thoughts)


class _Session:
    """A session's mutable state. Only the hub touches it, on the event loop."""

    def __init__(
        self,
        session_id: str,
        *,
        origin: str,
        first_frame: SwarmSnapshot,
        lease_ms: int,
        clock: Clock,
        capacity: int,
        backlog: int,
        now_ms: int,
    ) -> None:
        self.session_id = session_id
        self.origin = origin
        self.locks = AsyncLockManager(lease_ms=lease_ms, clock=clock, on_event=self._relay_lock)
        # Serializes writers across the await on the bus, so workers see commits in order.
        self.mutex = asyncio.Lock()
        self.closed = False
        self.last_activity_ms = now_ms
        self.listeners: set[SessionListener] = set()
        self.agents: dict[str, AgentCursorFrame] = {}
        self.deltas: list[StreamDelta] = []
        self.snapshots = [first_frame]
        self.reserved_delta_seq = -1
        self.pending_frames: dict[str, AgentCursorFrame] = {}
        self.pending_notices: deque[LockNotice] = deque(maxlen=capacity)
        # Unbounded unlike the notices: anything dropped here is lost to every other worker.
        self.pending_thoughts: deque[ThoughtEvent] = deque()
        self.thought_log: deque[ThoughtEvent] = deque(maxlen=backlog)
        self.next_thought_seq = 0
        self.pending_presence: dict[str, PresenceEvent] = {}
        self.outbox_ready = asyncio.Event()

    @property
    def latest(self) -> SwarmSnapshot:
        return self.snapshots[-1]

    @property
    def last_delta_seq(self) -> int:
        return self.deltas[-1].delta_seq if self.deltas else -1

    def fan_out(self, event: SwarmEvent) -> None:
        for listener in self.listeners:
            listener.offer(event)

    def record(self, delta: StreamDelta, snapshot: SwarmSnapshot) -> None:
        self.deltas.append(delta)
        self.snapshots.append(snapshot)

    def timeline(self) -> ReplayTimeline:
        # Built on demand: revalidating an ever-growing model per delta would cost quadratic time.
        return ReplayTimeline(
            session_id=self.session_id, deltas=tuple(self.deltas), snapshots=tuple(self.snapshots)
        )

    def require_locks(self, agent_id: str, target_ids: Iterable[str], now_ms: int) -> None:
        """Reject the agent unless it has a live hold on every target, naming the first it lacks."""
        for target_id in sorted(target_ids):
            holder = self.locks.live_holder(target_id, now_ms)
            if holder != agent_id:
                raise ElementLockConflictError(
                    competing_agent_ids=(agent_id,) if holder is None else (agent_id, holder),
                    resource_uri=element_resource_uri(target_id),
                )

    def frames_for_snapshot(self) -> tuple[AgentCursorFrame, ...]:
        """Return the latest frames, clearing locks that were released since each was sent.

        A frame is the agent's own claim and goes stale when a lock moves on. Left in place, a stale
        claim would fail the snapshot's validation and block an otherwise valid delta.
        """
        active = self.locks.active_locks()
        return tuple(
            frame
            if frame.active_lock is None or frame.active_lock in active
            else frame.model_copy(update={"active_lock": None})
            for _, frame in sorted(self.agents.items())
        )

    def _relay_lock(self, event: LockEvent) -> None:
        notice = LockNotice(origin=self.origin, session_id=self.session_id, event=event)
        self.fan_out(notice)
        self.pending_notices.append(notice)
        self.outbox_ready.set()


class SwarmSessionHub:
    """Owns the sessions of one worker: their state, locks, replay logs and listeners.

    ``sweep`` must be called periodically by whatever hosts the hub. Without it, lapsed leases are
    only reclaimed when another agent contends for them, and idle sessions stay open.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        bus: EventBus | None = None,
        clock: Clock = system_now_ms,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        listener_capacity: int = DEFAULT_LISTENER_CAPACITY,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_bus = bus is None
        self._bus = create_event_bus(self._settings) if bus is None else bus
        self._clock = clock
        self._sleep = sleep
        self._listener_capacity = listener_capacity
        self._engine = PatchEngine(
            max_patch_size_bytes=self._settings.max_patch_size_bytes, clock=clock
        )
        self._origin = uuid.uuid4().hex
        self._sessions: dict[str, _Session] = {}
        self._pumps: dict[str, asyncio.Task[None]] = {}
        self._bus_handlers: dict[str, BusHandler] = {}

    async def create_session(
        self, session_id: str, initial_bundle: FullstackProjectBundle
    ) -> SwarmSnapshot:
        """Open a session and return its first frame, the base state its replay starts from."""
        # Events embed the id, so a bad one would otherwise only surface inside the frame pump.
        _SESSION_ID.validate_python(session_id)
        if session_id in self._sessions:
            raise SessionAlreadyExistsError(session_id=session_id)
        now_ms = self._clock()
        session = _Session(
            session_id,
            origin=self._origin,
            first_frame=self._engine.generate_snapshot(
                initial_bundle, 0, _SESSION_CREATED_STEP, (), (), timestamp_ms=now_ms
            ),
            lease_ms=self._settings.lock_lease_ms,
            clock=self._clock,
            capacity=self._listener_capacity,
            backlog=self._settings.thought_backlog_size,
            now_ms=now_ms,
        )
        handler = partial(self._on_bus_event, session)
        self._sessions[session_id] = session
        self._pumps[session_id] = asyncio.create_task(self._pump(session))
        self._bus_handlers[session_id] = handler
        try:
            await self._bus.subscribe(session_channel(session_id), handler)
        except BaseException:
            await self._discard(session)
            raise
        return session.latest

    async def close_session(self, session_id: str) -> ReplayTimeline:
        """End a session once in-flight deltas finish, and return its complete replay log."""
        async with self._exclusive(session_id) as session:
            handler = await self._discard(session)
            await self._flush_thoughts(session)
            await self._bus.unsubscribe(session_channel(session_id), handler)
            return session.timeline()

    async def shutdown(self) -> None:
        """Close every session, and the bus if this hub created it."""
        for session_id in tuple(self._sessions):
            with contextlib.suppress(SessionNotFoundError):
                await self.close_session(session_id)
        if self._owns_bus:
            await self._bus.close()

    async def broadcast_frame(
        self, session_id: str, frame: AgentCursorFrame, *, touch: bool = True
    ) -> None:
        """Queue a cursor update for the session's frame pump, without waiting for any consumer.

        Pass ``touch=False`` for frames nobody acted to produce, such as a simulator's idle drift.
        Those must not keep an otherwise abandoned session from expiring.
        """
        session = self._require(session_id)
        if touch:
            session.last_activity_ms = self._clock()
        session.agents[frame.agent_id] = frame
        session.pending_frames[frame.agent_id] = frame
        session.outbox_ready.set()

    async def broadcast_thought(self, session_id: str, thought: AgentThoughtStream) -> ThoughtEvent:
        """Deliver a thought to every local listener now, and queue it for other workers.

        Nothing suspends here, so concurrent agents' thoughts get their ``seq`` in delivery order.
        """
        session = self._require(session_id)
        session.last_activity_ms = self._clock()
        event = ThoughtEvent(
            origin=self._origin,
            session_id=session_id,
            seq=session.next_thought_seq,
            thought=thought,
        )
        session.next_thought_seq += 1
        session.thought_log.append(event)
        session.fan_out(event)
        session.pending_thoughts.append(event)
        session.outbox_ready.set()
        return event

    async def apply_delta(self, session_id: str, delta: StreamDelta) -> SwarmSnapshot:
        """Commit a delta, record it for replay and announce the new state.

        Nothing about the session changes unless the delta is fully accepted. Its agent must hold
        a live lock on ``target_id``, on everything each pointer in the patch resolves to, and on
        everything the result turns out to change, which is how reordering is caught, since no
        pointer names it. Otherwise this raises ``ElementLockConflictError``. No await separates
        these checks from the commit, so a lease cannot be lost in between.
        """
        if delta.session_id != session_id:
            raise DeltaRejectedError(
                f"it is addressed to session {delta.session_id}",
                session_id=session_id,
                delta_seq=delta.delta_seq,
            )
        async with self._exclusive(session_id) as session:
            if delta.delta_seq <= session.last_delta_seq:
                raise DeltaRejectedError(
                    f"its sequence number does not follow {session.last_delta_seq}",
                    session_id=session_id,
                    delta_seq=delta.delta_seq,
                )
            now_ms = self._clock()
            session.require_locks(delta.agent_id, (delta.target_id,), now_ms)
            prior = session.latest.bundle_state
            bundle, _ = self._engine.apply_patch(
                prior,
                delta,
                authorize=lambda touched: session.require_locks(delta.agent_id, touched, now_ms),
            )
            session.require_locks(delta.agent_id, changed_elements(prior, bundle), now_ms)
            snapshot = self._engine.generate_snapshot(
                bundle,
                session.latest.frame_seq + 1,
                _DELTA_APPLIED_STEP,
                session.frames_for_snapshot(),
                session.locks.active_locks(),
                # A wall clock can step backwards, but a replay's frames must not.
                timestamp_ms=max(now_ms, session.latest.timestamp_ms),
            )
            session.record(delta, snapshot)
            session.last_activity_ms = now_ms
            event = StateEvent(
                origin=self._origin, session_id=session_id, delta=delta, snapshot=snapshot
            )
            session.fan_out(event)
            await self._bus.publish(session_channel(session_id), event)
            return snapshot

    def touch(self, session_id: str) -> None:
        """Count something a person did as activity, such as a heartbeat on an open connection."""
        self._require(session_id).last_activity_ms = self._clock()

    async def broadcast_presence(
        self,
        session_id: str,
        user_id: str,
        *,
        x: int = 0,
        y: int = 0,
        selected_id: str | None = None,
        active: bool = True,
    ) -> None:
        """Queue a person's position or selection. Like a cursor, only the latest one matters."""
        session = self._require(session_id)
        session.last_activity_ms = self._clock()
        session.pending_presence[user_id] = PresenceEvent(
            origin=self._origin,
            session_id=session_id,
            user_id=user_id,
            x=x,
            y=y,
            selected_id=selected_id,
            active=active,
        )
        session.outbox_ready.set()

    def thought_backlog(self, session_id: str, after_seq: int | None = None) -> ThoughtBacklog:
        """Return retained thoughts, all of them or only those after what a reader has seen.

        A reader claiming to be ahead of the log cannot be trusted, so it gets everything retained
        and is told about the gap. ``gap`` also reports what the ring evicted, and what another
        worker's Pub/Sub never delivered here. It is never hidden behind an apparently whole list.
        """
        session = self._require(session_id)
        retained = tuple(session.thought_log)
        latest = session.next_thought_seq - 1
        reported = None if latest < 0 else latest
        if after_seq is None or after_seq > latest:
            whole = len(retained) == latest + 1
            return ThoughtBacklog(retained, after_seq is not None or not whole, reported)
        missed = tuple(event for event in retained if event.seq > after_seq)
        return ThoughtBacklog(missed, len(missed) != latest - after_seq, reported)

    def subscribe(self, session_id: str, *, lossless: bool = False) -> SessionListener:
        """Add a listener. A lossless one drops no state, so its consumer must police its pace."""
        session = self._require(session_id)
        capacity = None if lossless else self._listener_capacity
        listener = SessionListener(capacity, session.listeners.discard)
        session.listeners.add(listener)
        return listener

    def lock_manager(self, session_id: str) -> AsyncLockManager:
        return self._require(session_id).locks

    @property
    def settings(self) -> Settings:
        return self._settings

    def now_ms(self) -> int:
        return self._clock()

    def latest_snapshot(self, session_id: str) -> SwarmSnapshot:
        return self._require(session_id).latest

    def reserve_delta_seq(self, session_id: str) -> int:
        """Return the next sequence number for concurrent writers to one session.

        Writers queue on the session mutex in arrival order, so numbers stay in commit order as long
        as the caller passes this to ``apply_delta`` without suspending in between. A rejected delta
        leaves a gap, which the log allows.
        """
        session = self._require(session_id)
        session.reserved_delta_seq = max(session.reserved_delta_seq, session.last_delta_seq) + 1
        return session.reserved_delta_seq

    def timeline(self, session_id: str) -> ReplayTimeline:
        return self._require(session_id).timeline()

    async def sweep(self, now_ms: int | None = None) -> tuple[str, ...]:
        """Reclaim lapsed locks in every session, close idle ones and return the closed ids."""
        now = self._clock() if now_ms is None else now_ms
        ttl_ms = self._settings.session_ttl_seconds * 1000
        idle: list[str] = []
        for session_id, session in tuple(self._sessions.items()):
            await session.locks.sweep_expired(now)
            if now - session.last_activity_ms >= ttl_ms:
                idle.append(session_id)
        for session_id in idle:
            with contextlib.suppress(SessionNotFoundError):
                await self.close_session(session_id)
        return tuple(idle)

    def _require(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id=session_id)
        return session

    @asynccontextmanager
    async def _exclusive(self, session_id: str) -> AsyncIterator[_Session]:
        session = self._require(session_id)
        async with session.mutex:
            # The session may have closed while this caller queued for the mutex.
            if session.closed:
                raise SessionNotFoundError(session_id=session_id)
            yield session

    async def _discard(self, session: _Session) -> BusHandler:
        """Take a session out of service and return the handler still to unsubscribe."""
        session.closed = True
        del self._sessions[session.session_id]
        pump = self._pumps.pop(session.session_id)
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
        for listener in tuple(session.listeners):
            listener.close()
        return self._bus_handlers.pop(session.session_id)

    async def _pump(self, session: _Session) -> None:
        """Flush at most ``broadcast_fps`` times a second, on the leading and the trailing edge.

        The first update after a quiet spell goes out at once. Updates during the pause conflate
        into one flush after it, so an agent's final position is never lost, and a slow bus delays
        this task alone, never the callers of ``broadcast_frame``.
        """
        channel = session_channel(session.session_id)
        interval_s = 1 / self._settings.broadcast_fps
        while True:
            await session.outbox_ready.wait()
            session.outbox_ready.clear()
            await self._flush_thoughts(session)
            frames, session.pending_frames = session.pending_frames, {}
            presence, session.pending_presence = session.pending_presence, {}
            cursors: list[CursorEvent | PresenceEvent] = [
                CursorEvent(origin=self._origin, session_id=session.session_id, frame=frame)
                for frame in frames.values()
            ]
            cursors.extend(presence.values())
            for cursor in cursors:
                session.fan_out(cursor)
            notices = tuple(session.pending_notices)
            session.pending_notices.clear()
            outgoing: tuple[SwarmEvent, ...] = (*cursors, *notices)
            for event in outgoing:
                await self._bus.publish(channel, event)
            await self._sleep(interval_s)

    async def _flush_thoughts(self, session: _Session) -> None:
        """Publish queued thoughts in order, removing each only after its publish returns.

        Being cancelled mid-publish leaves that thought queued for the flush on close. The worst
        case is a repeat, which the receiver discards by ``seq``, rather than a loss.
        """
        channel = session_channel(session.session_id)
        while session.pending_thoughts:
            await self._bus.publish(channel, session.pending_thoughts[0])
            session.pending_thoughts.popleft()

    def _on_bus_event(self, session: _Session, event: SwarmEvent) -> None:
        if event.origin == self._origin or event.session_id != session.session_id:
            return
        # A replica cannot tell a real frame from idle drift, so no cursor counts. Real work always
        # shows up as thoughts and state, and two ticking workers would otherwise never expire.
        if not isinstance(event, CursorEvent):
            session.last_activity_ms = self._clock()
        if isinstance(event, CursorEvent):
            session.agents[event.frame.agent_id] = event.frame
        elif isinstance(event, ThoughtEvent):
            if event.seq < session.next_thought_seq:
                return
            session.next_thought_seq = event.seq + 1
            session.thought_log.append(event)
        elif isinstance(event, StateEvent) and not self._adopt(session, event):
            return
        session.fan_out(event)

    @staticmethod
    def _adopt(session: _Session, event: StateEvent) -> bool:
        """Take another worker's commit into this replica, unless it would rewrite history.

        Locks are not checked here: they live on the writing worker, which already enforced them.
        """
        snapshot = event.snapshot
        if (
            event.delta.delta_seq <= session.last_delta_seq
            or snapshot.frame_seq <= session.latest.frame_seq
        ):
            logger.warning(
                "ignoring out-of-order delta %d for session %s",
                event.delta.delta_seq,
                session.session_id,
            )
            return False
        # Workers' clocks skew, but this replica's own timeline has to stay ordered.
        timestamp_ms = max(snapshot.timestamp_ms, session.latest.timestamp_ms)
        session.record(event.delta, snapshot.model_copy(update={"timestamp_ms": timestamp_ms}))
        return True
