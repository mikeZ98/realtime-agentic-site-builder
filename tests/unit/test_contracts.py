"""Boundaries and contracts that mutation testing showed no other test pinned down.

Each test names the property it holds, not the mutant that prompted it: a test written only to
kill a mutant, with no property behind it, would be noise.
"""

import asyncio
import dataclasses
import json
import time
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import SecretStr

from realtime_agentic_site_builder.canvas.compositor import (
    BADGE_ENTER_MS,
    BADGE_HOLD_MS,
    MAX_SNIPPET_LENGTH,
    AgentSprite,
    BadgePhase,
    _AgentTrack,
    _Badge,
    _badge_view,
    _HumanTrack,
    _motion_fields,
    human_color,
)
from realtime_agentic_site_builder.canvas.projection import (
    CONVERGENCE_MS,
    EXTRAPOLATION_LIMIT_MS,
    MOTION_RESET_MS,
    DeadReckoner,
    Dock,
    DockSide,
    MotionState,
    _Knot,
)
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import (
    CursorEvent,
    InMemoryEventBus,
    PresenceEvent,
    SwarmSessionHub,
    ThoughtBacklog,
    ThoughtEvent,
    session_channel,
)
from realtime_agentic_site_builder.core.lock_manager import AsyncLockManager
from realtime_agentic_site_builder.core.patch_engine import (
    PatchEngine,
    PatchResult,
    bundle_size,
)
from realtime_agentic_site_builder.domain.backend import (
    BackendBlueprint,
    DatabaseDialect,
    FullstackProjectBundle,
)
from realtime_agentic_site_builder.domain.blueprint import SiteBlueprint
from realtime_agentic_site_builder.domain.errors import (
    ASTSyntaxCompilationError,
    BundleTooLargeError,
)
from realtime_agentic_site_builder.domain.swarm import (
    VIRTUAL_VIEWPORT_HEIGHT,
    VIRTUAL_VIEWPORT_WIDTH,
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    LockState,
    PatchOperation,
    StreamDelta,
    find_wait_cycle,
)
from realtime_agentic_site_builder.gateway.auth import Identity, TokenAuthenticator
from realtime_agentic_site_builder.simulation.motion import Point, Viewport
from realtime_agentic_site_builder.synthesis.exporter import ProjectArchive

WIDE = Viewport(1920, 1080)
DOCUMENT = "frontend:document"


def empty_bundle(title: str = "T") -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id="p",
        frontend=SiteBlueprint(site_id="s", title=title, theme="dark"),
        backend=BackendBlueprint(service_name="api", database_dialect=DatabaseDialect.SQLITE),
    )


def delta(*patches: dict[str, Any], seq: int = 0, session_id: str = "s") -> StreamDelta:
    return StreamDelta(
        session_id=session_id,
        agent_id="agent-coder",
        delta_seq=seq,
        target_domain="frontend",
        target_id=DOCUMENT,
        patches=tuple(PatchOperation.model_validate(patch) for patch in patches),
        emitted_at_ms=0,
    )


def retitle(title: str, seq: int = 0, session_id: str = "s") -> StreamDelta:
    patch = {"op": "replace", "path": "/frontend/title", "value": title}
    return delta(patch, seq=seq, session_id=session_id)


# --- documented values ----------------------------------------------------------------------


def test_documented_constants_hold_the_values_the_readme_promises() -> None:
    assert CONVERGENCE_MS == 120
    assert EXTRAPOLATION_LIMIT_MS == 250
    assert MAX_SNIPPET_LENGTH == 200
    assert (VIRTUAL_VIEWPORT_WIDTH, VIRTUAL_VIEWPORT_HEIGHT) == (3840, 65536)
    defaults = Settings()
    assert defaults.lock_lease_ms == 5000
    assert defaults.max_bundle_bytes == 2 * 1024 * 1024
    assert defaults.replay_history_bytes == 64 * 1024 * 1024


def test_a_persons_color_is_stable_across_releases() -> None:
    """The viewer shows it, so a change would recolor everyone already in a session."""
    assert human_color("alice") == "#98F797"
    assert human_color("bob") == "#97F7D0"


# --- exact boundaries -----------------------------------------------------------------------


def test_a_bundle_exactly_at_the_size_limit_is_accepted() -> None:
    target = retitle("Exactly")
    size = PatchEngine().apply(empty_bundle(), target).size_bytes
    assert PatchEngine(max_bundle_bytes=size).apply(empty_bundle(), target).size_bytes == size
    with pytest.raises(BundleTooLargeError):
        PatchEngine(max_bundle_bytes=size - 1).apply(empty_bundle(), target)


async def test_a_starting_bundle_exactly_at_the_size_limit_is_accepted() -> None:
    start = empty_bundle()
    size = bundle_size(start)
    secret = SecretStr("x" * 40)
    at_limit = SwarmSessionHub(Settings(gateway_secret=secret, max_bundle_bytes=size))
    await at_limit.create_session("s1", start)
    over = SwarmSessionHub(Settings(gateway_secret=secret, max_bundle_bytes=size - 1))
    with pytest.raises(BundleTooLargeError):
        await over.create_session("s1", start)
    await at_limit.shutdown()
    await over.shutdown()


def test_a_patch_exactly_at_the_size_limit_is_accepted() -> None:
    target = retitle("Fits")
    with pytest.raises(ASTSyntaxCompilationError) as caught:
        PatchEngine(max_patch_size_bytes=1).apply(empty_bundle(), target)
    assert caught.value.extensions["line_number"] == 1, "a size refusal points at the patch start"
    size = len(str(caught.value.extensions["code_snippet"]).encode())
    assert PatchEngine(max_patch_size_bytes=size).apply(empty_bundle(), target)
    with pytest.raises(ASTSyntaxCompilationError):
        PatchEngine(max_patch_size_bytes=size - 1).apply(empty_bundle(), target)


async def test_a_replay_log_exactly_at_its_budget_keeps_every_frame() -> None:
    start = empty_bundle("T0")
    engine, state, frames = PatchEngine(), start, [bundle_size(start)]
    for index in range(1, 4):
        state = engine.apply(state, retitle(f"T{index}", index - 1)).bundle
        frames.append(bundle_size(state))
    secret = SecretStr("x" * 40)
    hub = SwarmSessionHub(Settings(gateway_secret=secret, replay_history_bytes=sum(frames)))
    await hub.create_session("s1", start)
    await hub.lock_manager("s1").acquire(DOCUMENT, "agent-coder", 60, hub.now_ms())
    for index in range(1, 4):
        await hub.apply_delta("s1", retitle(f"T{index}", index - 1, session_id="s1"))
    assert len(hub.timeline("s1").snapshots) == 4
    assert hub.history_bytes("s1") == sum(frames)
    await hub.shutdown()


def test_a_silence_of_exactly_the_reset_interval_does_not_restart_a_track() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 0.0))
    tracker.observe(MOTION_RESET_MS, Point(500.0, 0.0))
    # Interpolated, so between the two: the Hermite midpoint with tangents 0 and 1 px/ms.
    # A restart would instead hold the new position, 500, for every earlier moment.
    midpoint = tracker.sample(MOTION_RESET_MS / 2, WIDE).position.x
    assert midpoint == pytest.approx(250.0 + MOTION_RESET_MS * (0.0 - 1.0) / 8)


def test_a_middle_knot_takes_the_slope_across_both_of_its_neighbours() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 0.0))
    tracker.observe(100.0, Point(100.0, 0.0))
    tracker.observe(300.0, Point(700.0, 0.0))
    assert tracker.sample(100.0, WIDE).velocity.x == pytest.approx(700.0 / 300.0)


def test_sampling_exactly_at_the_newest_knot_still_eases_the_next_correction() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 500.0))
    tracker.observe(100.0, Point(1000.0, 500.0))
    assert tracker.sample(100.0, WIDE).position.x == pytest.approx(1000.0)
    tracker.observe(133.0, Point(1000.0, 500.0))  # It stopped: the spline now bends back.
    shown = tracker.sample(101.0, WIDE).position.x
    expected = 1000.0 + 10.0 * (1.0 - 1.0 / (2 * EXTRAPOLATION_LIMIT_MS))
    assert shown == pytest.approx(expected, abs=0.01), "it continues from what was on screen"


# --- the lock protocol ----------------------------------------------------------------------


def test_a_lock_never_outranks_itself() -> None:
    lock = ElementLock(
        target_id="hero",
        locked_by_agent_id="agent-coder",
        state=LockState.WAITING,
        priority=60,
        acquired_at_ms=0,
    )
    assert not lock.outranks(lock)


async def test_lamport_timestamps_count_requests_from_one() -> None:
    manager = AsyncLockManager(clock=lambda: 0)
    first = await manager.acquire("a", "agent-coder", 60, 0)
    second = await manager.acquire("b", "agent-coder", 60, 0)
    assert (first.lamport_ts, second.lamport_ts) == (1, 2)


def test_deadlock_search_stays_linear_on_a_graph_with_exponentially_many_paths() -> None:
    """Forty levels of two agents, each waiting for both of the next: 2**40 paths, no cycle."""
    levels = 40
    agents = [(f"a{level}", f"b{level}") for level in range(levels)]
    locks = [
        ElementLock(
            target_id=f"t-{agent}",
            locked_by_agent_id=agent,
            state=LockState.ACQUIRED,
            priority=50,
            acquired_at_ms=0,
        )
        for pair in agents
        for agent in pair
    ]
    for upper, lower in zip(agents, agents[1:], strict=False):
        for waiter in upper:
            locks.extend(
                ElementLock(
                    target_id=f"t-{holder}",
                    locked_by_agent_id=waiter,
                    state=LockState.WAITING,
                    priority=50,
                    acquired_at_ms=0,
                )
                for holder in lower
            )
    started = time.perf_counter()
    assert find_wait_cycle(locks) == ()
    assert time.perf_counter() - started < 1.0


# --- patches --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/frontend/components", {}), ("/frontend", [])],
    ids=["array-against-object", "object-against-array"],
)
def test_a_test_operation_comparing_different_containers_fails_cleanly(
    path: str, expected: Any
) -> None:
    """An RFC 6902 ``test`` between an array and an object is false, never an internal error.

    Both are empty here, which is where a check on length alone would wrongly call them equal.
    """
    with pytest.raises(ASTSyntaxCompilationError, match="test"):
        PatchEngine().apply(empty_bundle(), delta({"op": "test", "path": path, "value": expected}))


def test_a_schema_violation_reports_neither_documentation_links_nor_the_input() -> None:
    with pytest.raises(ASTSyntaxCompilationError) as caught:
        PatchEngine().apply(
            empty_bundle(),
            delta({"op": "replace", "path": "/frontend/theme", "value": "secret-theme-value"}),
        )
    detail = caught.value.detail
    assert "violates its schema at frontend.theme" in detail
    assert "http" not in detail
    assert "secret-theme-value" not in detail


# --- the canvas -----------------------------------------------------------------------------


def test_badges_change_phase_exactly_at_their_boundaries() -> None:
    badge = _Badge("text", born_ms=1000.0, fade_from_ms=1000.0 + BADGE_ENTER_MS + BADGE_HOLD_MS)
    assert badge.view(1000.0) == (BadgePhase.ENTERING, 0.0)
    assert badge.view(1000.0 + BADGE_ENTER_MS / 4) == (BadgePhase.ENTERING, 0.25)
    assert badge.view(1000.0 + BADGE_ENTER_MS) == (BadgePhase.VISIBLE, 1.0)
    assert badge.view(badge.fade_from_ms) == (BadgePhase.EXITING, 1.0)


def test_motion_is_sent_rounded_to_a_tenth_with_velocity_in_pixels_per_second() -> None:
    fields = _motion_fields(MotionState(Point(12.34, -6.78), Point(0.12344, -0.06789)))
    assert fields == {"x": 12.3, "y": -6.8, "vx": 123.4, "vy": -67.9}


def test_badge_opacity_is_sent_to_three_places() -> None:
    agent = AgentSprite(
        id="agent-coder",
        label="Coder",
        role=AgentRole.FRONTEND_CODER,
        color="#3B82F6",
        x=1.0,
        y=2.0,
        vx=0.0,
        vy=0.0,
        action=AgentAction.IDLE,
    )
    assert _badge_view(agent, "t", BadgePhase.ENTERING, 1 / 3).opacity == 0.333


# --- identity, defaults and immutability ----------------------------------------------------


async def test_a_token_lasts_an_hour_unless_told_otherwise() -> None:
    authenticator = TokenAuthenticator(SecretStr("x" * 40), clock=lambda: 5_000_000)
    identity = authenticator.verify(authenticator.issue("alice"))
    assert identity.expires_at_s == 5_000 + 3600


def test_user_ids_up_to_120_characters_are_accepted() -> None:
    authenticator = TokenAuthenticator(SecretStr("x" * 40), clock=lambda: 0)
    assert authenticator.verify(authenticator.issue("x" * 120)).user_id == "x" * 120
    with pytest.raises(ValueError, match="120"):
        authenticator.issue("x" * 121)


def test_wire_defaults_are_part_of_the_schema() -> None:
    thought = AgentThoughtStream(
        agent_id="agent-coder", role=AgentRole.FRONTEND_CODER, thought="Hi", timestamp_ms=1
    )
    assert (thought.tokens_used, thought.execution_latency_ms) == (0, 0.0)
    lock = ElementLock(
        target_id="hero",
        locked_by_agent_id="agent-coder",
        state=LockState.WAITING,
        priority=60,
        acquired_at_ms=0,
    )
    assert lock.lamport_ts == 0
    presence = PresenceEvent(origin="w", session_id="s", user_id="alice")
    assert (presence.x, presence.y, presence.selected_id, presence.active) == (0, 0, None, True)


async def test_presence_defaults_to_an_active_person_at_the_origin() -> None:
    hub = SwarmSessionHub(Settings(gateway_secret=SecretStr("x" * 40)))
    await hub.create_session("s1", empty_bundle())
    listener = hub.subscribe("s1")
    await hub.broadcast_presence("s1", "alice")
    async with asyncio.timeout(2):
        while (event := listener.poll()) is None:
            await asyncio.sleep(0.001)
    assert isinstance(event, PresenceEvent)
    assert (event.x, event.y, event.active) == (0, 0, True)
    await hub.shutdown()


FROZEN: list[Callable[[], object]] = [
    lambda: Identity(user_id="alice", expires_at_s=1),
    lambda: ThoughtBacklog(events=(), gap=False, latest_seq=None),
    lambda: Dock(side=DockSide.TOP, anchor=Point(0, 0), label_at=Point(0, 0)),
    lambda: PatchResult(bundle=empty_bundle(), state_hash="h", size_bytes=1),
    lambda: ProjectArchive("a.zip", "application/zip", b""),
]
SLOTTED: list[Callable[[], object]] = [
    *FROZEN,
    lambda: _Badge("t", 0.0, 0.0),
    lambda: _Knot(0.0, Point(0, 0), Point(0, 0)),
    lambda: _HumanTrack(DeadReckoner(0.0, Point(0, 0)), None),
    lambda: _AgentTrack(None, DeadReckoner(0.0, Point(0, 0))),  # type: ignore[arg-type]
]


@pytest.mark.parametrize("build", FROZEN)
def test_values_meant_to_be_immutable_are(build: Callable[[], object]) -> None:
    """An Identity above all: a connection's identity must not change after verification."""
    value = build()
    field = dataclasses.fields(value)[0].name  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, None)


@pytest.mark.parametrize("build", SLOTTED)
def test_records_kept_per_cursor_or_value_carry_no_per_instance_dict(
    build: Callable[[], object],
) -> None:
    assert not hasattr(build(), "__dict__")


# --- copy budgets ---------------------------------------------------------------------------


def test_a_copy_that_exactly_spends_the_patch_budget_is_allowed() -> None:
    """Copies can double a document, so what they copy counts against the patch size limit."""
    copy = delta({"op": "copy", "from": "/frontend/title", "path": "/frontend/title"})
    encoded = len(json.dumps([op.model_dump() for op in copy.patches], separators=(",", ":")))
    copied = len(json.dumps("T"))
    exact = encoded + copied
    assert PatchEngine(max_patch_size_bytes=exact).apply(empty_bundle(), copy)
    with pytest.raises(ASTSyntaxCompilationError, match="copied data exceeds"):
        PatchEngine(max_patch_size_bytes=exact - 1).apply(empty_bundle(), copy)


# --- listeners, backlogs and sequence numbers -----------------------------------------------


def secret_settings(**overrides: Any) -> Settings:
    return Settings(gateway_secret=SecretStr("x" * 40), **overrides)


async def committed_state_event(hub: SwarmSessionHub, session_id: str = "s1") -> Any:
    """Commit one title change, and return the state event a listener received for it."""
    listener = hub.subscribe(session_id, lossless=True)
    await hub.lock_manager(session_id).acquire(DOCUMENT, "agent-coder", 60, hub.now_ms())
    seq = hub.reserve_delta_seq(session_id)
    await hub.apply_delta(session_id, retitle(f"T{seq}", seq, session_id=session_id))
    event = listener.poll()
    listener.close()
    return event


async def test_a_newer_cursor_frame_counts_as_dropping_the_one_it_replaced() -> None:
    hub = SwarmSessionHub(secret_settings())
    await hub.create_session("s1", empty_bundle())
    listener = hub.subscribe("s1")
    for agent_id, x in (("agent-coder", 1), ("agent-coder", 2), ("agent-styler", 3)):
        listener.offer(
            CursorEvent(
                origin="w",
                session_id="s1",
                frame=AgentCursorFrame(
                    agent_id=agent_id,
                    name=agent_id,
                    role=AgentRole.FRONTEND_CODER,
                    color="#3B82F6",
                    x=x,
                    y=0,
                ),
            )
        )
    assert listener.dropped == 1, "only the replaced frame, not the first of another agent"
    await hub.shutdown()


async def test_pending_lossless_events_are_counted_across_both_lanes() -> None:
    hub = SwarmSessionHub(secret_settings())
    await hub.create_session("s1", empty_bundle())
    state = await committed_state_event(hub)
    listener = hub.subscribe("s1", lossless=True)
    listener.offer(state)
    await hub.broadcast_thought(
        "s1",
        AgentThoughtStream(
            agent_id="agent-coder", role=AgentRole.FRONTEND_CODER, thought="Hi", timestamp_ms=1
        ),
    )
    assert listener.lossless_pending == 2
    await hub.shutdown()


async def test_the_first_thought_is_reported_as_sequence_zero() -> None:
    hub = SwarmSessionHub(secret_settings())
    await hub.create_session("s1", empty_bundle())
    assert hub.thought_backlog("s1").latest_seq is None
    await hub.broadcast_thought(
        "s1",
        AgentThoughtStream(
            agent_id="agent-coder", role=AgentRole.FRONTEND_CODER, thought="Hi", timestamp_ms=1
        ),
    )
    assert hub.thought_backlog("s1").latest_seq == 0
    await hub.shutdown()


async def test_reserved_delta_numbers_are_consecutive() -> None:
    hub = SwarmSessionHub(secret_settings())
    await hub.create_session("s1", empty_bundle())
    assert [hub.reserve_delta_seq("s1") for _ in range(3)] == [0, 1, 2]
    await hub.shutdown()


# --- replicas -------------------------------------------------------------------------------


async def replica_pair(clock: Callable[[], int] = lambda: 0) -> tuple[Any, Any, Any]:
    bus = InMemoryEventBus()
    writer = SwarmSessionHub(secret_settings(), bus=bus)
    replica = SwarmSessionHub(secret_settings(), bus=bus, clock=clock)
    for hub in (writer, replica):
        await hub.create_session("s1", empty_bundle())
    return bus, writer, replica


async def test_a_replica_counts_real_work_as_activity_but_not_cursor_drift() -> None:
    now = [0]
    bus, writer, replica = await replica_pair(lambda: now[0])
    session = replica._sessions["s1"]
    now[0] = 1_000
    frame = AgentCursorFrame(
        agent_id="agent-coder",
        name="Coder",
        role=AgentRole.FRONTEND_CODER,
        color="#3B82F6",
        x=1,
        y=1,
    )
    await bus.publish(session_channel("s1"), CursorEvent(origin="w", session_id="s1", frame=frame))
    assert session.last_activity_ms == 0, "a ticking simulator must not keep a session alive"
    thought = AgentThoughtStream(
        agent_id="agent-coder", role=AgentRole.FRONTEND_CODER, thought="Hi", timestamp_ms=1
    )
    event = ThoughtEvent(origin="w", session_id="s1", seq=0, thought=thought)
    await bus.publish(session_channel("s1"), event)
    assert session.last_activity_ms == 1_000
    await writer.shutdown()
    await replica.shutdown()


async def test_a_replica_refuses_a_repeated_or_reordered_commit() -> None:
    bus, writer, replica = await replica_pair()
    first = await committed_state_event(writer)
    assert len(replica.timeline("s1").deltas) == 1, "the commit itself arrives over the bus"
    channel = session_channel("s1")

    await bus.publish(channel, first.model_copy(update={"origin": "elsewhere"}))
    assert len(replica.timeline("s1").deltas) == 1, "a repeat of the same commit"

    later_delta = first.delta.model_copy(update={"delta_seq": 5})
    same_frame = first.model_copy(update={"origin": "elsewhere", "delta": later_delta})
    await bus.publish(channel, same_frame)
    assert len(replica.timeline("s1").deltas) == 1, "a newer delta cannot reuse a frame number"

    later_snapshot = first.snapshot.model_copy(update={"frame_seq": 9})
    stale_delta = first.model_copy(update={"origin": "elsewhere", "snapshot": later_snapshot})
    await bus.publish(channel, stale_delta)
    assert len(replica.timeline("s1").deltas) == 1, "an old delta cannot claim a newer frame"
    await writer.shutdown()
    await replica.shutdown()
