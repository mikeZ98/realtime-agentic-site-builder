import asyncio
import base64
import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.agents.provider import MockLLMProvider
from realtime_agentic_site_builder.canvas.compositor import (
    BADGE_ENTER_MS,
    BADGE_EXIT_MS,
    BADGE_HOLD_MS,
    MAX_SNIPPET_LENGTH,
    MAX_THOUGHT_HISTORY,
    SWARM_PALETTE,
    BadgePhase,
    CanvasCompositor,
    OverlayState,
    VisualFrame,
    human_color,
    participant,
    thought_entry,
)
from realtime_agentic_site_builder.canvas.projection import (
    BADGE_GAP_PX,
    CONVERGENCE_MS,
    DOCK_HYSTERESIS_PX,
    EXTRAPOLATION_LIMIT_MS,
    MAX_SPEED_PX_PER_MS,
    MOTION_RESET_MS,
    SMOOTHING_DELAY_MS,
    DeadReckoner,
    DockSide,
    box_gap,
    elastic_viewport,
    hermite_controls,
    normalize_box,
    resolve_dock,
)
from realtime_agentic_site_builder.canvas.viewer import (
    CLIENT_BUFFER_MS,
    CONTENT_SECURITY_POLICY,
    HEARTBEAT_MS,
    MAX_VIEWER_BYTES,
    VIEWER_BYTES,
    VIEWER_HTML,
)
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import (
    CursorEvent,
    LockNotice,
    PresenceEvent,
    StateEvent,
    ThoughtEvent,
)
from realtime_agentic_site_builder.core.lock_manager import (
    AsyncLockManager,
    LockEvent,
    LockEventKind,
)
from realtime_agentic_site_builder.domain.backend import (
    BackendBlueprint,
    DatabaseDialect,
    FullstackProjectBundle,
)
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    ComponentType,
    SiteBlueprint,
)
from realtime_agentic_site_builder.domain.swarm import (
    VIRTUAL_VIEWPORT_HEIGHT,
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    LockState,
    PatchOperation,
    StreamDelta,
    SwarmSnapshot,
)
from realtime_agentic_site_builder.gateway.auth import SUBPROTOCOL, TOKEN_SUBPROTOCOL_PREFIX
from realtime_agentic_site_builder.gateway.service import GatewayService, create_app
from realtime_agentic_site_builder.simulation.motion import Point, Viewport, cubic_bezier

SESSIONS = "/api/v1/sessions"
WAIT_S = 2.0
ALICE = "alice"
BOB = "bob"
TICK_MS = 1000 / 30
Message = dict[str, Any]
WIDE = Viewport(1920, 1080)


def box(x: int = 0, y: int = 0, width: int = 100, height: int = 50) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def node(component_id: str, bounds: BoundingBox) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.FEATURES,
        title=f"Title {component_id}",
        code="<p>x</p>",
        box=bounds,
    )


def bundle_with(*components: ComponentNode) -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id="p",
        frontend=SiteBlueprint(site_id="s", title="T", theme="dark", components=components),
        backend=BackendBlueprint(service_name="api", database_dialect=DatabaseDialect.SQLITE),
    )


def page() -> FullstackProjectBundle:
    return bundle_with(
        node("hero", box(0, 0, 1000, 200)),
        node("features", box(0, 300, 1000, 200)),
    )


def agent_frame(
    role: AgentRole = AgentRole.FRONTEND_CODER,
    x: int = 500,
    y: int = 100,
    action: AgentAction = AgentAction.IDLE,
) -> AgentCursorFrame:
    identity = IDENTITIES[role]
    return AgentCursorFrame(
        agent_id=identity.agent_id,
        name=identity.name,
        role=role,
        color=identity.color,
        x=x,
        y=y,
        current_action=action,
    )


def thought(
    seq: int = 0, role: AgentRole = AgentRole.FRONTEND_CODER, text: str = "Thinking"
) -> ThoughtEvent:
    return ThoughtEvent(
        origin="w",
        session_id="s",
        seq=seq,
        thought=AgentThoughtStream(
            agent_id=IDENTITIES[role].agent_id,
            role=role,
            thought=text,
            tool_name="llm.generate",
            timestamp_ms=1,
        ),
    )


def presence(
    user_id: str = ALICE,
    x: int = 10,
    y: int = 20,
    *,
    active: bool = True,
    selected: str | None = None,
) -> PresenceEvent:
    return PresenceEvent(
        origin="w", session_id="s", user_id=user_id, x=x, y=y, active=active, selected_id=selected
    )


def lock(
    target: str = "hero",
    agent: str = "agent-coder",
    state: LockState = LockState.ACQUIRED,
    acquired_at: int = 1000,
    timeout: int = 5000,
) -> ElementLock:
    return ElementLock(
        target_id=target,
        locked_by_agent_id=agent,
        state=state,
        priority=50,
        acquired_at_ms=acquired_at,
        timeout_ms=timeout,
    )


# --- the elastic viewport -------------------------------------------------------------------


def test_canvas_starts_at_full_hd() -> None:
    assert elastic_viewport(()) == Viewport(1920, 1080)


@pytest.mark.parametrize(
    ("bottom", "height"),
    [
        (1, 1080),
        (1080, 1080),
        (1081, 1081),
        (5000, 5000),
        (VIRTUAL_VIEWPORT_HEIGHT, VIRTUAL_VIEWPORT_HEIGHT),
        (VIRTUAL_VIEWPORT_HEIGHT + 1, VIRTUAL_VIEWPORT_HEIGHT),
    ],
)
def test_canvas_grows_downward_with_content_up_to_what_a_frame_can_carry(
    bottom: int, height: int
) -> None:
    viewport = elastic_viewport((box(0, 0, 100, bottom),))
    assert viewport.width == 1920
    assert viewport.height == height


def test_canvas_never_shrinks_under_cursors_still_working_down_the_page() -> None:
    tall = elastic_viewport((box(0, 0, 100, 4000),))
    assert elastic_viewport((box(0, 0, 100, 10),), tall).height == 4000


@pytest.mark.parametrize("height", [1080, 2160, 9999, VIRTUAL_VIEWPORT_HEIGHT])
def test_normalized_boxes_map_back_to_the_same_pixels_at_any_height(height: int) -> None:
    viewport = Viewport(1920, height)
    bounds = box(192, 108, 960, 540)
    normalized = normalize_box(bounds, viewport)
    assert normalized.x * viewport.width == pytest.approx(bounds.x)
    assert normalized.y * viewport.height == pytest.approx(bounds.y)
    assert normalized.width * viewport.width == pytest.approx(bounds.width)
    assert normalized.height * viewport.height == pytest.approx(bounds.height)


def test_a_box_keeps_its_place_as_the_canvas_grows_below_it() -> None:
    bounds = box(0, 540, 1920, 540)
    assert normalize_box(bounds, Viewport(1920, 1080)).y == 0.5
    assert normalize_box(bounds, Viewport(1920, 2160)).y == 0.25


def test_whole_canvas_is_exactly_the_unit_square() -> None:
    normalized = normalize_box(box(0, 0, 1920, 1080), Viewport(1920, 1080))
    assert (normalized.x, normalized.y, normalized.width, normalized.height) == (0, 0, 1, 1)


@pytest.mark.parametrize(
    "bounds",
    [box(0, 60000, 100, 10000), box(0, 70000, 100, 10), box(3000, 0, 2000, 10)],
    ids=["straddles", "beyond", "too-wide"],
)
def test_what_lies_past_the_largest_canvas_is_clipped_never_out_of_range(
    bounds: BoundingBox,
) -> None:
    viewport = elastic_viewport((bounds,))
    normalized = normalize_box(bounds, viewport)
    assert 0 <= normalized.x <= 1
    assert 0 <= normalized.y <= 1
    assert normalized.x + normalized.width <= 1
    assert normalized.y + normalized.height <= 1


# --- Hermite and Bézier motion --------------------------------------------------------------


def hermite_basis(p0: float, v0: float, p1: float, v1: float, duration: float, u: float) -> float:
    """The textbook polynomial form, to check the Bézier form against."""
    return (
        (2 * u**3 - 3 * u**2 + 1) * p0
        + (u**3 - 2 * u**2 + u) * duration * v0
        + (-2 * u**3 + 3 * u**2) * p1
        + (u**3 - u**2) * duration * v1
    )


@pytest.mark.parametrize("u", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
def test_bezier_controls_describe_the_same_curve_as_the_hermite_polynomial(u: float) -> None:
    start, end = Point(10.0, 20.0), Point(300.0, -40.0)
    start_velocity, end_velocity = Point(2.0, -1.0), Point(0.5, 3.0)
    point = cubic_bezier(*hermite_controls(start, start_velocity, end, end_velocity, 50.0), u)
    assert point.x == pytest.approx(hermite_basis(10, 2, 300, 0.5, 50, u))
    assert point.y == pytest.approx(hermite_basis(20, -1, -40, 3, 50, u))


def truth(t_ms: float) -> Point:
    """A cursor's path: curved, with continuously varying speed, in the middle of the canvas."""
    return Point(
        960 + 500 * math.sin(t_ms / 400),
        540 + 300 * math.sin(t_ms / 260 + 1),
    )


@dataclass
class Replay:
    median_error: float
    worst_error: float
    worst_jump: float
    worst_gap_ms: float


def replay(keep: Callable[[int], bool], delay_ms: float = CLIENT_BUFFER_MS) -> Replay:
    """Render a path at 120 Hz through a lossy 30 Hz stream, the way a viewer does.

    A jump is how much further the cursor moved between two rendered frames than the real cursor
    did, which is the part a person would see as a stutter.
    """
    tracker = DeadReckoner(0.0, truth(0.0))
    errors: list[float] = []
    worst_jump = worst_gap = last_arrival = 0.0
    previous: tuple[float, Point] | None = None
    next_tick = 1
    for step in range(3600):
        now = step * 1000 / 120
        while next_tick * TICK_MS <= now:
            arrival = next_tick * TICK_MS
            if keep(next_tick):
                worst_gap = max(worst_gap, arrival - last_arrival)
                last_arrival = arrival
                tracker.observe(arrival, truth(arrival))
            next_tick += 1
        at = now - delay_ms
        if at <= 0:
            continue
        position = tracker.sample(at, WIDE).position
        errors.append(math.dist(position, truth(at)))
        if previous is not None:
            moved = math.dist(position, previous[1])
            worst_jump = max(worst_jump, moved - math.dist(truth(at), truth(previous[0])))
        previous = (at, position)
    return Replay(statistics.median(errors), max(errors), worst_jump, worst_gap)


def lossy(seed: int) -> Callable[[int], bool]:
    rng = random.Random(seed)
    pattern = [rng.random() >= 0.5 for _ in range(1000)]
    return lambda tick: pattern[tick]


def test_every_sample_arriving_tracks_the_path_almost_exactly() -> None:
    result = replay(lambda _tick: True)
    assert result.median_error < 0.05
    assert result.worst_jump < 4.0, "only the first motion, out of the rest it starts from"


def test_every_other_frame_conflated_away_stays_within_a_pixel_and_never_jumps() -> None:
    result = replay(lambda tick: tick % 2 == 0)
    assert result.worst_gap_ms == pytest.approx(2 * TICK_MS)
    assert result.median_error < 0.5
    assert result.worst_jump < 5.0


@pytest.mark.parametrize("seed", range(5))
def test_half_the_frames_lost_at_random_stays_accurate_and_without_a_snap(seed: int) -> None:
    result = replay(lossy(seed))
    assert result.worst_gap_ms > 4 * TICK_MS, "the loss has to include real bursts"
    assert result.median_error < 1.5
    # A coast has been wrong by this much by the time a burst ends. It must not show as one jump.
    assert result.worst_error > 40
    assert result.worst_jump < result.worst_error / 8


def test_without_convergence_those_bursts_would_be_visible_snaps() -> None:
    """Pins down what convergence is for: this spline alone does jump on the same input."""
    tracker = DeadReckoner(0.0, truth(0.0))
    tracker.observe(TICK_MS, truth(TICK_MS))
    resume = TICK_MS + EXTRAPOLATION_LIMIT_MS
    coasted = tracker.sample(resume, WIDE).position
    tracker.observe(resume + 1, truth(resume + 1))
    corrected = tracker.sample(resume + 1, WIDE).position
    target = truth(resume + 1)
    assert math.dist(coasted, target) > 100, "the guess has gone badly wrong"
    assert math.dist(corrected, coasted) < 15, "yet the next frame barely moves"
    later = resume + 1 + CONVERGENCE_MS
    assert tracker.sample(later, WIDE).position == tracker._spline(later, WIDE).position, (
        "and then it has fully caught up"
    )


def test_rendering_just_ahead_of_the_samples_cannot_hold_a_track_back() -> None:
    """Each frame starts a correction here. The lag they cause settles, and does not accumulate."""
    tracker = DeadReckoner(0.0, Point(0.0, 500.0))
    lags = []
    for tick in range(1, 55):  # Which keeps it short of the canvas's right edge.
        at = tick * TICK_MS
        tracker.observe(at, Point(at, 500.0))
        lags.append(at + 2 - tracker.sample(at + 2, WIDE).position.x)
    assert max(lags) < 15, "at one pixel per millisecond, that is 15 ms behind at worst"
    assert lags[-1] == pytest.approx(lags[30], abs=0.5), "and it stays the same from then on"


def test_a_correction_in_progress_absorbs_the_next_one_without_a_jump() -> None:
    tracker = DeadReckoner(0.0, Point(100.0, 100.0))
    tracker.observe(30.0, Point(160.0, 100.0))
    shown = tracker.sample(200.0, WIDE).position
    tracker.observe(220.0, Point(100.0, 100.0))
    during = tracker.sample(222.0, WIDE).position
    # Motion is still allowed, at no more than the speed limit plus the eased share of the error.
    assert math.dist(during, shown) < 40
    tracker.observe(240.0, Point(100.0, 400.0))
    assert math.dist(tracker.sample(241.0, WIDE).position, during) < 40


def test_a_spline_passes_exactly_through_the_samples() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 0.0))
    points = [(33.0, Point(10.0, 5.0)), (66.0, Point(30.0, 25.0)), (130.0, Point(20.0, 90.0))]
    for t_ms, point in points:
        tracker.observe(t_ms, point)
    for t_ms, point in points:
        assert tracker.sample(t_ms, WIDE).position == pytest.approx(point)


def test_velocity_is_the_derivative_of_position() -> None:
    tracker = DeadReckoner(0.0, truth(0.0))
    for index in range(1, 10):
        tracker.observe(index * TICK_MS, truth(index * TICK_MS))
    at = 4.5 * TICK_MS
    state = tracker.sample(at, WIDE)
    before = tracker.sample(at - 0.01, WIDE).position
    after = tracker.sample(at + 0.01, WIDE).position
    assert state.velocity.x == pytest.approx((after.x - before.x) / 0.02, rel=1e-3)
    assert state.velocity.y == pytest.approx((after.y - before.y) / 0.02, rel=1e-3)


def test_a_late_or_repeated_sample_is_ignored() -> None:
    tracker = DeadReckoner(100.0, Point(50.0, 50.0))
    tracker.observe(100.0, Point(999.0, 999.0))
    tracker.observe(50.0, Point(999.0, 999.0))
    assert tracker.sample(100.0, WIDE).position == Point(50.0, 50.0)


def test_before_the_first_sample_a_track_waits_there() -> None:
    tracker = DeadReckoner(100.0, Point(50.0, 50.0))
    state = tracker.sample(0.0, WIDE)
    assert state.position == Point(50.0, 50.0)
    assert state.velocity == Point(0.0, 0.0)


def test_coasting_eases_to_a_stop_and_stays_there() -> None:
    tracker = DeadReckoner(0.0, Point(100.0, 100.0))
    tracker.observe(100.0, Point(200.0, 100.0))
    assert tracker.sample(100.0, WIDE).velocity.x == pytest.approx(1.0)
    midway = tracker.sample(100.0 + EXTRAPOLATION_LIMIT_MS / 2, WIDE)
    assert midway.velocity.x == pytest.approx(0.5)
    assert midway.position.x > 200.0
    stopped = tracker.sample(100.0 + EXTRAPOLATION_LIMIT_MS, WIDE)
    assert stopped.velocity.x == 0
    assert stopped.position.x == pytest.approx(200.0 + EXTRAPOLATION_LIMIT_MS / 2)
    much_later = tracker.sample(100.0 + EXTRAPOLATION_LIMIT_MS * 50, WIDE)
    assert much_later.position == stopped.position


def test_coasting_never_leaves_the_canvas() -> None:
    tracker = DeadReckoner(0.0, Point(1900.0, 1060.0))
    tracker.observe(33.0, Point(1919.0, 1079.0))
    position = tracker.sample(33.0 + EXTRAPOLATION_LIMIT_MS, WIDE).position
    assert position == Point(1920.0, 1080.0)


def test_a_teleport_cannot_fling_the_curve() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 0.0))
    tracker.observe(1.0, Point(1920.0, 1080.0))
    velocity = tracker.sample(1.0, WIDE).velocity
    assert math.hypot(*velocity) == pytest.approx(MAX_SPEED_PX_PER_MS)


def test_a_slow_reader_at_exactly_the_speed_limit_is_not_altered() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 0.0))
    tracker.observe(10.0, Point(MAX_SPEED_PX_PER_MS * 10, 0.0))
    assert tracker.sample(10.0, WIDE).velocity == Point(MAX_SPEED_PX_PER_MS, 0.0)


def test_a_long_silence_restarts_the_track_instead_of_animating_across_it() -> None:
    """A person who rested and moved on was not gliding the whole time."""
    tracker = DeadReckoner(0.0, Point(10.0, 10.0))
    tracker.observe(30.0, Point(20.0, 10.0))
    arrival = 30.0 + MOTION_RESET_MS + 1
    tracker.observe(arrival, Point(900.0, 900.0))
    state = tracker.sample(arrival, WIDE)
    assert state.position == Point(900.0, 900.0)
    assert state.velocity == Point(0.0, 0.0)
    assert tracker.sample(arrival - 100, WIDE).position == Point(900.0, 900.0)


def test_history_is_bounded_however_long_a_track_lives() -> None:
    tracker = DeadReckoner(0.0, Point(0.0, 0.0))
    for index in range(1, 10_000):
        tracker.observe(float(index), Point(float(index % 100), 0.0))
    assert tracker.sample(9_999.0, WIDE).position.x == pytest.approx(99.0)
    assert len(tracker._knots) == 16


# --- docking --------------------------------------------------------------------------------

DOCK_BOX = box(100, 100, 400, 200)


@pytest.mark.parametrize(
    ("cursor", "side", "anchor"),
    [
        (Point(300, 110), DockSide.TOP, Point(300, 100)),
        (Point(490, 200), DockSide.RIGHT, Point(500, 200)),
        (Point(300, 290), DockSide.BOTTOM, Point(300, 300)),
        (Point(110, 200), DockSide.LEFT, Point(100, 200)),
        (Point(300, 20), DockSide.TOP, Point(300, 100)),
        (Point(900, 200), DockSide.RIGHT, Point(500, 200)),
        (Point(300, 900), DockSide.BOTTOM, Point(300, 300)),
        (Point(5, 200), DockSide.LEFT, Point(100, 200)),
    ],
)
def test_cursor_docks_on_the_nearest_border(cursor: Point, side: DockSide, anchor: Point) -> None:
    dock = resolve_dock(cursor, DOCK_BOX, WIDE)
    assert dock.side is side
    assert dock.anchor == anchor


def test_every_anchor_lies_on_the_border_and_every_label_outside_it() -> None:
    for x in range(0, 700, 37):
        for y in range(0, 500, 29):
            dock = resolve_dock(Point(x, y), DOCK_BOX, WIDE)
            on_vertical = dock.anchor.x in (100, 500) and 100 <= dock.anchor.y <= 300
            on_horizontal = dock.anchor.y in (100, 300) and 100 <= dock.anchor.x <= 500
            assert on_vertical or on_horizontal
            assert box_gap(dock.label_at, DOCK_BOX) == pytest.approx(BADGE_GAP_PX)


def test_label_is_kept_on_the_canvas_when_a_box_fills_it() -> None:
    dock = resolve_dock(Point(960, 1), box(0, 0, 1920, 1080), WIDE)
    assert dock.side is DockSide.TOP
    assert dock.label_at == Point(960, 0)


def test_drifting_across_a_sections_midline_does_not_flip_the_dock() -> None:
    wide = box(0, 0, 1000, 400)
    above = resolve_dock(Point(500, 199), wide, WIDE)
    assert above.side is DockSide.TOP
    assert resolve_dock(Point(500, 201), wide, WIDE).side is DockSide.BOTTOM, "no memory, it flips"
    assert resolve_dock(Point(500, 201), wide, WIDE, above.side).side is DockSide.TOP
    just_short = 200 + DOCK_HYSTERESIS_PX / 2
    assert resolve_dock(Point(500, just_short), wide, WIDE, DockSide.TOP).side is DockSide.TOP


def test_a_clearly_nearer_side_takes_over() -> None:
    wide = box(0, 0, 1000, 400)
    far = 200 + DOCK_HYSTERESIS_PX
    assert resolve_dock(Point(500, far), wide, WIDE, DockSide.TOP).side is DockSide.BOTTOM


def test_gap_is_zero_inside_and_the_distance_outside() -> None:
    assert box_gap(Point(300, 200), DOCK_BOX) == 0
    assert box_gap(Point(530, 340), DOCK_BOX) == 50


# --- compositor: identities -----------------------------------------------------------------


def composed(
    compositor: CanvasCompositor,
    now_ms: float = 1000.0,
    locks: tuple[ElementLock, ...] = (),
    lock_time_ms: int = 1000,
) -> VisualFrame:
    return compositor.compose(now_ms, locks, lock_time_ms)


def test_swarm_palette_is_the_specified_one() -> None:
    assert SWARM_PALETTE == {
        AgentRole.ARCHITECT: "#8B5CF6",
        AgentRole.FRONTEND_CODER: "#3B82F6",
        AgentRole.BACKEND_ENGINEER: "#10B981",
        AgentRole.STYLER: "#EC4899",
        AgentRole.CRITIC_LINTER: "#F59E0B",
        AgentRole.COPYWRITER: "#06B6D4",
    }


def test_all_six_agents_appear_in_their_colors_in_a_stable_order() -> None:
    compositor = CanvasCompositor(page())
    for role in reversed(tuple(IDENTITIES)):
        compositor.place_agent(agent_frame(role), 0.0)
    agents = composed(compositor).agents
    assert [agent.id for agent in agents] == sorted(i.agent_id for i in IDENTITIES.values())
    for agent in agents:
        assert agent.color == SWARM_PALETTE[agent.role]


def test_a_role_outside_the_palette_keeps_the_color_its_frame_declares() -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(
        AgentCursorFrame(
            agent_id="agent-optimizer",
            name="Optimizer",
            role=AgentRole.OPTIMIZER,
            color="#123456",
            x=1,
            y=1,
        ),
        0.0,
    )
    assert composed(compositor).agents[0].color == "#123456"


def test_agents_carry_a_velocity_vector_in_pixels_per_second_along_their_motion() -> None:
    compositor = CanvasCompositor(page())
    for index in range(6):
        compositor.place_agent(agent_frame(x=100 + index * 10, y=500), index * 33.0)
    agent = composed(compositor, 5 * 33.0 + SMOOTHING_DELAY_MS).agents[0]
    assert agent.vx == pytest.approx(10 / 33 * 1000, rel=0.01)
    assert agent.vy == 0


def test_a_frame_is_time_stamped_behind_now_so_it_can_interpolate() -> None:
    compositor = CanvasCompositor(page())
    assert composed(compositor, 5000.0).t_ms == 5000.0 - SMOOTHING_DELAY_MS


# --- compositor: people ---------------------------------------------------------------------


def test_people_are_indicated_apart_from_agents_with_no_role() -> None:
    compositor = CanvasCompositor(page())
    compositor.apply(presence(ALICE, 10, 20, selected="hero"), 0.0)
    frame = composed(compositor)
    assert frame.agents == ()
    human = frame.humans[0]
    assert human.id == human.label == "user:alice"
    assert (human.x, human.y) == (10, 20)
    assert human.selected_id == "hero"
    assert "role" not in human.model_dump()
    assert human.color == human_color(ALICE)


def test_a_persons_color_is_stable_valid_and_never_an_agents() -> None:
    assert human_color(ALICE) == human_color(ALICE)
    assert human_color(ALICE) != human_color(BOB)
    palette = set(SWARM_PALETTE.values())
    for index in range(500):
        color = human_color(f"user-{index}")
        assert re.fullmatch(r"#[0-9A-F]{6}", color)
        assert color not in palette


def test_moving_updates_a_person_and_selection_follows() -> None:
    compositor = CanvasCompositor(page())
    compositor.apply(presence(ALICE, 10, 20), 0.0)
    compositor.apply(presence(ALICE, 40, 20, selected="features"), 100.0)
    human = composed(compositor, 100.0 + SMOOTHING_DELAY_MS).humans[0]
    assert human.x == 40
    assert human.selected_id == "features"


def test_a_person_leaving_is_removed_and_an_unknown_departure_is_harmless() -> None:
    compositor = CanvasCompositor(page())
    compositor.apply(presence(ALICE), 0.0)
    compositor.apply(presence(BOB), 0.0)
    compositor.apply(presence(ALICE, active=False), 1.0)
    compositor.apply(presence("never-seen", active=False), 1.0)
    assert [human.id for human in composed(compositor).humans] == ["user:bob"]


# --- compositor: lock overlays --------------------------------------------------------------


def overlay_of(frame: VisualFrame, component_id: str = "hero") -> Any:
    return next(item for item in frame.scene.overlays if item.id == component_id)


async def test_overlay_follows_a_lock_through_contention_and_release() -> None:
    now = 1000
    manager = AsyncLockManager(lease_ms=5000, clock=lambda: now)
    compositor = CanvasCompositor(page())

    def state() -> Any:
        return overlay_of(composed(compositor, locks=manager.active_locks(), lock_time_ms=now))

    assert state().state is OverlayState.FREE
    assert state().owner is None

    await manager.acquire("hero", "agent-coder", 60, now)
    held = state()
    assert held.state is OverlayState.LOCKED
    assert held.owner.label == "Frontend Coder"
    assert held.owner.color == "#3B82F6"
    assert held.waiters == ()

    await manager.acquire("hero", "agent-styler", 40, now)
    contended = state()
    assert contended.state is OverlayState.WAITING
    assert contended.owner.label == "Frontend Coder", "the badge still says who holds it"
    assert [waiter.label for waiter in contended.waiters] == ["Styler"]

    await manager.release("hero", "agent-coder")
    promoted = state()
    assert promoted.state is OverlayState.LOCKED
    assert promoted.owner.label == "Styler"
    assert promoted.waiters == ()

    await manager.release("hero", "agent-styler")
    assert state().state is OverlayState.FREE


def test_a_lapsed_lease_already_counts_as_free_before_any_sweep() -> None:
    compositor = CanvasCompositor(page())
    held = lock(acquired_at=1000, timeout=5000)
    assert overlay_of(composed(compositor, locks=(held,), lock_time_ms=5999)).state is (
        OverlayState.LOCKED
    )
    assert overlay_of(composed(compositor, locks=(held,), lock_time_ms=6000)).state is (
        OverlayState.FREE
    )


def test_a_waiter_outliving_a_lapsed_holder_still_pulses_with_no_owner() -> None:
    locks = (
        lock(acquired_at=0, timeout=10),
        lock(agent="agent-styler", state=LockState.WAITING, acquired_at=900),
    )
    overlay = overlay_of(composed(CanvasCompositor(page()), locks=locks, lock_time_ms=1000))
    assert overlay.state is OverlayState.WAITING
    assert overlay.owner is None


def test_a_person_owning_a_lock_is_badged_as_a_person() -> None:
    overlay = overlay_of(
        composed(CanvasCompositor(page()), locks=(lock(agent="user:alice"),), lock_time_ms=1000)
    )
    assert overlay.owner.label == "user:alice"
    assert overlay.owner.color == human_color(ALICE)


def test_an_owner_who_is_neither_a_known_agent_nor_a_person_is_still_named() -> None:
    assert participant("agent-unknown").label == "agent-unknown"


def test_locks_on_things_with_no_place_on_the_canvas_draw_nothing() -> None:
    locks = (lock(target="frontend:document"), lock(target="api_endpoint_post_leads"))
    frame = composed(CanvasCompositor(page()), locks=locks, lock_time_ms=1000)
    assert {item.state for item in frame.scene.overlays} == {OverlayState.FREE}


def test_finished_lock_records_do_not_show() -> None:
    locks = (lock(state=LockState.CONFLICT_RESOLVED), lock(state=LockState.TIMEOUT_RELEASED))
    overlay = overlay_of(composed(CanvasCompositor(page()), locks=locks, lock_time_ms=1000))
    assert overlay.state is OverlayState.FREE


def test_overlays_carry_normalized_boxes_and_titles() -> None:
    overlay = overlay_of(composed(CanvasCompositor(page())), "features")
    assert overlay.title == "Title features"
    assert overlay.box.y == pytest.approx(300 / 1080)
    assert overlay.box.width == pytest.approx(1000 / 1920)


# --- compositor: docking agents -------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        AgentAction.EDITING,
        AgentAction.NAVIGATING,
        AgentAction.BLOCKED_WAITING,
        AgentAction.PATCHING_AST,
    ],
)
def test_active_agents_dock_on_the_nearest_component_without_a_lock(action: AgentAction) -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=500, y=480, action=action), 0.0)
    dock = composed(compositor).agents[0].dock
    assert dock is not None
    assert dock.target_id == "features"
    assert dock.side is DockSide.BOTTOM
    assert (dock.x, dock.y) == (500, 500 + BADGE_GAP_PX)


@pytest.mark.parametrize("action", [AgentAction.IDLE, AgentAction.THINKING])
def test_agents_not_at_a_component_do_not_dock(action: AgentAction) -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(action=action), 0.0)
    assert composed(compositor).agents[0].dock is None


def test_nothing_to_dock_on_means_no_dock() -> None:
    compositor = CanvasCompositor(bundle_with())
    compositor.place_agent(agent_frame(action=AgentAction.EDITING), 0.0)
    assert composed(compositor).agents[0].dock is None


def test_what_an_agent_holds_decides_its_dock_over_what_is_nearest() -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=500, y=100, action=AgentAction.EDITING), 0.0)
    frame = composed(compositor, locks=(lock("features"),))
    assert cast("Any", frame.agents[0].dock).target_id == "features"


def test_holding_outranks_waiting_when_choosing_a_dock() -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=500, y=480, action=AgentAction.EDITING), 0.0)
    locks = (lock("features", state=LockState.WAITING), lock("hero"))
    assert cast("Any", composed(compositor, locks=locks).agents[0].dock).target_id == "hero"


def test_a_waiting_agent_docks_at_what_it_waits_for() -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=500, y=100, action=AgentAction.BLOCKED_WAITING), 0.0)
    locks = (lock("features", state=LockState.WAITING),)
    assert cast("Any", composed(compositor, locks=locks).agents[0].dock).target_id == "features"


def test_locks_of_others_and_off_canvas_locks_do_not_decide_a_dock() -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=500, y=100, action=AgentAction.EDITING), 0.0)
    locks = (lock("features", agent="agent-styler"), lock("frontend:document"))
    assert cast("Any", composed(compositor, locks=locks).agents[0].dock).target_id == "hero"


def test_compositor_remembers_a_dock_side_while_docked_and_forgets_it_when_idle() -> None:
    compositor = CanvasCompositor(bundle_with(node("wide", box(0, 0, 1000, 400))))
    compositor.place_agent(agent_frame(x=500, y=199, action=AgentAction.EDITING), 0.0)
    assert cast("Any", composed(compositor, 100.0).agents[0].dock).side is DockSide.TOP
    compositor.place_agent(agent_frame(x=500, y=201, action=AgentAction.EDITING), 100.0)
    assert cast("Any", composed(compositor, 100.0 + SMOOTHING_DELAY_MS).agents[0].dock).side is (
        DockSide.TOP
    )
    compositor.place_agent(agent_frame(x=500, y=201, action=AgentAction.IDLE), 200.0)
    assert composed(compositor, 200.0 + SMOOTHING_DELAY_MS).agents[0].dock is None
    compositor.place_agent(agent_frame(x=500, y=201, action=AgentAction.EDITING), 300.0)
    assert cast("Any", composed(compositor, 300.0 + SMOOTHING_DELAY_MS).agents[0].dock).side is (
        DockSide.BOTTOM
    )


# --- compositor: thoughts -------------------------------------------------------------------


def with_cursor(action: AgentAction = AgentAction.EDITING) -> CanvasCompositor:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=500, y=20, action=action), 0.0)
    return compositor


def test_a_badge_enters_holds_exits_and_is_forgotten() -> None:
    compositor = with_cursor()
    compositor.apply(thought(text="Writing"), 1000.0)

    entering = composed(compositor, 1000.0 + BADGE_ENTER_MS / 2).badges[0]
    assert entering.phase is BadgePhase.ENTERING
    assert entering.opacity == pytest.approx(0.5)
    assert entering.text == "Writing"
    assert entering.color == "#3B82F6"

    visible = composed(compositor, 1000.0 + BADGE_ENTER_MS + 1).badges[0]
    assert visible.phase is BadgePhase.VISIBLE
    assert visible.opacity == 1

    fade_start = 1000.0 + BADGE_ENTER_MS + BADGE_HOLD_MS
    assert composed(compositor, fade_start - 1).badges[0].phase is BadgePhase.VISIBLE
    exiting = composed(compositor, fade_start + BADGE_EXIT_MS / 2).badges[0]
    assert exiting.phase is BadgePhase.EXITING
    assert exiting.opacity == pytest.approx(0.5)

    assert composed(compositor, fade_start + BADGE_EXIT_MS).badges == ()
    assert compositor._badges == {}


def test_a_new_thought_fades_its_predecessor_out_and_nothing_older_is_kept() -> None:
    compositor = with_cursor()
    compositor.apply(thought(0, text="First"), 1000.0)
    compositor.apply(thought(1, text="Second"), 2000.0)
    badges = composed(compositor, 2000.0 + BADGE_ENTER_MS / 2).badges
    assert [(badge.text, badge.phase) for badge in badges] == [
        ("First", BadgePhase.EXITING),
        ("Second", BadgePhase.ENTERING),
    ]
    compositor.apply(thought(2, text="Third"), 2001.0)
    compositor.apply(thought(3, text="Fourth"), 2002.0)
    assert [badge.text for badge in composed(compositor, 2003.0).badges] == ["Third", "Fourth"]


def test_a_docked_badge_sits_at_the_dock_and_an_undocked_one_at_the_cursor() -> None:
    docked = with_cursor()
    docked.apply(thought(), 1000.0)
    badge = composed(docked, 1100.0).badges[0]
    assert (badge.x, badge.y) == (500, 0), "kept on the canvas above a box at the very top"
    assert badge.side is DockSide.TOP

    free = with_cursor(AgentAction.IDLE)
    free.apply(thought(), 1000.0)
    badge = composed(free, 1100.0).badges[0]
    assert (badge.x, badge.y) == (500, 20)
    assert badge.side is None


def test_a_thought_with_no_cursor_gets_no_badge_and_is_still_cleaned_up() -> None:
    compositor = CanvasCompositor(page())
    compositor.apply(thought(), 1000.0)
    assert composed(compositor, 1001.0).badges == ()
    composed(compositor, 1000.0 + BADGE_ENTER_MS + BADGE_HOLD_MS + BADGE_EXIT_MS)
    assert compositor._badges == {}


def test_a_thought_becomes_a_feed_entry_that_names_its_author() -> None:
    entry = thought_entry(thought(7, AgentRole.ARCHITECT, "Planning"))
    assert entry.model_dump() == {
        "kind": "canvas.thought",
        "seq": 7,
        "agent_id": "agent-architect",
        "label": "Architect",
        "color": "#8B5CF6",
        "text": "Planning",
        "tool_name": "llm.generate",
    }


def test_long_thoughts_become_bounded_snippets() -> None:
    compositor = with_cursor()
    compositor.apply(thought(text="x" * 4000), 1000.0)
    badge = composed(compositor, 1001.0).badges[0]
    assert len(badge.text) == MAX_SNIPPET_LENGTH
    assert badge.text.endswith("…")
    assert thought_entry(thought(text="x" * 4000)).text == badge.text
    assert thought_entry(thought(text="y" * MAX_SNIPPET_LENGTH)).text == "y" * MAX_SNIPPET_LENGTH


# --- compositor: events ---------------------------------------------------------------------


def test_cursor_events_move_agents() -> None:
    compositor = CanvasCompositor(page())
    event = CursorEvent(origin="w", session_id="s", frame=agent_frame(x=7, y=8))
    compositor.apply(event, 0.0)
    assert (composed(compositor).agents[0].x, composed(compositor).agents[0].y) == (7, 8)


def test_state_events_bring_new_layout_and_grow_the_canvas_but_never_shrink_it() -> None:
    compositor = CanvasCompositor(page())

    def state_event(bundle: FullstackProjectBundle) -> StateEvent:
        return StateEvent(
            origin="w",
            session_id="s",
            delta=StreamDelta(
                session_id="s",
                agent_id="agent-coder",
                delta_seq=1,
                target_domain="frontend",
                target_id="hero",
                patches=(
                    PatchOperation.model_validate(
                        {"op": "replace", "path": "/frontend/title", "value": "T"}
                    ),
                ),
                emitted_at_ms=1,
            ),
            snapshot=SwarmSnapshot.capture(
                frame_seq=1, step_name="DELTA_APPLIED", timestamp_ms=1, bundle_state=bundle
            ),
        )

    compositor.apply(state_event(bundle_with(node("tall", box(0, 0, 100, 5000)))), 0.0)
    scene = composed(compositor).scene
    assert [overlay.id for overlay in scene.overlays] == ["tall"]
    assert (scene.viewport.width, scene.viewport.height) == (1920, 5000)
    assert overlay_of(composed(compositor), "tall").box.height == 1

    compositor.apply(state_event(bundle_with(node("short", box(0, 0, 100, 500)))), 0.0)
    assert compositor.viewport.height == 5000
    assert overlay_of(composed(compositor), "short").box.height == pytest.approx(0.1)


def test_lock_notices_change_nothing_because_the_table_is_authoritative() -> None:
    compositor = CanvasCompositor(page())
    before = composed(compositor)
    compositor.apply(
        LockNotice(
            origin="w",
            session_id="s",
            event=LockEvent(kind=LockEventKind.PROMOTED, lock=lock()),
        ),
        0.0,
    )
    assert composed(compositor) == before


def test_agents_stay_on_the_canvas_when_sampled() -> None:
    compositor = CanvasCompositor(page())
    compositor.place_agent(agent_frame(x=1919, y=1079), 0.0)
    compositor.place_agent(agent_frame(x=1920, y=1080), 33.0)
    agent = composed(compositor, 33.0 + EXTRAPOLATION_LIMIT_MS + SMOOTHING_DELAY_MS).agents[0]
    assert agent.x <= 1920
    assert agent.y <= 1080


# --- the viewer asset -----------------------------------------------------------------------


def script_of() -> str:
    return VIEWER_HTML[VIEWER_HTML.index("<script>") + 8 : VIEWER_HTML.rindex("</script>")]


def style_of() -> str:
    return VIEWER_HTML[VIEWER_HTML.index("<style>") + 7 : VIEWER_HTML.rindex("</style>")]


def test_viewer_fits_its_budget_and_has_nothing_left_to_fill_in() -> None:
    assert len(VIEWER_BYTES) < MAX_VIEWER_BYTES
    assert "@@" not in VIEWER_HTML


def test_viewer_loads_nothing_from_anywhere() -> None:
    assert not re.search(r"https?://", VIEWER_HTML)
    assert not re.search(r"\b(?:src|href)\s*=", VIEWER_HTML)
    assert not re.search(r"@import|url\(", style_of())
    assert VIEWER_HTML.count("<script") == 1
    assert VIEWER_HTML.count("<style") == 1


def test_policy_hashes_cover_exactly_the_inline_code_and_allow_nothing_else() -> None:
    def source_hash(source: str) -> str:
        return base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()

    directives = dict(item.split(" ", 1) for item in CONTENT_SECURITY_POLICY.split("; "))
    assert directives == {
        "default-src": "'none'",
        "script-src": f"'sha256-{source_hash(script_of())}'",
        "style-src": f"'sha256-{source_hash(style_of())}'",
        "connect-src": "'self'",
        "base-uri": "'none'",
        "form-action": "'none'",
        "frame-ancestors": "'none'",
    }
    assert "unsafe" not in CONTENT_SECURITY_POLICY


def test_nothing_in_the_page_depends_on_what_the_policy_forbids() -> None:
    markup = VIEWER_HTML[: VIEWER_HTML.index("<script>")]
    assert not re.search(r"\son[a-z]+\s*=", markup), "inline handlers"
    assert not re.search(r"\sstyle\s*=", markup), "style attributes"
    assert not re.search(r"\.innerHTML|outerHTML|insertAdjacentHTML|document\.write", script_of())
    assert not re.search(r"\beval\b|new Function", script_of())


def test_session_text_reaches_the_page_only_through_safe_sinks() -> None:
    assert "textContent" in script_of()
    assert "fillText" in script_of()


def test_viewer_runs_the_servers_motion_constants_rather_than_a_copy() -> None:
    script = script_of()
    assert f"COAST_MS = {EXTRAPOLATION_LIMIT_MS}" in script
    assert f"RESET_MS = {MOTION_RESET_MS}" in script
    assert f"CONVERGE_MS = {CONVERGENCE_MS}" in script
    assert f"BUFFER_MS = {CLIENT_BUFFER_MS}" in script
    assert f"HEARTBEAT_MS = {HEARTBEAT_MS}" in script


def test_viewer_heartbeats_often_enough_to_keep_its_connection() -> None:
    assert Settings().ws_heartbeat_timeout_seconds / 2 > HEARTBEAT_MS / 1000


def test_viewer_takes_its_token_from_the_fragment_and_hands_it_over_as_a_subprotocol() -> None:
    script = script_of()
    assert "location.hash" in script
    assert "history.replaceState" in script, "the token must leave the address bar"
    assert f'["{SUBPROTOCOL}", `{TOKEN_SUBPROTOCOL_PREFIX}${{token}}`]' in script
    assert "?view=canvas" in script
    assert "location.search" not in script.replace("location.pathname + location.search", "")
    assert not re.search(r"localStorage|sessionStorage|document\.cookie", script)


def test_viewer_only_ever_sends_read_and_presence_messages() -> None:
    sent = set(re.findall(r'send\(\{ type: "([a-z.]+)"', script_of()))
    assert sent == {"ping", "cursor", "select"}


# --- the viewer route -----------------------------------------------------------------------


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        **{
            "gateway_secret": SecretStr("canvas-signing-key-for-unit-tests-only"),
            "broadcast_fps": 240,
            **overrides,
        }
    )


@dataclass
class Harness:
    app: FastAPI
    service: GatewayService
    http: httpx.AsyncClient

    def token(self, user_id: str = ALICE) -> str:
        return self.service.authenticator.issue(user_id)

    def auth(self, user_id: str = ALICE) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(user_id)}"}

    async def create_session(self, user_id: str = ALICE, **body: Any) -> str:
        response = await self.http.post(SESSIONS, json=body, headers=self.auth(user_id))
        assert response.status_code == 201, response.text
        session_id: str = response.json()["session_id"]
        return session_id

    async def staged(self, user_id: str = ALICE) -> str:
        return await self.create_session(user_id, bundle=page().model_dump(mode="json"))

    def socket(
        self,
        session_id: str,
        user_id: str = ALICE,
        *,
        query: str = "view=canvas",
        send_gate: Callable[[Message], Awaitable[None]] | None = None,
    ) -> "CanvasDriver":
        return CanvasDriver(
            self.app,
            f"/ws/sessions/{session_id}",
            query,
            [SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}{self.token(user_id)}"],
            send_gate,
        )


@asynccontextmanager
async def running(settings: Settings | None = None) -> AsyncIterator[Harness]:
    app = create_app(settings or make_settings(), provider=MockLLMProvider())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://gateway") as http,
    ):
        yield Harness(app, app.state.gateway, http)


class CanvasDriver:
    """Speaks ASGI to the application in this event loop, leaving ``send`` in the test's hands."""

    def __init__(
        self,
        app: FastAPI,
        path: str,
        query: str,
        subprotocols: list[str],
        send_gate: Callable[[Message], Awaitable[None]] | None,
    ) -> None:
        self._app = app
        self._scope: dict[str, Any] = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "http_version": "1.1",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": query.encode(),
            "headers": [],
            "subprotocols": subprotocols,
            "client": ("client", 1),
            "server": ("gateway", 80),
        }
        self._to_app: asyncio.Queue[Message] = asyncio.Queue()
        self.from_app: asyncio.Queue[Message] = asyncio.Queue()
        self._send_gate = send_gate
        self.task: asyncio.Task[None] | None = None

    async def _send(self, message: Message) -> None:
        if self._send_gate is not None and message["type"] == "websocket.send":
            await self._send_gate(message)
        await self.from_app.put(message)

    async def open(self) -> Message:
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(
            self._app(self._scope, cast("Any", self._to_app.get), cast("Any", self._send))
        )
        return await asyncio.wait_for(self.from_app.get(), WAIT_S)

    async def connect(self) -> dict[str, Any]:
        answer = await self.open()
        assert answer["type"] == "websocket.accept", answer
        assert answer["subprotocol"] == SUBPROTOCOL
        return await self.receive()

    async def receive(self) -> dict[str, Any]:
        message = await asyncio.wait_for(self.from_app.get(), WAIT_S)
        assert message["type"] == "websocket.send", message
        parsed: dict[str, Any] = json.loads(message["text"])
        return parsed

    async def frame_where(self, condition: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        """Return the next canvas frame meeting a condition, tracking the latest scene seen."""
        async with asyncio.timeout(WAIT_S):
            while True:
                frame = await self.receive()
                if frame["kind"] == "canvas" and condition(frame):
                    return frame

    async def scene_where(self, condition: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        frame = await self.frame_where(lambda item: "scene" in item and condition(item["scene"]))
        scene: dict[str, Any] = frame["scene"]
        return scene

    async def reply(self, kind: str) -> dict[str, Any]:
        async with asyncio.timeout(WAIT_S):
            while True:
                frame = await self.receive()
                if frame["kind"] == kind:
                    return frame

    def send(self, payload: dict[str, Any]) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    async def closed(self) -> tuple[int, str]:
        async with asyncio.timeout(WAIT_S):
            while True:
                message = await self.from_app.get()
                if message["type"] == "websocket.close":
                    return message["code"], message.get("reason", "")

    async def disconnect(self) -> None:
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1000})
        assert self.task is not None
        await asyncio.wait_for(self.task, WAIT_S)


async def test_viewer_is_served_as_utf8_html_with_every_protective_header() -> None:
    async with running() as harness:
        response = await harness.http.get("/canvas/s-anything")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cross-origin-opener-policy"] == "same-origin"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == VIEWER_BYTES


async def test_viewer_is_the_same_bytes_for_anyone_and_reveals_nothing_about_a_session() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        real = await harness.http.get(f"/canvas/{session_id}")
        missing = await harness.http.get("/canvas/s-does-not-exist")
        signed_in = await harness.http.get(f"/canvas/{session_id}", headers=harness.auth())
    assert real.status_code == missing.status_code == signed_in.status_code == 200
    assert real.content == missing.content == signed_in.content
    assert session_id.encode() not in real.content


@pytest.mark.parametrize(
    "path",
    [
        "/canvas/%3Cscript%3Ealert(1)",
        "/canvas/x?token=secret-in-query&a=%3Cscript%3E",
        "/canvas/%22%3E%3Cimg%20onerror=alert(1)%3E",
    ],
)
async def test_viewer_reflects_nothing_from_the_request(path: str) -> None:
    async with running() as harness:
        response = await harness.http.get(path)
    assert response.content == VIEWER_BYTES
    assert "secret-in-query" not in str(response.headers)


async def test_a_path_that_misses_the_route_is_an_inert_problem_rather_than_a_page() -> None:
    async with running() as harness:
        response = await harness.http.get("/canvas/%3Cscript%3Ealert(1)%3C%2Fscript%3E")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


async def test_viewer_is_not_part_of_the_api_document() -> None:
    async with running() as harness:
        document = (await harness.http.get("/openapi.json")).json()
    assert not any(path.startswith("/canvas") for path in document["paths"])


# --- the canvas stream ----------------------------------------------------------------------


async def test_canvas_view_opens_with_a_hello_and_the_full_scene() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id)
        hello = await socket.connect()
        assert hello == {
            "kind": "canvas.hello",
            "session_id": session_id,
            "user_id": ALICE,
            "agent_id": "user:alice",
            "is_owner": True,
            "latest_thought_seq": None,
            "replayed_thoughts": 0,
            "thought_gap": False,
        }
        frame = await socket.receive()
        assert frame["kind"] == "canvas"
        assert [overlay["id"] for overlay in frame["scene"]["overlays"]] == ["hero", "features"]
        assert frame["scene"]["viewport"] == {"width": 1920, "height": 1080}
        assert {agent["color"] for agent in frame["agents"]} <= set(SWARM_PALETTE.values())
        await socket.disconnect()


async def test_scene_is_only_resent_when_it_changes() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id)
        await socket.connect()
        assert "scene" in await socket.receive()
        await socket.frame_where(lambda frame: frame["agents"])
        for _ in range(5):
            frame = await socket.receive()
            assert frame["kind"] == "canvas"
            assert "scene" not in frame
        await socket.disconnect()


async def test_a_plain_lock_change_reaches_the_overlay_although_it_emits_no_event() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        hub = harness.service.hub
        manager = hub.lock_manager(session_id)
        socket = harness.socket(session_id)
        await socket.connect()
        await socket.scene_where(lambda _scene: True)

        await manager.acquire("hero", "agent-coder", 60, hub.now_ms())
        scene = await socket.scene_where(lambda s: s["overlays"][0]["state"] == "LOCKED")
        assert scene["overlays"][0]["owner"]["label"] == "Frontend Coder"

        await manager.acquire("hero", "agent-styler", 40, hub.now_ms())
        scene = await socket.scene_where(lambda s: s["overlays"][0]["state"] == "WAITING")
        assert scene["overlays"][0]["waiters"][0]["label"] == "Styler"

        await manager.release("hero", "agent-coder")
        scene = await socket.scene_where(
            lambda s: (
                s["overlays"][0]["state"] == "LOCKED"
                and s["overlays"][0]["owner"]["label"] == "Styler"
            )
        )
        assert scene["overlays"][0]["waiters"] == []

        await manager.release("hero", "agent-styler")
        await socket.scene_where(lambda s: s["overlays"][0]["state"] == "FREE")
        await socket.disconnect()


async def test_a_viewer_can_watch_but_every_change_is_refused_over_the_canvas_socket() -> None:
    async with running() as harness:
        session_id = await harness.staged(ALICE)
        viewer = harness.socket(session_id, BOB)
        assert (await viewer.connect())["is_owner"] is False
        viewer.send({"type": "prompt", "text": "Spend it", "request_id": "p"})
        refused = await viewer.reply("error")
        assert refused["request_id"] == "p"
        assert refused["problem"]["status"] == 403
        viewer.send({"type": "lock.acquire", "target_id": "hero", "request_id": "l"})
        assert (await viewer.reply("error"))["problem"]["status"] == 403
        assert harness.service._sessions[session_id].run is None
        viewer.send({"type": "ping", "request_id": "still-here"})
        assert (await viewer.reply("pong"))["request_id"] == "still-here"
        await viewer.disconnect()


async def test_canvas_socket_requires_a_token_like_any_other() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        anonymous = CanvasDriver(
            harness.app, f"/ws/sessions/{session_id}", "view=canvas", [SUBPROTOCOL], None
        )
        answer = await anonymous.open()
        assert answer["type"] == "websocket.close"
        assert answer["code"] == 1008
        forged = CanvasDriver(
            harness.app,
            f"/ws/sessions/{session_id}",
            "view=canvas",
            [SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}forged.token"],
            None,
        )
        assert (await forged.open())["code"] == 1008


async def test_the_subprotocol_token_is_never_echoed_back() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id)
        answer = await socket.open()
        assert answer["subprotocol"] == SUBPROTOCOL
        assert harness.token() not in json.dumps(answer)
        await socket.disconnect()


async def test_unknown_view_is_refused_and_unknown_session_has_its_own_close_code() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        wrong = harness.socket(session_id, query="view=nonsense")
        answer = await wrong.open()
        assert answer["type"] == "websocket.close"
        assert answer["code"] == 1008
        missing = harness.socket("s-missing")
        assert (await missing.open())["type"] == "websocket.accept"
        assert (await missing.closed())[0] == 4404


async def test_the_regular_stream_is_unchanged_without_the_view_parameter() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id, query="")
        assert (await socket.connect())["kind"] == "sync"
        assert (await socket.reply("cursor"))["kind"] == "cursor"
        await socket.disconnect()


async def test_presence_of_others_is_composited_and_a_person_who_leaves_disappears() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        watcher = harness.socket(session_id, ALICE)
        other = harness.socket(session_id, BOB)
        await watcher.connect()
        await other.connect()
        other.send({"type": "cursor", "x": 321, "y": 123})
        frame = await watcher.frame_where(lambda item: item["humans"])
        human = frame["humans"][0]
        assert (human["id"], human["x"], human["y"]) == ("user:bob", 321, 123)
        await other.disconnect()
        await watcher.frame_where(lambda item: not item["humans"])
        await watcher.disconnect()


async def test_someone_joining_late_sees_who_is_already_here() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        unrelated = await harness.staged()
        early = harness.socket(session_id, BOB)
        silent = harness.socket(session_id, "carol")
        elsewhere = harness.socket(unrelated, "dave")
        for socket in (early, silent, elsewhere):
            await socket.connect()
        early.send({"type": "cursor", "x": 50, "y": 60, "request_id": "moved"})
        await early.reply("ack")
        elsewhere.send({"type": "cursor", "x": 1, "y": 1, "request_id": "moved"})
        await elsewhere.reply("ack")

        late = harness.socket(session_id, ALICE)
        await late.connect()
        frame = await late.receive()
        assert [human["id"] for human in frame["humans"]] == ["user:bob"], (
            "not someone who has not moved, and not someone in another session"
        )
        for socket in (early, silent, elsewhere, late):
            await socket.disconnect()


async def test_closing_a_second_tab_does_not_make_a_person_vanish() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        watcher = harness.socket(session_id, ALICE)
        first_tab = harness.socket(session_id, BOB)
        second_tab = harness.socket(session_id, BOB)
        for socket in (watcher, first_tab, second_tab):
            await socket.connect()
        first_tab.send({"type": "cursor", "x": 9, "y": 9})
        await watcher.frame_where(lambda item: item["humans"])

        await first_tab.disconnect()
        for _ in range(30):
            assert (await watcher.frame_where(lambda _item: True))["humans"], "bob is still here"

        await second_tab.disconnect()
        await watcher.frame_where(lambda item: not item["humans"])
        await watcher.disconnect()


async def test_disconnecting_releases_a_persons_locks_and_the_overlay_clears() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        watcher = harness.socket(session_id, BOB)
        owner = harness.socket(session_id, ALICE)
        await watcher.connect()
        await owner.connect()
        owner.send({"type": "lock.acquire", "target_id": "hero"})
        scene = await watcher.scene_where(lambda s: s["overlays"][0]["state"] == "LOCKED")
        assert scene["overlays"][0]["owner"]["label"] == "user:alice"
        await owner.disconnect()
        await watcher.scene_where(lambda s: s["overlays"][0]["state"] == "FREE")
        await watcher.disconnect()


async def think(harness: Harness, session_id: str, text: str) -> None:
    await harness.service.hub.broadcast_thought(
        session_id,
        AgentThoughtStream(
            agent_id="agent-coder", role=AgentRole.FRONTEND_CODER, thought=text, timestamp_ms=1
        ),
    )


async def test_a_new_viewer_gets_recent_history_then_every_thought_live() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        for index in range(MAX_THOUGHT_HISTORY + 5):
            await think(harness, session_id, f"Before {index}")
        socket = harness.socket(session_id)
        hello = await socket.connect()
        assert hello["replayed_thoughts"] == MAX_THOUGHT_HISTORY
        assert hello["latest_thought_seq"] == MAX_THOUGHT_HISTORY + 4
        assert hello["thought_gap"] is False, "history is not a gap, only a missed resume is"
        replayed = [await socket.receive() for _ in range(MAX_THOUGHT_HISTORY)]
        assert [entry["kind"] for entry in replayed] == ["canvas.thought"] * MAX_THOUGHT_HISTORY
        assert [entry["seq"] for entry in replayed] == list(range(5, MAX_THOUGHT_HISTORY + 5))

        await think(harness, session_id, "After joining")
        live = await socket.reply("canvas.thought")
        assert (live["seq"], live["text"]) == (MAX_THOUGHT_HISTORY + 5, "After joining")
        await socket.frame_where(lambda item: item["badges"])
        await socket.disconnect()


async def test_a_reconnecting_viewer_gets_exactly_what_it_missed() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        for index in range(3):
            await think(harness, session_id, f"Seen {index}")
        for index in range(120):
            await think(harness, session_id, f"Missed {index}")
        socket = harness.socket(session_id, query="view=canvas&last_seen_thought_seq=2")
        hello = await socket.connect()
        assert (hello["replayed_thoughts"], hello["thought_gap"]) == (120, False)
        seqs = [(await socket.receive())["seq"] for _ in range(120)]
        assert seqs == list(range(3, 123))
        await socket.disconnect()


async def test_resuming_from_beyond_the_backlog_is_reported_as_a_gap() -> None:
    async with running(make_settings(thought_backlog_size=10)) as harness:
        session_id = await harness.staged()
        for index in range(30):
            await think(harness, session_id, f"Thought {index}")
        socket = harness.socket(session_id, query="view=canvas&last_seen_thought_seq=0")
        hello = await socket.connect()
        assert (hello["replayed_thoughts"], hello["thought_gap"]) == (10, True)
        assert [(await socket.receive())["seq"] for _ in range(10)] == list(range(20, 30))
        await socket.disconnect()


async def test_thought_text_stays_sanitized_on_the_way_to_the_page() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        await harness.service.hub.broadcast_thought(
            session_id,
            AgentThoughtStream(
                agent_id="agent-coder",
                role=AgentRole.FRONTEND_CODER,
                thought="<img src=x onerror=alert(1)>Harmless",
                timestamp_ms=1,
            ),
        )
        socket = harness.socket(session_id)
        await socket.connect()
        assert (await socket.receive())["text"] == "Harmless"
        await socket.disconnect()


async def test_a_swarm_run_is_visible_in_the_canvas_view() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        harness.service.start_run(session_id, "Build a landing page")
        await socket.scene_where(
            lambda scene: {"hero", "contact"} <= {item["id"] for item in scene["overlays"]}
        )
        tools = set()
        while "run.completed" not in tools:
            tools.add((await socket.reply("canvas.thought"))["tool_name"])
        assert {"graph.enter", "llm.generate", "hub.apply_delta"} <= tools
        await socket.disconnect()


async def test_a_slow_client_gets_fewer_frames_but_every_thought_in_order() -> None:
    release = asyncio.Event()
    sent: list[dict[str, Any]] = []

    async def gate(message: Message) -> None:
        sent.append(json.loads(message["text"]))
        if len(sent) >= 3:
            await release.wait()

    async with running(make_settings(ws_catchup_deadline_ms=60_000)) as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id, send_gate=gate)
        await socket.connect()
        await asyncio.wait_for(_until(lambda: len(sent) >= 3), WAIT_S)
        for index in range(200):
            await think(harness, session_id, f"Thought {index}")
        await asyncio.sleep(0.05)
        release.set()
        while not any(item.get("seq") == 199 for item in sent):
            await socket.receive()
        thoughts = [item["seq"] for item in sent if item["kind"] == "canvas.thought"]
        assert thoughts == list(range(200))
        frames = [item for item in sent if item["kind"] == "canvas"]
        assert len(frames) < 10, "the stall folded motion into a few frames, not 50 ms of them"
        await socket.disconnect()


async def _until(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0.001)


async def test_a_stalled_client_is_still_evicted() -> None:
    stall = asyncio.Event()
    sent = 0

    async def gate(_message: Message) -> None:
        nonlocal sent
        sent += 1
        if sent >= 2:
            await stall.wait()

    async with running(make_settings(ws_catchup_deadline_ms=30)) as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id, send_gate=gate)
        await socket.connect()
        assert await socket.closed() == (1008, "Slow Consumer")
        assert socket.task is not None
        await asyncio.wait_for(socket.task, WAIT_S)
        stall.set()


async def test_session_end_closes_the_canvas_socket_normally() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id)
        await socket.connect()
        await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert await socket.closed() == (1000, "Session closed")


async def test_shutdown_tells_canvas_clients_it_is_going_away() -> None:
    async with running() as harness:
        session_id = await harness.staged()
        socket = harness.socket(session_id)
        await socket.connect()
        await socket.frame_where(lambda _item: True)
        for connection in tuple(harness.service._connections):
            connection.drain()
        assert await socket.closed() == (1001, "Server shutting down")
