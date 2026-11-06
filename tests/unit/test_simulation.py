import asyncio
import logging
import math
import random
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from realtime_agentic_site_builder.agents.graph import create_swarm_graph
from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.agents.provider import (
    LLMClient,
    LLMRequest,
    LLMResult,
    MockLLMProvider,
)
from realtime_agentic_site_builder.agents.runtime import NodeContext, SwarmRuntime
from realtime_agentic_site_builder.agents.state import initial_state
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import (
    CursorEvent,
    InMemoryEventBus,
    SessionListener,
    SwarmEvent,
    SwarmSessionHub,
    ThoughtEvent,
    session_channel,
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
from realtime_agentic_site_builder.domain.errors import SessionNotFoundError
from realtime_agentic_site_builder.domain.swarm import (
    VIRTUAL_VIEWPORT_HEIGHT,
    VIRTUAL_VIEWPORT_WIDTH,
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    LockState,
)
from realtime_agentic_site_builder.simulation.controller import (
    HOVER_AMPLITUDE_PX,
    JITTER_AMPLITUDE_PX,
    WANDER_AMPLITUDE_PX,
    CursorMode,
    CursorSimulator,
)
from realtime_agentic_site_builder.simulation.motion import (
    CLEARANCE_PX,
    DEFAULT_VIEWPORT_HEIGHT,
    DEFAULT_VIEWPORT_WIDTH,
    FOCAL_INSET_PX,
    HOVER_GAP_PX,
    Point,
    Rect,
    Trajectory,
    Viewport,
    build_trajectory,
    cubic_bezier,
    distance,
    ease_in_out,
    fitts_duration_ms,
    focal_point,
    hover_point,
    plan_route,
    seed_unit,
    wander,
)
from realtime_agentic_site_builder.simulation.ticker import (
    MAX_STEP_MS,
    SimulationTicker,
    monotonic_ms,
)

SESSION = "session-1"
CODER = AgentRole.FRONTEND_CODER
STYLER = AgentRole.STYLER
TICK_MS = 33
WAIT_S = 2.0
VIEW = Viewport()


def box(x: int = 400, y: int = 300, width: int = 600, height: int = 200) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def component(component_id: str, target: BoundingBox) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.HERO,
        title="Section",
        code="<Section />",
        box=target,
    )


def bundle(*components: ComponentNode) -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id="project-1",
        frontend=SiteBlueprint(site_id="site-1", title="Acme", theme="dark", components=components),
        backend=BackendBlueprint(service_name="api", database_dialect=DatabaseDialect.SQLITE),
    )


def held(
    target_id: str,
    agent_id: str,
    *,
    state: LockState = LockState.ACQUIRED,
) -> ElementLock:
    return ElementLock(
        target_id=target_id,
        locked_by_agent_id=agent_id,
        state=state,
        priority=50,
        acquired_at_ms=0,
    )


def run_for(simulator: CursorSimulator, duration_ms: int, tick_ms: int = TICK_MS) -> None:
    for _ in range(duration_ms // tick_ms + 1):
        simulator.step(tick_ms)


def agent_frame(
    role: AgentRole,
    action: AgentAction,
    *,
    thought: AgentThoughtStream | None = None,
    lock: ElementLock | None = None,
) -> AgentCursorFrame:
    identity = IDENTITIES[role]
    return AgentCursorFrame(
        agent_id=identity.agent_id,
        name=identity.name,
        role=role,
        color=identity.color,
        x=1,
        y=1,
        current_action=action,
        current_thought=thought,
        active_lock=lock,
    )


def assert_in_bounds(frames: Sequence[AgentCursorFrame], viewport: Viewport) -> None:
    for frame in frames:
        assert 0 <= frame.x <= viewport.width, frame
        assert 0 <= frame.y <= viewport.height, frame


# --- bezier and easing ----------------------------------------------------------------------


def test_cubic_bezier_starts_and_ends_on_its_endpoints() -> None:
    p0, p1, p2, p3 = Point(0, 0), Point(10, 40), Point(30, -20), Point(50, 10)
    assert cubic_bezier(p0, p1, p2, p3, 0.0) == p0
    assert cubic_bezier(p0, p1, p2, p3, 1.0) == p3


def test_cubic_bezier_matches_its_formula_at_the_midpoint() -> None:
    point = cubic_bezier(Point(0, 0), Point(0, 8), Point(8, 8), Point(8, 0), 0.5)
    assert point == pytest.approx((4.0, 6.0))


def test_cubic_bezier_stays_in_the_hull_of_its_control_points() -> None:
    controls = (Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100))
    for step in range(101):
        point = cubic_bezier(*controls, step / 100)
        assert 0 <= point.x <= 100
        assert 0 <= point.y <= 100


def test_easing_is_monotonic_with_flat_ends() -> None:
    values = [ease_in_out(step / 1000) for step in range(1001)]
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))
    assert values[1] - values[0] < 0.0001, "should start from rest"
    assert values[500] - values[499] > values[1] - values[0], "should accelerate"
    assert values[-1] - values[-2] < 0.0001, "should arrive at rest"


@pytest.mark.parametrize("progress, expected", [(-1.0, 0.0), (2.0, 1.0), (0.5, 0.5)])
def test_easing_clamps_its_input(progress: float, expected: float) -> None:
    assert ease_in_out(progress) == expected


def test_fitts_law_grows_with_distance_and_shrinks_with_target_size() -> None:
    assert fitts_duration_ms(0) == pytest.approx(120.0)
    assert fitts_duration_ms(1000) > fitts_duration_ms(100)
    assert fitts_duration_ms(500, target_width_px=20) > fitts_duration_ms(500, target_width_px=400)
    assert fitts_duration_ms(500, target_width_px=0) > 0


# --- trajectories ---------------------------------------------------------------------------


def test_trajectory_progress_is_monotonic_along_its_length() -> None:
    path = build_trajectory(Point(10, 10), Point(1800, 900), (), VIEW)
    travelled = []
    previous = path.position_at(0)
    total = 0.0
    for elapsed in range(0, round(path.duration_ms) + 1):
        current = path.position_at(elapsed)
        total += distance(previous, current)
        travelled.append(total)
        previous = current
    assert all(later >= earlier for earlier, later in zip(travelled, travelled[1:], strict=False))
    assert total == pytest.approx(path.length, rel=0.02)


def test_trajectory_lands_exactly_on_its_goal_and_starts_at_rest() -> None:
    start, goal = Point(100, 200), Point(1500, 800)
    path = build_trajectory(start, goal, (), VIEW)
    assert path.position_at(0) == start
    assert path.position_at(path.duration_ms) == goal
    assert path.position_at(path.duration_ms * 5) == goal
    assert distance(path.position_at(0), path.position_at(1)) < 1.0


def test_speed_peaks_mid_flight() -> None:
    path = build_trajectory(Point(0, 0), Point(1900, 0), (), VIEW)
    step = path.duration_ms / 20
    speeds = [
        distance(path.position_at(step * i), path.position_at(step * (i + 1))) for i in range(20)
    ]
    assert 5 <= speeds.index(max(speeds)) <= 14
    assert speeds[0] < max(speeds) / 3
    assert speeds[-1] < max(speeds) / 3


def test_trajectory_arcs_rather_than_moving_in_a_straight_line() -> None:
    path = build_trajectory(Point(100, 500), Point(1500, 500), (), VIEW, bow_unit=0.95)
    assert max(abs(point.y - 500) for point in path.sample()) > 20


def test_bow_unit_picks_the_side_of_the_arc() -> None:
    up = build_trajectory(Point(100, 500), Point(1500, 500), (), VIEW, bow_unit=0.0)
    down = build_trajectory(Point(100, 500), Point(1500, 500), (), VIEW, bow_unit=0.99)
    assert up.position_at(up.duration_ms / 2).y < 500 < down.position_at(down.duration_ms / 2).y


def test_zero_length_trajectory_has_arrived_already() -> None:
    path = build_trajectory(Point(50, 50), Point(50, 50), (), VIEW)
    assert path.length < 0.5
    assert path.duration_ms == 0
    assert path.fraction_at(0) == 1.0
    assert path.position_at(0) == pytest.approx(Point(50, 50))


@pytest.mark.parametrize(
    "start, goal",
    [
        (Point(0, 0), Point(1920, 0)),
        (Point(0, 1080), Point(1920, 1080)),
        (Point(0, 0), Point(0, 1080)),
        (Point(1920, 0), Point(1920, 1080)),
    ],
)
@pytest.mark.parametrize("bow_unit", [0.0, 0.99])
def test_arc_along_an_edge_stays_inside_the_viewport(
    start: Point, goal: Point, bow_unit: float
) -> None:
    path = build_trajectory(start, goal, (), VIEW, bow_unit=bow_unit)
    assert all(VIEW.contains(point) for point in path.sample())


@pytest.mark.parametrize("bow_unit", [0.0, 0.99])
def test_length_describes_the_motion_even_when_an_arc_hugs_an_edge(bow_unit: float) -> None:
    """Clamping only the positions would flatten the arc and leave its speed profile wrong."""
    path = build_trajectory(Point(0, 0), Point(1920, 0), (), VIEW, bow_unit=bow_unit)
    points = [path.point_at_length_fraction(step / 2000) for step in range(2001)]
    travelled = sum(distance(a, b) for a, b in zip(points, points[1:], strict=False))
    assert travelled == pytest.approx(path.length, rel=0.01)


def test_endpoints_outside_the_viewport_are_brought_inside() -> None:
    path = build_trajectory(Point(-500, -500), Point(99999, 99999), (), VIEW)
    assert path.position_at(0) == Point(0, 0)
    assert path.position_at(path.duration_ms) == Point(1920, 1080)


def test_sample_after_covers_only_what_remains() -> None:
    path = build_trajectory(Point(0, 500), Point(1000, 500), (), VIEW, bow_unit=0.5)
    remaining = path.sample(after=0.5)
    assert remaining[0].x == pytest.approx(500, abs=1)
    assert remaining[-1] == Point(1000, 500)


def test_sampling_is_dense_enough_not_to_step_over_a_thin_box() -> None:
    path = build_trajectory(Point(0, 500), Point(1900, 500), (), VIEW, bow_unit=0.5)
    gaps = [distance(a, b) for a, b in zip(path.sample(), path.sample()[1:], strict=False)]
    assert max(gaps) <= 5.0
    assert path.crosses([Rect(950, 490, 956, 510)])


# --- avoidance ------------------------------------------------------------------------------


def test_segment_blocking_treats_the_boundary_as_clear() -> None:
    rect = Rect(100, 100, 200, 200)
    assert rect.blocks_segment(Point(0, 150), Point(300, 150))
    assert rect.blocks_segment(Point(150, 0), Point(150, 300))
    assert rect.blocks_segment(Point(0, 0), Point(300, 300))
    assert rect.blocks_segment(Point(150, 150), Point(160, 160))
    assert not rect.blocks_segment(Point(0, 100), Point(300, 100))
    assert not rect.blocks_segment(Point(100, 0), Point(100, 300))
    assert not rect.blocks_segment(Point(0, 0), Point(50, 50))
    assert not rect.blocks_segment(Point(0, 250), Point(300, 250))
    assert not rect.blocks_segment(Point(0, 190), Point(190, 0)), "a diagonal passing the corner"


def test_rect_contains_excludes_its_boundary() -> None:
    rect = Rect(0, 0, 10, 10)
    assert rect.contains(Point(5, 5))
    assert not rect.contains(Point(0, 5))
    assert not rect.contains(Point(10, 10))


def test_clear_path_has_no_detour() -> None:
    assert plan_route(Point(0, 0), Point(100, 0), [Rect(0, 200, 100, 300)], VIEW) == (
        Point(0, 0),
        Point(100, 0),
    )


def test_route_goes_around_the_obstacle_by_its_nearest_corners() -> None:
    rect = Rect(800, 300, 1100, 900)
    route = plan_route(Point(500, 400), Point(1400, 400), [rect], VIEW)
    assert len(route) > 2
    assert route[0] == Point(500, 400)
    assert route[-1] == Point(1400, 400)
    inflated = rect.inflated(CLEARANCE_PX)
    for a, b in zip(route, route[1:], strict=False):
        assert not inflated.blocks_segment(a, b)
    assert all(point.y == 300 - CLEARANCE_PX for point in route[1:-1]), "over the top is shorter"


def test_route_threads_several_obstacles() -> None:
    rects = [Rect(400, 0, 600, 700), Rect(900, 300, 1100, 1080)]
    route = plan_route(Point(100, 400), Point(1500, 400), rects, VIEW)
    for a, b in zip(route, route[1:], strict=False):
        for rect in rects:
            assert not rect.inflated(CLEARANCE_PX).blocks_segment(a, b)


def route_length(route: Sequence[Point]) -> float:
    return sum(distance(a, b) for a, b in zip(route, route[1:], strict=False))


def shortest_by_relaxation(start: Point, goal: Point, rects: Sequence[Rect]) -> float:
    """Work out the optimum a different way: Bellman-Ford over the same visibility graph."""
    inflated = [rect.inflated(CLEARANCE_PX) for rect in rects]
    nodes = [start, goal, *(c for r in inflated for c in r.corners() if VIEW.contains(c))]
    best = [math.inf] * len(nodes)
    best[0] = 0.0
    for _ in nodes:
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j and not any(r.blocks_segment(a, b) for r in inflated):
                    best[j] = min(best[j], best[i] + distance(a, b))
    return best[1]


@pytest.mark.parametrize("seed", range(40))
def test_route_is_the_shortest_one_available(seed: int) -> None:
    rng = random.Random(seed)  # nosec B311
    rects = []
    for _ in range(rng.randint(2, 6)):
        left, top = rng.randint(100, 1500), rng.randint(100, 800)
        rects.append(Rect(left, top, left + rng.randint(20, 300), top + rng.randint(20, 300)))
    start, goal = Point(20, rng.randint(20, 1060)), Point(1900, rng.randint(20, 1060))
    if any(r.inflated(CLEARANCE_PX).contains(p) for r in rects for p in (start, goal)):
        pytest.skip("an endpoint is inside an obstacle, which planning deliberately ignores")
    expected = shortest_by_relaxation(start, goal, rects)
    assert route_length(plan_route(start, goal, rects, VIEW)) == pytest.approx(expected)


def test_route_is_deterministic() -> None:
    rects = [Rect(800, 300, 1100, 900), Rect(300, 100, 500, 600)]
    assert plan_route(Point(100, 300), Point(1500, 500), rects, VIEW) == plan_route(
        Point(100, 300), Point(1500, 500), rects, VIEW
    )


def test_route_ignores_an_obstacle_it_starts_in_or_ends_in() -> None:
    rect = Rect(100, 100, 500, 500)
    direct_out = plan_route(Point(300, 300), Point(1000, 300), [rect], VIEW)
    direct_in = plan_route(Point(1000, 300), Point(300, 300), [rect], VIEW)
    assert direct_out == (Point(300, 300), Point(1000, 300))
    assert direct_in == (Point(1000, 300), Point(300, 300))


def test_wall_across_the_whole_viewport_falls_back_to_crossing_it() -> None:
    wall = Rect(-10, 400, 2000, 600)
    route = plan_route(Point(500, 100), Point(500, 900), [wall], VIEW)
    assert route == (Point(500, 100), Point(500, 900))


def test_stacked_page_sections_are_passed_through_the_side_gutter() -> None:
    """Sections are 1280 wide on a wider canvas, so the gutter is the only way past them."""
    sections = [Rect(0, 0, 1280, 400), Rect(0, 400, 1280, 800)]
    route = plan_route(Point(100, 900), Point(100, -0.0), sections, VIEW)
    assert any(point.x >= 1280 for point in route)


def test_path_diverts_around_a_lock_that_is_directly_in_its_way() -> None:
    start, goal = Point(100, 540), Point(1800, 540)
    locked = Rect(800, 400, 1100, 700)
    straight = build_trajectory(start, goal, (), VIEW, bow_unit=0.5)
    diverted = build_trajectory(start, goal, [locked], VIEW, bow_unit=0.5)

    assert straight.crosses([locked])
    assert not diverted.crosses([locked])
    assert diverted.length > straight.length
    assert max(abs(point.y - 540) for point in diverted.sample()) > 140


@pytest.mark.parametrize("bow_unit", [0.0, 0.25, 0.5, 0.75, 0.99])
def test_no_arc_cuts_through_the_lock_it_was_routed_around(bow_unit: float) -> None:
    locked = Rect(800, 400, 1100, 700)
    path = build_trajectory(Point(100, 540), Point(1800, 540), [locked], VIEW, bow_unit=bow_unit)
    assert not path.crosses([locked])
    assert all(VIEW.contains(point) for point in path.sample())


def test_tight_gap_straightens_the_curve_rather_than_clipping_a_box() -> None:
    """Between these two boxes there is only a narrow corridor, too tight for a wide arc."""
    rects = [Rect(600, 0, 700, 515), Rect(600, 565, 700, 1080)]
    path = build_trajectory(Point(300, 100), Point(1000, 1000), rects, VIEW, bow_unit=0.99)
    assert not path.crosses(rects)


def test_curve_is_abandoned_when_every_tension_clips_a_box() -> None:
    """A sliver right on the straight route gets no clearance, so only straight legs can pass."""
    sliver = Rect(500, 410, 1400, 412)
    start, goal = Point(100, 420), Point(1800, 420)
    path = build_trajectory(start, goal, [sliver], VIEW, bow_unit=0.99)
    assert not path.crosses([sliver])


def test_trajectory_with_explicit_segments_reports_length() -> None:
    a, b = Point(0, 0), Point(30, 40)
    path = Trajectory([(a, a, b, b)], VIEW)
    assert path.length == pytest.approx(50, rel=0.01)


# --- viewport -------------------------------------------------------------------------------


def test_default_viewport_is_the_blueprints_canvas() -> None:
    assert (VIEW.width, VIEW.height) == (1920, 1080)
    assert (DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT) == (1920, 1080)


@pytest.mark.parametrize(
    "point, expected",
    [
        (Point(-5, -5), Point(0, 0)),
        (Point(5000, 5000), Point(1920, 1080)),
        (Point(100, 200), Point(100, 200)),
        (Point(1920, 1080), Point(1920, 1080)),
    ],
)
def test_clamp(point: Point, expected: Point) -> None:
    assert VIEW.clamp(point) == expected
    assert VIEW.contains(expected)


@pytest.mark.parametrize(
    "width, height",
    [(0, 100), (100, 0), (VIRTUAL_VIEWPORT_WIDTH + 1, 100), (100, VIRTUAL_VIEWPORT_HEIGHT + 1)],
)
def test_viewport_rejects_sizes_a_frame_cannot_carry(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="viewport"):
        Viewport(width, height)


def test_viewport_grows_to_fit_content_and_never_shrinks() -> None:
    assert VIEW.fitted_to([box(0, 0, 100, 100)]) == VIEW
    assert VIEW.fitted_to([]) == VIEW
    grown = VIEW.fitted_to([box(0, 2000, 1280, 400)])
    assert grown == Viewport(1920, 2400)
    assert grown.fitted_to([box(0, 0, 10, 10)]) == grown
    assert VIEW.fitted_to([box(0, 0, 3000, 10)]).width == 3000


def test_viewport_growth_stops_at_what_a_frame_can_carry() -> None:
    huge = VIEW.fitted_to([box(0, 70000, 9000, 100)])
    assert (huge.width, huge.height) == (VIRTUAL_VIEWPORT_WIDTH, VIRTUAL_VIEWPORT_HEIGHT)


# --- focal and hover points -----------------------------------------------------------------


def test_styler_enters_at_the_top_left_header_and_coder_at_the_center() -> None:
    target = box(100, 200, 800, 400)
    assert focal_point(target, STYLER) == Point(100 + FOCAL_INSET_PX, 200 + FOCAL_INSET_PX)
    assert focal_point(target, CODER) == Point(500, 400)


def test_roles_work_in_different_places_inside_the_box() -> None:
    target = box()
    points = {focal_point(target, role) for role in IDENTITIES}
    assert len(points) == len(IDENTITIES)
    rect = Rect.around(target)
    assert all(rect.contains(point) for point in points)


@pytest.mark.parametrize("role", list(AgentRole))
def test_focal_point_stays_inside_a_tiny_box(role: AgentRole) -> None:
    target = box(10, 10, 4, 4)
    point = focal_point(target, role)
    assert 10 <= point.x <= 14
    assert 10 <= point.y <= 14


def test_hover_point_is_outside_the_boundary_and_past_the_clearance_margin() -> None:
    target = box()
    point = hover_point(target, VIEW)
    assert point == Point(target.right + HOVER_GAP_PX, 400)
    assert not Rect.around(target, CLEARANCE_PX).contains(point)
    assert HOVER_GAP_PX > CLEARANCE_PX


def test_hover_point_moves_to_the_next_free_side() -> None:
    wide = box(100, 300, 1820, 200)
    assert hover_point(wide, VIEW) == Point(100 - HOVER_GAP_PX, 400)
    full_width = box(0, 300, 1920, 200)
    assert hover_point(full_width, VIEW) == Point(960, 300 - HOVER_GAP_PX)
    occupied_above = [Rect(0, 0, 1920, 299)]
    assert hover_point(full_width, VIEW, occupied_above) == Point(960, 500 + HOVER_GAP_PX)


def test_hover_point_falls_back_inside_the_viewport_when_no_side_is_free() -> None:
    everything = box(0, 0, 1920, 1080)
    assert VIEW.contains(hover_point(everything, VIEW))


# --- wander and seeding ---------------------------------------------------------------------


def test_wander_starts_at_zero_and_is_bounded() -> None:
    assert wander(1, "a", 0, 10.0) == Point(0.0, 0.0)
    for elapsed in range(0, 60_000, 37):
        offset = wander(1, "a", elapsed, 10.0)
        assert abs(offset.x) <= 10.0
        assert abs(offset.y) <= 10.0


def test_wander_is_deterministic_and_varies_by_seed_and_agent() -> None:
    assert wander(1, "a", 500, 10.0) == wander(1, "a", 500, 10.0)
    assert wander(1, "a", 500, 10.0) != wander(2, "a", 500, 10.0)
    assert wander(1, "a", 500, 10.0) != wander(1, "b", 500, 10.0)


def test_wander_is_smooth() -> None:
    previous = wander(1, "a", 0, 10.0)
    for elapsed in range(33, 10_000, 33):
        current = wander(1, "a", elapsed, 10.0)
        assert distance(previous, current) < 1.0
        previous = current


def test_seed_unit_is_stable_across_processes_and_in_range() -> None:
    # Pinned: the built-in hash() is salted per process, and could not reproduce this value.
    assert seed_unit("fixed") == pytest.approx(0.5983058971727964)
    assert all(0 <= seed_unit(i) < 1 for i in range(1000))


# --- controller: state machine --------------------------------------------------------------


def test_all_six_active_roles_are_simulated_and_framed_in_a_stable_order() -> None:
    simulator = CursorSimulator()
    assert len(simulator.roles) == 6
    assert set(simulator.roles) == {
        AgentRole.ARCHITECT,
        AgentRole.FRONTEND_CODER,
        AgentRole.BACKEND_ENGINEER,
        AgentRole.STYLER,
        AgentRole.COPYWRITER,
        AgentRole.CRITIC_LINTER,
    }
    frames = simulator.step(TICK_MS)
    assert [frame.role for frame in frames] == list(simulator.roles)
    assert len({frame.agent_id for frame in frames}) == 6
    for frame in frames:
        position = simulator.position_of(frame.role)
        assert (frame.x, frame.y) == (round(position.x), round(position.y))
    assert_in_bounds(frames, simulator.viewport)


def test_cursors_start_apart_and_idle() -> None:
    simulator = CursorSimulator()
    positions = {simulator.position_of(role) for role in simulator.roles}
    assert len(positions) == 6
    assert all(simulator.mode_of(role) is CursorMode.IDLE for role in simulator.roles)
    assert all(frame.current_action is AgentAction.IDLE for frame in simulator.step(0))


def test_identical_scripts_give_identical_frames() -> None:
    def script() -> list[tuple[AgentCursorFrame, ...]]:
        simulator = CursorSimulator(seed=7)
        out = [simulator.step(TICK_MS)]
        simulator.edit(CODER, box())
        simulator.block_on(STYLER, box())
        out.extend(simulator.step(TICK_MS) for _ in range(120))
        simulator.idle(CODER)
        out.extend(simulator.step(TICK_MS) for _ in range(30))
        return out

    assert script() == script()


def test_seed_changes_the_run_and_where_cursors_start() -> None:
    first, second = CursorSimulator(seed=1), CursorSimulator(seed=2)
    assert first.position_of(CODER).x != second.position_of(CODER).x
    assert first.position_of(CODER).y != second.position_of(CODER).y
    assert first.step(TICK_MS) != second.step(TICK_MS)


@pytest.mark.parametrize("ticks", [1, 2, 3, 7, 16, 33, 99])
def test_outcome_does_not_depend_on_how_time_is_split_into_ticks(ticks: int) -> None:
    def position_after_splitting() -> Point:
        simulator = CursorSimulator(seed=3)
        simulator.step(10)
        simulator.edit(CODER, box())
        total = 2970
        for _ in range(ticks):
            simulator.step(total // ticks)
        simulator.step(total - (total // ticks) * ticks)
        return simulator.position_of(CODER)

    reference = CursorSimulator(seed=3)
    reference.step(10)
    reference.edit(CODER, box())
    reference.step(2970)
    assert position_after_splitting() == pytest.approx(reference.position_of(CODER), abs=0.5)


def test_time_cannot_run_backwards() -> None:
    with pytest.raises(ValueError, match="backwards"):
        CursorSimulator().step(-1)


def test_zero_step_changes_nothing() -> None:
    simulator = CursorSimulator()
    simulator.step(TICK_MS)
    assert simulator.step(0) == simulator.step(0)


def test_idle_cursor_floats_gently_around_where_it_stopped() -> None:
    simulator = CursorSimulator()
    home = simulator.position_of(CODER)
    seen = []
    for _ in range(600):
        simulator.step(TICK_MS)
        seen.append(simulator.position_of(CODER))
    assert len(set(seen)) > 100, "an idle cursor should not freeze"
    assert all(distance(home, point) <= WANDER_AMPLITUDE_PX * math.sqrt(2) for point in seen)
    steps = [distance(a, b) for a, b in zip(seen, seen[1:], strict=False)]
    assert max(steps) < 3.0, "the float should be gentle"


def test_navigating_moves_along_the_path_then_settles() -> None:
    simulator = CursorSimulator()
    simulator.step(TICK_MS)
    start = simulator.position_of(CODER)
    goal = focal_point(box(), CODER)

    simulator.navigate_to(CODER, box())
    frames = simulator.step(TICK_MS)
    assert simulator.mode_of(CODER) is CursorMode.NAVIGATING
    coder = next(frame for frame in frames if frame.role is CODER)
    assert coder.current_action is AgentAction.NAVIGATING

    distances = []
    while simulator.mode_of(CODER) is CursorMode.NAVIGATING:
        simulator.step(TICK_MS)
        distances.append(distance(simulator.position_of(CODER), goal))
    assert distance(start, goal) > 100
    assert distances[0] < distance(start, goal)
    assert simulator.mode_of(CODER) is CursorMode.IDLE
    assert distance(simulator.position_of(CODER), goal) <= WANDER_AMPLITUDE_PX * math.sqrt(2)


def test_editing_arrives_then_makes_small_rapid_movements() -> None:
    simulator = CursorSimulator()
    goal = focal_point(box(), CODER)
    simulator.edit(CODER, box())
    run_for(simulator, 3000)
    assert simulator.mode_of(CODER) is CursorMode.EDITING

    seen = []
    for _ in range(300):
        simulator.step(TICK_MS)
        seen.append(simulator.position_of(CODER))
    assert all(distance(goal, point) <= JITTER_AMPLITUDE_PX * math.sqrt(2) for point in seen)
    assert len(set(seen)) > 100, "typing should be visible"
    frame = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert frame.current_action is AgentAction.EDITING


def test_jitter_is_smaller_and_livelier_than_the_idle_float() -> None:
    def motion(editing: bool) -> tuple[float, float]:
        """Return how far the cursor ranges, and how much of that it covers per tick."""
        simulator = CursorSimulator()
        if editing:
            simulator.edit(CODER, box())
            run_for(simulator, 3000)
        points = []
        for _ in range(300):
            simulator.step(TICK_MS)
            points.append(simulator.position_of(CODER))
        reach = max(distance(points[0], point) for point in points)
        path = sum(distance(a, b) for a, b in zip(points, points[1:], strict=False))
        return reach, path / len(points) / reach

    jitter_reach, jitter_pace = motion(True)
    float_reach, float_pace = motion(False)
    assert jitter_reach < float_reach
    assert jitter_pace > float_pace


def test_blocked_cursor_hovers_outside_the_component_with_its_lock_showing() -> None:
    simulator = CursorSimulator()
    target = box()
    waiting = held("hero", IDENTITIES[CODER].agent_id, state=LockState.WAITING)
    simulator.block_on(CODER, target, waiting)
    run_for(simulator, 3000)

    assert simulator.mode_of(CODER) is CursorMode.BLOCKED_WAITING
    rect = Rect.around(target)
    for _ in range(300):
        frames = simulator.step(TICK_MS)
        assert not rect.contains(simulator.position_of(CODER))
    frame = next(f for f in frames if f.role is CODER)
    assert frame.current_action is AgentAction.BLOCKED_WAITING
    assert frame.active_lock == waiting
    assert distance(simulator.position_of(CODER), hover_point(target, VIEW)) <= (
        HOVER_AMPLITUDE_PX * math.sqrt(2)
    )


def test_lock_indicator_clears_when_the_agent_goes_idle() -> None:
    simulator = CursorSimulator()
    simulator.block_on(CODER, box(), held("hero", IDENTITIES[CODER].agent_id))
    simulator.idle(CODER)
    frame = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert frame.active_lock is None
    assert simulator.mode_of(CODER) is CursorMode.IDLE


def test_stopping_midway_then_returning_really_travels_back() -> None:
    """Regression: a cursor that stopped short must not believe it is docked at its old goal."""
    simulator = CursorSimulator()
    target = box()
    simulator.navigate_to(CODER, target)
    simulator.step(30)
    simulator.idle(CODER)
    stopped_at = simulator.position_of(CODER)
    simulator.navigate_to(CODER, target)
    run_for(simulator, 3000)
    goal = focal_point(target, CODER)
    assert distance(stopped_at, goal) > 50
    assert distance(simulator.position_of(CODER), goal) < 40


def test_repeating_a_destination_does_not_restart_the_journey() -> None:
    simulator = CursorSimulator()
    target = box()
    simulator.navigate_to(CODER, target)
    run_for(simulator, 100)
    before = simulator.position_of(CODER)
    simulator.navigate_to(CODER, target)
    simulator.step(TICK_MS)
    assert simulator.mode_of(CODER) is CursorMode.NAVIGATING
    assert distance(before, simulator.position_of(CODER)) > 0


def test_changing_what_happens_on_arrival_keeps_the_journey_going() -> None:
    simulator = CursorSimulator()
    simulator.navigate_to(CODER, box())
    run_for(simulator, 100)
    simulator.edit(CODER, box())
    assert simulator.mode_of(CODER) is CursorMode.NAVIGATING
    run_for(simulator, 3000)
    assert simulator.mode_of(CODER) is CursorMode.EDITING


def test_docked_cursor_switches_mode_in_place() -> None:
    simulator = CursorSimulator()
    target = box()
    simulator.navigate_to(CODER, target)
    run_for(simulator, 3000)
    before = simulator.position_of(CODER)
    simulator.edit(CODER, target)
    assert simulator.mode_of(CODER) is CursorMode.EDITING
    simulator.step(1)
    assert distance(before, simulator.position_of(CODER)) < 2.0


def test_repeating_the_settled_state_changes_nothing() -> None:
    simulator = CursorSimulator()
    target = box()
    simulator.edit(CODER, target)
    run_for(simulator, 3000)
    simulator.step(TICK_MS)
    before = simulator.position_of(CODER)
    simulator.edit(CODER, target)
    simulator.step(0)
    assert simulator.position_of(CODER) == before


def test_retargeting_mid_flight_never_teleports() -> None:
    simulator = CursorSimulator()
    simulator.navigate_to(CODER, box(0, 0, 100, 100))
    run_for(simulator, 150)
    before = simulator.position_of(CODER)
    simulator.navigate_to(CODER, box(1500, 800, 100, 100))
    simulator.step(1)
    assert distance(before, simulator.position_of(CODER)) < 5.0


def test_arrival_leaves_no_position_jump() -> None:
    simulator = CursorSimulator()
    simulator.navigate_to(CODER, box())
    previous = simulator.position_of(CODER)
    biggest = 0.0
    for _ in range(200):
        simulator.step(5)
        current = simulator.position_of(CODER)
        biggest = max(biggest, distance(previous, current))
        previous = current
    assert biggest < 40


# --- controller: locks as obstacles ---------------------------------------------------------


def two_section_layout() -> FullstackProjectBundle:
    return bundle(component("free", box(1300, 100)), component("locked", box(800, 300, 300, 500)))


def test_cursor_steers_around_a_box_locked_by_someone_else() -> None:
    simulator = CursorSimulator()
    layout = two_section_layout()
    simulator.sync(layout, [held("locked", "agent-other")])
    rect = Rect(800, 300, 1100, 800)
    # Starts far to the left of the locked box, with its target just beyond it.
    run_for(simulator, 0)
    simulator.navigate_to(CODER, box(40, 500, 10, 10))
    run_for(simulator, 2500)
    simulator.navigate_to(CODER, box(1300, 500, 400, 100))
    while simulator.mode_of(CODER) is CursorMode.NAVIGATING:
        simulator.step(TICK_MS)
        assert not rect.contains(simulator.position_of(CODER))
    assert simulator.position_of(CODER).x > 1300


def test_cursor_does_not_steer_around_its_own_lock() -> None:
    def travel(owner: str) -> float:
        simulator = CursorSimulator()
        simulator.sync(two_section_layout(), [held("locked", owner)])
        simulator.navigate_to(CODER, box(40, 500, 10, 10))
        run_for(simulator, 2500)
        simulator.navigate_to(CODER, box(1300, 500, 400, 100))
        highest = 2000.0
        while simulator.mode_of(CODER) is CursorMode.NAVIGATING:
            simulator.step(TICK_MS)
            highest = min(highest, simulator.position_of(CODER).y)
        return highest

    assert travel(IDENTITIES[CODER].agent_id) > travel("agent-other")


def test_cursor_replans_when_a_lock_lands_in_its_path_mid_flight() -> None:
    simulator = CursorSimulator()
    layout = two_section_layout()
    simulator.sync(layout, [])
    simulator.navigate_to(CODER, box(40, 500, 10, 10))
    run_for(simulator, 2500)
    simulator.navigate_to(CODER, box(1300, 500, 400, 100))
    simulator.step(TICK_MS)

    simulator.sync(layout, [held("locked", "agent-other")])

    rect = Rect(800, 300, 1100, 800)
    while simulator.mode_of(CODER) is CursorMode.NAVIGATING:
        simulator.step(TICK_MS)
        assert not rect.contains(simulator.position_of(CODER))


def test_lock_out_of_the_way_does_not_interrupt_the_journey() -> None:
    def journey(locks: list[ElementLock]) -> list[Point]:
        simulator = CursorSimulator()
        layout = bundle(component("far", box(1500, 900, 100, 100)))
        simulator.sync(layout, [])
        simulator.navigate_to(CODER, box(40, 40, 10, 10))
        points = [simulator.position_of(CODER)]
        for _ in range(5):
            simulator.step(TICK_MS)
        simulator.sync(layout, locks)
        for _ in range(20):
            simulator.step(TICK_MS)
            points.append(simulator.position_of(CODER))
        return points

    assert journey([held("far", "agent-other")]) == journey([])


def test_lock_on_the_destination_itself_does_not_force_a_replan() -> None:
    def journey(locks: list[ElementLock]) -> list[Point]:
        target = box(1300, 500, 400, 100)
        layout = bundle(component("dest", target))
        simulator = CursorSimulator()
        simulator.sync(layout, [])
        simulator.navigate_to(CODER, target)
        simulator.step(TICK_MS)
        simulator.sync(layout, locks)
        points = []
        for _ in range(10):
            simulator.step(TICK_MS)
            points.append(simulator.position_of(CODER))
        return points

    assert journey([held("dest", "agent-other")]) == journey([])


def test_settled_cursor_is_not_disturbed_by_a_new_lock() -> None:
    simulator = CursorSimulator()
    layout = two_section_layout()
    simulator.sync(layout, [])
    run_for(simulator, 100)
    simulator.sync(layout, [held("locked", "agent-other")])
    assert all(simulator.mode_of(role) is CursorMode.IDLE for role in simulator.roles)


@pytest.mark.parametrize(
    "lock",
    [
        held("frontend:document", "agent-other"),
        held("Lead", "agent-other"),
        held("locked", "agent-other", state=LockState.WAITING),
    ],
)
def test_only_a_held_lock_on_a_component_is_an_obstacle(lock: ElementLock) -> None:
    simulator = CursorSimulator()
    simulator.sync(two_section_layout(), [lock])
    assert simulator._obstacles == ()


def test_sync_grows_the_canvas_to_fit_a_tall_page() -> None:
    simulator = CursorSimulator()
    assert simulator.viewport == Viewport()
    lower = box(0, 2000, 1280, 400)
    simulator.sync(bundle(component("lower", lower)), [])
    assert simulator.viewport == Viewport(1920, 2400)
    simulator.navigate_to(STYLER, lower)
    run_for(simulator, 3000)
    assert simulator.position_of(STYLER).y > 2000, "a lower section must be reachable"


def test_without_sync_the_default_canvas_is_a_hard_limit() -> None:
    simulator = CursorSimulator()
    simulator.navigate_to(STYLER, box(0, 2000, 1280, 400))
    run_for(simulator, 3000)
    assert_in_bounds(simulator.step(TICK_MS), Viewport())


# --- controller: agent intent ---------------------------------------------------------------


def make_thought(role: AgentRole, text: str = "Working") -> AgentThoughtStream:
    return AgentThoughtStream(
        agent_id=IDENTITIES[role].agent_id, role=role, thought=text, timestamp_ms=1
    )


@pytest.mark.parametrize(
    "action, settled_mode, settled_action",
    [
        (AgentAction.EDITING, CursorMode.EDITING, AgentAction.EDITING),
        (AgentAction.PATCHING_AST, CursorMode.EDITING, AgentAction.PATCHING_AST),
        (AgentAction.THINKING, CursorMode.IDLE, AgentAction.THINKING),
        (AgentAction.ACQUIRING_LOCK, CursorMode.IDLE, AgentAction.ACQUIRING_LOCK),
        (AgentAction.NAVIGATING, CursorMode.IDLE, AgentAction.IDLE),
    ],
)
def test_frame_with_a_box_travels_there_then_shows_its_action(
    action: AgentAction, settled_mode: CursorMode, settled_action: AgentAction
) -> None:
    simulator = CursorSimulator()
    target = box()
    simulator.direct(agent_frame(CODER, action), target)
    frames = simulator.step(TICK_MS)
    coder = next(f for f in frames if f.role is CODER)
    assert coder.current_action is AgentAction.NAVIGATING
    assert (coder.x, coder.y) != (1, 1), "the frame's own coordinates are not used"

    run_for(simulator, 3000)
    coder = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert simulator.mode_of(CODER) is settled_mode
    assert coder.current_action is settled_action


def test_blocked_frame_with_a_box_hovers_and_shows_the_lock() -> None:
    simulator = CursorSimulator()
    waiting = held("hero", IDENTITIES[CODER].agent_id, state=LockState.WAITING)
    simulator.direct(agent_frame(CODER, AgentAction.BLOCKED_WAITING, lock=waiting), box())
    run_for(simulator, 3000)
    assert simulator.mode_of(CODER) is CursorMode.BLOCKED_WAITING
    assert not Rect.around(box()).contains(simulator.position_of(CODER))
    assert next(f for f in simulator.step(0) if f.role is CODER).active_lock == waiting


def test_idle_frame_stops_the_cursor() -> None:
    simulator = CursorSimulator()
    simulator.navigate_to(CODER, box())
    run_for(simulator, 100)
    simulator.direct(agent_frame(CODER, AgentAction.IDLE), None)
    assert simulator.mode_of(CODER) is CursorMode.IDLE


def test_frame_without_a_box_changes_what_is_shown_but_not_where() -> None:
    simulator = CursorSimulator()
    simulator.step(TICK_MS)
    thought = make_thought(CODER)
    simulator.direct(agent_frame(CODER, AgentAction.THINKING, thought=thought), None)
    coder = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert coder.current_action is AgentAction.THINKING
    assert coder.current_thought == thought
    assert simulator.mode_of(CODER) is CursorMode.IDLE


def test_entering_a_node_without_a_destination_does_not_claim_to_be_moving() -> None:
    simulator = CursorSimulator()
    simulator.direct(agent_frame(CODER, AgentAction.NAVIGATING), None)
    coder = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert coder.current_action is AgentAction.IDLE


def test_status_without_a_box_does_not_interrupt_a_journey() -> None:
    simulator = CursorSimulator()
    simulator.edit(CODER, box())
    simulator.step(TICK_MS)
    simulator.direct(agent_frame(CODER, AgentAction.THINKING), None)
    coder = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert coder.current_action is AgentAction.NAVIGATING
    assert simulator.mode_of(CODER) is CursorMode.NAVIGATING
    run_for(simulator, 3000)
    coder = next(f for f in simulator.step(TICK_MS) if f.role is CODER)
    assert coder.current_action is AgentAction.EDITING, "what it set out to do still happens"


# --- controller: stability ------------------------------------------------------------------


def random_box(rng: random.Random, viewport: Viewport) -> BoundingBox:
    return BoundingBox(
        x=rng.randint(0, viewport.width),
        y=rng.randint(0, viewport.height),
        width=rng.randint(1, 800),
        height=rng.randint(1, 800),
    )


@pytest.mark.parametrize("seed", range(5))
def test_chaotic_script_never_leaves_the_viewport_or_teleports(seed: int) -> None:
    rng = random.Random(seed)  # nosec B311
    simulator = CursorSimulator(seed=seed)
    roles = list(simulator.roles)
    previous = {role: simulator.position_of(role) for role in roles}
    # Long diagonals at peak speed set this bound; anything larger would be a discontinuity.
    limit_per_ms = 8.0

    for _ in range(3000):
        action = rng.random()
        role = rng.choice(roles)
        if action < 0.05:
            simulator.navigate_to(role, random_box(rng, simulator.viewport))
        elif action < 0.1:
            simulator.edit(role, random_box(rng, simulator.viewport))
        elif action < 0.15:
            simulator.block_on(role, random_box(rng, simulator.viewport))
        elif action < 0.2:
            simulator.idle(role)
        delta = rng.choice((0, 1, 5, 16, 33, 100, 250))
        frames = simulator.step(delta)
        assert_in_bounds(frames, simulator.viewport)
        for each in roles:
            current = simulator.position_of(each)
            assert simulator.viewport.contains(current)
            assert distance(previous[each], current) <= limit_per_ms * max(delta, 1) + 8
            previous[each] = current


@pytest.mark.parametrize(
    "corner",
    [box(0, 0, 1, 1), box(1919, 0, 1, 1), box(0, 1079, 1, 1), box(1919, 1079, 1, 1)],
)
@pytest.mark.parametrize("role", [CODER, STYLER])
def test_cursors_docked_in_a_corner_stay_inside_while_they_jitter(
    corner: BoundingBox, role: AgentRole
) -> None:
    simulator = CursorSimulator()
    simulator.edit(role, corner)
    for _ in range(600):
        assert_in_bounds(simulator.step(TICK_MS), simulator.viewport)


# --- ticker ---------------------------------------------------------------------------------


class ManualTime:
    """Advances only when the ticker sleeps, so a tick's duration is exactly what a test says."""

    def __init__(self, step_ms: int = TICK_MS) -> None:
        self.now = 0
        self.step_ms = step_ms
        self.pauses: list[float] = []

    def clock(self) -> int:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += self.step_ms
        await asyncio.sleep(0)


def make_settings(**overrides: Any) -> Settings:
    return Settings(**{"broadcast_fps": 30, **overrides})


async def make_hub(**settings: Any) -> SwarmSessionHub:
    hub = SwarmSessionHub(make_settings(**settings), bus=InMemoryEventBus())
    await hub.create_session(SESSION, bundle())
    return hub


async def wait_until(condition: Any, timeout: float = WAIT_S) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


async def test_ticker_publishes_every_agents_frame_to_listeners() -> None:
    hub = await make_hub(broadcast_fps=240)
    listener = hub.subscribe(SESSION)
    async with SimulationTicker(hub, SESSION) as ticker:
        seen: set[str] = set()
        while len(seen) < 6:
            event = await asyncio.wait_for(anext(listener), WAIT_S)
            assert isinstance(event, CursorEvent)
            seen.add(event.frame.agent_id)
        assert ticker.running
    assert not ticker.running
    assert seen == {identity.agent_id for identity in IDENTITIES.values()}
    await hub.shutdown()


async def test_ticker_starts_stops_and_restarts_cleanly() -> None:
    hub = await make_hub()
    time_source = ManualTime()
    ticker = SimulationTicker(hub, SESSION, clock=time_source.clock, sleep=time_source.sleep)
    assert not ticker.running

    ticker.start()
    assert ticker.running
    await wait_until(lambda: ticker.ticks >= 3)
    await ticker.stop()
    assert not ticker.running

    count = ticker.ticks
    await asyncio.sleep(0)
    assert ticker.ticks == count

    ticker.start()
    await wait_until(lambda: ticker.ticks > count)
    await ticker.stop()
    await hub.shutdown()


async def test_starting_twice_keeps_one_loop_and_stopping_twice_is_safe() -> None:
    hub = await make_hub()
    ticker = SimulationTicker(hub, SESSION)
    before = asyncio.all_tasks()
    ticker.start()
    ticker.start()
    assert len(asyncio.all_tasks() - before) == 1
    await ticker.stop()
    await ticker.stop()
    assert asyncio.all_tasks() - before == set()
    await hub.shutdown()


async def test_stopping_before_starting_is_safe() -> None:
    hub = await make_hub()
    await SimulationTicker(hub, SESSION).stop()
    await hub.shutdown()


@pytest.mark.parametrize("fps", [30, 60])
async def test_ticker_interval_follows_broadcast_fps(fps: int) -> None:
    hub = await make_hub(broadcast_fps=fps)
    interval_ms = 1000 / fps
    pauses: list[float] = []

    async def record(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) > 3:
            await asyncio.Event().wait()

    ticker = SimulationTicker(hub, SESSION, clock=lambda: 0, sleep=record)
    ticker.start()
    await wait_until(lambda: len(pauses) > 3)
    await ticker.stop()
    assert pauses[0] == pytest.approx(interval_ms / 1000)
    assert pauses[1] == pytest.approx(2 * interval_ms / 1000), "the schedule must not drift"
    await hub.shutdown()


class RecordingSimulator(CursorSimulator):
    def __init__(self) -> None:
        super().__init__()
        self.deltas: list[int] = []

    def step(self, delta_time_ms: int) -> tuple[AgentCursorFrame, ...]:
        self.deltas.append(delta_time_ms)
        return super().step(delta_time_ms)


async def test_ticker_steps_by_the_time_that_really_passed() -> None:
    hub = await make_hub()
    time_source = ManualTime(step_ms=40)
    simulator = RecordingSimulator()
    ticker = SimulationTicker(
        hub, SESSION, simulator, clock=time_source.clock, sleep=time_source.sleep
    )
    ticker.start()
    await wait_until(lambda: len(simulator.deltas) >= 3)
    await ticker.stop()
    assert simulator.deltas[:3] == [40, 40, 40]
    await hub.shutdown()


async def test_long_stall_resumes_without_a_leap_or_a_burst() -> None:
    hub = await make_hub()
    time_source = ManualTime(step_ms=60_000)
    simulator = RecordingSimulator()
    ticker = SimulationTicker(
        hub, SESSION, simulator, clock=time_source.clock, sleep=time_source.sleep
    )
    ticker.start()
    await wait_until(lambda: len(simulator.deltas) >= 3)
    await ticker.stop()
    assert simulator.deltas[:3] == [MAX_STEP_MS] * 3
    assert time_source.pauses[1] == pytest.approx((1000 / 30) / 1000), "missed frames are skipped"
    await hub.shutdown()


async def test_clock_stepping_backwards_never_reverses_the_simulation() -> None:
    hub = await make_hub()
    readings = iter([1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 40, 30, 20, 10, 0])
    simulator = RecordingSimulator()
    idle = asyncio.Event()

    def clock() -> int:
        return next(readings, 0)

    async def sleep(_seconds: float) -> None:
        if len(simulator.deltas) >= 3:
            await idle.wait()
        await asyncio.sleep(0)

    ticker = SimulationTicker(hub, SESSION, simulator, clock=clock, sleep=sleep)
    ticker.start()
    await wait_until(lambda: len(simulator.deltas) >= 3)
    await ticker.stop()
    assert all(delta == 0 for delta in simulator.deltas)
    await hub.shutdown()


def test_monotonic_clock_advances() -> None:
    first = monotonic_ms()
    assert monotonic_ms() >= first > 0


async def test_ticker_ends_quietly_when_its_session_closes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hub = await make_hub(broadcast_fps=240)
    ticker = SimulationTicker(hub, SESSION)
    ticker.start()
    await wait_until(lambda: ticker.ticks > 0)
    with caplog.at_level(logging.INFO):
        await hub.close_session(SESSION)
        await wait_until(lambda: not ticker.running)
    assert "ended with session" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    await ticker.stop()
    await hub.shutdown()


async def test_ticker_on_a_missing_session_ends_immediately() -> None:
    hub = await make_hub()
    ticker = SimulationTicker(hub, "missing", sleep=lambda _s: asyncio.sleep(0))
    ticker.start()
    await wait_until(lambda: not ticker.running)
    assert ticker.ticks == 0
    await hub.shutdown()


class BrokenSimulator(CursorSimulator):
    def step(self, delta_time_ms: int) -> tuple[AgentCursorFrame, ...]:
        raise RuntimeError(f"simulator bug at {delta_time_ms}")


async def test_failed_ticker_reports_itself_and_does_not_hurt_the_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hub = await make_hub()
    ticker = SimulationTicker(hub, SESSION, BrokenSimulator(), sleep=lambda _s: asyncio.sleep(0))
    with caplog.at_level(logging.ERROR):
        ticker.start()
        await wait_until(lambda: not ticker.running)
        await asyncio.sleep(0)
    assert "simulation ticker failed" in caplog.text
    await ticker.stop()
    assert hub.latest_snapshot(SESSION).frame_seq == 0
    await hub.shutdown()


async def test_cancelled_stop_still_finishes_the_shutdown() -> None:
    hub = await make_hub()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def sleep_with_slow_teardown(_seconds: float) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            entered.set()
            await release.wait()

    ticker = SimulationTicker(hub, SESSION, sleep=sleep_with_slow_teardown)
    ticker.start()
    await asyncio.sleep(0)
    task = ticker._task
    assert task is not None

    stopping = asyncio.create_task(ticker.stop())
    await asyncio.wait_for(entered.wait(), WAIT_S)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    assert not task.done(), "the ticker is still shutting down"
    assert not ticker.running
    release.set()
    await asyncio.wait_for(asyncio.wait({task}), WAIT_S)
    assert task.cancelled()
    await hub.shutdown()


async def test_stop_after_a_finished_task_does_nothing() -> None:
    hub = await make_hub()
    ticker = SimulationTicker(hub, "missing", sleep=lambda _s: asyncio.sleep(0))
    ticker.start()
    await wait_until(lambda: not ticker.running)
    await ticker.stop()
    assert ticker._task is None
    await hub.shutdown()


# --- ticker and the hub ---------------------------------------------------------------------


class SharedClock:
    """A hub clock the test moves by hand, so ticks can land after a chosen moment."""

    def __init__(self, now: int = 1_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


TTL_MS = 1_000


async def test_ticker_does_not_keep_an_abandoned_session_alive() -> None:
    clock = SharedClock()
    hub = SwarmSessionHub(
        make_settings(broadcast_fps=240, session_ttl_seconds=1),
        bus=InMemoryEventBus(),
        clock=clock,
    )
    await hub.create_session(SESSION, bundle())
    ticker = SimulationTicker(hub, SESSION)
    ticker.start()
    await wait_until(lambda: ticker.ticks >= 1)

    # Ticks land after the session has gone quiet. Any of them refreshing it would hide the expiry.
    clock.now += TTL_MS
    ticks = ticker.ticks
    await wait_until(lambda: ticker.ticks >= ticks + 3)

    assert await hub.sweep() == (SESSION,)
    await wait_until(lambda: not ticker.running)
    with pytest.raises(SessionNotFoundError):
        hub.latest_snapshot(SESSION)
    await ticker.stop()
    await hub.shutdown()


async def test_agent_frames_still_count_as_activity() -> None:
    clock = SharedClock()
    hub = SwarmSessionHub(make_settings(session_ttl_seconds=1), bus=InMemoryEventBus(), clock=clock)
    await hub.create_session(SESSION, bundle())
    clock.now += TTL_MS - 1
    await hub.broadcast_frame(SESSION, agent_frame(CODER, AgentAction.IDLE))
    clock.now += TTL_MS - 1
    assert await hub.sweep() == ()
    clock.now += 1
    assert await hub.sweep() == (SESSION,)
    await hub.shutdown()


async def test_two_ticking_workers_do_not_keep_each_other_alive() -> None:
    bus = InMemoryEventBus()
    clock = SharedClock()
    settings = make_settings(broadcast_fps=240, session_ttl_seconds=1)
    first = SwarmSessionHub(settings, bus=bus)
    second = SwarmSessionHub(settings, bus=bus, clock=clock)
    await first.create_session(SESSION, bundle())
    await second.create_session(SESSION, bundle())
    # Subscribed after the replica, so anything seen here has already reached the replica too.
    arrivals: list[SwarmEvent] = []
    await bus.subscribe(session_channel(SESSION), arrivals.append)
    async with SimulationTicker(first, SESSION):
        await wait_until(lambda: len(arrivals) > 0)
        clock.now += TTL_MS
        arrived = len(arrivals)
        await wait_until(lambda: len(arrivals) > arrived)
        assert isinstance(arrivals[-1], CursorEvent)
        assert await second.sweep() == (SESSION,)
    await first.shutdown()
    await second.shutdown()


async def test_real_work_on_another_worker_keeps_a_replica_alive() -> None:
    bus = InMemoryEventBus()
    clock = SharedClock()
    settings = make_settings(session_ttl_seconds=1)
    writer = SwarmSessionHub(settings, bus=bus)
    replica = SwarmSessionHub(settings, bus=bus, clock=clock)
    await writer.create_session(SESSION, bundle())
    await replica.create_session(SESSION, bundle())
    received = replica.subscribe(SESSION)

    clock.now += TTL_MS
    await writer.broadcast_thought(SESSION, make_thought(CODER))
    assert isinstance(await asyncio.wait_for(anext(received), WAIT_S), ThoughtEvent)

    assert await replica.sweep() == ()
    await writer.shutdown()
    await replica.shutdown()


async def test_ticker_steers_around_locks_held_in_the_session() -> None:
    hub = SwarmSessionHub(make_settings(), bus=InMemoryEventBus())
    layout = two_section_layout()
    await hub.create_session(SESSION, layout)
    await hub.lock_manager(SESSION).acquire("locked", "agent-other", 50, hub.now_ms())
    simulator = CursorSimulator()
    time_source = ManualTime()
    ticker = SimulationTicker(
        hub, SESSION, simulator, clock=time_source.clock, sleep=time_source.sleep
    )
    ticker.start()
    await wait_until(lambda: ticker.ticks >= 1)
    await ticker.stop()
    assert [owner for owner, _ in simulator._obstacles] == ["agent-other"]
    await hub.shutdown()


async def test_many_tickers_on_many_sessions_stay_valid_under_stress() -> None:
    hub = SwarmSessionHub(make_settings(broadcast_fps=240), bus=InMemoryEventBus())
    sessions = [f"session-{index}" for index in range(30)]
    tickers = []
    listeners = []
    for session_id in sessions:
        await hub.create_session(session_id, bundle())
        listeners.append(hub.subscribe(session_id))
        tickers.append(SimulationTicker(hub, session_id))
    baseline = asyncio.all_tasks()
    for ticker in tickers:
        ticker.start()
    await wait_until(lambda: all(ticker.ticks >= 5 for ticker in tickers), 10)
    # Half are stopped while the rest keep running, then the remainder all at once.
    await asyncio.gather(*(ticker.stop() for ticker in tickers[::2]))
    assert not any(ticker.running for ticker in tickers[::2])
    assert all(ticker.running for ticker in tickers[1::2])
    await asyncio.gather(*(ticker.stop() for ticker in tickers[1::2]))
    assert asyncio.all_tasks() - baseline == set()

    for listener, session_id in zip(listeners, sessions, strict=True):
        event = await asyncio.wait_for(anext(listener), WAIT_S)
        assert isinstance(event, CursorEvent)
        assert event.session_id == session_id
        assert 0 <= event.frame.x <= DEFAULT_VIEWPORT_WIDTH
        assert 0 <= event.frame.y <= DEFAULT_VIEWPORT_HEIGHT
    await hub.shutdown()


# --- the whole swarm under simulation -------------------------------------------------------


async def collect(listener: SessionListener) -> list[Any]:
    """Read as a live client would. Reading only at the end would see cursors fully conflated."""
    return [event async for event in listener]


class PacedProvider:
    """Gives the ticker real time between steps, as a real model's latency would."""

    def __init__(self, inner: MockLLMProvider, delay_s: float) -> None:
        self._inner = inner
        self._delay_s = delay_s

    def client_for(self, role: AgentRole) -> LLMClient:
        return _PacedClient(self._inner.client_for(role), self._delay_s)


class _PacedClient:
    def __init__(self, inner: LLMClient, delay_s: float) -> None:
        self._inner = inner
        self._delay_s = delay_s

    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        await asyncio.sleep(self._delay_s)
        return await self._inner.generate_structured(request)


async def test_swarm_under_simulation_has_one_frame_writer_and_a_lossless_thought_stream() -> None:
    hub = SwarmSessionHub(make_settings(broadcast_fps=240), bus=InMemoryEventBus())
    await hub.create_session(SESSION, bundle())
    listener = hub.subscribe(SESSION)
    simulator = CursorSimulator(seed=1)
    graph = create_swarm_graph(hub, PacedProvider(MockLLMProvider(), 0.02), cursors=simulator)

    reader = asyncio.create_task(collect(listener))
    async with SimulationTicker(hub, SESSION, simulator) as ticker:
        state = await graph.ainvoke(initial_state(SESSION, "Build it", bundle()))
        await asyncio.sleep(0.05)
    assert ticker.ticks > 0
    assert state["completed"] is True
    listener.close()
    received = await asyncio.wait_for(reader, WAIT_S)
    thoughts = [event for event in received if isinstance(event, ThoughtEvent)]
    cursors = [event for event in received if isinstance(event, CursorEvent)]
    assert len(thoughts) == len(state["agent_thoughts"])
    assert [event.seq for event in thoughts] == list(range(len(thoughts)))

    # An agent's own frame sits at (0, 0) until it has a box. None of those may reach a listener.
    assert not [event for event in cursors if (event.frame.x, event.frame.y) == (0, 0)]
    by_agent: dict[str, list[AgentCursorFrame]] = {}
    for event in cursors:
        by_agent.setdefault(event.frame.agent_id, []).append(event.frame)
    assert len(by_agent) == 6
    moved = [
        agent
        for agent, frames in by_agent.items()
        if len({(frame.x, frame.y) for frame in frames}) > 1
    ]
    assert len(moved) == 6
    assert any(
        frame.current_action is AgentAction.NAVIGATING
        for frames in by_agent.values()
        for frame in frames
    ), "agents should be seen travelling between components"
    await hub.shutdown()


async def test_without_a_simulator_agents_still_publish_their_own_frames() -> None:
    frames: list[AgentCursorFrame] = []

    class Spy(SwarmSessionHub):
        async def broadcast_frame(
            self, session_id: str, frame: AgentCursorFrame, *, touch: bool = True
        ) -> None:
            frames.append(frame)
            await super().broadcast_frame(session_id, frame, touch=touch)

    hub = Spy(make_settings(broadcast_fps=240), bus=InMemoryEventBus())
    await hub.create_session(SESSION, bundle())
    graph = create_swarm_graph(hub, MockLLMProvider())
    await graph.ainvoke(initial_state(SESSION, "Build it", bundle()))
    assert (0, 0) in {(frame.x, frame.y) for frame in frames}, "at (0, 0) until it has a box"
    await hub.shutdown()


async def test_agent_waiting_for_a_lock_hovers_with_its_request_showing() -> None:
    hub = SwarmSessionHub(make_settings(broadcast_fps=240), bus=InMemoryEventBus())
    target = box(100, 100, 400, 200)
    await hub.create_session(SESSION, bundle(component("hero", target)))
    manager = hub.lock_manager(SESSION)
    await manager.acquire("hero", "someone-else", 99, hub.now_ms())
    simulator = CursorSimulator()

    runtime = SwarmRuntime(hub, MockLLMProvider(), cursors=simulator)
    state = initial_state(SESSION, "x", bundle())
    context = NodeContext(runtime, state, CODER)

    async def wait_for_it() -> None:
        async with context.locks() as locks:
            await locks.acquire("hero")

    waiter = asyncio.create_task(wait_for_it())
    await wait_until(lambda: simulator._cursors[CODER].lock is not None)
    run_for(simulator, 3000)
    assert simulator.mode_of(CODER) is CursorMode.BLOCKED_WAITING
    assert not Rect.around(target).contains(simulator.position_of(CODER))
    shown = next(frame for frame in simulator.step(0) if frame.role is CODER).active_lock
    assert shown is not None
    assert shown.state is LockState.WAITING

    await manager.release("hero", "someone-else")
    await asyncio.wait_for(waiter, WAIT_S)
    await hub.shutdown()
