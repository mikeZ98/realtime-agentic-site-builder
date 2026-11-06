"""Pure geometry for human-like cursor motion. Everything here is a deterministic function.

Avoidance is planned, not a repulsive force. Page sections are wide and stacked, so a force pushing
away from a section's nearest edge would flip direction at its midline and make the cursor jump. A
route around the obstacle's corners has no such discontinuity, and can be checked for clearance.
"""

import heapq
import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

from realtime_agentic_site_builder.domain.base import sha256_hex
from realtime_agentic_site_builder.domain.blueprint import BoundingBox
from realtime_agentic_site_builder.domain.swarm import (
    VIRTUAL_VIEWPORT_HEIGHT,
    VIRTUAL_VIEWPORT_WIDTH,
    AgentRole,
)

DEFAULT_VIEWPORT_WIDTH: Final = 1920
DEFAULT_VIEWPORT_HEIGHT: Final = 1080
# How far a route stays from an obstacle, and where a blocked cursor hovers. A hover point has to
# sit outside the clearance margin, or a route to it could never be clear of the box it waits on.
CLEARANCE_PX: Final = 16.0
HOVER_GAP_PX: Final = 24.0
FOCAL_INSET_PX: Final = 24.0
# Fitts's law: time grows with the logarithm of distance over target size.
FITTS_BASE_MS: Final = 120.0
FITTS_SLOPE_MS: Final = 150.0
DEFAULT_TARGET_WIDTH_PX: Final = 100.0
_MAX_BOW_FRACTION: Final = 0.12
_HANDLE_FRACTION: Final = 0.33
_TENSIONS: Final = (1.0, 0.5)
_SAMPLES_PER_SEGMENT: Final = 24
_MIN_SAMPLES: Final = 16
_MIN_TRAVEL_PX: Final = 0.5
_SAMPLE_SPACING_PX: Final = 4.0
_WANDER_PERIODS_MS: Final = (2300.0, 3700.0, 5900.0)


class Point(NamedTuple):
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Rect:
    """Unlike a ``BoundingBox``, this can extend past the origin once inflated."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def around(cls, box: BoundingBox, margin: float = 0.0) -> "Rect":
        return cls(box.x - margin, box.y - margin, box.right + margin, box.bottom + margin)

    def inflated(self, margin: float) -> "Rect":
        return Rect(
            self.left - margin, self.top - margin, self.right + margin, self.bottom + margin
        )

    def contains(self, point: Point) -> bool:
        """Report strict containment. A point on the boundary is clear of the rectangle."""
        return self.left < point.x < self.right and self.top < point.y < self.bottom

    def corners(self) -> tuple[Point, ...]:
        return (
            Point(self.left, self.top),
            Point(self.right, self.top),
            Point(self.left, self.bottom),
            Point(self.right, self.bottom),
        )

    def blocks_segment(self, start: Point, end: Point) -> bool:
        """Clip the segment against each axis, and report any stretch through the interior."""
        low, high = 0.0, 1.0
        for origin, delta, minimum, maximum in (
            (start.x, end.x - start.x, self.left, self.right),
            (start.y, end.y - start.y, self.top, self.bottom),
        ):
            if delta == 0:
                if not minimum < origin < maximum:
                    return False
                continue
            first, second = (minimum - origin) / delta, (maximum - origin) / delta
            low = max(low, min(first, second))
            high = min(high, max(first, second))
        return low < high


@dataclass(frozen=True, slots=True)
class Viewport:
    width: int = DEFAULT_VIEWPORT_WIDTH
    height: int = DEFAULT_VIEWPORT_HEIGHT

    def __post_init__(self) -> None:
        if not 1 <= self.width <= VIRTUAL_VIEWPORT_WIDTH:
            raise ValueError(f"viewport width must be within 1..{VIRTUAL_VIEWPORT_WIDTH}")
        if not 1 <= self.height <= VIRTUAL_VIEWPORT_HEIGHT:
            raise ValueError(f"viewport height must be within 1..{VIRTUAL_VIEWPORT_HEIGHT}")

    def clamp(self, point: Point) -> Point:
        return Point(min(max(point.x, 0.0), self.width), min(max(point.y, 0.0), self.height))

    def contains(self, point: Point) -> bool:
        return 0 <= point.x <= self.width and 0 <= point.y <= self.height

    def fitted_to(self, boxes: Sequence[BoundingBox]) -> "Viewport":
        """Grow to hold every box, up to what a frame can carry. A viewport never shrinks here.

        A stacked page outgrows the default canvas quickly, and its lower sections would otherwise
        be unreachable, with every cursor bound for them pinned to the bottom edge.
        """
        right = max((box.right for box in boxes), default=0)
        bottom = max((box.bottom for box in boxes), default=0)
        return Viewport(
            min(max(self.width, right), VIRTUAL_VIEWPORT_WIDTH),
            min(max(self.height, bottom), VIRTUAL_VIEWPORT_HEIGHT),
        )


def distance(start: Point, end: Point) -> float:
    return math.hypot(end.x - start.x, end.y - start.y)


def ease_in_out(progress: float) -> float:
    """Accelerate, then decelerate. Monotonic, and flat at both ends so motion starts at rest."""
    bounded = min(max(progress, 0.0), 1.0)
    return bounded * bounded * (3.0 - 2.0 * bounded)


def cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    inverse = 1.0 - t
    a, b, c, d = inverse**3, 3 * inverse**2 * t, 3 * inverse * t**2, t**3
    return Point(
        a * p0.x + b * p1.x + c * p2.x + d * p3.x,
        a * p0.y + b * p1.y + c * p2.y + d * p3.y,
    )


def fitts_duration_ms(travel_px: float, target_width_px: float = DEFAULT_TARGET_WIDTH_PX) -> float:
    return FITTS_BASE_MS + FITTS_SLOPE_MS * math.log2(1.0 + travel_px / max(target_width_px, 1.0))


def seed_unit(*parts: object) -> float:
    """Map values to a stable number in [0, 1). The built-in ``hash`` is salted per process."""
    digest = sha256_hex("|".join(str(part) for part in parts))
    return int(digest[:12], 16) / 16**12


def wander(seed: int, salt: str, elapsed_ms: float, amplitude_px: float) -> Point:
    """Smooth drift that starts at the origin, so it can begin from any position without a jump.

    It sums incommensurate sines and subtracts their starting value, so it depends only on the
    elapsed time and never accumulates. The offset is at most ``amplitude_px`` on each axis.
    """
    x_total = y_total = 0.0
    scale = amplitude_px / (2 * len(_WANDER_PERIODS_MS))
    for index, period in enumerate(_WANDER_PERIODS_MS):
        x_phase = math.tau * seed_unit(seed, salt, "x", index)
        y_phase = math.tau * seed_unit(seed, salt, "y", index)
        turn = math.tau * elapsed_ms / period
        x_total += math.sin(turn + x_phase) - math.sin(x_phase)
        y_total += math.sin(turn + y_phase) - math.sin(y_phase)
    return Point(x_total * scale, y_total * scale)


def focal_point(box: BoundingBox, role: AgentRole) -> Point:
    """Choose where a role naturally works in a component.

    The inset never passes the box's center, so a small box still yields a point inside it.
    """
    inset_x = min(FOCAL_INSET_PX, box.width / 2)
    inset_y = min(FOCAL_INSET_PX, box.height / 2)
    left, right = box.x + inset_x, box.right - inset_x
    top, bottom = box.y + inset_y, box.bottom - inset_y
    center_x, center_y = box.x + box.width / 2, box.y + box.height / 2
    return {
        AgentRole.STYLER: Point(left, top),
        AgentRole.ARCHITECT: Point(center_x, top),
        AgentRole.COPYWRITER: Point(left, center_y),
        AgentRole.BACKEND_ENGINEER: Point(right, center_y),
        AgentRole.CRITIC_LINTER: Point(right, bottom),
    }.get(role, Point(center_x, center_y))


def hover_point(box: BoundingBox, viewport: Viewport, obstacles: Sequence[Rect] = ()) -> Point:
    """Find where a blocked cursor waits: just outside the box, and clear of other locked boxes.

    Sections are usually stacked, so the side gutters are often the only free space.
    """
    center_x, center_y = box.x + box.width / 2, box.y + box.height / 2
    candidates = (
        Point(box.right + HOVER_GAP_PX, center_y),
        Point(box.x - HOVER_GAP_PX, center_y),
        Point(center_x, box.y - HOVER_GAP_PX),
        Point(center_x, box.bottom + HOVER_GAP_PX),
    )
    for candidate in candidates:
        if viewport.contains(candidate) and not any(o.contains(candidate) for o in obstacles):
            return candidate
    return viewport.clamp(candidates[0])


def plan_route(
    start: Point, goal: Point, obstacles: Sequence[Rect], viewport: Viewport
) -> tuple[Point, ...]:
    """Return the shortest clear polyline, via obstacle corners, from start to goal.

    Obstacles are inflated by the clearance margin. One that already contains an endpoint cannot be
    avoided, so it is ignored. If no route exists, as when a section spans the whole viewport, this
    returns the direct line: a cursor crossing a locked box is cosmetic, and being stuck is not.
    """
    inflated = [rect.inflated(CLEARANCE_PX) for rect in obstacles]
    active = [r for r in inflated if not r.contains(start) and not r.contains(goal)]
    direct = (start, goal)
    if not any(rect.blocks_segment(start, goal) for rect in active):
        return direct
    nodes = [start, goal]
    nodes.extend(c for rect in active for c in rect.corners() if viewport.contains(c))

    def clear(a: Point, b: Point) -> bool:
        return not any(rect.blocks_segment(a, b) for rect in active)

    best = {0: 0.0}
    previous: dict[int, int] = {}
    queue = [(0.0, 0)]
    while queue:
        cost, current = heapq.heappop(queue)
        if current == 1:
            route = [goal]
            while current in previous:
                current = previous[current]
                route.append(nodes[current])
            return tuple(reversed(route))
        for index, node in enumerate(nodes):
            if index != current and clear(nodes[current], node):
                total = cost + distance(nodes[current], node)
                if total < best.get(index, math.inf):
                    best[index] = total
                    previous[index] = current
                    heapq.heappush(queue, (total, index))
    return direct


class Trajectory:
    """A chain of cubic Béziers traversed by arc length, with eased speed over its duration.

    Every control point is clamped into the viewport. A Bézier stays inside the convex hull of its
    control points, and a rectangle is convex, so the curve only leaves the viewport by rounding
    error. Positions are clamped as well, to cover that: ``1920.0000000000002`` does happen.
    """

    __slots__ = ("_lengths", "_segments", "_viewport", "duration_ms", "length")

    def __init__(
        self, segments: Sequence[tuple[Point, Point, Point, Point]], viewport: Viewport
    ) -> None:
        self._segments = tuple(segments)
        self._viewport = viewport
        lengths = [0.0]
        for segment in self._segments:
            before = segment[0]
            for step in range(1, _SAMPLES_PER_SEGMENT + 1):
                point = cubic_bezier(*segment, step / _SAMPLES_PER_SEGMENT)
                lengths.append(lengths[-1] + distance(before, point))
                before = point
        self._lengths = lengths
        self.length = lengths[-1]
        # Rounding gives even a trip to the same spot a tiny length, which is no journey at all.
        self.duration_ms = fitts_duration_ms(self.length) if self.length >= _MIN_TRAVEL_PX else 0.0

    def point_at_length_fraction(self, fraction: float) -> Point:
        """Return the position a given share of the way along the curve's length."""
        bounded = min(max(fraction, 0.0), 1.0)
        target = bounded * self.length
        sample = min(bisect_right(self._lengths, target) - 1, len(self._lengths) - 2)
        span = self._lengths[sample + 1] - self._lengths[sample]
        local = (target - self._lengths[sample]) / span if span > 0 else 0.0
        segment_index = min(sample // _SAMPLES_PER_SEGMENT, len(self._segments) - 1)
        within = sample - segment_index * _SAMPLES_PER_SEGMENT
        point = cubic_bezier(
            *self._segments[segment_index], (within + local) / _SAMPLES_PER_SEGMENT
        )
        return self._viewport.clamp(point)

    def fraction_at(self, elapsed_ms: float) -> float:
        """Return the share of the length covered after a time, with speed eased in and out."""
        if self.duration_ms == 0:
            return 1.0
        return ease_in_out(elapsed_ms / self.duration_ms)

    def position_at(self, elapsed_ms: float) -> Point:
        return self.point_at_length_fraction(self.fraction_at(elapsed_ms))

    def sample(self, *, after: float = 0.0) -> tuple[Point, ...]:
        """Return points along the curve from ``after``, a share of its length, to the end.

        They are spaced by distance, not count, so a long path cannot step over a thin box.
        """
        span = 1.0 - after
        count = max(_MIN_SAMPLES, int(self.length * span / _SAMPLE_SPACING_PX) + 2)
        return tuple(
            self.point_at_length_fraction(after + span * index / (count - 1))
            for index in range(count)
        )

    def crosses(self, rects: Sequence[Rect], *, after: float = 0.0) -> bool:
        return any(rect.contains(point) for point in self.sample(after=after) for rect in rects)


def _smooth_segments(
    route: Sequence[Point], viewport: Viewport, tension: float, bow: float
) -> list[tuple[Point, Point, Point, Point]]:
    """Build one cubic per leg, with shared tangents at joints so the path has no corners."""
    tangents: list[Point] = []
    for index in range(len(route)):
        before = route[max(index - 1, 0)]
        after = route[min(index + 1, len(route) - 1)]
        length = distance(before, after)
        tangents.append(
            Point((after.x - before.x) / length, (after.y - before.y) / length)
            if length > 0
            else Point(0.0, 0.0)
        )
    segments = []
    for index in range(len(route) - 1):
        start, end = route[index], route[index + 1]
        handle = _HANDLE_FRACTION * distance(start, end) * tension
        normal = Point(-(end.y - start.y), end.x - start.x)
        first = Point(
            start.x + tangents[index].x * handle + normal.x * bow,
            start.y + tangents[index].y * handle + normal.y * bow,
        )
        second = Point(
            end.x - tangents[index + 1].x * handle + normal.x * bow,
            end.y - tangents[index + 1].y * handle + normal.y * bow,
        )
        segments.append((start, viewport.clamp(first), viewport.clamp(second), end))
    return segments


def build_trajectory(
    start: Point,
    goal: Point,
    obstacles: Sequence[Rect],
    viewport: Viewport,
    *,
    bow_unit: float = 0.5,
) -> Trajectory:
    """Plan a clear route, then curve it as much as clearance allows.

    ``bow_unit`` in [0, 1) picks the arc's direction and size, so each agent has its own hand. The
    curve is checked against the real boxes and straightened if it cuts through one. With no tension
    and no bow, a route is straight legs, already proven clear of the larger inflated boxes.
    """
    origin, destination = viewport.clamp(start), viewport.clamp(goal)
    route = plan_route(origin, destination, obstacles, viewport)
    # Only obstacles the route actually avoided are held against the curve.
    avoidable = [
        rect
        for rect in obstacles
        if not rect.inflated(CLEARANCE_PX).contains(origin)
        and not rect.inflated(CLEARANCE_PX).contains(destination)
    ]
    bow = (bow_unit * 2.0 - 1.0) * _MAX_BOW_FRACTION
    for tension in _TENSIONS:
        trajectory = Trajectory(_smooth_segments(route, viewport, tension, bow), viewport)
        if not trajectory.crosses(avoidable):
            return trajectory
    return Trajectory(_smooth_segments(route, viewport, 0.0, 0.0), viewport)
