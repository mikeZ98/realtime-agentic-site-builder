"""Pure geometry for the canvas: the elastic viewport, smooth motion and docking on components.

Telemetry arrives at about 30 FPS at best, and conflation or a poor network drops frames from that.
A renderer that drew samples as they came would stutter, so positions are reconstructed as a cubic
Hermite spline through the samples that did arrive, using their real timestamps. The viewer runs the
same arithmetic between frames, so these constants are compiled into it rather than repeated.
"""

from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from typing import Final, NamedTuple

from realtime_agentic_site_builder.domain.base import DomainModel
from realtime_agentic_site_builder.domain.blueprint import BoundingBox
from realtime_agentic_site_builder.simulation.motion import (
    Point,
    Viewport,
    cubic_bezier,
    distance,
    ease_in_out,
)

# Rendering this far in the past means there is usually a later sample to interpolate towards.
SMOOTHING_DELAY_MS: Final = 34
# How long a track coasts past its last sample. Beyond this, guessing looks worse than stopping.
EXTRAPOLATION_LIMIT_MS: Final = 250
# With no sample for this long, a cursor was not moving between its samples, so it restarts there.
MOTION_RESET_MS: Final = 500
# How long a track takes to absorb the error found when a sample finally ends a stretch of coasting.
CONVERGENCE_MS: Final = 120
MAX_SPEED_PX_PER_MS: Final = 10.0
DOCK_HYSTERESIS_PX: Final = 24.0
BADGE_GAP_PX: Final = 16.0
_MAX_KNOTS: Final = 16


class NormalizedBox(DomainModel):
    """A box as fractions of the canvas, which stay correct at any display scale."""

    x: float
    y: float
    width: float
    height: float


def elastic_viewport(boxes: tuple[BoundingBox, ...], current: Viewport | None = None) -> Viewport:
    """Grow the canvas to hold the content. It starts at 1920x1080, and never shrinks.

    Cursors may still be working down the page, and a canvas that shrank under them would make
    every normalized box on it jump.
    """
    return (current or Viewport()).fitted_to(boxes)


def normalize_box(box: BoundingBox, viewport: Viewport) -> NormalizedBox:
    """Map a box to fractions of the canvas. Whatever lies past the largest canvas is clipped."""
    left = min(box.x, viewport.width)
    top = min(box.y, viewport.height)
    return NormalizedBox(
        x=left / viewport.width,
        y=top / viewport.height,
        width=(min(box.right, viewport.width) - left) / viewport.width,
        height=(min(box.bottom, viewport.height) - top) / viewport.height,
    )


def hermite_controls(
    start: Point, start_velocity: Point, end: Point, end_velocity: Point, duration_ms: float
) -> tuple[Point, Point, Point, Point]:
    """Express a Hermite segment as Bézier control points, which is how the viewer draws trails."""
    third = duration_ms / 3.0
    return (
        start,
        Point(start.x + start_velocity.x * third, start.y + start_velocity.y * third),
        Point(end.x - end_velocity.x * third, end.y - end_velocity.y * third),
        end,
    )


class MotionState(NamedTuple):
    """Where a track is, and how fast it is going in pixels per millisecond."""

    position: Point
    velocity: Point


@dataclass(slots=True)
class _Knot:
    t_ms: float
    position: Point
    velocity: Point


def _limited(velocity: Point) -> Point:
    """Cap a tangent, so a teleport between samples cannot fling the curve off the canvas."""
    speed = distance(Point(0.0, 0.0), velocity)
    if speed <= MAX_SPEED_PX_PER_MS:
        return velocity
    return Point(
        velocity.x * MAX_SPEED_PX_PER_MS / speed,
        velocity.y * MAX_SPEED_PX_PER_MS / speed,
    )


def _slope(earlier: _Knot, later: _Knot) -> Point:
    elapsed = later.t_ms - earlier.t_ms
    return _limited(
        Point(
            (later.position.x - earlier.position.x) / elapsed,
            (later.position.y - earlier.position.y) / elapsed,
        )
    )


class DeadReckoner:
    """One track's recent samples, as a spline passing exactly through each of them.

    Tangents use the real time between neighbours (Catmull-Rom), so a dropped frame just widens a
    segment. It does not distort the speed on either side of it.

    A burst of losses is different, because it outlasts any render delay and the track has to guess.
    The next sample then shows how wrong the guess was, by hundreds of pixels for a fast cursor. So
    what is shown converges on the spline instead of jumping to it. The correction is only ever the
    difference between the two predictions for one instant, so it is near zero in normal running
    and cannot hold a track back. Samples have to be requested in time order, as a render loop does.
    """

    __slots__ = ("_coasting_from", "_correction", "_correction_at", "_knots")

    def __init__(self, t_ms: float, position: Point) -> None:
        self._knots: deque[_Knot] = deque(
            [_Knot(t_ms, position, Point(0.0, 0.0))], maxlen=_MAX_KNOTS
        )
        self._coasting_from: _Knot | None = None
        self._correction = Point(0.0, 0.0)
        self._correction_at = 0.0

    def observe(self, t_ms: float, position: Point) -> None:
        """Record a sample. One that is not newer than the last one is out of date, and ignored."""
        last = self._knots[-1]
        if t_ms <= last.t_ms:
            return
        if t_ms - last.t_ms > MOTION_RESET_MS:
            self._knots.clear()
            self._knots.append(_Knot(t_ms, position, Point(0.0, 0.0)))
            return
        knot = _Knot(t_ms, position, Point(0.0, 0.0))
        knot.velocity = _slope(last, knot)
        if len(self._knots) >= 2:
            last.velocity = _slope(self._knots[-2], knot)
        self._knots.append(knot)

    def sample(self, t_ms: float, viewport: Viewport) -> MotionState:
        """Report where the track appears to be, including what is left of any correction."""
        state = self._spline(t_ms, viewport)
        last = self._knots[-1]
        if self._coasting_from is not None and self._coasting_from.t_ms != last.t_ms:
            # New data has ended a coast. What it was about to show at this instant is the start.
            coasted = self._coast(self._coasting_from, t_ms - self._coasting_from.t_ms, viewport)
            expected = self._corrected(coasted.position, t_ms)
            self._correction = Point(expected.x - state.position.x, expected.y - state.position.y)
            self._correction_at = t_ms
        # A copy, because a knot's tangent changes once its successor is known.
        self._coasting_from = replace(last) if t_ms >= last.t_ms else None
        return MotionState(viewport.clamp(self._corrected(state.position, t_ms)), state.velocity)

    def _corrected(self, position: Point, t_ms: float) -> Point:
        remaining = 1.0 - ease_in_out((t_ms - self._correction_at) / CONVERGENCE_MS)
        return Point(
            position.x + self._correction.x * remaining,
            position.y + self._correction.y * remaining,
        )

    def _spline(self, t_ms: float, viewport: Viewport) -> MotionState:
        """Interpolate within the samples, or coast past the last one and ease to a stop."""
        first, last = self._knots[0], self._knots[-1]
        if t_ms <= first.t_ms:
            return MotionState(viewport.clamp(first.position), first.velocity)
        if t_ms >= last.t_ms:
            return self._coast(last, t_ms - last.t_ms, viewport)
        earlier, later = next(pair for pair in pairwise(self._knots) if t_ms <= pair[1].t_ms)
        duration = later.t_ms - earlier.t_ms
        u = (t_ms - earlier.t_ms) / duration
        controls = hermite_controls(
            earlier.position, earlier.velocity, later.position, later.velocity, duration
        )
        return MotionState(
            viewport.clamp(cubic_bezier(*controls, u)), _bezier_velocity(controls, u, duration)
        )

    @staticmethod
    def _coast(last: _Knot, past_ms: float, viewport: Viewport) -> MotionState:
        """Decelerate evenly, which keeps both position and velocity continuous at every instant."""
        coast = min(past_ms, EXTRAPOLATION_LIMIT_MS)
        travel = coast - coast * coast / (2 * EXTRAPOLATION_LIMIT_MS)
        remaining = 1.0 - coast / EXTRAPOLATION_LIMIT_MS
        return MotionState(
            viewport.clamp(
                Point(
                    last.position.x + last.velocity.x * travel,
                    last.position.y + last.velocity.y * travel,
                )
            ),
            Point(last.velocity.x * remaining, last.velocity.y * remaining),
        )


def _bezier_velocity(
    controls: tuple[Point, Point, Point, Point], u: float, duration_ms: float
) -> Point:
    p0, p1, p2, p3 = controls
    inverse = 1.0 - u
    a, b, c = 3 * inverse * inverse, 6 * inverse * u, 3 * u * u
    return Point(
        (a * (p1.x - p0.x) + b * (p2.x - p1.x) + c * (p3.x - p2.x)) / duration_ms,
        (a * (p1.y - p0.y) + b * (p2.y - p1.y) + c * (p3.y - p2.y)) / duration_ms,
    )


class DockSide(StrEnum):
    TOP = "TOP"
    RIGHT = "RIGHT"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"


_OUTWARD: Final = {
    DockSide.TOP: Point(0.0, -1.0),
    DockSide.RIGHT: Point(1.0, 0.0),
    DockSide.BOTTOM: Point(0.0, 1.0),
    DockSide.LEFT: Point(-1.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class Dock:
    side: DockSide
    anchor: Point
    label_at: Point


def box_gap(point: Point, box: BoundingBox) -> float:
    """Return how far a point is from a box, which is zero anywhere inside it."""
    return distance(point, _clamp_into(point, box))


def _clamp_into(point: Point, box: BoundingBox) -> Point:
    return Point(min(max(point.x, box.x), box.right), min(max(point.y, box.y), box.bottom))


def resolve_dock(
    cursor: Point, box: BoundingBox, viewport: Viewport, previous: DockSide | None = None
) -> Dock:
    """Find where on a component's border a cursor docks, sliding along it as the cursor moves.

    Inside a wide section, the nearest edge flips at the midline, and a cursor drifting across that
    line would make its label jump between the two edges. So the side already in use is kept until
    another is nearer by a clear margin.
    """
    inside = _clamp_into(cursor, box)
    gaps = {
        DockSide.TOP: inside.y - box.y,
        DockSide.RIGHT: box.right - inside.x,
        DockSide.BOTTOM: box.bottom - inside.y,
        DockSide.LEFT: inside.x - box.x,
    }
    side = min(gaps, key=lambda candidate: gaps[candidate])
    if previous is not None and gaps[previous] <= gaps[side] + DOCK_HYSTERESIS_PX:
        side = previous
    anchor = {
        DockSide.TOP: Point(inside.x, box.y),
        DockSide.RIGHT: Point(box.right, inside.y),
        DockSide.BOTTOM: Point(inside.x, box.bottom),
        DockSide.LEFT: Point(box.x, inside.y),
    }[side]
    outward = _OUTWARD[side]
    return Dock(
        side=side,
        anchor=viewport.clamp(anchor),
        label_at=viewport.clamp(
            Point(anchor.x + outward.x * BADGE_GAP_PX, anchor.y + outward.y * BADGE_GAP_PX)
        ),
    )
