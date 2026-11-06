"""The virtual cursors of the swarm's agents, advanced one deterministic tick at a time."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from realtime_agentic_site_builder.agents.identity import IDENTITIES, AgentIdentity
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.blueprint import BoundingBox
from realtime_agentic_site_builder.domain.swarm import (
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    LockState,
)
from realtime_agentic_site_builder.simulation.motion import (
    Point,
    Rect,
    Trajectory,
    Viewport,
    build_trajectory,
    focal_point,
    hover_point,
    seed_unit,
    wander,
)

WANDER_AMPLITUDE_PX: Final = 18.0
HOVER_AMPLITUDE_PX: Final = 6.0
JITTER_AMPLITUDE_PX: Final = 2.5
# Typing is much quicker than a drifting hand, so time runs faster through the same smooth noise.
JITTER_SPEEDUP: Final = 14.0

_EDITING_ACTIONS: Final = frozenset({AgentAction.EDITING, AgentAction.PATCHING_AST})


class CursorMode(StrEnum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    EDITING = "EDITING"
    BLOCKED_WAITING = "BLOCKED_WAITING"


@dataclass(slots=True)
class _Cursor:
    identity: AgentIdentity
    position: Point
    anchor: Point
    goal: Point
    mode: CursorMode = CursorMode.IDLE
    settled_mode: CursorMode = CursorMode.IDLE
    entered_ms: int = 0
    trajectory: Trajectory | None = None
    action: AgentAction = AgentAction.IDLE
    thought: AgentThoughtStream | None = None
    lock: ElementLock | None = None


class CursorSimulator:
    """Owns where each agent's cursor is. Agents say what they are doing, and this makes it move.

    Each position is a pure function of the simulated time since the cursor last changed state, not
    of how the time was divided into ticks. Runs are reproducible for a given seed and script, and
    a late tick lands exactly where a punctual one would have.
    """

    def __init__(self, viewport: Viewport | None = None, *, seed: int = 0) -> None:
        self._viewport = viewport or Viewport()
        self._seed = seed
        self._now_ms = 0
        self._obstacles: tuple[tuple[str, Rect], ...] = ()
        self._cursors = {role: self._new_cursor(identity) for role, identity in IDENTITIES.items()}

    def _new_cursor(self, identity: AgentIdentity) -> _Cursor:
        start = Point(
            seed_unit(self._seed, identity.agent_id, "x") * self._viewport.width,
            seed_unit(self._seed, identity.agent_id, "y") * self._viewport.height,
        )
        return _Cursor(identity=identity, position=start, anchor=start, goal=start)

    @property
    def viewport(self) -> Viewport:
        return self._viewport

    @property
    def roles(self) -> tuple[AgentRole, ...]:
        return tuple(self._cursors)

    def mode_of(self, role: AgentRole) -> CursorMode:
        return self._cursors[role].mode

    def position_of(self, role: AgentRole) -> Point:
        return self._cursors[role].position

    def navigate_to(self, role: AgentRole, box: BoundingBox) -> None:
        """Travel to where this role works in the box, and float there."""
        self._travel(self._cursors[role], focal_point(box, role), CursorMode.IDLE, AgentAction.IDLE)

    def edit(self, role: AgentRole, box: BoundingBox) -> None:
        """Travel to the box, and make typing movements on arrival."""
        self._travel(
            self._cursors[role], focal_point(box, role), CursorMode.EDITING, AgentAction.EDITING
        )

    def block_on(self, role: AgentRole, box: BoundingBox, lock: ElementLock | None = None) -> None:
        """Travel to just outside a box this agent is waiting for, and hover there."""
        cursor = self._cursors[role]
        cursor.lock = lock
        goal = hover_point(box, self._viewport, self._rects_for(cursor))
        self._travel(cursor, goal, CursorMode.BLOCKED_WAITING, AgentAction.BLOCKED_WAITING)

    def idle(self, role: AgentRole) -> None:
        """Stop where it is, and float."""
        cursor = self._cursors[role]
        cursor.lock = None
        # It may have stopped short, and must not count as docked at where it had been heading.
        cursor.goal = cursor.position
        self._settle(cursor, CursorMode.IDLE, AgentAction.IDLE, self._now_ms)

    def direct(self, frame: AgentCursorFrame, box: BoundingBox | None) -> None:
        """Take an agent's frame as a statement of intent. Its coordinates are not used.

        Agents report what they are doing, and often where. A frame with no box leaves the cursor
        where it is, unless it says the agent has gone idle.
        """
        cursor = self._cursors[frame.role]
        cursor.thought = frame.current_thought
        cursor.lock = frame.active_lock
        action = frame.current_action
        if box is None:
            if action is AgentAction.IDLE:
                self.idle(frame.role)
            elif action is not AgentAction.NAVIGATING and cursor.mode is not CursorMode.NAVIGATING:
                cursor.action = action
            return
        if action is AgentAction.BLOCKED_WAITING:
            self.block_on(frame.role, box, frame.active_lock)
            return
        mode = CursorMode.EDITING if action in _EDITING_ACTIONS else CursorMode.IDLE
        settled = action if action is not AgentAction.NAVIGATING else AgentAction.IDLE
        self._travel(cursor, focal_point(box, frame.role), mode, settled)

    def sync(self, bundle: FullstackProjectBundle, locks: Iterable[ElementLock]) -> None:
        """Take the current layout and locks from the session, rerouting whoever they obstruct.

        Only a held lock on a component has a place on the canvas. Document locks and models do
        not, and a queued request does not stop anyone yet.
        """
        boxes: Mapping[str, BoundingBox] = {c.id: c.box for c in bundle.frontend.components}
        self._viewport = self._viewport.fitted_to(tuple(boxes.values()))
        obstacles = tuple(
            (lock.locked_by_agent_id, Rect.around(boxes[lock.target_id]))
            for lock in locks
            if lock.state is LockState.ACQUIRED and lock.target_id in boxes
        )
        if obstacles != self._obstacles:
            self._obstacles = obstacles
            for cursor in self._cursors.values():
                self._reroute_if_obstructed(cursor)

    def step(self, delta_time_ms: int) -> tuple[AgentCursorFrame, ...]:
        """Advance simulated time, and return every cursor's frame in a stable order."""
        if delta_time_ms < 0:
            raise ValueError("simulated time cannot run backwards")
        self._now_ms += delta_time_ms
        for cursor in self._cursors.values():
            self._advance(cursor)
        return tuple(self._frame(cursor) for cursor in self._cursors.values())

    def _rects_for(self, cursor: _Cursor) -> tuple[Rect, ...]:
        """Return what this cursor steers around: everyone's locked boxes except its own."""
        return tuple(rect for owner, rect in self._obstacles if owner != cursor.identity.agent_id)

    def _travel(self, cursor: _Cursor, goal: Point, mode: CursorMode, action: AgentAction) -> None:
        clamped = self._viewport.clamp(goal)
        if cursor.goal == clamped:
            # Already bound for or docked at this spot. Keep going and only update what happens
            # there, because restarting would visibly stall the cursor at every repeated frame.
            if cursor.mode is CursorMode.NAVIGATING:
                cursor.settled_mode, cursor.action = mode, action
            elif cursor.mode is not mode or cursor.action is not action:
                self._settle(cursor, mode, action, self._now_ms)
            return
        cursor.goal = clamped
        cursor.settled_mode, cursor.action = mode, action
        self._launch(cursor)

    def _launch(self, cursor: _Cursor) -> None:
        bow_unit = seed_unit(self._seed, cursor.identity.agent_id, "bow", self._now_ms)
        cursor.trajectory = build_trajectory(
            cursor.position,
            cursor.goal,
            self._rects_for(cursor),
            self._viewport,
            bow_unit=bow_unit,
        )
        cursor.mode = CursorMode.NAVIGATING
        cursor.entered_ms = self._now_ms

    def _reroute_if_obstructed(self, cursor: _Cursor) -> None:
        trajectory = cursor.trajectory
        if cursor.mode is not CursorMode.NAVIGATING or trajectory is None:
            return
        fraction = trajectory.fraction_at(self._now_ms - cursor.entered_ms)
        # A box containing the goal cannot be avoided, so it is no reason to replan.
        avoidable = [rect for rect in self._rects_for(cursor) if not rect.contains(cursor.goal)]
        if trajectory.crosses(avoidable, after=fraction):
            self._launch(cursor)

    def _settle(self, cursor: _Cursor, mode: CursorMode, action: AgentAction, at_ms: int) -> None:
        cursor.mode = cursor.settled_mode = mode
        cursor.action = action
        cursor.anchor = cursor.position
        cursor.entered_ms = at_ms
        cursor.trajectory = None

    def _advance(self, cursor: _Cursor) -> None:
        if cursor.mode is CursorMode.NAVIGATING and cursor.trajectory is not None:
            elapsed = self._now_ms - cursor.entered_ms
            cursor.position = cursor.trajectory.position_at(elapsed)
            if elapsed < cursor.trajectory.duration_ms:
                return
            # Time continues from the moment of arrival, not from whichever tick noticed it.
            arrived_ms = cursor.entered_ms + round(cursor.trajectory.duration_ms)
            self._settle(cursor, cursor.settled_mode, cursor.action, arrived_ms)
        cursor.position = self._viewport.clamp(self._settled_position(cursor))

    def _settled_position(self, cursor: _Cursor) -> Point:
        elapsed = self._now_ms - cursor.entered_ms
        agent = cursor.identity.agent_id
        if cursor.mode is CursorMode.EDITING:
            offset = wander(self._seed, agent, elapsed * JITTER_SPEEDUP, JITTER_AMPLITUDE_PX)
        elif cursor.mode is CursorMode.BLOCKED_WAITING:
            offset = wander(self._seed, agent, elapsed, HOVER_AMPLITUDE_PX)
        else:
            offset = wander(self._seed, agent, elapsed, WANDER_AMPLITUDE_PX)
        return Point(cursor.anchor.x + offset.x, cursor.anchor.y + offset.y)

    def _frame(self, cursor: _Cursor) -> AgentCursorFrame:
        identity = cursor.identity
        return AgentCursorFrame(
            agent_id=identity.agent_id,
            name=identity.name,
            role=identity.role,
            color=identity.color,
            x=round(cursor.position.x),
            y=round(cursor.position.y),
            current_action=(
                AgentAction.NAVIGATING if cursor.mode is CursorMode.NAVIGATING else cursor.action
            ),
            current_thought=cursor.thought,
            active_lock=cursor.lock,
        )
