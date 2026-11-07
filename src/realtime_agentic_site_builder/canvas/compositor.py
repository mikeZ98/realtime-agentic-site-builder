"""Merges a session's events and lock table into one visual frame, for the canvas to draw."""

import colorsys
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.canvas.projection import (
    SMOOTHING_DELAY_MS,
    DeadReckoner,
    DockSide,
    MotionState,
    NormalizedBox,
    box_gap,
    elastic_viewport,
    normalize_box,
    resolve_dock,
)
from realtime_agentic_site_builder.core.hub import (
    CursorEvent,
    PresenceEvent,
    StateEvent,
    SwarmEvent,
    ThoughtEvent,
)
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.base import DomainModel
from realtime_agentic_site_builder.domain.blueprint import ComponentNode
from realtime_agentic_site_builder.domain.swarm import (
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    ElementLock,
    LockState,
)
from realtime_agentic_site_builder.gateway.auth import AGENT_ID_PREFIX
from realtime_agentic_site_builder.simulation.motion import (
    DEFAULT_VIEWPORT_HEIGHT,
    DEFAULT_VIEWPORT_WIDTH,
    Point,
    Viewport,
    seed_unit,
)

SWARM_PALETTE: Final[Mapping[AgentRole, str]] = {
    role: identity.color for role, identity in IDENTITIES.items()
}
UNOWNED_COLOR: Final = "#94A3B8"
DOCKING_ACTIONS: Final = frozenset(
    {
        AgentAction.EDITING,
        AgentAction.NAVIGATING,
        AgentAction.BLOCKED_WAITING,
        AgentAction.PATCHING_AST,
    }
)
# What a fresh viewer is shown of the past. One that reconnects asks for what it missed instead.
MAX_THOUGHT_HISTORY: Final = 50
MAX_SNIPPET_LENGTH: Final = 200
BADGE_ENTER_MS: Final = 200
BADGE_HOLD_MS: Final = 6000
BADGE_EXIT_MS: Final = 400
# A new thought fades its predecessor out rather than replacing it, and nothing older is kept.
_BADGES_PER_AGENT: Final = 2
_NAMES_BY_AGENT_ID: Final[Mapping[str, tuple[str, str]]] = {
    identity.agent_id: (identity.name, identity.color) for identity in IDENTITIES.values()
}


class OverlayState(StrEnum):
    FREE = "FREE"
    LOCKED = "LOCKED"
    WAITING = "WAITING"


class BadgePhase(StrEnum):
    ENTERING = "ENTERING"
    VISIBLE = "VISIBLE"
    EXITING = "EXITING"


class Participant(DomainModel):
    id: str
    label: str
    color: str


class DockView(DomainModel):
    target_id: str
    side: DockSide
    x: float
    y: float


class AgentSprite(DomainModel):
    id: str
    label: str
    role: AgentRole
    color: str
    x: float
    y: float
    vx: float
    vy: float
    action: AgentAction
    dock: DockView | None = None


class HumanSprite(DomainModel):
    id: str
    label: str
    color: str
    x: float
    y: float
    vx: float
    vy: float
    selected_id: str | None = None


class ThoughtBadge(DomainModel):
    agent_id: str
    color: str
    text: str
    x: float
    y: float
    side: DockSide | None
    phase: BadgePhase
    opacity: float


class ThoughtEntry(DomainModel):
    """One thought for a viewer's feed. Each is sent once, in order, and never conflated."""

    kind: Literal["canvas.thought"] = "canvas.thought"
    seq: int
    agent_id: str
    label: str
    color: str
    text: str
    tool_name: str | None = None


class LockOverlay(DomainModel):
    id: str
    title: str
    box: NormalizedBox
    state: OverlayState
    owner: Participant | None = None
    waiters: tuple[Participant, ...] = ()


class ViewportSize(DomainModel):
    width: int
    height: int


class CanvasScene(DomainModel):
    """What changes rarely. A connection only sends it again when it differs from the last one."""

    viewport: ViewportSize
    overlays: tuple[LockOverlay, ...]


class VisualFrame(DomainModel):
    kind: Literal["canvas"] = "canvas"
    t_ms: float
    agents: tuple[AgentSprite, ...]
    humans: tuple[HumanSprite, ...]
    badges: tuple[ThoughtBadge, ...]
    scene: CanvasScene


def human_id(user_id: str) -> str:
    return f"{AGENT_ID_PREFIX}{user_id}"


def human_color(user_id: str) -> str:
    """Give each person a stable pastel, lighter than any saturated color the agents use."""
    red, green, blue = colorsys.hls_to_rgb(seed_unit("presence", user_id), 0.78, 0.85)
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def participant(agent_id: str) -> Participant:
    """Describe a lock's owner. People are told apart from agents by their id, not by a role."""
    if agent_id.startswith(AGENT_ID_PREFIX):
        user_id = agent_id.removeprefix(AGENT_ID_PREFIX)
        return Participant(id=agent_id, label=agent_id, color=human_color(user_id))
    name, color = _NAMES_BY_AGENT_ID.get(agent_id, (agent_id, UNOWNED_COLOR))
    return Participant(id=agent_id, label=name, color=color)


def _snippet(text: str) -> str:
    if len(text) <= MAX_SNIPPET_LENGTH:
        return text
    return f"{text[: MAX_SNIPPET_LENGTH - 1].rstrip()}…"


def thought_entry(event: ThoughtEvent) -> ThoughtEntry:
    thought = event.thought
    who = participant(thought.agent_id)
    return ThoughtEntry(
        seq=event.seq,
        agent_id=who.id,
        label=who.label,
        color=who.color,
        text=_snippet(thought.thought),
        tool_name=thought.tool_name,
    )


@dataclass(slots=True)
class _Badge:
    text: str
    born_ms: float
    fade_from_ms: float

    def view(self, now_ms: float) -> tuple[BadgePhase, float] | None:
        """Work out where this is in its lifecycle, or ``None`` once it has gone."""
        if now_ms >= self.fade_from_ms:
            remaining = 1.0 - (now_ms - self.fade_from_ms) / BADGE_EXIT_MS
            return None if remaining <= 0 else (BadgePhase.EXITING, remaining)
        age = now_ms - self.born_ms
        if age < BADGE_ENTER_MS:
            return BadgePhase.ENTERING, max(age, 0.0) / BADGE_ENTER_MS
        return BadgePhase.VISIBLE, 1.0


@dataclass(slots=True)
class _AgentTrack:
    frame: AgentCursorFrame
    motion: DeadReckoner
    side: DockSide | None = None


@dataclass(slots=True)
class _HumanTrack:
    motion: DeadReckoner
    selected_id: str | None


class CanvasCompositor:
    """One viewer's picture of a session. It only uses the time and events it is given.

    Locks are read from the lock table on every frame rather than followed through events, because
    the lock manager only announces changes an agent did not ask for. A plain acquire or release
    emits nothing, so an overlay built from events alone would be wrong until the next state change.
    """

    def __init__(self, bundle: FullstackProjectBundle) -> None:
        self._components: tuple[ComponentNode, ...] = ()
        self._viewport = Viewport(DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT)
        self._agents: dict[str, _AgentTrack] = {}
        self._humans: dict[str, _HumanTrack] = {}
        self._badges: dict[str, deque[_Badge]] = {}
        self._adopt(bundle)

    @property
    def viewport(self) -> Viewport:
        return self._viewport

    def _adopt(self, bundle: FullstackProjectBundle) -> None:
        self._components = bundle.frontend.components
        self._viewport = elastic_viewport(
            tuple(component.box for component in self._components), self._viewport
        )

    def apply(self, event: SwarmEvent, now_ms: float) -> None:
        """Take in one event. Lock notices need no handling, because the table is read directly."""
        if isinstance(event, CursorEvent):
            self.place_agent(event.frame, now_ms)
        elif isinstance(event, PresenceEvent):
            if event.active:
                self.place_human(event.user_id, event.x, event.y, event.selected_id, now_ms)
            else:
                self._humans.pop(event.user_id, None)
        elif isinstance(event, ThoughtEvent):
            self.add_thought(event, now_ms)
        elif isinstance(event, StateEvent):
            self._adopt(event.snapshot.bundle_state)

    def place_agent(self, frame: AgentCursorFrame, now_ms: float) -> None:
        position = Point(float(frame.x), float(frame.y))
        track = self._agents.get(frame.agent_id)
        if track is None:
            self._agents[frame.agent_id] = _AgentTrack(frame, DeadReckoner(now_ms, position))
            return
        track.frame = frame
        track.motion.observe(now_ms, position)

    def place_human(
        self, user_id: str, x: int, y: int, selected_id: str | None, now_ms: float
    ) -> None:
        position = Point(float(x), float(y))
        track = self._humans.get(user_id)
        if track is None:
            self._humans[user_id] = _HumanTrack(DeadReckoner(now_ms, position), selected_id)
            return
        track.selected_id = selected_id
        track.motion.observe(now_ms, position)

    def add_thought(self, event: ThoughtEvent, now_ms: float) -> None:
        """Badge a new thought. The thought itself reaches the feed as its own frame."""
        entry = thought_entry(event)
        badges = self._badges.setdefault(entry.agent_id, deque(maxlen=_BADGES_PER_AGENT))
        for badge in badges:
            badge.fade_from_ms = min(badge.fade_from_ms, now_ms)
        badges.append(_Badge(entry.text, now_ms, now_ms + BADGE_ENTER_MS + BADGE_HOLD_MS))

    def compose(
        self, now_ms: float, locks: Iterable[ElementLock], lock_time_ms: int
    ) -> VisualFrame:
        """Draw everything as of ``now_ms``. ``lock_time_ms`` is on the clock lock leases use.

        A lapsed lease stays in the table until the next sweep, and already counts as nobody's.
        """
        live = tuple(lock for lock in locks if not lock.is_expired(lock_time_ms))
        t_ms = now_ms - SMOOTHING_DELAY_MS
        agents = tuple(
            self._agent_sprite(track, t_ms, live) for _, track in sorted(self._agents.items())
        )
        humans = tuple(
            self._human_sprite(user_id, track, t_ms)
            for user_id, track in sorted(self._humans.items())
        )
        size = ViewportSize(width=self._viewport.width, height=self._viewport.height)
        return VisualFrame(
            t_ms=t_ms,
            agents=agents,
            humans=humans,
            badges=self._badge_views(now_ms, agents),
            scene=CanvasScene(viewport=size, overlays=self._overlays(live)),
        )

    def _agent_sprite(
        self, track: _AgentTrack, t_ms: float, locks: tuple[ElementLock, ...]
    ) -> AgentSprite:
        frame = track.frame
        motion = track.motion.sample(t_ms, self._viewport)
        role_color = SWARM_PALETTE.get(frame.role, frame.color)
        return AgentSprite(
            id=frame.agent_id,
            label=frame.name,
            role=frame.role,
            color=role_color,
            action=frame.current_action,
            dock=self._dock(track, motion.position, locks),
            **_motion_fields(motion),
        )

    def _dock(
        self, track: _AgentTrack, position: Point, locks: tuple[ElementLock, ...]
    ) -> DockView | None:
        target = self._dock_target(track.frame, position, locks)
        if track.frame.current_action not in DOCKING_ACTIONS or target is None:
            track.side = None
            return None
        dock = resolve_dock(position, target.box, self._viewport, track.side)
        track.side = dock.side
        return DockView(target_id=target.id, side=dock.side, x=dock.label_at.x, y=dock.label_at.y)

    def _dock_target(
        self, frame: AgentCursorFrame, position: Point, locks: tuple[ElementLock, ...]
    ) -> ComponentNode | None:
        """Prefer what the agent holds, then what it waits for, then the nearest component."""
        by_id = {component.id: component for component in self._components}
        own = [
            by_id[lock.target_id]
            for state in (LockState.ACQUIRED, LockState.WAITING)
            for lock in locks
            if lock.locked_by_agent_id == frame.agent_id
            and lock.state is state
            and lock.target_id in by_id
        ]
        if own:
            return own[0]
        return min(
            self._components,
            key=lambda component: box_gap(position, component.box),
            default=None,
        )

    def _human_sprite(self, user_id: str, track: _HumanTrack, t_ms: float) -> HumanSprite:
        return HumanSprite(
            id=human_id(user_id),
            label=human_id(user_id),
            color=human_color(user_id),
            selected_id=track.selected_id,
            **_motion_fields(track.motion.sample(t_ms, self._viewport)),
        )

    def _badge_views(
        self, now_ms: float, agents: tuple[AgentSprite, ...]
    ) -> tuple[ThoughtBadge, ...]:
        """Dock badges to their agents, and forget those that have finished fading."""
        by_id = {agent.id: agent for agent in agents}
        views: list[ThoughtBadge] = []
        for agent_id in sorted(self._badges):
            badges = self._badges[agent_id]
            agent = by_id.get(agent_id)
            for badge in tuple(badges):
                state = badge.view(now_ms)
                if state is None:
                    badges.remove(badge)
                elif agent is not None:
                    views.append(_badge_view(agent, badge.text, *state))
            if not badges:
                del self._badges[agent_id]
        return tuple(views)

    def _overlays(self, locks: tuple[ElementLock, ...]) -> tuple[LockOverlay, ...]:
        holders: dict[str, str] = {}
        waiters: dict[str, list[str]] = {}
        for lock in locks:
            if lock.state is LockState.ACQUIRED:
                holders[lock.target_id] = lock.locked_by_agent_id
            elif lock.state is LockState.WAITING:
                waiters.setdefault(lock.target_id, []).append(lock.locked_by_agent_id)
        return tuple(
            _overlay(
                component,
                self._viewport,
                holders.get(component.id),
                waiters.get(component.id, ()),
            )
            for component in self._components
        )


def _motion_fields(motion: MotionState) -> dict[str, float]:
    """Round for the wire, with velocity in pixels per second, which is plenty for an arrow."""
    return {
        "x": round(motion.position.x, 1),
        "y": round(motion.position.y, 1),
        "vx": round(motion.velocity.x * 1000.0, 1),
        "vy": round(motion.velocity.y * 1000.0, 1),
    }


def _badge_view(agent: AgentSprite, text: str, phase: BadgePhase, opacity: float) -> ThoughtBadge:
    dock = agent.dock
    return ThoughtBadge(
        agent_id=agent.id,
        color=agent.color,
        text=text,
        x=agent.x if dock is None else dock.x,
        y=agent.y if dock is None else dock.y,
        side=None if dock is None else dock.side,
        phase=phase,
        opacity=round(opacity, 3),
    )


def _overlay(
    component: ComponentNode,
    viewport: Viewport,
    holder: str | None,
    queued: Iterable[str],
) -> LockOverlay:
    """Contention outranks a plain hold, as the pulse is what tells people to look there."""
    waiting = tuple(participant(agent_id) for agent_id in queued)
    if waiting:
        state = OverlayState.WAITING
    elif holder is not None:
        state = OverlayState.LOCKED
    else:
        state = OverlayState.FREE
    return LockOverlay(
        id=component.id,
        title=component.title,
        box=normalize_box(component.box, viewport),
        state=state,
        owner=None if holder is None else participant(holder),
        waiters=waiting,
    )
