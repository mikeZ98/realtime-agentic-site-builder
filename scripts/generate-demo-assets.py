"""Render the README's demo assets from the platform's own simulation, deterministically.

The GIF is not an illustration. It runs the real cursor simulator, lock manager, compositor and
spline code against a scripted session, and draws the frames the canvas viewer would receive. The
SVG is a hand-laid diagram of the gateway and event mesh. Nothing touches the network or a clock:
the same Pillow version produces byte-identical files on every run.

    uv run python scripts/generate-demo-assets.py
"""

import asyncio
import hashlib
import io
import itertools
import math
import random
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from PIL import Image, ImageChops, ImageDraw, ImageFont
from pydantic import SecretStr

from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.canvas.compositor import (
    AgentSprite,
    CanvasCompositor,
    LockOverlay,
    ThoughtBadge,
    VisualFrame,
    human_color,
)
from realtime_agentic_site_builder.canvas.mirror import corrections
from realtime_agentic_site_builder.canvas.projection import DeadReckoner, hermite_controls
from realtime_agentic_site_builder.canvas.viewer import (
    CLIENT_BUFFER_MS,
    GLOW_DECAY_MS,
    GRAPH_NODES,
    HISTORY_CHECKPOINT_EVERY,
    RADAR_PULSE_MS,
    SPRING_DAMPING,
    SPRING_STIFFNESS,
)
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import SwarmSessionHub, ThoughtEvent
from realtime_agentic_site_builder.core.lock_manager import AsyncLockManager, LockEvent
from realtime_agentic_site_builder.core.patch_engine import (
    PatchEngine,
    bundle_size,
    replay_operations,
)
from realtime_agentic_site_builder.domain.backend import (
    EMAIL_COMPATIBLE_TYPE,
    ApiEndpointNode,
    BackendBlueprint,
    DatabaseDialect,
    DataModelSchema,
    FieldDefinition,
    FullstackProjectBundle,
    HttpMethod,
)
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    ComponentType,
    SiteBlueprint,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentRole,
    AgentThoughtStream,
    PatchOperation,
    StreamDelta,
)
from realtime_agentic_site_builder.simulation.controller import CursorSimulator
from realtime_agentic_site_builder.simulation.motion import Point, Viewport, cubic_bezier

ROOT: Final = Path(__file__).resolve().parent.parent
ASSETS: Final = ROOT / "docs" / "assets"
GIF_PATH: Final = ASSETS / "swarm-live-demo.gif"
SVG_PATH: Final = ASSETS / "architecture-topology.svg"
CONVERGENCE_PATH: Final = ASSETS / "dead-reckoning-convergence.svg"
LOCKS_PATH: Final = ASSETS / "lock-contention-matrix.svg"
FORKING_PATH: Final = ASSETS / "time-travel-forking.svg"

WORLD: Final = Viewport(1920, 1080)
OUT_W, OUT_H = 800, 450
SUPERSAMPLE: Final = 2
FPS: Final = 30
SECONDS: Final = 5
FRAMES: Final = FPS * SECONDS
TRAIL_MS: Final = 700.0
PALETTE_SIZE: Final = 96
TRANSPARENT_INDEX: Final = PALETTE_SIZE - 1
MAX_GIF_BYTES: Final = 2 * 1024 * 1024
HUMAN: Final = "alice"
PULSE_MS: Final = RADAR_PULSE_MS

BG: Final = (11, 15, 25)
PANEL: Final = (17, 24, 39)
LINE: Final = (51, 65, 85)
MUTED: Final = (100, 116, 139)
TEXT: Final = (229, 231, 235)

ARCHITECT, CODER, BACKEND = (
    AgentRole.ARCHITECT,
    AgentRole.FRONTEND_CODER,
    AgentRole.BACKEND_ENGINEER,
)
STYLER, CRITIC, COPYWRITER = AgentRole.STYLER, AgentRole.CRITIC_LINTER, AgentRole.COPYWRITER
# The graph node each role runs, which is what the HUD's radar shows. The Copywriter has none.
NODES: Final = {
    ARCHITECT: GRAPH_NODES[0],
    CODER: GRAPH_NODES[1],
    BACKEND: GRAPH_NODES[2],
    STYLER: GRAPH_NODES[3],
    CRITIC: GRAPH_NODES[4],
}


def _component(
    component_id: str,
    kind: ComponentType,
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
    endpoint: str | None = None,
) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=kind,
        title=title,
        code="export default function Section() { return <section />; }",
        box=BoundingBox(x=x, y=y, width=width, height=height),
        connected_endpoint_id=endpoint,
    )


# What the Backend Engineer is writing in the script: the endpoint the contact form posts to.
LEADS: Final = BackendBlueprint(
    service_name="api",
    database_dialect=DatabaseDialect.SQLITE,
    models=(
        DataModelSchema(
            name="Lead",
            table_name="leads",
            fields={"email": FieldDefinition(python_type=EMAIL_COMPATIBLE_TYPE)},
        ),
    ),
    endpoints=(
        ApiEndpointNode(
            id="api_endpoint_post_leads",
            path="/api/v1/leads",
            method=HttpMethod.POST,
            summary="Create a lead",
            request_model_ref="Lead",
            handler_code="async def handler() -> dict[str, str]:\n    return {'status': 'ok'}",
        ),
    ),
)


BUNDLE: Final = FullstackProjectBundle(
    project_id="demo",
    frontend=SiteBlueprint(
        site_id="demo",
        title="SaaS landing page",
        theme="dark",
        components=(
            _component("navbar", ComponentType.NAVBAR, "Acme", 60, 40, 1800, 90),
            _component("hero", ComponentType.HERO, "Hero", 60, 160, 1800, 300),
            _component("features", ComponentType.FEATURES, "Features", 60, 490, 880, 260),
            _component("pricing", ComponentType.PRICING, "Pricing", 980, 490, 880, 260),
            _component(
                "contact",
                ComponentType.CONTACT_FORM,
                "Contact form",
                60,
                780,
                1100,
                250,
                "api_endpoint_post_leads",
            ),
            _component("footer", ComponentType.FOOTER, "Footer", 1200, 780, 660, 250),
        ),
    ),
    backend=LEADS,
)
BOXES: Final = {component.id: component.box for component in BUNDLE.frontend.components}


@dataclass(frozen=True, slots=True)
class Cue:
    """One scripted moment: a lock change, a cursor intent, and what the agent says about it.

    ``title`` commits a new title for the target through the real patch engine, which the DOM
    mirror then applies from the patches and corrections a viewer would be sent. ``verdict`` is
    the critic's, which the HUD's radar pulses on.
    """

    at_ms: int
    role: AgentRole
    move: str  # "edit", "go", "wait" or "idle"
    target: str | None = None
    lock: str | None = None  # "take", "want" or "drop"
    thought: str | None = None
    title: str | None = None
    verdict: bool | None = None


SCRIPT: Final = (
    Cue(0, ARCHITECT, "go", "hero", thought="Planning six sections from the brief"),
    Cue(150, COPYWRITER, "go", "navbar"),
    Cue(250, CODER, "edit", "hero", "take", "Writing the hero layout"),
    Cue(550, BACKEND, "edit", "contact", "take", "POST /api/v1/leads handler"),
    Cue(900, STYLER, "edit", "features", "take", "Tuning the feature grid"),
    Cue(1200, ARCHITECT, "go", "pricing"),
    Cue(1350, COPYWRITER, "wait", "features", "want", "Waiting for Styler on features"),
    Cue(1700, CRITIC, "wait", "hero", "want", "Queued to review the hero"),
    Cue(2100, CODER, "edit", "hero", None, "Committing the hero", title="Ship faster with Acme"),
    Cue(2300, CODER, "go", "pricing", "drop", "Hero committed, releasing the lock"),
    Cue(2300, CRITIC, "edit", "hero", None, "Reviewing hero contrast"),
    Cue(2700, STYLER, "go", "footer", "drop"),
    Cue(2700, COPYWRITER, "edit", "features", None, "Rewriting feature copy"),
    Cue(2900, BACKEND, "edit", "contact", None, "Wiring the form", title="Talk to sales"),
    Cue(3000, ARCHITECT, "edit", "pricing", "take", "Adjusting the pricing tiers"),
    Cue(3150, CRITIC, "edit", "hero", None, "Feature copy runs long", verdict=False),
    Cue(3400, BACKEND, "go", "footer", "drop", "Endpoint and schema committed"),
    Cue(3450, COPYWRITER, "edit", "features", None, "Tightened", title="Everything you need"),
    Cue(3600, ARCHITECT, "edit", "pricing", None, "Three tiers", title="Simple, fair pricing"),
    Cue(3900, CRITIC, "go", "contact", "drop", "Approved", verdict=True),
    Cue(4300, ARCHITECT, "idle", None, "drop", "Run complete, all checks passed"),
    Cue(4300, COPYWRITER, "idle", None, "drop"),
    Cue(4300, CODER, "idle"),
    Cue(4300, STYLER, "idle"),
    Cue(4300, BACKEND, "idle"),
    Cue(4300, CRITIC, "idle"),
)


def human_path(t_ms: float) -> Point:
    """A person watching, drifting across the canvas the way a mouse idles over a page."""
    return Point(1500 + 260 * math.sin(t_ms / 900), 620 + 140 * math.sin(t_ms / 620 + 0.8))


@dataclass(slots=True)
class Spring:
    """One mirrored section's motion and halo, integrated exactly as the viewer page does."""

    s: float = 1.0
    v: float = 0.0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    glow: float = 0.0
    halo: str = "#334155"

    def kick(self, color: str, strength: float, vx: float, vy: float) -> None:
        self.glow, self.halo = max(self.glow, strength), color
        speed = math.hypot(vx, vy)
        self.v += strength * (0.35 + min(speed / 800, 0.65))
        if speed > 1:
            push = strength * 110 * min(speed, 1200) / 1200
            self.vx += vx / speed * push
            self.vy += vy / speed * push

    def step(self, dt: float) -> None:
        self.v += (-SPRING_STIFFNESS * (self.s - 1) - SPRING_DAMPING * self.v) * dt
        self.s += self.v * dt
        self.vx += (-SPRING_STIFFNESS * self.x - SPRING_DAMPING * self.vx) * dt
        self.x += self.vx * dt
        self.vy += (-SPRING_STIFFNESS * self.y - SPRING_DAMPING * self.vy) * dt
        self.y += self.vy * dt
        self.glow *= math.exp(-dt * 1000 / GLOW_DECAY_MS)


@dataclass(frozen=True, slots=True)
class Section:
    """What the mirror draws of one component in one frame."""

    id: str
    kind: str
    title: str
    box: BoundingBox
    scale: float
    dx: float
    dy: float
    glow: float
    halo: str


@dataclass(frozen=True, slots=True)
class Hud:
    visits: dict[str, int]
    active: frozenset[str]
    pulses: tuple[tuple[float, bool], ...]
    rejected: int
    approved: int
    frame_seq: int
    commits: int
    held: int


@dataclass(frozen=True, slots=True)
class Shot:
    frame: VisualFrame
    sections: tuple[Section, ...]
    hud: Hud


@dataclass
class Session:
    """The scripted session: the real simulator, lock manager, compositor and patch engine."""

    simulator: CursorSimulator
    locks: AsyncLockManager
    compositor: CanvasCompositor
    bundle: FullstackProjectBundle = BUNDLE
    mirror: dict[str, object] = field(
        default_factory=lambda: {"frontend": BUNDLE.frontend.model_dump(mode="json")}
    )
    springs: dict[str, Spring] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)
    pulses: list[tuple[float, bool]] = field(default_factory=list)
    rejected: int = 0
    approved: int = 0
    frame_seq: int = 0
    thought_seq: int = 0
    owners: dict[str, str | None] = field(default_factory=dict)
    last: VisualFrame | None = None

    def velocity(self, agent_id: str) -> tuple[float, float]:
        sprites = self.last.agents if self.last else ()
        sprite = next((sprite for sprite in sprites if sprite.id == agent_id), None)
        return (sprite.vx, sprite.vy) if sprite else (0.0, 0.0)

    def kick(self, target: str, agent_id: str, color: str, strength: float) -> None:
        self.springs.setdefault(target, Spring()).kick(color, strength, *self.velocity(agent_id))

    def commit(self, cue: Cue, now: int) -> None:
        """Commit a title through the engine, and mirror it the way a viewer's page does."""
        assert cue.target is not None and cue.title is not None
        identity = IDENTITIES[cue.role]
        index = [item.id for item in self.bundle.frontend.components].index(cue.target)
        patches = (
            PatchOperation.model_validate(
                {"op": "replace", "path": f"/frontend/components/{index}/title", "value": cue.title}
            ),
        )
        delta = StreamDelta(
            session_id="demo",
            agent_id=identity.agent_id,
            delta_seq=self.frame_seq,
            target_domain="frontend",
            target_id=cue.target,
            patches=patches,
            emitted_at_ms=now,
        )
        before = self.bundle.frontend.model_dump(mode="json")
        self.bundle, _ = PatchEngine().apply_patch(self.bundle, delta)
        after = self.bundle.frontend.model_dump(mode="json")
        replay_operations(self.mirror, (*patches, *corrections(before, patches, after)))
        if self.mirror["frontend"] != after:
            raise SystemExit("the mirror diverged from the committed layout")
        self.frame_seq += 1
        self.kick(cue.target, identity.agent_id, identity.color, 1.0)

    def tally(self, cue: Cue, now: int) -> None:
        node = NODES.get(cue.role)
        if node is not None and cue.move == "edit" and node not in self.active:
            self.active.add(node)
            self.visits[node] = self.visits.get(node, 0) + 1
        elif node is not None and (cue.lock == "drop" or cue.move == "idle"):
            self.active.discard(node)
        if cue.verdict is not None:
            self.pulses.append((float(now), cue.verdict))
            self.rejected += not cue.verdict
            self.approved += cue.verdict

    async def perform(self, cue: Cue, now: int) -> None:
        identity = IDENTITIES[cue.role]
        box = BOXES[cue.target] if cue.target else None
        held = [
            lock.target_id
            for lock in self.locks.active_locks()
            if lock.locked_by_agent_id == identity.agent_id
        ]
        if cue.title is not None:
            self.commit(cue, now)
        if cue.lock == "drop":
            for target in held:
                await self.locks.release(target, identity.agent_id)
        elif cue.lock in {"take", "want"} and cue.target:
            await self.locks.acquire(cue.target, identity.agent_id, identity.priority, now)
        if cue.move == "edit" and box is not None:
            self.simulator.edit(cue.role, box)
        elif cue.move == "go" and box is not None:
            self.simulator.navigate_to(cue.role, box)
        elif cue.move == "wait" and box is not None:
            self.simulator.block_on(cue.role, box)
        else:
            self.simulator.idle(cue.role)
        self.tally(cue, now)
        if cue.thought is None:
            return
        thought = AgentThoughtStream(
            agent_id=identity.agent_id, role=cue.role, thought=cue.thought, timestamp_ms=now
        )
        self.compositor.add_thought(
            ThoughtEvent(origin="demo", session_id="demo", seq=self.thought_seq, thought=thought),
            now,
        )
        self.thought_seq += 1

    def shoot(self, frame: VisualFrame, t_ms: float) -> Shot:
        """Settle the springs for this frame, as the page's animation loop does, and record it."""
        self.last = frame
        for overlay in frame.scene.overlays:
            owner = overlay.owner
            if owner is not None and self.owners.get(overlay.id) != owner.id:
                self.kick(overlay.id, owner.id, owner.color, 0.8)
            self.owners[overlay.id] = owner.id if owner else None
        sections = []
        overlays = {overlay.id: overlay for overlay in frame.scene.overlays}
        components = self.mirror["frontend"]["components"]  # type: ignore[index]
        for item in components:
            spring = self.springs.setdefault(item["id"], Spring())
            spring.step(1 / FPS)
            held = overlays.get(item["id"])
            rest, rest_color = 0.0, spring.halo
            if held is not None and held.state.value != "FREE":
                lead = held.owner or held.waiters[0]
                rest, rest_color = 0.35, lead.color
                if held.state.value == "WAITING":
                    rest = 0.22 + 0.16 * math.sin(t_ms / 180)
            flashing = spring.glow > rest
            sections.append(
                Section(
                    id=item["id"],
                    kind=item["component_type"],
                    title=item["title"],
                    box=BoundingBox.model_validate(item["box"]),
                    scale=spring.s,
                    dx=spring.x,
                    dy=spring.y,
                    glow=max(spring.glow, rest),
                    halo=spring.halo if flashing else rest_color,
                )
            )
        self.pulses = [pulse for pulse in self.pulses if t_ms - pulse[0] < PULSE_MS]
        hud = Hud(
            visits=dict(self.visits),
            active=frozenset(self.active),
            pulses=tuple(self.pulses),
            rejected=self.rejected,
            approved=self.approved,
            frame_seq=self.frame_seq,
            commits=self.frame_seq,
            held=sum(lock.state.value == "ACQUIRED" for lock in self.locks.active_locks()),
        )
        return Shot(frame, tuple(sections), hud)


async def simulate() -> list[Shot]:
    """Run the script through the real simulator, lock manager, compositor and patch engine."""
    now = 0
    session = Session(
        CursorSimulator(WORLD, seed=11),
        AsyncLockManager(lease_ms=30_000, clock=lambda: now),
        CanvasCompositor(BUNDLE),
    )
    pending = deque(sorted(SCRIPT, key=lambda cue: cue.at_ms))
    shots: list[Shot] = []
    elapsed = 0
    for index in range(FRAMES):
        target_ms = round((index + 1) * 1000 / FPS)
        while pending and pending[0].at_ms <= target_ms:
            await session.perform(pending.popleft(), now)
        session.simulator.sync(BUNDLE, session.locks.active_locks())
        for frame in session.simulator.step(target_ms - elapsed):
            session.compositor.place_agent(frame, float(target_ms))
        elapsed = now = target_ms
        position = human_path(target_ms)
        session.compositor.place_human(
            HUMAN, round(position.x), round(position.y), "pricing", target_ms
        )
        composed = session.compositor.compose(float(target_ms), session.locks.active_locks(), now)
        shots.append(session.shoot(composed, float(target_ms)))
    return shots


# --- drawing --------------------------------------------------------------------------------

Color = tuple[int, int, int]
SCALE: Final = OUT_W * SUPERSAMPLE / WORLD.width


def _hex(color: str) -> Color:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _sx(x: float) -> float:
    return x * SCALE


def _font(size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.load_default(size=size * SUPERSAMPLE)
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise SystemExit("Pillow was built without FreeType, which the assets need")
    return font


FONT_SMALL: Final = _font(9)
FONT_LABEL: Final = _font(10)
FONT_TITLE: Final = _font(11)
FONT_HERO: Final = _font(17)


def _pill(
    draw: ImageDraw.ImageDraw, text: str, x: float, y: float, color: Color, alpha: float = 1.0
) -> None:
    width = draw.textlength(text, font=FONT_SMALL) + 12 * SUPERSAMPLE
    height = 15 * SUPERSAMPLE
    left = min(max(x, 2.0), OUT_W * SUPERSAMPLE - width - 2)
    fill = (*color, round(255 * alpha))
    draw.rounded_rectangle((left, y, left + width, y + height), radius=height / 2, fill=fill)
    draw.text(
        (left + 6 * SUPERSAMPLE, y + height / 2),
        text,
        font=FONT_SMALL,
        fill=(*BG, round(255 * alpha)),
        anchor="lm",
    )


def _draw_overlay(draw: ImageDraw.ImageDraw, overlay: LockOverlay, t_ms: float) -> None:
    """Who holds what, outlined over the mirror, which draws the section itself."""
    box = overlay.box
    left, top = _sx(box.x * WORLD.width), _sx(box.y * WORLD.height)
    right, bottom = left + _sx(box.width * WORLD.width), top + _sx(box.height * WORLD.height)
    if overlay.state.value == "FREE":
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=7 * SUPERSAMPLE,
            outline=(*LINE, 110),
            width=SUPERSAMPLE,
        )
        return
    leader = overlay.owner or overlay.waiters[0]
    color = _hex(leader.color)
    pulse = 0.55 + 0.45 * math.sin(t_ms / 180) if overlay.state.value == "WAITING" else 1.0
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=7 * SUPERSAMPLE,
        outline=(*color, round(255 * pulse)),
        width=2 * SUPERSAMPLE,
    )
    queue = f" +{len(overlay.waiters)} waiting" if overlay.waiters else ""
    label = f"{overlay.owner.label}{queue}" if overlay.owner else f"{len(overlay.waiters)} waiting"
    _pill(draw, label, left + 8 * SUPERSAMPLE, bottom - 22 * SUPERSAMPLE, color)


# The page's dark theme, which the mirror draws each section in.
PAGE_BG: Final = _hex("#0F172A")
CARD: Final = _hex("#1E293B")
DIM: Final = _hex("#334155")
ACCENT: Final = _hex("#6366F1")
INK_PAGE: Final = _hex("#E2E8F0")


def _draw_section(draw: ImageDraw.ImageDraw, section: Section) -> None:
    """One mirrored section: its real title over the page's skeleton for its kind, never its code.

    Geometry passes through the section's spring: scaled about its center and nudged along the
    path of the cursor that last kicked it.
    """
    box = section.box
    cx, cy = box.x + box.width / 2 + section.dx, box.y + box.height / 2 + section.dy

    def at(x: float, y: float) -> tuple[float, float]:
        return _sx(cx + (x - box.x - box.width / 2) * section.scale), _sx(
            cy + (y - box.y - box.height / 2) * section.scale
        )

    def rect(x: float, y: float, w: float, h: float, **style: object) -> None:
        (x0, y0), (x1, y1) = at(x, y), at(x + w, y + h)
        draw.rounded_rectangle((x0, y0, x1, y1), **style)  # type: ignore[arg-type]

    left, top, width, height = box.x, box.y, box.width, box.height
    if section.glow > 0.02:
        halo = _hex(section.halo)
        for spread, alpha in ((22, 0.10), (12, 0.22), (5, 0.4)):
            grow = spread * section.glow
            rect(
                left - grow,
                top - grow,
                width + 2 * grow,
                height + 2 * grow,
                radius=(16 + grow) * SCALE,
                outline=(*halo, round(255 * alpha * section.glow)),
                width=round(3 * SUPERSAMPLE),
            )
    rect(left, top, width, height, radius=16 * SCALE, fill=CARD)
    pad = 26
    tx, ty = at(left + pad, top + pad)
    font = FONT_HERO if section.kind == "HERO" else FONT_TITLE
    draw.text((tx, ty), section.title, font=font, fill=INK_PAGE)
    inner_top = top + pad + (88 if section.kind == "HERO" else 54)
    inner_w = width - 2 * pad

    def bars(x: float, y: float, w: float, count: int) -> None:
        for index in range(count):
            rect(x, y + index * 26, w * (1, 0.74, 0.52)[index % 3], 12, radius=6 * SCALE, fill=DIM)

    def button(x: float, y: float) -> None:
        rect(x, y, 150, 40, radius=9 * SCALE, fill=ACCENT)

    def cards(count: int, fill: Callable[[float, float, float], None]) -> None:
        gap = 20
        card_w = (inner_w - gap * (count - 1)) / count
        for index in range(count):
            x = left + pad + index * (card_w + gap)
            rect(x, inner_top, card_w, top + height - pad - inner_top, radius=10 * SCALE, fill=DIM)
            fill(x + 16, inner_top + 18, card_w - 32)

    if section.kind == "NAVBAR":
        for index in range(3):
            rect(left + width - 560 + index * 120, top + 40, 90, 12, radius=6 * SCALE, fill=DIM)
        button(left + width - 190, top + 25)
    elif section.kind == "HERO":
        bars(left + pad, inner_top, inner_w * 0.6, 2)
        button(left + pad, inner_top + 70)
    elif section.kind == "FEATURES":
        cards(3, lambda x, y, w: bars(x, y, w, 3))
    elif section.kind == "PRICING":

        def tier(x: float, y: float, w: float) -> None:
            rect(x, y, w * 0.5, 26, radius=6 * SCALE, fill=INK_PAGE)
            bars(x, y + 44, w, 2)

        cards(3, tier)
    elif section.kind == "CONTACT_FORM":
        for index in range(3):
            rect(
                left + pad,
                inner_top + index * 46,
                inner_w * 0.7,
                34,
                radius=7 * SCALE,
                outline=DIM,
                width=SUPERSAMPLE,
            )
    else:
        for index in range(4):
            rect(left + pad + index * 150, inner_top + 20, 110, 12, radius=6 * SCALE, fill=DIM)


HUD_W, HUD_H = 168, 206


def _draw_hud(draw: ImageDraw.ImageDraw, hud: Hud, t_ms: float) -> None:
    """The heads-up display, with what a simulation can derive: no spend, no latency."""
    s = SUPERSAMPLE
    # Top right, over the hero's empty side, so the titles the mirror updates stay in view.
    x0, y0 = (OUT_W - 8 - HUD_W) * s, 8 * s
    draw.rounded_rectangle(
        (x0, y0, x0 + HUD_W * s, y0 + HUD_H * s),
        radius=9 * s,
        fill=(*BG, 235),
        outline=(*LINE, 255),
        width=s,
    )
    draw.text((x0 + 10 * s, y0 + 10 * s), "GRAPH STATE", font=FONT_SMALL, fill=MUTED)
    cx, cy, radius = x0 + HUD_W * s / 2, y0 + 78 * s, 40 * s
    count = len(GRAPH_NODES)

    def corner(index: int, r: float) -> tuple[float, float]:
        angle = -math.pi / 2 + index * 2 * math.pi / count
        return cx + math.cos(angle) * r, cy + math.sin(angle) * r

    for ring in (1 / 3, 2 / 3, 1):
        draw.polygon([corner(i, radius * ring) for i in range(count)], outline=(*LINE, 255))
    for i in range(count):
        draw.line((cx, cy, *corner(i, radius)), fill=(*LINE, 255), width=1)
    for at, ok in hud.pulses:
        age = (t_ms - at) / PULSE_MS
        color = (16, 185, 129) if ok else (248, 113, 113)
        r = radius * (0.15 + age)
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            outline=(*color, round(255 * (1 - age))),
            width=round((2 if ok else 2 + 3 * (1 - age)) * s),
        )
    most = max(1, *hud.visits.values()) if hud.visits else 1
    shape = [
        corner(i, radius * max(hud.visits.get(node, 0) / most, 0.06))
        for i, node in enumerate(GRAPH_NODES)
    ]
    draw.polygon(shape, fill=(139, 92, 246, 60), outline=(139, 92, 246, 255), width=s)
    for i, node in enumerate(GRAPH_NODES):
        if node in hud.active:
            px, py = corner(i, radius)
            glow = 0.6 + 0.4 * math.sin(t_ms / 140)
            dot = 4 * s
            draw.ellipse(
                (px - dot, py - dot, px + dot, py + dot), fill=(196, 181, 253, round(255 * glow))
            )
        lx, ly = corner(i, radius + 13 * s)
        draw.text(
            (lx, ly), node, font=FONT_SMALL, fill=TEXT if node in hud.active else MUTED, anchor="mm"
        )
    rows = (
        ("Critic", f"{hud.rejected} rejected, {hud.approved} approved"),
        ("Mirror frame", str(hud.frame_seq)),
        ("Commits", str(hud.commits)),
        ("Locks held", str(hud.held)),
    )
    for index, (name, value) in enumerate(rows):
        y = y0 + (140 + index * 15) * s
        draw.text((x0 + 10 * s, y), name, font=FONT_SMALL, fill=MUTED)
        draw.text((x0 + (HUD_W - 10) * s, y), value, font=FONT_SMALL, fill=TEXT, anchor="ra")


def _draw_trail(
    draw: ImageDraw.ImageDraw, history: Sequence[tuple[float, AgentSprite]], color: Color
) -> None:
    newest = history[-1][0]
    for (t0, a), (t1, b) in itertools.pairwise(history):
        age = newest - t1
        if age > TRAIL_MS or t1 <= t0:
            continue
        span = t1 - t0
        controls = hermite_controls(
            Point(a.x, a.y),
            Point(a.vx / 1000, a.vy / 1000),
            Point(b.x, b.y),
            Point(b.vx / 1000, b.vy / 1000),
            span,
        )
        points = [cubic_bezier(*controls, step / 6) for step in range(7)]
        alpha = round(150 * max(1 - age / TRAIL_MS, 0))
        draw.line(
            [(_sx(p.x), _sx(p.y)) for p in points], fill=(*color, alpha), width=2 * SUPERSAMPLE
        )


def _draw_agent(draw: ImageDraw.ImageDraw, agent: AgentSprite) -> None:
    color = _hex(agent.color)
    x, y = _sx(agent.x), _sx(agent.y)
    if agent.dock is not None:
        draw.line(
            (x, y, _sx(agent.dock.x), _sx(agent.dock.y)), fill=(*color, 120), width=SUPERSAMPLE
        )
        r = 3 * SUPERSAMPLE
        draw.ellipse(
            (
                _sx(agent.dock.x) - r,
                _sx(agent.dock.y) - r,
                _sx(agent.dock.x) + r,
                _sx(agent.dock.y) + r,
            ),
            fill=color,
        )
    speed = math.hypot(agent.vx, agent.vy)
    if speed > 40:
        reach = min(speed * 0.03, 30) * SUPERSAMPLE
        draw.line(
            (x, y, x + agent.vx / speed * reach, y + agent.vy / speed * reach),
            fill=color,
            width=2 * SUPERSAMPLE,
        )
    s = SUPERSAMPLE
    arrow = [(x, y), (x + 3.5 * s, y + 15 * s), (x + 7 * s, y + 9.5 * s), (x + 13 * s, y + 9 * s)]
    draw.polygon(arrow, fill=color, outline=BG, width=s)
    _pill(draw, agent.label, x + 10 * s, y + 14 * s, color)


def _draw_human(draw: ImageDraw.ImageDraw, x: float, y: float, label: str, color: Color) -> None:
    sx, sy, r = _sx(x), _sx(y), 7 * SUPERSAMPLE
    draw.ellipse((sx - r, sy - r, sx + r, sy + r), outline=color, width=2 * SUPERSAMPLE)
    dot = 2 * SUPERSAMPLE
    draw.ellipse((sx - dot, sy - dot, sx + dot, sy + dot), fill=color)
    _pill(draw, label, sx + 10 * SUPERSAMPLE, sy + 8 * SUPERSAMPLE, color)


def _draw_badge(draw: ImageDraw.ImageDraw, badge: ThoughtBadge) -> None:
    lift = {"TOP": -24, "BOTTOM": 4}.get(badge.side.value if badge.side else "", -8)
    _pill(
        draw,
        badge.text,
        _sx(badge.x) - 20 * SUPERSAMPLE,
        _sx(badge.y) + lift * SUPERSAMPLE,
        _hex(badge.color),
        badge.opacity,
    )


def _draw_legend(draw: ImageDraw.ImageDraw) -> None:
    x: float = OUT_W * SUPERSAMPLE - 8 * SUPERSAMPLE
    y = OUT_H * SUPERSAMPLE - 12 * SUPERSAMPLE
    for identity in reversed(tuple(IDENTITIES.values())):
        width = draw.textlength(identity.name, font=FONT_SMALL)
        x -= width
        draw.text((x, y), identity.name, font=FONT_SMALL, fill=MUTED, anchor="lm")
        r = 3 * SUPERSAMPLE
        draw.ellipse(
            (x - 10 * SUPERSAMPLE - r, y - r, x - 10 * SUPERSAMPLE + r, y + r),
            fill=_hex(identity.color),
        )
        x -= 22 * SUPERSAMPLE


def render(shots: Sequence[Shot]) -> list[Image.Image]:
    images: list[Image.Image] = []
    trails: dict[str, deque[tuple[float, AgentSprite]]] = {}
    size = (OUT_W * SUPERSAMPLE, OUT_H * SUPERSAMPLE)
    for shot in shots:
        frame = shot.frame
        canvas = Image.new("RGBA", size, (*BG, 255))
        backdrop = ImageDraw.Draw(canvas)
        backdrop.rectangle((0, 0, _sx(WORLD.width), _sx(WORLD.height)), fill=PAGE_BG)
        page = Image.new("RGBA", size, (0, 0, 0, 0))
        mirror = ImageDraw.Draw(page)
        for section in shot.sections:
            _draw_section(mirror, section)
        canvas.alpha_composite(page)
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for overlay in frame.scene.overlays:
            _draw_overlay(draw, overlay, frame.t_ms)
        for agent in frame.agents:
            history = trails.setdefault(agent.id, deque())
            history.append((frame.t_ms, agent))
            while history and frame.t_ms - history[0][0] > TRAIL_MS:
                history.popleft()
            if len(history) > 1:
                _draw_trail(draw, history, _hex(agent.color))
        for human in frame.humans:
            _draw_human(draw, human.x, human.y, human.label, _hex(human.color))
        for agent in frame.agents:
            _draw_agent(draw, agent)
        for badge in frame.badges:
            _draw_badge(draw, badge)
        _draw_hud(draw, shot.hud, frame.t_ms)
        draw.text(
            (10 * SUPERSAMPLE, (OUT_H - 12) * SUPERSAMPLE),
            "Swarm Canvas  \N{MIDDLE DOT}  live DOM mirror, telemetry and HUD  \N{MIDDLE DOT}"
            "  scripted session replay",
            font=FONT_LABEL,
            fill=MUTED,
            anchor="lm",
        )
        _draw_legend(draw)
        canvas.alpha_composite(layer)
        images.append(canvas.convert("RGB").resize((OUT_W, OUT_H), Image.Resampling.LANCZOS))
    return images


def encode_gif(images: Sequence[Image.Image]) -> bytes:
    """Quantize to one shared palette, and send each frame as only what changed.

    A shared palette keeps colors from shimmering between frames. Cursors are spread across the
    canvas, so the box around a frame's changes covers most of it. Pixels that did not change are
    therefore marked transparent instead, and the previous frame shows through them. Those long
    runs of one index compress to almost nothing.
    """
    sample = images[:: max(len(images) // 12, 1)]
    montage = Image.new("RGB", (OUT_W, OUT_H * len(sample)))
    for index, image in enumerate(sample):
        montage.paste(image, (0, OUT_H * index))
    palette = montage.quantize(colors=PALETTE_SIZE - 1, method=Image.Quantize.MEDIANCUT)
    colors = palette.getpalette() or []
    colors = colors[: 3 * (PALETTE_SIZE - 1)] + [255, 0, 255]  # The reserved index, never shown.
    keep = Image.new("L", (OUT_W, OUT_H), TRANSPARENT_INDEX)
    frames: list[Image.Image] = []
    previous: Image.Image | None = None
    for image in images:
        indices = image.quantize(palette=palette, dither=Image.Dither.NONE)
        current = Image.frombytes("L", indices.size, indices.tobytes())
        shown = current
        if previous is not None:
            changed = ImageChops.difference(current, previous).point(lambda v: 255 if v else 0)
            shown = Image.composite(current, keep, changed)
        previous = current
        frame = Image.frombytes("P", shown.size, shown.tobytes())
        frame.putpalette(colors)
        frames.append(frame)
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=False,
        transparency=TRANSPARENT_INDEX,
        disposal=1,
    )
    return buffer.getvalue()


# --- the topology diagram -------------------------------------------------------------------


def _box(x: int, y: int, w: int, h: int, title: str, lines: Sequence[str], accent: str) -> str:
    body = "".join(
        f'<text x="{x + 14}" y="{y + 44 + 17 * i}" class="body">{line}</text>'
        for i, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="box"/>'
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>'
        f'<text x="{x + 14}" y="{y + 24}" class="title">{title}</text>{body}'
    )


def _arrow(
    points: Sequence[tuple[int, int]],
    label: str = "",
    label_at: tuple[int, int] | None = None,
    dashed: bool = False,
) -> str:
    path = " ".join(f"{'M' if i == 0 else 'L'} {x} {y}" for i, (x, y) in enumerate(points))
    style = ' stroke-dasharray="6 5"' if dashed else ""
    text = ""
    if label and label_at:
        text = f'<text x="{label_at[0]}" y="{label_at[1]}" class="edge">{label}</text>'
    return f'<path d="{path}" class="wire"{style} marker-end="url(#head)"/>{text}'


def topology_svg() -> str:
    violet, blue, green, pink, amber, cyan = (
        identity.color
        for identity in (
            IDENTITIES[ARCHITECT],
            IDENTITIES[CODER],
            IDENTITIES[BACKEND],
            IDENTITIES[STYLER],
            IDENTITIES[CRITIC],
            IDENTITIES[COPYWRITER],
        )
    )
    parts = [
        # Clients
        _box(
            30,
            90,
            230,
            92,
            "Owner client",
            ["rasb.v1 WebSocket", "prompt · lock · delta · presence"],
            violet,
        ),
        _box(
            30,
            202,
            230,
            92,
            "Canvas viewer",
            ["GET /canvas/{id}: static, strict CSP", "WS ?view=canvas · read only"],
            cyan,
        ),
        _box(
            30, 340, 230, 92, "REST client", ["Bearer token", "sessions · timeline · export"], blue
        ),
        # Gateway
        _box(
            320,
            90,
            270,
            150,
            "WebSocket gateway",
            [
                "Token in rasb.token.* subprotocol",
                "Owner-only: prompt · lock · delta",
                "Token bucket · heartbeat · expiry",
                "Catch-up deadline, then 1008",
                "Resume from last_seen_thought_seq",
            ],
            blue,
        ),
        _box(
            320,
            258,
            270,
            78,
            "Canvas compositor",
            ["Per canvas client: dead reckoning,", "dock anchors and lock overlays"],
            cyan,
        ),
        _box(
            320,
            354,
            270,
            116,
            "REST ingress",
            [
                "Body limit before parsing",
                "Bearer token, verified identity",
                "RFC 9457 problem documents",
            ],
            blue,
        ),
        # Session core
        _box(
            650,
            90,
            300,
            106,
            "Listener lanes (per client)",
            [
                "State and locks: bounded queue",
                "Thoughts: lossless, in order",
                "Cursors and presence: conflated",
            ],
            violet,
        ),
        _box(
            650,
            216,
            300,
            150,
            "SwarmSessionHub",
            [
                "PatchEngine: RFC 6902, AST-checked",
                "Path-level two-phase locking",
                "Deadlocks broken by rank",
                "Replay log, bundle hash per frame",
                "Thought backlog with sequence numbers",
                "Frame pump at broadcast_fps",
            ],
            violet,
        ),
        # Producers
        _box(
            1010,
            90,
            250,
            106,
            "Lock priority",
            [
                "Architect 90 · Owner 80",
                "Backend 70 · Coder 60",
                "Copywriter 50 · Styler 40 · Critic 30",
            ],
            amber,
        ),
        _box(
            1010,
            216,
            250,
            106,
            "LangGraph swarm",
            [
                "planner, then coder and backend,",
                "then styler, then critic and retry",
                "Tiered LLM provider",
            ],
            green,
        ),
        _box(
            1010,
            342,
            250,
            80,
            "Simulation ticker",
            ["Bézier cursor motion", "Obstacle-aware routing"],
            pink,
        ),
        # Bottom row
        _box(
            320,
            500,
            270,
            80,
            "Synthesis engine",
            ["Reads the latest snapshot", "Deterministic ZIP or tar.gz"],
            green,
        ),
        _box(
            650,
            500,
            300,
            80,
            "Event bus",
            ["In-memory, or Redis Pub/Sub", "Fan-out to replica workers"],
            amber,
        ),
        _box(
            1010,
            500,
            250,
            80,
            "Other workers",
            ["Adopt state as replicas", "Deduplicate thoughts by seq"],
            amber,
        ),
    ]
    wires = [
        _arrow([(260, 136), (320, 136)]),
        _arrow([(260, 232), (320, 232)]),
        _arrow([(260, 386), (320, 386)]),
        _arrow([(590, 228), (650, 228)], "commands", (594, 220)),
        _arrow([(590, 358), (650, 358)]),
        _arrow([(650, 140), (590, 140)], "events", (600, 132)),
        _arrow([(800, 216), (800, 196)], "fan-out", (808, 211)),
        _arrow([(455, 240), (455, 258)]),
        _arrow([(1010, 269), (950, 269)], "deltas", (957, 261)),
        _arrow([(1010, 356), (950, 356)], "frames", (957, 348)),
        _arrow([(800, 366), (800, 500)], "publish", (808, 438)),
        _arrow([(950, 540), (1010, 540)], dashed=True),
        _arrow([(455, 470), (455, 500)], "export", (463, 490)),
    ]
    headers = "".join(
        f'<text x="{x}" y="66" class="sub">{label}</text>'
        for x, label in ((30, "Clients"), (320, "Gateway"), (650, "Session core"), (1010, "Agents"))
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}"'
            f' width="{SVG_W}" height="{SVG_H}" role="img" aria-labelledby="t d">',
            '<title id="t">realtime-agentic-site-builder topology</title>',
            f'<desc id="d">{SVG_DESCRIPTION}</desc>',
            "<defs>",
            '<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"'
            ' markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0 0 L10 5 L0 10 z" fill="#94A3B8"/></marker>',
            f"<style>{SVG_STYLE}</style>",
            "</defs>",
            f'<rect width="{SVG_W}" height="{SVG_H}" rx="14" fill="#0B0F19"/>',
            '<text x="30" y="44" class="head">Gateway pipeline and event mesh</text>',
            headers,
            "".join(wires),
            "".join(parts),
            "</svg>",
            "",
        ]
    )


SVG_W, SVG_H = 1290, 610
SVG_FONT: Final = 'system-ui, -apple-system, "Segoe UI", sans-serif'
SVG_STYLE: Final = f"""
.box {{ fill: #111827; stroke: #334155; stroke-width: 1; }}
.title {{ fill: #F1F5F9; font: 600 14px {SVG_FONT}; }}
.body {{ fill: #94A3B8; font: 12px {SVG_FONT}; }}
.edge {{ fill: #CBD5E1; font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }}
.wire {{ fill: none; stroke: #94A3B8; stroke-width: 1.5; }}
.head {{ fill: #E5E7EB; font: 700 18px {SVG_FONT}; }}
.sub {{ fill: #64748B; font: 12px {SVG_FONT}; }}
"""
SVG_DESCRIPTION: Final = (
    "Clients reach an authenticated WebSocket gateway and REST ingress, which command a session"
    " hub. The hub applies lock-checked RFC 6902 deltas from the LangGraph swarm and cursor"
    " frames from the simulation ticker, and fans events out through per-client listener lanes"
    " to the gateway and through an event bus to other workers. Canvas clients get frames from"
    " a per-connection compositor, and the export route feeds the latest snapshot to the"
    " synthesis engine."
)


# --- shared chart styling -------------------------------------------------------------------

INK: Final = "#E5E7EB"
INK_MUTED: Final = "#94A3B8"
INK_FAINT: Final = "#64748B"
GRID: Final = "#1F2937"
SURFACE: Final = "#0B0F19"
# Validated for the dark surface with the dataviz palette checker: lightness band, chroma,
# colour-vision separation (worst delta E 32) and contrast all pass.
WITHOUT_CONVERGENCE: Final = "#EA580C"
WITH_CONVERGENCE: Final = "#8B5CF6"
CRITICAL: Final = "#EF4444"


def _svg_open(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' width="{width}" height="{height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{title}</title>',
        f'<desc id="d">{description}</desc>',
        f"<style>text {{ font-family: {SVG_FONT}; }}"
        " .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }</style>",
        f'<rect width="{width}" height="{height}" rx="14" fill="{SURFACE}"/>',
    ]


def _text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    fill: str = INK_MUTED,
    weight: int = 400,
    anchor: str = "start",
    mono: bool = False,
) -> str:
    klass = ' class="mono"' if mono else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}"'
        f' fill="{fill}" text-anchor="{anchor}"{klass}>{text}</text>'
    )


# --- dead reckoning convergence ---------------------------------------------------------------

CONVERGENCE_SECONDS: Final = 3.0
TICK_MS: Final = 1000 / 30


def _truth(t_ms: float) -> Point:
    return Point(960 + 620 * math.sin(t_ms / 380), 540 + 260 * math.sin(t_ms / 300 + 1))


@dataclass(frozen=True, slots=True)
class ConvergenceSeries:
    times_s: list[float]
    truth: list[float]
    without: list[float]
    with_convergence: list[float]
    losses: list[tuple[float, float]]


def convergence_series() -> ConvergenceSeries:
    """Render one lossy 30 Hz stream at 120 Hz twice: the bare spline, and with convergence."""
    rng = random.Random(4)
    bursts = set(range(20, 29)) | set(range(55, 69))  # About 300 ms and 470 ms of total loss.
    ticks = round(CONVERGENCE_SECONDS * 1000 / TICK_MS) + 4
    kept = [tick == 0 or (tick not in bursts and rng.random() >= 0.25) for tick in range(ticks)]
    bare, converging = DeadReckoner(0.0, _truth(0.0)), DeadReckoner(0.0, _truth(0.0))
    series = ConvergenceSeries([], [], [], [], [])
    next_tick, last_arrival = 1, 0.0
    for step in range(round((CONVERGENCE_SECONDS * 1000 + CLIENT_BUFFER_MS) / (1000 / 120))):
        now = step * 1000 / 120
        while next_tick * TICK_MS <= now:
            arrival = next_tick * TICK_MS
            if kept[next_tick]:
                if arrival - last_arrival > 4 * TICK_MS:
                    series.losses.append((last_arrival / 1000, arrival / 1000))
                last_arrival = arrival
                for tracker in (bare, converging):
                    tracker.observe(arrival, _truth(arrival))
            next_tick += 1
        at = now - CLIENT_BUFFER_MS
        if at < 0:
            continue
        series.times_s.append(at / 1000)
        series.truth.append(_truth(at).x)
        # The bare spline: the same samples and coasting, but nothing easing out a correction.
        series.without.append(bare._spline(at, WORLD).position.x)
        series.with_convergence.append(converging.sample(at, WORLD).position.x)
    return series


def _worst_jump(times: Sequence[float], values: Sequence[float], truth: Sequence[float]) -> float:
    return max(
        abs(values[i] - values[i - 1]) - abs(truth[i] - truth[i - 1]) for i in range(1, len(times))
    )


def convergence_svg() -> str:
    data = convergence_series()
    width, height = 1000, 580
    left, right = 70, 970
    panels = ((122, 350), (410, 530))  # Position, then error, sharing one time axis.

    def sx(t_s: float) -> float:
        return left + (right - left) * t_s / CONVERGENCE_SECONDS

    def path(values: Sequence[float], top: int, bottom: int, low: float, high: float) -> str:
        def sy(value: float) -> float:
            return bottom - (bottom - top) * (value - low) / (high - low)

        return " ".join(
            f"{'M' if i == 0 else 'L'}{sx(t):.1f},{sy(v):.1f}"
            for i, (t, v) in enumerate(zip(data.times_s, values, strict=True))
        )

    def line(x1: float, y1: float, x2: float, y2: float, color: str, extra: str = "") -> str:
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{color}" stroke-width="{2 if extra else 1}"{extra}/>'
        )

    errors_without = [abs(a - b) for a, b in zip(data.without, data.truth, strict=True)]
    errors_with = [abs(a - b) for a, b in zip(data.with_convergence, data.truth, strict=True)]
    jump_without = _worst_jump(data.times_s, data.without, data.truth)
    jump_with = _worst_jump(data.times_s, data.with_convergence, data.truth)
    parts = _svg_open(
        width,
        height,
        "Dead reckoning with and without convergence",
        "A cursor's x position over three seconds of a 30 Hz stream with two long bursts of "
        f"packet loss. Without convergence the track snaps by up to {jump_without:.0f} px in one "
        f"rendered frame when a burst ends. With 120 ms convergence the worst frame-to-frame jump "
        f"is {jump_with:.0f} px, at the cost of staying off the real position slightly longer.",
    )
    parts.append(_text(30, 40, "Dead reckoning through packet loss", size=18, weight=700, fill=INK))
    parts.append(
        _text(
            30,
            62,
            f"Worst jump in one 120 Hz frame: {jump_without:.0f} px without convergence,"
            f" {jump_with:.0f} px with it. The price is staying off the real position for about"
            " 120 ms longer after a burst.",
            size=12,
            fill=INK_FAINT,
        )
    )
    legend_x = 30.0
    for label, color, dashed in (
        ("Real position", INK_FAINT, True),
        ("Without convergence", WITHOUT_CONVERGENCE, False),
        ("With 120 ms convergence", WITH_CONVERGENCE, False),
    ):
        dash = ' stroke-dasharray="5 4"' if dashed else ' stroke-dasharray="none"'
        parts.append(line(legend_x, 88, legend_x + 18, 88, color, dash))
        parts.append(_text(legend_x + 24, 92, label, size=11))
        legend_x += 24 + len(label) * 6.4 + 22
    for start, end in data.losses:
        x0, x1 = sx(max(start, 0)), sx(min(end, CONVERGENCE_SECONDS))
        span = panels[1][1] - panels[0][0]
        parts.append(
            f'<rect x="{x0:.1f}" y="{panels[0][0]}" width="{x1 - x0:.1f}" height="{span}"'
            ' fill="#F8FAFC" fill-opacity="0.05"/>'
        )
        if x1 - x0 >= 80:
            label_x = (x0 + x1) / 2
            parts.append(
                _text(label_x, panels[0][0] + 14, "packets lost", size=10, anchor="middle")
            )

    def axes(top: int, bottom: int, low: float, high: float, step: float) -> None:
        value = low
        while value <= high + 1e-9:
            y = bottom - (bottom - top) * (value - low) / (high - low)
            parts.append(line(left, y, right, y, GRID))
            parts.append(_text(left - 8, y + 4, f"{value:.0f}", size=10, anchor="end"))
            value += step

    positions = data.truth + data.without + data.with_convergence
    low, high = math.floor(min(positions) / 200) * 200, math.ceil(max(positions) / 200) * 200
    axes(*panels[0], low, high, 200)
    parts.append(_text(left, panels[0][0] - 8, "Cursor x (px)", size=11))
    error_high = math.ceil(max(errors_without) / 100) * 100
    axes(*panels[1], 0, error_high, error_high / 2)
    parts.append(_text(left, panels[1][0] - 8, "Distance from the real position (px)", size=11))
    for second in range(int(CONVERGENCE_SECONDS) + 1):
        parts.append(_text(sx(second), panels[1][1] + 18, f"{second} s", size=10, anchor="middle"))
    for values, color, dashed in (
        (data.truth, INK_FAINT, True),
        (data.without, WITHOUT_CONVERGENCE, False),
        (data.with_convergence, WITH_CONVERGENCE, False),
    ):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        parts.append(
            f'<path d="{path(values, *panels[0], low, high)}" fill="none" stroke="{color}"'
            f' stroke-width="2" stroke-linejoin="round"{dash}/>'
        )
    for values, color in ((errors_without, WITHOUT_CONVERGENCE), (errors_with, WITH_CONVERGENCE)):
        parts.append(
            f'<path d="{path(values, *panels[1], 0, error_high)}" fill="none" stroke="{color}"'
            ' stroke-width="2" stroke-linejoin="round"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# --- lock contention matrix -------------------------------------------------------------------

LOCK_PARTICIPANTS: Final = (
    ("agent-architect", "Architect", 90),
    ("user:alice", "Owner (person)", 80),
    ("agent-coder", "Frontend Coder", 60),
    ("agent-styler", "Styler", 40),
)
_LOCK_COLORS: Final = {
    "agent-architect": IDENTITIES[ARCHITECT].color,
    "user:alice": human_color("alice"),
    "agent-coder": IDENTITIES[CODER].color,
    "agent-styler": IDENTITIES[STYLER].color,
}


_SHORT: Final = {
    "agent-architect": "Architect",
    "user:alice": "Owner",
    "agent-coder": "Coder",
    "agent-styler": "Styler",
}


@dataclass(frozen=True, slots=True)
class LockStep:
    action: str
    states: dict[str, list[tuple[str, str]]]  # participant -> [(target, state)]
    events: list[str]
    preempted: frozenset[str]


async def lock_scenario() -> list[LockStep]:
    """Drive the real lock manager through contention, a priority handoff and a deadlock."""
    now = 1_000
    events: list[LockEvent] = []
    manager = AsyncLockManager(lease_ms=30_000, clock=lambda: now, on_event=events.append)
    priority = {agent: rank for agent, _, rank in LOCK_PARTICIPANTS}
    script: list[tuple[str, str, str, str]] = [
        ("Owner takes hero", "acquire", "user:alice", "hero"),
        ("Styler queues for hero", "acquire", "agent-styler", "hero"),
        ("Coder queues for hero", "acquire", "agent-coder", "hero"),
        ("Architect queues for hero", "acquire", "agent-architect", "hero"),
        ("Owner releases hero", "release", "user:alice", "hero"),
        ("Coder takes the title", "acquire", "agent-coder", "document"),
        ("Architect wants the title", "acquire", "agent-architect", "document"),
        ("Architect releases hero", "release", "agent-architect", "hero"),
    ]
    steps: list[LockStep] = []
    for action, verb, agent, target in script:
        events.clear()
        if verb == "acquire":
            await manager.acquire(target, agent, priority[agent], now)
        else:
            await manager.release(target, agent)
        now += 100
        states: dict[str, list[tuple[str, str]]] = {
            agent_id: [] for agent_id, _, _ in LOCK_PARTICIPANTS
        }
        for lock in manager.active_locks():
            states[lock.locked_by_agent_id].append((lock.target_id, lock.state.value))
        notes = [
            f"✕ {_SHORT[event.lock.locked_by_agent_id]} loses {event.lock.target_id}"
            if event.kind.value == "PREEMPTED"
            else f"→ {_SHORT[event.lock.locked_by_agent_id]} gets {event.lock.target_id}"
            for event in events
        ]
        preempted = frozenset(
            event.lock.locked_by_agent_id for event in events if event.kind.value == "PREEMPTED"
        )
        steps.append(LockStep(action, states, notes, preempted))
    return steps


def locks_svg() -> str:
    steps = asyncio.run(lock_scenario())
    width, label_w, cell_w, row_h = 1280, 170, 136, 64
    top = 150
    height = top + row_h * len(LOCK_PARTICIPANTS) + 150
    parts = _svg_open(
        width,
        height,
        "Lock contention matrix",
        "Eight steps through the real lock manager. Three agents queue for a component an owner "
        "holds. On release it passes to the highest priority waiter, the Architect. The Coder "
        "then takes the title, the Architect asks for it, and the wait cycle is broken by "
        "preempting the lower ranked Coder, which loses everything it held or waited for.",
    )
    parts.append(_text(30, 40, "Two-phase locking under contention", size=18, weight=700, fill=INK))
    parts.append(
        _text(
            30,
            62,
            "Each cell is the real lock table after the step above it. Priority decides who is"
            " promoted next and who gives way in a deadlock. It never takes a lock from a live"
            " holder.",
            size=12,
            fill=INK_FAINT,
        )
    )
    for column, step in enumerate(steps):
        x = label_w + column * cell_w
        parts.append(
            _text(x + 8, top - 40, f"{column + 1}", size=11, weight=700, fill=INK, mono=True)
        )
        words, line = step.action.split(), ""
        rows: list[str] = []
        for word in words:
            if len(line) + len(word) > 17:
                rows.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        rows.append(line)
        for index, text in enumerate(rows):
            parts.append(_text(x + 8, top - 24 + index * 14, text, size=11, fill=INK_MUTED))
    for row, (agent_id, name, rank) in enumerate(LOCK_PARTICIPANTS):
        y = top + row * row_h + 18
        color = _LOCK_COLORS[agent_id]
        parts.append(f'<circle cx="38" cy="{y + row_h / 2 - 4:.1f}" r="5" fill="{color}"/>')
        parts.append(_text(52, y + row_h / 2 - 6, name, size=13, weight=600, fill=INK))
        parts.append(_text(52, y + row_h / 2 + 10, f"priority {rank}", size=11, fill=INK_FAINT))
        for column, step in enumerate(steps):
            x = label_w + column * cell_w
            parts.append(
                f'<rect x="{x + 2}" y="{y}" width="{cell_w - 4}" height="{row_h - 6}" rx="6"'
                f' fill="#111827" stroke="{GRID}"/>'
            )
            for index, (target, state) in enumerate(step.states[agent_id]):
                cy = y + 8 + index * 24
                label = "holds" if state == "ACQUIRED" else "waits"
                if state == "ACQUIRED":
                    parts.append(
                        f'<rect x="{x + 8}" y="{cy}" width="{cell_w - 16}" height="20" rx="10"'
                        f' fill="{color}" fill-opacity="0.9"/>'
                    )
                    parts.append(
                        _text(
                            x + 16, cy + 14, f"{label} {target}", size=11, weight=600, fill=SURFACE
                        )
                    )
                else:
                    parts.append(
                        f'<rect x="{x + 8.5}" y="{cy + 0.5}" width="{cell_w - 17}" height="19"'
                        f' rx="10" fill="none" stroke="{color}" stroke-dasharray="4 3"/>'
                    )
                    parts.append(_text(x + 16, cy + 14, f"{label} {target}", size=11, fill=INK))
            if agent_id in step.preempted:
                parts.append(
                    f'<rect x="{x + 8.5}" y="{y + 8.5}" width="{cell_w - 17}" height="19" rx="10"'
                    f' fill="none" stroke="{CRITICAL}"/>'
                )
                parts.append(
                    _text(x + 16, y + 22, "✕ preempted", size=11, weight=600, fill=CRITICAL)
                )
    notes_y = top + row_h * len(LOCK_PARTICIPANTS) + 44
    parts.append(
        _text(30, notes_y, "Events the lock manager emitted", size=12, weight=600, fill=INK)
    )
    for column, step in enumerate(steps):
        x = label_w + column * cell_w
        for index, note in enumerate(step.events):
            fill = CRITICAL if note.startswith("✕") else INK_MUTED
            parts.append(_text(x + 8, notes_y + index * 16, note, size=10, fill=fill))
    legend_y = height - 28
    parts.append(
        f'<rect x="30" y="{legend_y - 13}" width="44" height="18" rx="9" fill="{INK_MUTED}"/>'
    )
    parts.append(_text(84, legend_y, "holds the lock", size=11))
    parts.append(
        f'<rect x="200.5" y="{legend_y - 12.5}" width="44" height="17" rx="9" fill="none"'
        f' stroke="{INK_MUTED}" stroke-dasharray="4 3"/>'
    )
    parts.append(_text(254, legend_y, "queued, waiting", size=11))
    parts.append(
        _text(
            380,
            legend_y,
            "✕ preempted: lost everything it held or waited for",
            size=11,
            fill=CRITICAL,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# --- time travel and forking ----------------------------------------------------------------

SOURCE_COMMITS: Final = 20
FORK_AT: Final = 11
FORK_COMMITS: Final = 3
SCRUB_TO: Final = 18
FORK_AUTHORS: Final = ("agent-coder", "agent-architect", "agent-styler")
FORK_TARGETS: Final = ("hero", "pricing", "features")


@dataclass(frozen=True, slots=True)
class ForkMeasurement:
    source_frames: int
    fork_frames: int
    shared_frames: int
    shared_patches: int
    shared_bytes: int
    own_frames: int
    source_unchanged: bool
    authors: tuple[str, ...]


def _retitle(session_id: str, agent_id: str, target: str, title: str, seq: int) -> StreamDelta:
    index = [component.id for component in BUNDLE.frontend.components].index(target)
    return StreamDelta(
        session_id=session_id,
        agent_id=agent_id,
        delta_seq=seq,
        target_domain="frontend",
        target_id=target,
        patches=(
            PatchOperation.model_validate(
                {"op": "replace", "path": f"/frontend/components/{index}/title", "value": title}
            ),
        ),
        emitted_at_ms=0,
    )


async def measure_fork() -> ForkMeasurement:
    """Fork a real session on the real hub, and count what the branch shares with its source."""
    hub = SwarmSessionHub(Settings(gateway_secret=SecretStr("asset-generation-only-" + "0" * 24)))
    await hub.create_session("source", BUNDLE)
    for agent, target in zip(FORK_AUTHORS, FORK_TARGETS, strict=True):
        await hub.lock_manager("source").acquire(target, agent, 70, hub.now_ms())
    authors = []
    for seq in range(SOURCE_COMMITS):
        agent, target = FORK_AUTHORS[seq % 3], FORK_TARGETS[seq % 3]
        authors.append(agent)
        await hub.apply_delta("source", _retitle("source", agent, target, f"Draft {seq}", seq))
    before = hub.timeline("source")
    await hub.fork_session("source", "fork", FORK_AT)
    await hub.lock_manager("fork").acquire("pricing", "user:alice", 95, hub.now_ms())
    for seq in range(FORK_COMMITS):
        delta = _retitle("fork", "user:alice", "pricing", f"Alice {seq}", 100 + seq)
        await hub.apply_delta("fork", delta)
    source, fork = hub.timeline("source"), hub.timeline("fork")
    shared = sum(a is b for a, b in zip(fork.snapshots, source.snapshots, strict=False))
    measurement = ForkMeasurement(
        source_frames=len(source.snapshots),
        fork_frames=len(fork.snapshots),
        shared_frames=shared,
        shared_patches=sum(
            a.patches is b.patches for a, b in zip(fork.deltas, source.deltas, strict=False)
        ),
        shared_bytes=sum(bundle_size(frame.bundle_state) for frame in fork.snapshots[:shared]),
        own_frames=len(fork.snapshots) - shared,
        source_unchanged=source == before,
        authors=tuple(authors),
    )
    await hub.shutdown()
    if not measurement.source_unchanged or measurement.shared_frames != FORK_AT + 1:
        raise SystemExit(f"the fork did not behave as documented: {measurement}")
    return measurement


def forking_svg() -> str:
    m = asyncio.run(measure_fork())
    width, height = 1280, 610
    left, step = 120, 52
    source_y, fork_y, scrub_y = 190, 330, 470
    colors = {identity.agent_id: identity.color for identity in IDENTITIES.values()}
    alice = human_color("alice")

    def fx(frame: int) -> float:
        return left + frame * step

    parts = _svg_open(
        width,
        height,
        "Time travel and reality forking",
        f"A session of {SOURCE_COMMITS} commits on the real hub, forked at frame {FORK_AT}. The"
        f" fork shares frames 0 to {FORK_AT} with its source by reference, {m.shared_frames}"
        f" frames and {m.shared_bytes:,} bytes of canonical state, none of it copied, then"
        f" commits {m.own_frames} frames of its own while the source stays unchanged. Below, the"
        f" scrubber rebuilds frame {SCRUB_TO} from the checkpoint at frame"
        f" {SCRUB_TO - SCRUB_TO % HISTORY_CHECKPOINT_EVERY} plus"
        f" {SCRUB_TO % HISTORY_CHECKPOINT_EVERY} patch replays.",
    )
    parts.append(_text(30, 40, "Time travel and reality forking", size=18, weight=700, fill=INK))
    parts.append(
        _text(
            30,
            62,
            f"Measured on the real hub: a {SOURCE_COMMITS}-commit session forked at frame"
            f" {FORK_AT}. Frames are immutable, so a fork costs a reference per frame, not a copy.",
            size=12,
            fill=INK_FAINT,
        )
    )
    # The shared prefix, bracketed across both lanes.
    shared_right = fx(FORK_AT) + 18
    parts.append(
        f'<rect x="{left - 22}" y="{source_y - 34}" width="{shared_right - left + 22}"'
        f' height="{fork_y - source_y + 62}" rx="14" fill="#8B5CF6" fill-opacity="0.07"'
        f' stroke="#8B5CF6" stroke-opacity="0.45" stroke-dasharray="5 4"/>'
    )
    parts.append(
        _text(
            left - 10,
            source_y - 44,
            f"Shared by reference: frames 0–{FORK_AT}  ·  {m.shared_frames} frames,"
            f" {m.shared_patches} patch lists, {m.shared_bytes:,} bytes of state"
            "  ·  0 bytes copied",
            size=12,
            weight=600,
            fill="#C4B5FD",
        )
    )
    for label, y, sub in (
        ("Source", source_y, f"{m.source_frames} frames"),
        ("Fork", fork_y, f"{m.fork_frames} frames"),
    ):
        parts.append(_text(30, y - 2, label, size=13, weight=700, fill=INK))
        parts.append(_text(30, y + 14, sub, size=11, fill=INK_FAINT))
    parts.append(
        f'<line x1="{fx(0)}" y1="{source_y}" x2="{fx(SOURCE_COMMITS)}" y2="{source_y}"'
        f' stroke="{GRID}" stroke-width="2"/>'
    )
    for frame in range(SOURCE_COMMITS + 1):
        color = INK_FAINT if frame == 0 else colors[m.authors[frame - 1]]
        parts.append(f'<circle cx="{fx(frame)}" cy="{source_y}" r="9" fill="{color}"/>')
        parts.append(
            _text(
                fx(frame),
                source_y + 28,
                str(frame),
                size=10,
                fill=INK_FAINT,
                anchor="middle",
                mono=True,
            )
        )
    # The fork's lane: the shared prefix as ghosts of the same objects, then its own frames.
    for frame in range(FORK_AT + 1):
        parts.append(
            f'<line x1="{fx(frame)}" y1="{source_y + 11}" x2="{fx(frame)}" y2="{fork_y - 11}"'
            f' stroke="#8B5CF6" stroke-opacity="0.35" stroke-dasharray="2 4"/>'
        )
        parts.append(
            f'<circle cx="{fx(frame)}" cy="{fork_y}" r="8.5" fill="none" stroke="#8B5CF6"'
            f' stroke-opacity="0.7" stroke-dasharray="3 2"/>'
        )
    last = FORK_AT + m.own_frames
    parts.append(
        f'<path d="M {fx(FORK_AT)} {source_y + 10} C {fx(FORK_AT) + 30} {source_y + 70},'
        f' {fx(FORK_AT + 1) - 20} {fork_y - 60}, {fx(FORK_AT + 1)} {fork_y}" fill="none"'
        f' stroke="{alice}" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{fx(FORK_AT + 1)}" y1="{fork_y}" x2="{fx(last)}" y2="{fork_y}"'
        f' stroke="{alice}" stroke-width="2"/>'
    )
    for frame in range(FORK_AT + 1, last + 1):
        parts.append(f'<circle cx="{fx(frame)}" cy="{fork_y}" r="9" fill="{alice}"/>')
        parts.append(
            _text(
                fx(frame),
                fork_y + 28,
                f"{frame}′",
                size=10,
                fill=INK_FAINT,
                anchor="middle",
                mono=True,
            )
        )
    parts.append(
        _text(
            fx(last) + 22,
            fork_y - 4,
            f"POST /api/v1/sessions/source/fork?from_seq={FORK_AT}",
            size=11,
            fill=INK,
            mono=True,
        )
    )
    parts.append(
        _text(
            fx(last) + 22,
            fork_y + 12,
            f"{m.own_frames} commits of its own, by a person at priority 95.",
            size=11,
            fill=INK_MUTED,
        )
    )
    parts.append(
        _text(
            fx(last) + 22,
            fork_y + 27,
            "The source is unchanged, frame for frame.",
            size=11,
            fill=INK_MUTED,
        )
    )
    # The scrubber, rebuilding a frame from the nearest checkpoint.
    parts.append(_text(30, scrub_y - 2, "Scrubber", size=13, weight=700, fill=INK))
    parts.append(_text(30, scrub_y + 14, "GET …/history", size=11, fill=INK_FAINT, mono=True))
    parts.append(
        f'<rect x="{fx(0)}" y="{scrub_y - 3}" width="{fx(SOURCE_COMMITS) - fx(0)}" height="6"'
        f' rx="3" fill="{GRID}"/>'
    )
    parts.append(
        f'<rect x="{fx(0)}" y="{scrub_y - 3}" width="{fx(SCRUB_TO) - fx(0)}" height="6" rx="3"'
        f' fill="#8B5CF6"/>'
    )
    checkpoint = SCRUB_TO - SCRUB_TO % HISTORY_CHECKPOINT_EVERY
    for frame in range(0, SOURCE_COMMITS + 1, HISTORY_CHECKPOINT_EVERY):
        parts.append(
            f'<rect x="{fx(frame) - 5}" y="{scrub_y - 14}" width="10" height="10" rx="2"'
            f' fill="{INK if frame == checkpoint else INK_FAINT}"'
            f' transform="rotate(45 {fx(frame)} {scrub_y - 9})"/>'
        )
    parts.append(
        f'<path d="M {fx(checkpoint)} {scrub_y + 14} C {fx(checkpoint)} {scrub_y + 34},'
        f' {fx(SCRUB_TO)} {scrub_y + 34}, {fx(SCRUB_TO)} {scrub_y + 14}" fill="none"'
        f' stroke="{INK_MUTED}" stroke-dasharray="3 3"/>'
    )
    parts.append(f'<circle cx="{fx(SCRUB_TO)}" cy="{scrub_y}" r="8" fill="{INK}"/>')
    note_x = (fx(checkpoint) + fx(SCRUB_TO)) / 2
    parts.append(
        _text(
            note_x,
            scrub_y + 52,
            f"Frame {SCRUB_TO} = checkpoint {checkpoint} + {SCRUB_TO - checkpoint} patch replays",
            size=11,
            fill=INK,
            anchor="middle",
        )
    )
    parts.append(
        _text(
            note_x,
            scrub_y + 68,
            f"A checkpoint every {HISTORY_CHECKPOINT_EVERY} frames: never more than"
            f" {HISTORY_CHECKPOINT_EVERY - 1} replays.",
            size=11,
            fill=INK_MUTED,
            anchor="middle",
        )
    )
    legend_y = height - 22
    x = 30.0
    for identity_id in FORK_AUTHORS:
        name = next(i.name for i in IDENTITIES.values() if i.agent_id == identity_id)
        parts.append(
            f'<circle cx="{x + 5}" cy="{legend_y - 4}" r="5" fill="{colors[identity_id]}"/>'
        )
        parts.append(_text(x + 16, legend_y, name, size=11))
        x += 30 + 7 * len(name)
    parts.append(f'<circle cx="{x + 5}" cy="{legend_y - 4}" r="5" fill="{alice}"/>')
    parts.append(_text(x + 16, legend_y, "alice, a person", size=11))
    parts.append(
        f'<rect x="{x + 140}" y="{legend_y - 12}" width="10" height="10" rx="2" fill="{INK_FAINT}"'
        f' transform="rotate(45 {x + 145} {legend_y - 7})"/>'
    )
    parts.append(_text(x + 160, legend_y, "history checkpoint", size=11))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    frames = asyncio.run(simulate())
    gif = encode_gif(render(frames))
    if len(gif) > MAX_GIF_BYTES:
        raise SystemExit(f"{GIF_PATH.name} is {len(gif)} bytes, over the {MAX_GIF_BYTES} limit")
    GIF_PATH.write_bytes(gif)
    SVG_PATH.write_text(topology_svg())
    CONVERGENCE_PATH.write_text(convergence_svg())
    LOCKS_PATH.write_text(locks_svg())
    FORKING_PATH.write_text(forking_svg())
    for path in (GIF_PATH, SVG_PATH, CONVERGENCE_PATH, LOCKS_PATH, FORKING_PATH):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size:>9,} bytes  sha256 {digest}")


if __name__ == "__main__":
    main()
