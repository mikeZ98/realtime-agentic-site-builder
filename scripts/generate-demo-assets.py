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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageChops, ImageDraw, ImageFont

from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.canvas.compositor import (
    AgentSprite,
    CanvasCompositor,
    LockOverlay,
    ThoughtBadge,
    VisualFrame,
    human_color,
)
from realtime_agentic_site_builder.canvas.projection import DeadReckoner, hermite_controls
from realtime_agentic_site_builder.canvas.viewer import CLIENT_BUFFER_MS
from realtime_agentic_site_builder.core.hub import ThoughtEvent
from realtime_agentic_site_builder.core.lock_manager import AsyncLockManager, LockEvent
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
from realtime_agentic_site_builder.domain.swarm import AgentRole, AgentThoughtStream
from realtime_agentic_site_builder.simulation.controller import CursorSimulator
from realtime_agentic_site_builder.simulation.motion import Point, Viewport, cubic_bezier

ROOT: Final = Path(__file__).resolve().parent.parent
ASSETS: Final = ROOT / "docs" / "assets"
GIF_PATH: Final = ASSETS / "swarm-live-demo.gif"
SVG_PATH: Final = ASSETS / "architecture-topology.svg"
CONVERGENCE_PATH: Final = ASSETS / "dead-reckoning-convergence.svg"
LOCKS_PATH: Final = ASSETS / "lock-contention-matrix.svg"

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


def _component(
    component_id: str, title: str, x: int, y: int, width: int, height: int
) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.FEATURES,
        title=title,
        code="export default function Section() { return <section />; }",
        box=BoundingBox(x=x, y=y, width=width, height=height),
    )


BUNDLE: Final = FullstackProjectBundle(
    project_id="demo",
    frontend=SiteBlueprint(
        site_id="demo",
        title="SaaS landing page",
        theme="dark",
        components=(
            _component("navbar", "Navbar", 60, 40, 1800, 90),
            _component("hero", "Hero", 60, 160, 1800, 300),
            _component("features", "Features", 60, 490, 880, 260),
            _component("pricing", "Pricing", 980, 490, 880, 260),
            _component("contact", "Contact form", 60, 780, 1100, 250),
            _component("footer", "Footer", 1200, 780, 660, 250),
        ),
    ),
    backend=BackendBlueprint(service_name="api", database_dialect=DatabaseDialect.SQLITE),
)
BOXES: Final = {component.id: component.box for component in BUNDLE.frontend.components}


@dataclass(frozen=True, slots=True)
class Cue:
    """One scripted moment: a lock change, a cursor intent, and what the agent says about it."""

    at_ms: int
    role: AgentRole
    move: str  # "edit", "go", "wait" or "idle"
    target: str | None = None
    lock: str | None = None  # "take", "want" or "drop"
    thought: str | None = None


SCRIPT: Final = (
    Cue(0, ARCHITECT, "go", "hero", thought="Planning six sections from the brief"),
    Cue(150, COPYWRITER, "go", "navbar"),
    Cue(250, CODER, "edit", "hero", "take", "Writing the hero layout"),
    Cue(550, BACKEND, "edit", "contact", "take", "POST /api/v1/leads handler"),
    Cue(900, STYLER, "edit", "features", "take", "Tuning the feature grid"),
    Cue(1200, ARCHITECT, "go", "pricing"),
    Cue(1350, COPYWRITER, "wait", "features", "want", "Waiting for Styler on features"),
    Cue(1700, CRITIC, "wait", "hero", "want", "Queued to review the hero"),
    Cue(2300, CODER, "go", "pricing", "drop", "Hero committed, releasing the lock"),
    Cue(2300, CRITIC, "edit", "hero", None, "Reviewing hero contrast"),
    Cue(2700, STYLER, "go", "footer", "drop"),
    Cue(2700, COPYWRITER, "edit", "features", None, "Rewriting feature copy"),
    Cue(3000, ARCHITECT, "edit", "pricing", "take", "Adjusting the pricing tiers"),
    Cue(3400, BACKEND, "go", "footer", "drop", "Endpoint and schema committed"),
    Cue(3900, CRITIC, "go", "contact", "drop", "Hero approved"),
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


async def simulate() -> list[VisualFrame]:
    """Run the script through the real simulator, lock manager and compositor."""
    now = 0
    simulator = CursorSimulator(WORLD, seed=11)
    locks = AsyncLockManager(lease_ms=30_000, clock=lambda: now)
    compositor = CanvasCompositor(BUNDLE)
    pending = deque(sorted(SCRIPT, key=lambda cue: cue.at_ms))
    thought_seq = 0
    frames: list[VisualFrame] = []
    elapsed = 0
    for index in range(FRAMES):
        target_ms = round((index + 1) * 1000 / FPS)
        while pending and pending[0].at_ms <= target_ms:
            cue = pending.popleft()
            thought_seq = await _perform(cue, simulator, locks, compositor, thought_seq, now)
        simulator.sync(BUNDLE, locks.active_locks())
        for frame in simulator.step(target_ms - elapsed):
            compositor.place_agent(frame, float(target_ms))
        elapsed = now = target_ms
        position = human_path(target_ms)
        compositor.place_human(HUMAN, round(position.x), round(position.y), "pricing", target_ms)
        frames.append(compositor.compose(float(target_ms), locks.active_locks(), now))
    return frames


async def _perform(
    cue: Cue,
    simulator: CursorSimulator,
    locks: AsyncLockManager,
    compositor: CanvasCompositor,
    thought_seq: int,
    now: int,
) -> int:
    identity = IDENTITIES[cue.role]
    box = BOXES[cue.target] if cue.target else None
    held = [
        lock.target_id
        for lock in locks.active_locks()
        if lock.locked_by_agent_id == identity.agent_id
    ]
    if cue.lock == "drop":
        for target in held:
            await locks.release(target, identity.agent_id)
    elif cue.lock in {"take", "want"} and cue.target:
        await locks.acquire(cue.target, identity.agent_id, identity.priority, now)
    if cue.move == "edit" and box is not None:
        simulator.edit(cue.role, box)
    elif cue.move == "go" and box is not None:
        simulator.navigate_to(cue.role, box)
    elif cue.move == "wait" and box is not None:
        simulator.block_on(cue.role, box)
    else:
        simulator.idle(cue.role)
    if cue.thought is None:
        return thought_seq
    thought = AgentThoughtStream(
        agent_id=identity.agent_id, role=cue.role, thought=cue.thought, timestamp_ms=now
    )
    compositor.add_thought(
        ThoughtEvent(origin="demo", session_id="demo", seq=thought_seq, thought=thought), now
    )
    return thought_seq + 1


# --- drawing --------------------------------------------------------------------------------

Color = tuple[int, int, int]
SCALE: Final = OUT_W * SUPERSAMPLE / WORLD.width


def _hex(color: str) -> Color:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _sx(x: float) -> float:
    return x * SCALE


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size * SUPERSAMPLE)


FONT_SMALL: Final = _font(9)
FONT_LABEL: Final = _font(10)
FONT_TITLE: Final = _font(11)


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
    box = overlay.box
    left, top = _sx(box.x * WORLD.width), _sx(box.y * WORLD.height)
    right, bottom = left + _sx(box.width * WORLD.width), top + _sx(box.height * WORLD.height)
    leader = overlay.owner or (overlay.waiters[0] if overlay.waiters else None)
    color = _hex(leader.color) if leader else LINE
    draw.rounded_rectangle((left, top, right, bottom), radius=6 * SUPERSAMPLE, fill=PANEL)
    if overlay.state.value == "FREE":
        draw.rounded_rectangle(
            (left, top, right, bottom), radius=6 * SUPERSAMPLE, outline=LINE, width=SUPERSAMPLE
        )
    else:
        pulse = 0.55 + 0.45 * math.sin(t_ms / 160) if overlay.state.value == "WAITING" else 1.0
        for spread, alpha in ((6, 0.12), (3, 0.25)):
            grow = spread * SUPERSAMPLE
            draw.rounded_rectangle(
                (left - grow, top - grow, right + grow, bottom + grow),
                radius=(6 + spread) * SUPERSAMPLE,
                outline=(*color, round(255 * alpha * pulse)),
                width=2 * SUPERSAMPLE,
            )
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=6 * SUPERSAMPLE,
            outline=(*color, round(255 * pulse)),
            width=2 * SUPERSAMPLE,
        )
    draw.text(
        (left + 8 * SUPERSAMPLE, top + 6 * SUPERSAMPLE), overlay.title, font=FONT_TITLE, fill=MUTED
    )
    if overlay.state.value != "FREE":
        queue = f" +{len(overlay.waiters)} waiting" if overlay.waiters else ""
        label = (
            f"{overlay.owner.label}{queue}" if overlay.owner else f"{len(overlay.waiters)} waiting"
        )
        _pill(
            draw,
            label,
            left + 8 * SUPERSAMPLE,
            bottom - 22 * SUPERSAMPLE,
            color,
        )


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
    x = OUT_W * SUPERSAMPLE - 8 * SUPERSAMPLE
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


def render(frames: Sequence[VisualFrame]) -> list[Image.Image]:
    images: list[Image.Image] = []
    trails: dict[str, deque[tuple[float, AgentSprite]]] = {}
    size = (OUT_W * SUPERSAMPLE, OUT_H * SUPERSAMPLE)
    for frame in frames:
        canvas = Image.new("RGBA", size, (*BG, 255))
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
        draw.text(
            (10 * SUPERSAMPLE, (OUT_H - 12) * SUPERSAMPLE),
            "Swarm Canvas  \N{MIDDLE DOT}  live session replay",
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
    for path in (GIF_PATH, SVG_PATH, CONVERGENCE_PATH, LOCKS_PATH):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size:>9,} bytes  sha256 {digest}")


if __name__ == "__main__":
    main()
