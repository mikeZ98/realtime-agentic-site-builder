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
)
from realtime_agentic_site_builder.canvas.projection import hermite_controls
from realtime_agentic_site_builder.core.hub import ThoughtEvent
from realtime_agentic_site_builder.core.lock_manager import AsyncLockManager
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


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    frames = asyncio.run(simulate())
    gif = encode_gif(render(frames))
    if len(gif) > MAX_GIF_BYTES:
        raise SystemExit(f"{GIF_PATH.name} is {len(gif)} bytes, over the {MAX_GIF_BYTES} limit")
    GIF_PATH.write_bytes(gif)
    SVG_PATH.write_text(topology_svg())
    for path in (GIF_PATH, SVG_PATH):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size:>9,} bytes  sha256 {digest}")


if __name__ == "__main__":
    main()
