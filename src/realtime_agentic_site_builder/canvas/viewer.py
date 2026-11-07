"""The self-contained canvas page, and the headers that keep it that way.

Authentication has one route through it. A browser cannot put a header on a page load, and a query
string ends up in access logs and ``Referer`` headers, so a link carries the token in its fragment,
which is never sent to any server. The page moves it into memory, erases it from the address bar and
presents it as a WebSocket subprotocol. The socket is what enforces access, not this page, and the
page holds nothing for anyone without a token: it is the same bytes for every session and caller.

Nothing from the request is reflected, which is what makes a hash-based policy possible. Text from
the session reaches the page only through ``textContent`` and ``fillText``, which make no markup.
"""

import base64
import hashlib
from collections.abc import Mapping
from typing import Final

from starlette.responses import Response

from realtime_agentic_site_builder.canvas.projection import (
    CONVERGENCE_MS,
    EXTRAPOLATION_LIMIT_MS,
    MOTION_RESET_MS,
)
from realtime_agentic_site_builder.gateway.auth import SUBPROTOCOL, TOKEN_SUBPROTOCOL_PREFIX

MAX_VIEWER_BYTES: Final = 64 * 1024
# On top of the compositor's own smoothing, this absorbs jitter in the network's delivery.
CLIENT_BUFFER_MS: Final = 90
HEARTBEAT_MS: Final = 20_000
PRESENCE_INTERVAL_MS: Final = 100

_STYLE: Final = """
:root { color-scheme: dark; --bg: #0b0f19; --panel: #111827; --line: #1f2937;
  --text: #e5e7eb; --muted: #9ca3af; --accent: #8b5cf6; --bad: #f87171; }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
#app { display: grid; grid-template-columns: minmax(0, 1fr) 320px; height: 100dvh; }
#stage { position: relative; min-width: 0; min-height: 0; overflow: hidden; }
canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%;
  touch-action: none; outline: none; }
#status { position: absolute; top: 12px; right: 12px; display: flex; align-items: center; gap: 8px;
  padding: 5px 10px; border: 1px solid var(--line); border-radius: 999px;
  background: rgba(17, 24, 39, .85); color: var(--muted); pointer-events: none; }
#dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
#status[data-state="live"] #dot { background: #10b981; }
#status[data-state="error"] #dot { background: var(--bad); }
aside { display: flex; flex-direction: column; min-height: 0; border-left: 1px solid var(--line);
  background: var(--panel); }
h1 { margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); font-size: 13px;
  font-weight: 600; }
#feed { flex: 1; min-height: 0; margin: 0; padding: 6px 0; overflow-y: auto; list-style: none;
  overflow-wrap: anywhere; }
#feed li { padding: 7px 14px; border-left: 3px solid var(--who, var(--muted)); margin: 2px 0; }
#feed b { display: block; color: var(--who, var(--muted)); font-size: 12px; font-weight: 600; }
#feed code { color: var(--muted); font-size: 11px; }
#gate { position: absolute; inset: 0; display: grid; place-items: center; padding: 16px;
  background: rgba(11, 15, 25, .9); }
#gate[hidden] { display: none; }
#gate div { display: grid; gap: 10px; width: min(360px, 100%); padding: 18px;
  border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
#gate h2 { margin: 0; font-size: 15px; }
#gate p { margin: 0; color: var(--muted); }
#gate input, #gate button { width: 100%; padding: 8px 10px; border: 1px solid var(--line);
  border-radius: 6px; font: inherit; color: var(--text); background: var(--bg); }
#gate button { background: var(--accent); border-color: var(--accent); color: #fff;
  font-weight: 600; cursor: pointer; }
@media (max-width: 760px) {
  #app { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) 34%; }
  aside { border-left: 0; border-top: 1px solid var(--line); }
}
"""

_MARKUP: Final = """
<div id="app">
  <main id="stage">
    <canvas id="view" tabindex="0" aria-label="Live view of the agents and people in this session">
    </canvas>
    <div id="status" data-state="idle" role="status"><span id="dot"></span>
      <span id="state">Starting</span></div>
    <form id="gate" hidden><div>
      <h2>Open this session</h2>
      <p id="reason">Enter an access token to watch this session.</p>
      <input id="token" type="password" autocomplete="off" spellcheck="false"
        aria-label="Access token" placeholder="Access token" required>
      <button type="submit">Connect</button>
    </div></form>
  </main>
  <aside aria-label="Agent thoughts">
    <h1>Thoughts</h1>
    <ol id="feed" role="log" aria-live="polite"></ol>
  </aside>
</div>
"""

_SCRIPT: Final = r"""
"use strict";
(() => {
  const BUFFER_MS = @@CLIENT_BUFFER_MS@@, COAST_MS = @@EXTRAPOLATION_LIMIT_MS@@;
  const RESET_MS = @@MOTION_RESET_MS@@, HEARTBEAT_MS = @@HEARTBEAT_MS@@;
  const PRESENCE_MS = @@PRESENCE_INTERVAL_MS@@, CONVERGE_MS = @@CONVERGENCE_MS@@, TRAIL_MS = 700;
  const FEED_LIMIT = 200;
  const TOKEN_SHAPE = /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/;
  const SESSION_ENDED = 1000, GOING_AWAY = 1001, POLICY = 1008, NOT_FOUND = 4404;
  const $ = (id) => document.getElementById(id);
  const canvas = $("view"), ctx = canvas.getContext("2d", { alpha: false });
  const calm = matchMedia("(prefers-reduced-motion: reduce)");

  const path = location.pathname.replace(/\/+$/, "");
  const cut = path.lastIndexOf("/canvas/");
  const base = cut < 0 ? "" : path.slice(0, cut);
  const sessionId = path.slice(path.lastIndexOf("/") + 1);

  let token = null, socket = null, retryTimer = 0, heartbeat = 0, attempts = 0, opened = false;
  let me = null, offset = null, scene = null, scrollY = 0, lastPresence = 0;
  let lastFeedSeq = -1, width = 1, height = 1, dpr = 1, badges = [], selected = null;
  const tracks = new Map();

  function takeToken() {
    const found = new URLSearchParams(location.hash.slice(1)).get("token");
    if (location.hash) history.replaceState(null, "", location.pathname + location.search);
    return found && TOKEN_SHAPE.test(found) ? found : null;
  }

  function status(state, text) {
    $("status").dataset.state = state;
    $("state").textContent = text;
  }

  function askForToken(reason) {
    token = null;
    $("reason").textContent = reason;
    $("token").value = "";
    $("gate").hidden = false;
    status("error", "Not connected");
    $("token").focus();
  }

  function connect() {
    clearTimeout(retryTimer);
    opened = false;
    status("idle", attempts ? "Reconnecting" : "Connecting");
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    // A reconnect asks for exactly what it missed, so the feed has no holes or repeats.
    const resume = lastFeedSeq >= 0 ? `&last_seen_thought_seq=${lastFeedSeq}` : "";
    const url = `${scheme}//${location.host}${base}/ws/sessions/${sessionId}?view=canvas${resume}`;
    socket = new WebSocket(url, ["@@SUBPROTOCOL@@", `@@TOKEN_PREFIX@@${token}`]);
    socket.onopen = () => {
      opened = true;
      attempts = 0;
      heartbeat = setInterval(() => send({ type: "ping" }), HEARTBEAT_MS);
    };
    socket.onmessage = (event) => receive(JSON.parse(event.data));
    socket.onclose = closed;
  }

  function send(message) {
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  }

  function closed(event) {
    clearInterval(heartbeat);
    me = null;
    offset = null;
    tracks.clear();
    badges = [];
    if (event.code === NOT_FOUND) return status("error", "This session does not exist");
    if (event.code === SESSION_ENDED) return status("error", "This session has ended");
    if (event.code === POLICY && event.reason === "Token expired") {
      return askForToken("Your access token has expired. Enter a new one to keep watching.");
    }
    // A refused handshake looks like any network failure to a page. It never reaches onopen.
    if (!opened && attempts >= 1 && event.code !== GOING_AWAY) {
      return askForToken("Could not connect. The token may be invalid or expired.");
    }
    attempts += 1;
    const delay = Math.min(30000, 500 * 2 ** attempts) * (0.75 + Math.random() / 2);
    status("error", "Connection lost, retrying");
    retryTimer = setTimeout(connect, delay);
  }

  function receive(message) {
    if (message.kind === "canvas.hello") {
      me = message;
      $("gate").hidden = true;
      status("live", message.is_owner ? "Live, owner" : "Live, viewing");
      if (message.thought_gap) note("Some thoughts were missed while disconnected.");
    } else if (message.kind === "canvas.thought") {
      addThought(message);
    } else if (message.kind === "canvas") {
      takeFrame(message);
    }
  }

  function takeFrame(frame) {
    const seen = performance.now() - frame.t_ms;
    // The lowest offset is the least delayed frame. It drifts up slowly, to follow clock drift.
    offset = offset === null || seen < offset ? seen : offset + (seen - offset) * 0.01;
    if (frame.scene) takeScene(frame.scene);
    badges = frame.badges;
    const live = new Set();
    for (const agent of frame.agents) live.add(update(agent, frame.t_ms, false));
    for (const human of frame.humans) {
      if (me && human.id === me.agent_id) continue;
      live.add(update(human, frame.t_ms, true));
    }
    for (const id of tracks.keys()) if (!live.has(id)) tracks.delete(id);
  }

  function update(sprite, t, isHuman) {
    let track = tracks.get(sprite.id);
    if (!track) {
      track = { knots: [], isHuman, coast: null, fixX: 0, fixY: 0, fixAt: -Infinity };
      tracks.set(sprite.id, track);
    }
    track.sprite = sprite;
    const knots = track.knots;
    const last = knots[knots.length - 1];
    if (last && t - last.t > RESET_MS) knots.length = 0;
    if (!last || t > last.t) {
      knots.push({ t, x: sprite.x, y: sprite.y, vx: sprite.vx / 1000, vy: sprite.vy / 1000 });
    }
    while (knots.length > 2 && t - knots[1].t > TRAIL_MS) knots.shift();
    return sprite.id;
  }

  function takeScene(next) {
    scene = next;
    clampScroll();
  }

  function appendToFeed(item) {
    const feed = $("feed");
    const pinned = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
    feed.append(item);
    while (feed.children.length > FEED_LIMIT) feed.firstElementChild.remove();
    if (pinned) feed.scrollTop = feed.scrollHeight;
  }

  function note(text) {
    const item = document.createElement("li");
    item.textContent = text;
    appendToFeed(item);
  }

  function addThought(entry) {
    if (entry.seq <= lastFeedSeq) return;
    lastFeedSeq = entry.seq;
    const item = document.createElement("li");
    item.style.setProperty("--who", entry.color);
    const who = document.createElement("b");
    who.textContent = entry.label;
    item.append(who, entry.text);
    if (entry.tool_name) {
      const tool = document.createElement("code");
      tool.textContent = ` ${entry.tool_name}`;
      item.append(tool);
    }
    appendToFeed(item);
  }

  function coasted(knot, t) {
    const coast = Math.min(t - knot.t, COAST_MS);
    const travel = coast - (coast * coast) / (2 * COAST_MS);
    return { x: knot.x + knot.vx * travel, y: knot.y + knot.vy * travel };
  }

  // The same spline as the server's: Bézier control points from Hermite tangents.
  function positionAt(knots, t) {
    const first = knots[0], last = knots[knots.length - 1];
    if (t <= first.t) return first;
    if (t >= last.t) return coasted(last, t);
    let index = knots.length - 1;
    while (t < knots[index - 1].t) index -= 1;
    const a = knots[index - 1], b = knots[index], span = b.t - a.t;
    const u = (t - a.t) / span, v = 1 - u;
    const p = v * v * v, q = 3 * v * v * u, r = 3 * v * u * u, s = u * u * u;
    return {
      x: p * a.x + q * (a.x + (a.vx * span) / 3) + r * (b.x - (b.vx * span) / 3) + s * b.x,
      y: p * a.y + q * (a.y + (a.vy * span) / 3) + r * (b.y - (b.vy * span) / 3) + s * b.y,
    };
  }

  function remainingFix(track, t) {
    const u = Math.min(Math.max((t - track.fixAt) / CONVERGE_MS, 0), 1);
    return 1 - u * u * (3 - 2 * u);
  }

  // Ends a coast by easing out the error it built up, rather than jumping back onto the spline.
  function settledAt(track, t) {
    const point = positionAt(track.knots, t);
    const last = track.knots[track.knots.length - 1];
    if (track.coast && track.coast.t !== last.t) {
      const guess = coasted(track.coast, t), left = remainingFix(track, t);
      track.fixX = guess.x + track.fixX * left - point.x;
      track.fixY = guess.y + track.fixY * left - point.y;
      track.fixAt = t;
    }
    track.coast = t >= last.t ? last : null;
    const left = remainingFix(track, t);
    return { x: point.x + track.fixX * left, y: point.y + track.fixY * left };
  }

  function worldScale() {
    return scene ? width / scene.viewport.width : 1;
  }

  function contentHeight() {
    return scene ? scene.viewport.height * worldScale() : height;
  }

  function clampScroll() {
    scrollY = Math.min(Math.max(scrollY, 0), Math.max(contentHeight() - height, 0));
  }

  function resize() {
    const box = canvas.getBoundingClientRect();
    width = Math.max(box.width, 1);
    height = Math.max(box.height, 1);
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    clampScroll();
  }

  const sx = (x) => x * worldScale();
  const sy = (y) => y * worldScale() - scrollY;

  function overlayRect(overlay) {
    const { width: w, height: h } = scene.viewport;
    return {
      x: sx(overlay.box.x * w),
      y: sy(overlay.box.y * h),
      w: overlay.box.width * w * worldScale(),
      h: overlay.box.height * h * worldScale(),
    };
  }

  function pill(text, x, y, color, alpha) {
    ctx.font = "600 11px system-ui, sans-serif";
    const w = Math.min(ctx.measureText(text).width, 280) + 12;
    const left = Math.min(Math.max(x, 2), Math.max(width - w - 2, 2));
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(left, y - 9, w, 18, 9);
    ctx.fill();
    ctx.fillStyle = "#0b0f19";
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + 6, y + 0.5, 280);
    ctx.globalAlpha = 1;
  }

  function drawOverlay(overlay, now) {
    const rect = overlayRect(overlay);
    if (rect.h <= 0 || rect.y > height || rect.y + rect.h < 0) return;
    const owner = overlay.owner;
    const leader = owner || overlay.waiters[0];
    const lead = leader ? leader.color : "#475569";
    ctx.fillStyle = "#111827";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.save();
    if (overlay.state === "FREE") {
      ctx.strokeStyle = overlay.id === selected ? "#e5e7eb" : "#334155";
      ctx.lineWidth = 1;
    } else {
      const pulse = overlay.state === "WAITING" && !calm.matches
        ? 0.55 + 0.45 * Math.sin(now / 180) : 1;
      ctx.strokeStyle = lead;
      ctx.lineWidth = 2;
      ctx.globalAlpha = pulse;
      ctx.shadowColor = lead;
      ctx.shadowBlur = overlay.state === "WAITING" ? 18 * pulse : 12;
      if (overlay.state === "WAITING") ctx.setLineDash([8, 6]);
    }
    ctx.strokeRect(rect.x + 1, rect.y + 1, Math.max(rect.w - 2, 0), Math.max(rect.h - 2, 0));
    ctx.restore();
    ctx.fillStyle = "#64748b";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText(overlay.title, rect.x + 10, rect.y + 8, Math.max(rect.w - 20, 1));
    if (overlay.state !== "FREE") {
      const queue = overlay.waiters.length ? ` +${overlay.waiters.length} waiting` : "";
      const label = owner ? `${owner.label}${queue}` : `${overlay.waiters.length} waiting`;
      pill(label, rect.x + 8, rect.y + rect.h - 14, lead, 1);
    }
  }

  function drawTrail(track, at, color) {
    if (calm.matches) return;
    const knots = track.knots;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (let i = 1; i < knots.length; i += 1) {
      const a = knots[i - 1], b = knots[i];
      if (a.t >= at) break;
      const age = at - b.t;
      if (age > TRAIL_MS) continue;
      const span = b.t - a.t;
      ctx.globalAlpha = 0.45 * Math.max(1 - age / TRAIL_MS, 0);
      ctx.beginPath();
      ctx.moveTo(sx(a.x), sy(a.y));
      ctx.bezierCurveTo(
        sx(a.x + (a.vx * span) / 3), sy(a.y + (a.vy * span) / 3),
        sx(b.x - (b.vx * span) / 3), sy(b.y - (b.vy * span) / 3),
        sx(b.x), sy(b.y),
      );
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawAgent(track, point) {
    const agent = track.sprite, x = sx(point.x), y = sy(point.y);
    if (agent.dock) {
      ctx.strokeStyle = agent.color;
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(sx(agent.dock.x), sy(agent.dock.y));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    const speed = Math.hypot(agent.vx, agent.vy);
    if (speed > 40 && !calm.matches) {
      const reach = Math.min(speed * 0.04, 36);
      ctx.strokeStyle = agent.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (agent.vx / speed) * reach, y + (agent.vy / speed) * reach);
      ctx.stroke();
    }
    ctx.fillStyle = agent.color;
    ctx.strokeStyle = "#0b0f19";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 3.5, y + 15);
    ctx.lineTo(x + 7, y + 9.5);
    ctx.lineTo(x + 13, y + 9);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    pill(agent.label, x + 12, y + 22, agent.color, 1);
  }

  function drawHuman(track, point) {
    const human = track.sprite, x = sx(point.x), y = sy(point.y);
    ctx.strokeStyle = human.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = human.color;
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fill();
    pill(human.label, x + 11, y + 14, human.color, 1);
  }

  function drawBadge(badge) {
    const above = badge.side === "TOP" ? -14 : badge.side === "BOTTOM" ? 14 : 0;
    pill(badge.text, sx(badge.x), sy(badge.y) + above, badge.color, badge.opacity);
  }

  function frame(now) {
    requestAnimationFrame(frame);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0b0f19";
    ctx.fillRect(0, 0, width, height);
    if (!scene) return;
    for (const overlay of scene.overlays) drawOverlay(overlay, now);
    const at = offset === null ? 0 : now - offset - BUFFER_MS;
    for (const track of tracks.values()) {
      const point = settledAt(track, at);
      drawTrail(track, at, track.sprite.color);
      if (track.isHuman) drawHuman(track, point);
      else drawAgent(track, point);
    }
    for (const badge of badges) drawBadge(badge);
  }

  function toWorld(event) {
    const box = canvas.getBoundingClientRect(), scale = worldScale();
    return {
      x: Math.round((event.clientX - box.left) / scale),
      y: Math.round((event.clientY - box.top + scrollY) / scale),
    };
  }

  function hit(point) {
    if (!scene) return null;
    const { width: w, height: h } = scene.viewport;
    const found = scene.overlays.find((overlay) => {
      const x = point.x / w - overlay.box.x, y = point.y / h - overlay.box.y;
      return x >= 0 && y >= 0 && x <= overlay.box.width && y <= overlay.box.height;
    });
    return found ? found.id : null;
  }

  canvas.addEventListener("pointermove", (event) => {
    if (!me || !scene || event.timeStamp - lastPresence < PRESENCE_MS) return;
    lastPresence = event.timeStamp;
    send({ type: "cursor", ...toWorld(event) });
  });
  canvas.addEventListener("click", (event) => {
    if (!me) return;
    selected = hit(toWorld(event));
    send({ type: "select", target_id: selected });
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    scrollY += event.deltaY;
    clampScroll();
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    const steps = { ArrowDown: 60, ArrowUp: -60, PageDown: height, PageUp: -height };
    if (steps[event.key] === undefined) return;
    event.preventDefault();
    scrollY += steps[event.key];
    clampScroll();
  });
  $("gate").addEventListener("submit", (event) => {
    event.preventDefault();
    const entered = $("token").value.trim();
    if (!TOKEN_SHAPE.test(entered)) {
      $("reason").textContent = "That does not look like an access token.";
      return;
    }
    token = entered;
    attempts = 0;
    $("gate").hidden = true;
    connect();
  });
  new ResizeObserver(resize).observe(canvas);

  resize();
  requestAnimationFrame(frame);
  token = takeToken();
  if (token) connect();
  else askForToken("Enter an access token to watch this session.");
})();
"""


def _render_script() -> str:
    script = _SCRIPT
    for name, value in (
        ("CLIENT_BUFFER_MS", CLIENT_BUFFER_MS),
        ("EXTRAPOLATION_LIMIT_MS", EXTRAPOLATION_LIMIT_MS),
        ("MOTION_RESET_MS", MOTION_RESET_MS),
        ("CONVERGENCE_MS", CONVERGENCE_MS),
        ("HEARTBEAT_MS", HEARTBEAT_MS),
        ("PRESENCE_INTERVAL_MS", PRESENCE_INTERVAL_MS),
        ("SUBPROTOCOL", SUBPROTOCOL),
        ("TOKEN_PREFIX", TOKEN_SUBPROTOCOL_PREFIX),
    ):
        script = script.replace(f"@@{name}@@", str(value))
    return script


def _source_hash(source: str) -> str:
    digest = hashlib.sha256(source.encode()).digest()
    return f"'sha256-{base64.b64encode(digest).decode()}'"


_RENDERED_SCRIPT: Final = _render_script()
VIEWER_HTML: Final = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    '<meta name="referrer" content="no-referrer">\n'
    "<title>Swarm Canvas</title>\n"
    f"<style>{_STYLE}</style>\n</head>\n<body>{_MARKUP}<script>{_RENDERED_SCRIPT}</script>\n"
    "</body>\n</html>\n"
)
VIEWER_BYTES: Final = VIEWER_HTML.encode()
CONTENT_SECURITY_POLICY: Final = "; ".join(
    (
        "default-src 'none'",
        f"script-src {_source_hash(_RENDERED_SCRIPT)}",
        f"style-src {_source_hash(_STYLE)}",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)
VIEWER_HEADERS: Final[Mapping[str, str]] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}


def viewer_response() -> Response:
    return Response(
        VIEWER_BYTES, media_type="text/html; charset=utf-8", headers=dict(VIEWER_HEADERS)
    )
