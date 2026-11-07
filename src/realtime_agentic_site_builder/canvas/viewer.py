"""The self-contained canvas page, and the headers that keep it that way.

Authentication has one route through it. A browser cannot put a header on a page load, and a query
string ends up in access logs and ``Referer`` headers, so a link carries the token in its fragment,
which is never sent to any server. The page moves it into memory, erases it from the address bar and
presents it as a WebSocket subprotocol, or as a bearer token to the history, verify and fork routes.
The server is what enforces access, not this page, and the page holds nothing for anyone without a
token: it is the same bytes for every session and caller.

Nothing from the request is reflected, which is what makes a hash-based policy possible. Text from
the session reaches the page only through ``textContent`` and ``fillText``, which make no markup.

The page draws three layers. At the back, a DOM mirror of the site being built: a skeleton per
component, laid out at its real coordinates and kept current by applying each commit's RFC 6902
patches in place. It never runs the generated code, so nothing the agents write can execute here.
Over it, the transparent telemetry canvas. On top, the heads-up display, the timeline scrubber and,
for the session's owner, the override control.
"""

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Final

from starlette.responses import Response

from realtime_agentic_site_builder.agents.graph import BACKEND, CODER, CRITIC, PLANNER, STYLER
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
# The HUD measures round trips with a ping a second, only while it is open.
PROBE_INTERVAL_MS: Final = 1_000
LATENCY_SAMPLES: Final = 240
# The order of the radar's axes is the order a run visits them.
GRAPH_NODES: Final = (PLANNER, CODER, BACKEND, STYLER, CRITIC)
# A component that is locked or edited takes an impulse, then rings down on this spring. Per second.
SPRING_STIFFNESS: Final = 260.0
SPRING_DAMPING: Final = 16.0
# The halo's time constant: a flash falls to about a third of its brightness in this long.
GLOW_DECAY_MS: Final = 650
# How long the radar's ring travels outward after a critic's verdict.
RADAR_PULSE_MS: Final = 900
# Scrubbing rebuilds a frame from the nearest checkpoint, so it never replays the whole window.
HISTORY_CHECKPOINT_EVERY: Final = 16


def spring_settle_ms(tolerance: float = 0.02) -> float:
    """How long an impulse takes to ring down to ``tolerance`` of its first swing."""
    return 1000 * math.log(1 / tolerance) / (SPRING_DAMPING / 2)


_STYLE: Final = """
:root { color-scheme: dark; --bg: #0b0f19; --panel: #111827; --line: #1f2937;
  --text: #e5e7eb; --muted: #9ca3af; --accent: #8b5cf6; --bad: #f87171; --good: #10b981; }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text);
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
#app { display: grid; grid-template-columns: minmax(0, 1fr) 320px; height: 100dvh; }
#stage { position: relative; min-width: 0; min-height: 0; overflow: hidden; background: var(--bg); }
#view { position: absolute; inset: 0; display: block; width: 100%; height: 100%;
  touch-action: none; outline: none; }
#mirror { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
#stage[data-travel] #mirror { filter: sepia(.45) saturate(.8); }
#page { position: absolute; left: 0; top: 0; transform-origin: 0 0; background: var(--pg-bg);
  color: var(--pg-fg); font: 16px/1.35 system-ui, sans-serif;
  --pg-bg: #0f172a; --pg-fg: #e2e8f0; --pg-card: #1e293b; --pg-dim: #334155; --pg-hi: #6366f1; }
#page[data-theme="light"] { --pg-bg: #f8fafc; --pg-fg: #0f172a; --pg-card: #fff;
  --pg-dim: #cbd5e1; --pg-hi: #2563eb; }
#page[data-theme="cyberpunk"] { --pg-bg: #0a0014; --pg-fg: #f0abfc; --pg-card: #1a0533;
  --pg-dim: #4c1d95; --pg-hi: #22d3ee; }
#page[data-theme="minimal"] { --pg-bg: #fafafa; --pg-fg: #171717; --pg-card: #fff;
  --pg-dim: #d4d4d4; --pg-hi: #171717; }
#page[data-theme="enterprise"] { --pg-bg: #f1f5f9; --pg-fg: #0f172a; --pg-card: #fff;
  --pg-dim: #cbd5e1; --pg-hi: #0f766e; }
.c { position: absolute; display: flex; flex-direction: column; gap: 12px; padding: 20px 24px;
  overflow: hidden; border-radius: 12px; background: var(--pg-card); transform-origin: 50% 50%;
  box-shadow: 0 0 calc(var(--g, 0) * 34px) calc(var(--g, 0) * 6px) var(--halo, transparent),
    inset 0 0 0 calc(var(--g, 0) * 2px) var(--halo, transparent); }
.c h3 { margin: 0; font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }
.c small { position: absolute; top: 8px; right: 12px; font: 600 11px ui-monospace, monospace;
  opacity: .5; }
.c i { display: block; height: 11px; border-radius: 6px; background: var(--pg-dim); }
.c i:nth-of-type(2n) { width: 74%; }
.c i:nth-of-type(3n) { width: 52%; }
.c em { display: block; width: 132px; height: 36px; border-radius: 8px; background: var(--pg-hi); }
.c u { display: block; height: 32px; border: 1px solid var(--pg-dim); border-radius: 6px; }
.c b { display: block; width: 28px; height: 28px; border-radius: 50%; background: var(--pg-hi); }
.c strong { display: block; width: 46%; height: 26px; border-radius: 6px; background: var(--pg-fg);
  opacity: .8; }
.c nav { display: flex; gap: 18px; flex: 1; justify-content: flex-end; }
.c nav i { width: 64px; }
.c pre { display: grid; gap: 8px; margin: 0; }
.c .row { display: flex; gap: 16px; flex: 1; min-height: 0; }
.c .row > div { display: flex; flex: 1; flex-direction: column; gap: 9px; padding: 14px;
  border-radius: 10px; background: color-mix(in srgb, var(--pg-dim) 35%, transparent); }
.c-NAVBAR { flex-direction: row; align-items: center; }
.c-HERO { justify-content: center; }
.c-HERO h3 { font-size: 40px; }
.c-FOOTER { flex-direction: row; align-items: center; background: transparent; }
.c-CUSTOM_DYNAMIC pre i { height: 9px; }
#status { position: absolute; top: 12px; right: 12px; display: flex; align-items: center; gap: 8px;
  padding: 5px 10px; border: 1px solid var(--line); border-radius: 999px;
  background: rgba(17, 24, 39, .85); color: var(--muted); pointer-events: none; }
#dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
#status[data-state="live"] #dot { background: var(--good); }
#status[data-state="error"] #dot { background: var(--bad); }
#tools { position: absolute; top: 12px; left: 12px; display: flex; flex-wrap: wrap; gap: 6px;
  max-width: calc(100% - 210px); }
.tb, #hud button { padding: 5px 11px; border: 1px solid var(--line); border-radius: 999px;
  background: rgba(17, 24, 39, .9); color: var(--text); font: inherit; cursor: pointer; }
.tb[aria-pressed="true"] { border-color: var(--accent); background: rgba(76, 29, 149, .6); }
.tb:focus-visible, #hud button:focus-visible, #seek:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
kbd { margin-left: 4px; padding: 0 4px; border: 1px solid var(--line); border-radius: 4px;
  color: var(--muted); font: 600 10px ui-monospace, monospace; }
#hud { position: absolute; top: 50px; left: 12px; width: 264px; max-width: calc(100% - 24px);
  max-height: calc(100% - 128px); overflow-y: auto; padding: 12px 14px;
  border: 1px solid var(--line); border-radius: 12px; background: rgba(11, 15, 25, .9); }
#hud h2 { margin: 10px 0 4px; color: var(--muted); font-size: 10px; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; }
#hud h2:first-child { margin-top: 0; }
#hud p { margin: 2px 0 6px; color: var(--muted); }
#radar { display: block; width: 216px; height: 176px; }
#hud dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; margin: 0;
  font-variant-numeric: tabular-nums; }
#hud dt { color: var(--muted); }
#hud dd { margin: 0; text-align: right; white-space: nowrap; }
#hud button { margin-top: 8px; border-radius: 6px; }
#checks { margin: 6px 0 0; padding: 0; list-style: none; font-size: 12px; }
#checks li { color: var(--good); overflow-wrap: anywhere; }
#checks li[data-ok="false"] { color: var(--bad); }
#scrub { position: absolute; right: 12px; bottom: 12px; left: 12px; display: flex; flex-wrap: wrap;
  align-items: center; gap: 10px; padding: 8px 12px; border: 1px solid var(--line);
  border-radius: 12px; background: rgba(11, 15, 25, .92); }
#seek { flex: 1; min-width: 120px; accent-color: var(--accent); }
#when { min-width: 150px; color: var(--muted); font-variant-numeric: tabular-nums; }
#toasts { position: absolute; bottom: 76px; left: 50%; display: grid; gap: 6px;
  width: min(440px, calc(100% - 24px)); transform: translateX(-50%); pointer-events: none; }
#toasts p { margin: 0; padding: 8px 12px; border-left: 3px solid var(--who, var(--accent));
  border-radius: 8px; background: rgba(17, 24, 39, .96); }
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
  #tools { max-width: calc(100% - 24px); top: 48px; }
  #hud { top: 86px; width: 248px; }
}
"""

_MARKUP: Final = """
<div id="app">
  <main id="stage">
    <div id="mirror" role="region" aria-label="Live preview of the site being built">
      <div id="page"></div></div>
    <canvas id="view" tabindex="0" aria-label="Live view of the agents and people in this session">
    </canvas>
    <div id="tools" role="toolbar" aria-label="View controls">
      <button class="tb" id="hud-toggle" type="button" aria-pressed="false" aria-controls="hud"
        >HUD<kbd>H</kbd></button>
      <button class="tb" id="travel-toggle" type="button" aria-pressed="false"
        aria-controls="scrub">Timeline<kbd>T</kbd></button>
    </div>
    <div id="status" data-state="idle" role="status"><span id="dot"></span>
      <span id="state">Starting</span></div>
    <section id="hud" hidden aria-label="Swarm telemetry">
      <h2>Graph state</h2>
      <canvas id="radar" width="216" height="176"
        aria-label="How often each graph node has run, and which are running now"></canvas>
      <p id="verdicts">No critic verdicts yet</p>
      <h2>Burn</h2>
      <dl><dt>Tokens</dt><dd id="m-tokens">0</dd><dt>Spend</dt><dd id="m-cost">unpriced</dd></dl>
      <h2>Latency</h2>
      <dl><dt>Socket RTT p95/p99</dt><dd id="m-rtt">-</dd>
        <dt>Frame→paint p95/p99</dt><dd id="m-paint">-</dd></dl>
      <h2>Timeline</h2>
      <dl><dt>Mirror frame</dt><dd id="m-frame">-</dd><dt>Commits</dt><dd id="m-commits">0</dd></dl>
      <button id="verify" type="button">Verify invariants</button>
      <ul id="checks"></ul>
      <div id="owner" hidden>
        <h2>Human override</h2>
        <p id="target">Select an element on the canvas.</p>
        <button id="take" type="button">Override<kbd>O</kbd></button>
        <button id="give" type="button">Release<kbd>R</kbd></button>
      </div>
    </section>
    <div id="scrub" hidden>
      <button class="tb" id="live" type="button">Live<kbd>Esc</kbd></button>
      <input id="seek" type="range" min="0" max="0" value="0" step="1" aria-label="Frame">
      <span id="when" aria-live="polite"></span>
      <button class="tb" id="fork" type="button">Fork from here</button>
    </div>
    <div id="toasts" role="status" aria-live="polite"></div>
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
  const PROBE_MS = @@PROBE_INTERVAL_MS@@, SAMPLES = @@LATENCY_SAMPLES@@;
  const GRAPH_NODES = @@GRAPH_NODES@@, CHECKPOINT = @@HISTORY_CHECKPOINT_EVERY@@;
  const SPRING_K = @@SPRING_STIFFNESS@@, SPRING_D = @@SPRING_DAMPING@@;
  const GLOW_MS = @@GLOW_DECAY_MS@@;
  const FEED_LIMIT = 200, PULSE_MS = @@RADAR_PULSE_MS@@, TOAST_MS = 4500, TOAST_LIMIT = 4;
  const TOKEN_SHAPE = /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/;
  const SESSION_ENDED = 1000, GOING_AWAY = 1001, POLICY = 1008, NOT_FOUND = 4404;
  const $ = (id) => document.getElementById(id);
  const canvas = $("view"), ctx = canvas.getContext("2d");
  const radar = $("radar"), rctx = radar.getContext("2d");
  const calm = matchMedia("(prefers-reduced-motion: reduce)");

  const path = location.pathname.replace(/\/+$/, "");
  const cut = path.lastIndexOf("/canvas/");
  const base = cut < 0 ? "" : path.slice(0, cut);
  const sessionId = path.slice(path.lastIndexOf("/") + 1);

  let token = null, socket = null, retryTimer = 0, heartbeat = 0, attempts = 0, opened = false;
  let me = null, offset = null, scene = null, scrollY = 0, lastPresence = 0;
  let lastFeedSeq = -1, width = 1, height = 1, dpr = 1, badges = [], selected = null;
  const tracks = new Map();
  // The mirror: the site as of mirrorSeq, and what the page has drawn of it.
  let site = null, mirrorSeq = -1, epoch = 0, resyncing = false, queued = [];
  let commits = 0, resyncs = 0, travel = null, lastTick = 0, pageKey = "";
  const views = new Map(), people = new Map(), holders = new Map(), overlays = new Map();
  const hud = { visits: new Map(), active: new Set(), pulses: [], approved: 0, rejected: 0,
    tokens: 0, priced: 0, cost: 0 };
  const rtt = [], paint = [], probes = new Map(), asked = new Map();
  let probeSeq = 0, askSeq = 0, arrival = null;

  // --- RFC 6902, applied to the mirror exactly as the server applied it ------------------------
  // <patch>
  const UNSAFE = new Set(["__proto__", "constructor", "prototype"]);
  function keysOf(pointer) {
    if (pointer === "" || pointer[0] !== "/") throw new Error(`bad pointer ${pointer}`);
    return pointer.slice(1).split("/").map((part) => {
      const key = part.replace(/~1/g, "/").replace(/~0/g, "~");
      if (UNSAFE.has(key)) throw new Error(`unsafe key ${key}`);
      return key;
    });
  }
  function slot(list, key, appending) {
    if (appending && key === "-") return list.length;
    if (!/^(0|[1-9][0-9]*)$/.test(key)) throw new Error(`bad index ${key}`);
    const index = Number(key);
    if (index > list.length || (!appending && index === list.length)) {
      throw new Error(`index ${key} out of range`);
    }
    return index;
  }
  function isObject(node) {
    return node !== null && typeof node === "object" && !Array.isArray(node);
  }
  function child(node, key) {
    if (Array.isArray(node)) return node[slot(node, key, false)];
    if (isObject(node) && Object.hasOwn(node, key)) return node[key];
    throw new Error(`no member ${key}`);
  }
  function resolve(root, pointer) {
    let node = root;
    for (const key of keysOf(pointer)) node = child(node, key);
    return node;
  }
  function parentOf(root, pointer) {
    const keys = keysOf(pointer), last = keys.pop();
    let node = root;
    for (const key of keys) node = child(node, key);
    return [node, last];
  }
  function insert(root, pointer, value) {
    const [node, key] = parentOf(root, pointer);
    if (Array.isArray(node)) node.splice(slot(node, key, true), 0, value);
    else if (isObject(node)) node[key] = value;
    else throw new Error(`cannot add under ${pointer}`);
  }
  function detach(root, pointer) {
    const [node, key] = parentOf(root, pointer);
    if (Array.isArray(node)) return node.splice(slot(node, key, false), 1)[0];
    if (isObject(node) && Object.hasOwn(node, key)) {
      const value = node[key];
      delete node[key];
      return value;
    }
    throw new Error(`nothing at ${pointer}`);
  }
  function same(a, b) {
    if (Array.isArray(a) || Array.isArray(b)) {
      return Array.isArray(a) && Array.isArray(b) && a.length === b.length
        && a.every((item, index) => same(item, b[index]));
    }
    if (isObject(a) && isObject(b)) {
      const keys = Object.keys(a);
      return keys.length === Object.keys(b).length
        && keys.every((key) => Object.hasOwn(b, key) && same(a[key], b[key]));
    }
    return a === b;
  }
  function applyOperation(root, operation) {
    const { op, path: at } = operation;
    if (op === "add") insert(root, at, structuredClone(operation.value));
    else if (op === "remove") detach(root, at);
    else if (op === "replace") {
      detach(root, at);
      insert(root, at, structuredClone(operation.value));
    } else if (op === "move") insert(root, at, detach(root, operation.from));
    else if (op === "copy") insert(root, at, structuredClone(resolve(root, operation.from)));
    else if (op === "test") {
      if (!same(resolve(root, at), operation.value)) throw new Error(`test failed at ${at}`);
    } else throw new Error(`unknown op ${op}`);
  }
  // Whole or not at all: a delta that fails part way leaves the document as it was.
  function applyPatches(root, patches) {
    const next = structuredClone(root);
    for (const operation of patches) applyOperation(next, operation);
    return next;
  }
  // </patch>

  // --- the connection ---------------------------------------------------------------------------

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

  // The same token opens the REST routes. It travels in a header, never in a URL.
  async function api(method, suffix) {
    const response = await fetch(`${base}/api/v1/sessions/${sessionId}${suffix}`, {
      method,
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error((body && (body.detail || body.title)) || `HTTP ${response.status}`);
    }
    return body;
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
    const kind = message.kind;
    if (kind === "canvas") takeFrame(message);
    else if (kind === "canvas.thought") addThought(message);
    else if (kind === "canvas.delta") takeDelta(message);
    else if (kind === "canvas.override") overridden(message);
    else if (kind === "canvas.hello") hello(message);
    else if (kind === "pong") pong(message);
    else if (kind === "ack") answered(message, null);
    else if (kind === "error") answered(message, message.problem);
  }

  function hello(message) {
    me = message;
    $("gate").hidden = true;
    $("owner").hidden = !message.is_owner;
    status("live", message.is_owner ? "Live, owner" : "Live, viewing");
    if (message.thought_gap) note("Some thoughts were missed while disconnected.");
    epoch += 1;
    site = { frontend: message.frontend };
    mirrorSeq = message.frame_seq;
    resyncing = false;
    queued = [];
    render();
  }

  // --- the live mirror --------------------------------------------------------------------------

  function takeDelta(delta) {
    if (delta.frame_seq <= mirrorSeq) return;
    if (resyncing) {
      queued.push(delta);
      return;
    }
    if (delta.frame_seq !== mirrorSeq + 1) return resync(delta);
    try {
      // The server's corrections carry what the page cannot derive: stamped versions and
      // normalized text. With them, the mirror equals the committed layout value for value.
      site = applyPatches(site, [...delta.patches, ...delta.corrections]);
    } catch (error) {
      return resync(delta);
    }
    mirrorSeq = delta.frame_seq;
    commits += 1;
    if (travel) return;
    render();
    for (const id of touched(delta)) kick(id, delta.agent_id, delta.color, 1);
  }

  // A gap means the mirror missed a commit. It rebuilds from the server's window, then catches up.
  async function resync(delta) {
    const current = epoch;
    resyncing = true;
    queued = [delta];
    resyncs += 1;
    try {
      const timeline = timelineOf(await api("GET", "/history"));
      if (current !== epoch) return;
      const last = timeline.frames.length - 1;
      site = stateAt(timeline, last);
      mirrorSeq = timeline.frames[last].seq;
    } catch (error) {
      if (current === epoch) toast(`The preview fell behind: ${error.message}`);
      return;
    } finally {
      if (current === epoch) resyncing = false;
    }
    const waiting = queued.sort((a, b) => a.frame_seq - b.frame_seq);
    queued = [];
    if (!travel) render();
    for (const next of waiting) takeDelta(next);
  }

  // What a delta changed: its target if it is a component, else every component its paths name.
  function touched(delta) {
    if (views.has(delta.target_id)) return [delta.target_id];
    const found = new Set(), components = site.frontend.components;
    for (const operation of delta.patches) {
      const match = /^\/frontend\/components\/(\d+)/.exec(operation.path);
      const component = match && components[Number(match[1])];
      if (component) found.add(component.id);
    }
    return found.size ? [...found] : [...views.keys()];
  }

  function shown() {
    return travel ? travel.state : site;
  }

  function el(tag, parent, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }
  function bars(parent, count) {
    for (let i = 0; i < count; i += 1) el("i", parent);
  }
  function cards(parent, count, fill) {
    const row = el("div", parent);
    row.className = "row";
    for (let i = 0; i < count; i += 1) fill(el("div", row));
  }
  // Skeletons, not the generated code: the shape of each kind of section, and its real title.
  const SKELETONS = {
    NAVBAR: (node) => { bars(el("nav", node), 3); el("em", node); },
    HERO: (node) => { bars(node, 2); el("em", node); },
    FEATURES: (node) => cards(node, 3, (card) => { el("b", card); bars(card, 2); }),
    PRICING: (node) => cards(node, 3, (card) => {
      el("strong", card);
      bars(card, 3);
      el("em", card);
    }),
    TESTIMONIALS: (node) => cards(node, 2, (card) => { bars(card, 3); el("b", card); }),
    CONTACT_FORM: (node) => { for (let i = 0; i < 3; i += 1) el("u", node); el("em", node); },
    FOOTER: (node) => bars(el("nav", node), 4),
    CUSTOM_DYNAMIC: (node) => bars(el("pre", node), 5),
  };

  function build(component) {
    const type = component.component_type;
    const node = el("section");
    node.className = `c c-${type}`;
    const view = { node, type, title: el("h3", node), tag: el("small", node), box: "",
      s: 1, v: 0, x: 0, y: 0, vx: 0, vy: 0, glow: 0, halo: "", resting: true,
      transform: "", level: "", color: "" };
    (Object.hasOwn(SKELETONS, type) ? SKELETONS[type] : SKELETONS.CUSTOM_DYNAMIC)(node);
    $("page").append(node);
    return view;
  }

  // Changes only what differs, so a commit costs a few DOM writes rather than a reload.
  function render() {
    const state = shown();
    if (!state) return;
    const page = $("page"), frontend = state.frontend, keep = new Set();
    if (page.dataset.theme !== frontend.theme) page.dataset.theme = frontend.theme;
    for (const component of frontend.components) {
      keep.add(component.id);
      let view = views.get(component.id);
      if (!view || view.type !== component.component_type) {
        if (view) view.node.remove();
        view = build(component);
        views.set(component.id, view);
      }
      if (view.title.textContent !== component.title) view.title.textContent = component.title;
      const version = `${component.component_type.toLowerCase()} v${component.ast_version}`;
      if (view.tag.textContent !== version) view.tag.textContent = version;
      const { x, y, width: w, height: h } = component.box, box = `${x},${y},${w},${h}`;
      if (view.box !== box) {
        Object.assign(view.node.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`,
          height: `${h}px` });
        view.box = box;
      }
    }
    for (const [id, view] of views) {
      if (!keep.has(id)) {
        view.node.remove();
        views.delete(id);
      }
    }
    pageKey = "";
  }

  function placePage() {
    const state = shown();
    if (!state) return;
    let w = scene ? scene.viewport.width : 1, h = scene ? scene.viewport.height : 1;
    for (const { box } of state.frontend.components) {
      w = Math.max(w, box.x + box.width);
      h = Math.max(h, box.y + box.height);
    }
    const key = `${w},${h},${worldScale()},${scrollY}`;
    if (key === pageKey) return;
    pageKey = key;
    Object.assign($("page").style, { width: `${w}px`, height: `${h}px`,
      transform: `translate3d(0, ${-scrollY}px, 0) scale(${worldScale()})` });
  }

  // The impulse grows with the speed of the cursor that delivered it, and pushes along its path.
  function kick(id, agentId, color, strength) {
    const view = views.get(id);
    if (!view) return;
    view.glow = Math.max(view.glow, strength);
    view.halo = color;
    if (calm.matches) return;
    const track = tracks.get(agentId), sprite = track ? track.sprite : null;
    const vx = sprite ? sprite.vx : 0, vy = sprite ? sprite.vy : 0, speed = Math.hypot(vx, vy);
    view.v += strength * (0.35 + Math.min(speed / 800, 0.65));
    if (speed > 1) {
      const push = (strength * 110 * Math.min(speed, 1200)) / 1200;
      view.vx += (vx / speed) * push;
      view.vy += (vy / speed) * push;
    }
    view.resting = false;
  }

  // What a component glows with when nothing has just happened to it: whoever holds it.
  function heldBy(id, now) {
    if (travel) {
      const lock = travel.frames[travel.index].locks.find(
        (item) => item.target_id === id && item.state === "ACQUIRED");
      return lock ? [0.35, colorOf(lock.locked_by_agent_id)] : [0, ""];
    }
    const overlay = overlays.get(id);
    if (!overlay || overlay.state === "FREE") return [0, ""];
    const lead = overlay.owner || overlay.waiters[0];
    if (overlay.state === "WAITING" && !calm.matches) {
      return [0.22 + 0.16 * Math.sin(now / 180), lead.color];
    }
    return [0.35, lead.color];
  }

  // Semi-implicit Euler: stable at these rates, and a frame that comes late is clamped.
  function step(view, dt) {
    view.v += (-SPRING_K * (view.s - 1) - SPRING_D * view.v) * dt;
    view.s += view.v * dt;
    view.vx += (-SPRING_K * view.x - SPRING_D * view.vx) * dt;
    view.x += view.vx * dt;
    view.vy += (-SPRING_K * view.y - SPRING_D * view.vy) * dt;
    view.y += view.vy * dt;
    const motion = Math.abs(view.s - 1) + Math.abs(view.v) + (Math.abs(view.x) + Math.abs(view.y)
      + (Math.abs(view.vx) + Math.abs(view.vy)) / 50) / 100;
    if (motion < 1e-3) {
      Object.assign(view, { s: 1, v: 0, x: 0, y: 0, vx: 0, vy: 0, resting: true });
    }
  }

  function animate(now) {
    const dt = Math.min(now - lastTick, 48) / 1000;
    lastTick = now;
    for (const [id, view] of views) {
      if (!view.resting) step(view, dt);
      view.glow *= Math.exp((-dt * 1000) / GLOW_MS);
      const [rest, restColor] = heldBy(id, now);
      const flashing = view.glow > rest;
      const level = Math.max(view.glow, rest).toFixed(2);
      const color = (flashing ? view.halo : restColor) || view.halo || "transparent";
      const transform = view.resting ? ""
        : `translate(${view.x.toFixed(2)}px, ${view.y.toFixed(2)}px) scale(${view.s.toFixed(4)})`;
      if (transform !== view.transform) view.node.style.transform = view.transform = transform;
      if (level !== view.level) view.node.style.setProperty("--g", view.level = level);
      if (color !== view.color) view.node.style.setProperty("--halo", view.color = color);
    }
  }

  // --- time travel ------------------------------------------------------------------------------

  function timelineOf(history) {
    const frames = [{ seq: history.base_frame_seq, t: null, locks: history.base_locks,
      delta: null }];
    for (const entry of history.entries) {
      const patches = entry.delta.target_domain === "frontend" ? entry.delta.patches : [];
      frames.push({ seq: entry.frame_seq, t: entry.timestamp_ms, locks: entry.locks,
        delta: entry.delta, patches: [...patches, ...entry.corrections] });
    }
    return { frames, checkpoints: [{ frontend: history.base_frontend }], index: 0, state: null };
  }

  function advance(state, frame) {
    return frame.patches.length ? applyPatches(state, frame.patches) : state;
  }

  function stateAt(timeline, index) {
    const block = Math.floor(index / CHECKPOINT), saved = timeline.checkpoints;
    while (saved.length <= block) {
      let state = saved[saved.length - 1];
      const start = (saved.length - 1) * CHECKPOINT;
      for (let i = start + 1; i <= start + CHECKPOINT; i += 1) {
        state = advance(state, timeline.frames[i]);
      }
      saved.push(state);
    }
    let state = saved[block];
    for (let i = block * CHECKPOINT + 1; i <= index; i += 1) {
      state = advance(state, timeline.frames[i]);
    }
    return state;
  }

  async function openTravel() {
    if (!token) return;
    try {
      const timeline = timelineOf(await api("GET", "/history"));
      const seek = $("seek"), last = timeline.frames.length - 1;
      seek.max = String(last);
      seek.value = String(last);
      travel = timeline;
      $("scrub").hidden = false;
      $("travel-toggle").setAttribute("aria-pressed", "true");
      $("stage").dataset.travel = "";
      scrubTo(last);
      seek.focus();
    } catch (error) {
      toast(`The timeline is unavailable: ${error.message}`);
    }
  }

  function closeTravel() {
    travel = null;
    $("scrub").hidden = true;
    $("travel-toggle").setAttribute("aria-pressed", "false");
    delete $("stage").dataset.travel;
    render();
  }

  function scrubTo(index) {
    const before = travel.index;
    try {
      travel.state = stateAt(travel, index);
    } catch (error) {
      toast(`This frame cannot be rebuilt: ${error.message}`);
      return closeTravel();
    }
    travel.index = index;
    const frame = travel.frames[index], last = travel.frames[travel.frames.length - 1];
    const when = frame.t === null ? "oldest held"
      : `${((frame.t - last.t) / 1000).toFixed(1)} s`;
    const held = frame.locks.filter((lock) => lock.state === "ACQUIRED").length;
    $("when").textContent = `frame ${frame.seq} · ${when} · ${held} held`;
    render();
    if (frame.delta && index === before + 1) {
      kick(frame.delta.target_id, frame.delta.agent_id, colorOf(frame.delta.agent_id), 0.8);
    }
  }

  // A fork belongs to whoever made it, so the same token opens it.
  async function fork() {
    if (!travel) return;
    const seq = travel.frames[travel.index].seq;
    try {
      const forked = await api("POST", `/fork?from_seq=${seq}`);
      toast(`Forked at frame ${seq}. Opening the new session.`);
      location.assign(`${base}/canvas/${encodeURIComponent(forked.session_id)}#token=${token}`);
    } catch (error) {
      toast(`The fork was refused: ${error.message}`);
    }
  }

  // --- the heads-up display ---------------------------------------------------------------------

  function colorOf(agentId) {
    const known = people.get(agentId);
    return known ? known.color : "#94a3b8";
  }

  function record(list, value) {
    list.push(Math.max(value, 0));
    if (list.length > SAMPLES) list.shift();
  }

  // Nearest rank, as the server's own percentiles are.
  function percentile(list, rank) {
    const sorted = [...list].sort((a, b) => a - b);
    return sorted[Math.max(Math.ceil((rank / 100) * sorted.length) - 1, 0)];
  }

  function spread(list) {
    if (!list.length) return "-";
    return `${percentile(list, 95).toFixed(1)} / ${percentile(list, 99).toFixed(1)} ms`;
  }

  function probe() {
    if ($("hud").hidden || !me) return;
    const id = `rtt-${(probeSeq += 1)}`;
    probes.set(id, performance.now());
    if (probes.size > 32) probes.delete(probes.keys().next().value);
    send({ type: "ping", request_id: id });
  }

  function pong(message) {
    const sent = probes.get(message.request_id);
    if (sent === undefined) return;
    probes.delete(message.request_id);
    record(rtt, performance.now() - sent);
  }

  function updateHud() {
    if ($("hud").hidden) return;
    $("m-tokens").textContent = hud.tokens.toLocaleString("en-US");
    const share = hud.tokens ? Math.round((100 * hud.priced) / hud.tokens) : 0;
    $("m-cost").textContent = hud.priced
      ? `$${hud.cost.toFixed(4)}${share < 100 ? ` (${share}% priced)` : ""}` : "unpriced";
    $("m-rtt").textContent = spread(rtt);
    $("m-paint").textContent = spread(paint);
    $("m-frame").textContent = mirrorSeq < 0 ? "-" : String(mirrorSeq);
    $("m-commits").textContent = resyncs ? `${commits}, ${resyncs} resynced` : String(commits);
    if (hud.approved + hud.rejected) {
      $("verdicts").textContent = `Critic: ${hud.rejected} rejected, ${hud.approved} approved`;
    }
  }

  function toggleHud() {
    const hidden = !$("hud").hidden;
    $("hud").hidden = hidden;
    $("hud-toggle").setAttribute("aria-pressed", String(!hidden));
    updateHud();
  }

  function corner(index, radius, cx, cy) {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / GRAPH_NODES.length;
    return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius];
  }

  function drawRadar(now) {
    const w = 216, h = 176, cx = w / 2, cy = h / 2 + 4, r = 58, n = GRAPH_NODES.length;
    if (radar.width !== w * dpr) {
      radar.width = w * dpr;
      radar.height = h * dpr;
    }
    rctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    rctx.clearRect(0, 0, w, h);
    rctx.strokeStyle = "#1f2937";
    rctx.lineWidth = 1;
    for (const ring of [1 / 3, 2 / 3, 1]) {
      rctx.beginPath();
      for (let i = 0; i <= n; i += 1) rctx.lineTo(...corner(i % n, r * ring, cx, cy));
      rctx.stroke();
    }
    for (let i = 0; i < n; i += 1) {
      rctx.beginPath();
      rctx.moveTo(cx, cy);
      rctx.lineTo(...corner(i, r, cx, cy));
      rctx.stroke();
    }
    // Each verdict sends a ring out from the center: red for a rejection, green for approval.
    hud.pulses = hud.pulses.filter((pulse) => now - pulse.at < PULSE_MS);
    for (const pulse of hud.pulses) {
      const age = (now - pulse.at) / PULSE_MS;
      rctx.globalAlpha = 1 - age;
      rctx.strokeStyle = pulse.ok ? "#10b981" : "#f87171";
      rctx.lineWidth = pulse.ok ? 2 : 2 + 3 * (1 - age);
      rctx.beginPath();
      // With reduced motion the ring fades where it is instead of travelling outward.
      rctx.arc(cx, cy, r * (calm.matches ? 1 : 0.15 + age), 0, Math.PI * 2);
      rctx.stroke();
    }
    rctx.globalAlpha = 1;
    const most = Math.max(1, ...GRAPH_NODES.map((node) => hud.visits.get(node) || 0));
    rctx.beginPath();
    GRAPH_NODES.forEach((node, i) => {
      rctx.lineTo(...corner(i, r * Math.max((hud.visits.get(node) || 0) / most, 0.06), cx, cy));
    });
    rctx.closePath();
    rctx.fillStyle = "rgba(139, 92, 246, .22)";
    rctx.fill();
    rctx.strokeStyle = "#8b5cf6";
    rctx.lineWidth = 1.5;
    rctx.stroke();
    rctx.font = "600 10px system-ui, sans-serif";
    rctx.textAlign = "center";
    rctx.textBaseline = "middle";
    GRAPH_NODES.forEach((node, i) => {
      const running = hud.active.has(node);
      if (running) {
        rctx.globalAlpha = calm.matches ? 1 : 0.6 + 0.4 * Math.sin(now / 140);
        rctx.fillStyle = "#c4b5fd";
        rctx.beginPath();
        rctx.arc(...corner(i, r, cx, cy), 5, 0, Math.PI * 2);
        rctx.fill();
        rctx.globalAlpha = 1;
      }
      rctx.fillStyle = running ? "#e5e7eb" : "#9ca3af";
      rctx.fillText(node, ...corner(i, r + 15, cx, cy));
    });
  }

  async function verify() {
    const list = $("checks");
    list.replaceChildren();
    try {
      const report = await api("GET", "/verify");
      for (const result of report.invariants) {
        const mark = result.holds ? "✓" : "✗";
        el("li", list, `${mark} ${result.name} (${result.checked})`).dataset.ok = result.holds;
        for (const violation of result.violations) {
          el("li", list, violation).dataset.ok = "false";
        }
      }
      toast(report.holds
        ? `Every invariant holds over frames ${report.first_frame} to ${report.last_frame}.`
        : "An invariant is violated. The HUD lists where.", report.holds ? "#10b981" : "#f87171");
    } catch (error) {
      toast(`Verification failed: ${error.message}`, "#f87171");
    }
  }

  // --- the human override -----------------------------------------------------------------------

  function titleOf(id) {
    const state = site && site.frontend.components.find((component) => component.id === id);
    return state ? state.title : id;
  }

  function ask(label) {
    const id = `cmd-${(askSeq += 1)}`;
    asked.set(id, label);
    return id;
  }

  function chosen() {
    if (!me || !me.is_owner) return null;
    if (!selected) toast("Select an element on the canvas first.");
    return selected;
  }

  function takeOver() {
    const target = chosen();
    if (target) {
      const request = ask(`Override of ${titleOf(target)}`);
      send({ type: "lock.override", target_id: target, request_id: request });
    }
  }

  function giveBack() {
    const target = chosen();
    if (target) {
      const request = ask(`Release of ${titleOf(target)}`);
      send({ type: "lock.release", target_id: target, request_id: request });
    }
  }

  function answered(message, problem) {
    const label = asked.get(message.request_id);
    if (label === undefined) return;
    asked.delete(message.request_id);
    if (problem) return toast(`${label} refused: ${problem.detail || problem.title}`, "#f87171");
    const lock = message.lock;
    toast(lock && lock.state !== "ACQUIRED" ? `${label}: waiting in line` : `${label}: done`);
  }

  function overridden(message) {
    toast(`${message.by.label} took over ${titleOf(message.target_id)}`, message.by.color);
    kick(message.target_id, message.by.id, message.by.color, 1.6);
  }

  function toast(text, color) {
    const shelf = $("toasts"), item = el("p", shelf, text);
    if (color) item.style.setProperty("--who", color);
    setTimeout(() => item.remove(), TOAST_MS);
    while (shelf.children.length > TOAST_LIMIT) shelf.firstElementChild.remove();
  }

  function showSelection() {
    $("target").textContent = selected ? `Selected: ${titleOf(selected)}`
      : "Select an element on the canvas.";
  }

  // --- the telemetry layer ----------------------------------------------------------------------

  function takeFrame(frame) {
    if (arrival === null) arrival = performance.now();
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
    people.set(sprite.id, sprite);
    const knots = track.knots;
    const last = knots[knots.length - 1];
    if (last && t - last.t > RESET_MS) knots.length = 0;
    if (!last || t > last.t) {
      knots.push({ t, x: sprite.x, y: sprite.y, vx: sprite.vx / 1000, vy: sprite.vy / 1000 });
    }
    while (knots.length > 2 && t - knots[1].t > TRAIL_MS) knots.shift();
    return sprite.id;
  }

  // A new holder is an event worth seeing: the element takes an impulse from whoever took it.
  function takeScene(next) {
    scene = next;
    overlays.clear();
    for (const overlay of next.overlays) {
      overlays.set(overlay.id, overlay);
      const owner = overlay.owner;
      if (owner) people.set(owner.id, owner);
      if (owner && holders.get(overlay.id) !== owner.id) {
        kick(overlay.id, owner.id, owner.color, 0.8);
      }
      holders.set(overlay.id, owner ? owner.id : null);
    }
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
    appendToFeed(el("li", null, text));
  }

  function addThought(entry) {
    if (entry.seq <= lastFeedSeq) return;
    lastFeedSeq = entry.seq;
    tally(entry);
    const item = el("li");
    item.style.setProperty("--who", entry.color);
    el("b", item, entry.label);
    item.append(entry.text);
    if (entry.tool_name) el("code", item, ` ${entry.tool_name}`);
    appendToFeed(item);
  }

  function tally(entry) {
    people.set(entry.agent_id, entry);
    hud.tokens += entry.tokens;
    if (entry.cost_usd !== null && entry.cost_usd !== undefined) {
      hud.cost += entry.cost_usd;
      hud.priced += entry.tokens;
    }
    if (entry.node && entry.tool_name === "graph.enter") {
      hud.visits.set(entry.node, (hud.visits.get(entry.node) || 0) + 1);
      hud.active.add(entry.node);
    } else if (entry.node) hud.active.delete(entry.node);
    if (entry.approved === true || entry.approved === false) {
      hud.pulses.push({ at: performance.now(), ok: entry.approved });
      if (entry.approved) hud.approved += 1;
      else hud.rejected += 1;
    }
  }

  function coasted(knot, t) {
    const coast = Math.min(t - knot.t, COAST_MS);
    const glide = coast - (coast * coast) / (2 * COAST_MS);
    return { x: knot.x + knot.vx * glide, y: knot.y + knot.vy * glide };
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

  // The mirror draws the sections themselves, so this layer only outlines who holds what.
  function outline(rect, state, color, label, now, picked) {
    if (rect.h <= 0 || rect.y > height || rect.y + rect.h < 0) return;
    ctx.save();
    if (state === "FREE") {
      ctx.strokeStyle = picked ? "#e5e7eb" : "rgba(148, 163, 184, .28)";
      ctx.lineWidth = picked ? 1.5 : 1;
    } else {
      const pulse = state === "WAITING" && !calm.matches ? 0.55 + 0.45 * Math.sin(now / 180) : 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = pulse;
      if (state === "WAITING") ctx.setLineDash([8, 6]);
    }
    ctx.beginPath();
    ctx.roundRect(rect.x + 1, rect.y + 1, Math.max(rect.w - 2, 0), Math.max(rect.h - 2, 0), 12);
    ctx.stroke();
    ctx.restore();
    if (label) pill(label, rect.x + 8, rect.y + rect.h - 14, color, 1);
  }

  function drawOverlay(overlay, now) {
    const { width: w, height: h } = scene.viewport, scale = worldScale();
    const rect = { x: sx(overlay.box.x * w), y: sy(overlay.box.y * h),
      w: overlay.box.width * w * scale, h: overlay.box.height * h * scale };
    const owner = overlay.owner, leader = owner || overlay.waiters[0];
    const queue = overlay.waiters.length ? ` +${overlay.waiters.length} waiting` : "";
    const label = overlay.state === "FREE" ? ""
      : owner ? `${owner.label}${queue}` : `${overlay.waiters.length} waiting`;
    outline(rect, overlay.state, leader ? leader.color : "#475569", label, now,
      overlay.id === selected);
  }

  // In the past, the outlines come from that frame's lock table rather than the live one.
  function drawHeld(now) {
    const scale = worldScale(), frame = travel.frames[travel.index];
    for (const lock of frame.locks) {
      if (lock.state !== "ACQUIRED") continue;
      const component = travel.state.frontend.components.find((item) => item.id === lock.target_id);
      if (!component) continue;
      const { x, y, width: w, height: h } = component.box;
      const who = people.get(lock.locked_by_agent_id);
      outline({ x: sx(x), y: sy(y), w: w * scale, h: h * scale }, "LOCKED",
        colorOf(lock.locked_by_agent_id), who ? who.label : lock.locked_by_agent_id, now, false);
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
    placePage();
    animate(now);
    if (!$("hud").hidden) drawRadar(now);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!scene) return;
    if (travel && travel.state) drawHeld(now);
    else for (const overlay of scene.overlays) drawOverlay(overlay, now);
    const at = offset === null ? 0 : now - offset - BUFFER_MS;
    for (const track of tracks.values()) {
      const point = settledAt(track, at);
      drawTrail(track, at, track.sprite.color);
      if (track.isHuman) drawHuman(track, point);
      else drawAgent(track, point);
    }
    if (!travel) for (const badge of badges) drawBadge(badge);
    // From the first frame that arrived since the last paint, to this paint's end.
    if (arrival !== null) {
      record(paint, performance.now() - arrival);
      arrival = null;
    }
  }

  // --- input ------------------------------------------------------------------------------------

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
    showSelection();
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
  addEventListener("keydown", (event) => {
    const typing = event.target instanceof HTMLInputElement && event.target.type !== "range";
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "h") toggleHud();
    else if (key === "t") travel ? closeTravel() : openTravel();
    else if (key === "o") takeOver();
    else if (key === "r") giveBack();
    else if (key === "escape" && travel) closeTravel();
    else return;
    event.preventDefault();
  });
  $("hud-toggle").addEventListener("click", toggleHud);
  $("travel-toggle").addEventListener("click", () => (travel ? closeTravel() : openTravel()));
  $("live").addEventListener("click", closeTravel);
  $("fork").addEventListener("click", fork);
  $("verify").addEventListener("click", verify);
  $("take").addEventListener("click", takeOver);
  $("give").addEventListener("click", giveBack);
  $("seek").addEventListener("input", (event) => {
    if (travel) scrubTo(Number(event.target.value));
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
  setInterval(probe, PROBE_MS);
  setInterval(updateHud, 250);

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
        ("PROBE_INTERVAL_MS", PROBE_INTERVAL_MS),
        ("LATENCY_SAMPLES", LATENCY_SAMPLES),
        ("GRAPH_NODES", json.dumps(list(GRAPH_NODES))),
        ("HISTORY_CHECKPOINT_EVERY", HISTORY_CHECKPOINT_EVERY),
        ("SPRING_STIFFNESS", SPRING_STIFFNESS),
        ("SPRING_DAMPING", SPRING_DAMPING),
        ("GLOW_DECAY_MS", GLOW_DECAY_MS),
        ("RADAR_PULSE_MS", RADAR_PULSE_MS),
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
