"""One session from creation to a runnable export, watched live by its owner and a second person."""

import asyncio
import hashlib
import math
import re
from typing import Any

from e2e_support import (
    OWNER,
    SESSIONS,
    VIEWER,
    Frame,
    GatedProvider,
    Gateway,
    Socket,
    deadline,
    frontend_problems,
    gateway,
    python_problems,
    unzip,
)

from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.canvas.compositor import SWARM_PALETTE
from realtime_agentic_site_builder.core.patch_engine import PatchEngine
from realtime_agentic_site_builder.domain.swarm import (
    AgentRole,
    JsonPatchOp,
    LockState,
    ReplayTimeline,
    StreamDelta,
    SwarmSnapshot,
)

PROMPT = "Build SaaS landing page with contact form"
NODE_ROLES = {
    "agent-architect",
    "agent-coder",
    "agent-backend",
    "agent-styler",
    "agent-critic",
}


async def until_run_ends(socket: Socket, kind: str) -> list[Frame]:
    """Collect everything a client receives up to the thought that announces the end of the run."""
    frames: list[Frame] = []
    while True:
        frame = await socket.receive()
        frames.append(frame)
        tool = (frame.get("thought") or {}).get("tool_name") or frame.get("tool_name")
        if frame["kind"] == kind and tool in {"run.completed", "run.failed"}:
            return frames


def thoughts_of(frames: list[Frame]) -> list[Frame]:
    return [frame for frame in frames if frame["kind"] == "thought"]


async def watch_run(harness: Gateway) -> tuple[str, list[Frame], list[Frame], Frame]:
    """Create a session, connect its owner and a viewer, run the swarm and record both streams."""
    session_id = await harness.create_session(OWNER)
    owner = harness.socket(session_id, OWNER)
    viewer = harness.socket(session_id, VIEWER, query="view=canvas")
    assert (await owner.connect())["kind"] == "sync"
    hello = await viewer.connect()
    owner.send({"type": "prompt", "text": PROMPT, "request_id": "go"})
    owned, watched = await asyncio.gather(
        until_run_ends(owner, "thought"), until_run_ends(viewer, "canvas.thought")
    )
    await owner.drop()
    await viewer.drop()
    return session_id, owned, watched, hello


@deadline(30)
async def test_a_prompt_runs_the_whole_swarm_under_locks_and_exports_a_valid_project() -> None:
    async with gateway(provider=GatedProvider(latency_s=0.08, open_gate=True)) as harness:
        session_id, owned, watched, hello = await watch_run(harness)

        ack = next(frame for frame in owned if frame["kind"] == "ack")
        assert (ack["request_id"], ack["type"]) == ("go", "prompt")
        completed = thoughts_of(owned)[-1]["thought"]
        assert completed["tool_name"] == "run.completed"
        assert completed["tool_args"]["completed"] is True

        # Every agent node ran. The copywriter has a cursor, but the graph gives it no node.
        entered = {
            frame["thought"]["agent_id"]
            for frame in thoughts_of(owned)
            if frame["thought"]["tool_name"] == "graph.enter"
        }
        assert entered == NODE_ROLES

        states = [frame for frame in owned if frame["kind"] == "state"]
        assert states, "the run must have committed state"
        _assert_every_commit_was_locked_and_ordered(states)

        timeline_response = await harness.http.get(
            f"{SESSIONS}/{session_id}/timeline", headers=harness.auth(VIEWER)
        )
        timeline = ReplayTimeline.model_validate(timeline_response.json()["timeline"])
        _assert_replay_reproduces_every_frame(timeline)
        assert len(timeline.deltas) == len(states)

        _assert_viewer_saw_the_same_run(owned, watched, hello)

        final = SwarmSnapshot.model_validate(
            (await harness.http.get(f"{SESSIONS}/{session_id}", headers=harness.auth())).json()
        )
        await _assert_export_is_a_valid_project(harness, session_id, final)


def _assert_every_commit_was_locked_and_ordered(states: list[Frame]) -> None:
    for index, frame in enumerate(states):
        delta = StreamDelta.model_validate(frame["delta"])
        snapshot = SwarmSnapshot.model_validate(frame["snapshot"])
        assert delta.delta_seq == index
        assert snapshot.frame_seq == index + 1
        holders = {
            lock.target_id: lock.locked_by_agent_id
            for lock in snapshot.active_locks
            if lock.state is LockState.ACQUIRED
        }
        # The hub refuses a commit unless its author holds its target, so the record must show it.
        assert holders[delta.target_id] == delta.agent_id, frame["delta"]
        assert all(isinstance(patch.op, JsonPatchOp) for patch in delta.patches)
        assert delta.agent_id in {identity.agent_id for identity in IDENTITIES.values()}


def _assert_replay_reproduces_every_frame(timeline: ReplayTimeline) -> None:
    """Re-apply the RFC 6902 log from the first frame, and land on every recorded hash."""
    engine = PatchEngine(max_patch_size_bytes=1 << 20)
    bundle = timeline.snapshots[0].bundle_state
    for delta, snapshot in zip(timeline.deltas, timeline.snapshots[1:], strict=True):
        bundle, state_hash = engine.apply_patch(bundle, delta)
        assert state_hash == snapshot.bundle_hash
        snapshot.verify_replayed_state(bundle)


def _assert_viewer_saw_the_same_run(owned: list[Frame], watched: list[Frame], hello: Frame) -> None:
    assert hello["kind"] == "canvas.hello"
    assert (hello["user_id"], hello["is_owner"]) == (VIEWER, False)

    owner_seqs = [frame["seq"] for frame in thoughts_of(owned)]
    viewer_seqs = [frame["seq"] for frame in watched if frame["kind"] == "canvas.thought"]
    assert viewer_seqs == list(range(len(viewer_seqs))), "contiguous, from the first"
    assert viewer_seqs == owner_seqs, "and the same thoughts the owner saw"

    frames = [frame for frame in watched if frame["kind"] == "canvas"]
    assert len(frames) > 10
    final_scene = next(frame["scene"] for frame in reversed(frames) if "scene" in frame)
    assert {"hero", "contact"} <= {overlay["id"] for overlay in final_scene["overlays"]}

    agents = {agent["id"]: agent for frame in frames for agent in frame["agents"]}
    assert set(agents) == {identity.agent_id for identity in IDENTITIES.values()}
    for agent in agents.values():
        assert agent["color"] == SWARM_PALETTE[AgentRole(agent["role"])]
    # Idle cursors only drift. One that worked travelled, and reported its velocity on the way.
    travelled: dict[str, float] = {}
    for previous, current in zip(frames, frames[1:], strict=False):
        before = {agent["id"]: agent for agent in previous["agents"]}
        for agent in current["agents"]:
            if agent["id"] in before:
                step = math.dist(
                    (agent["x"], agent["y"]), (before[agent["id"]]["x"], before[agent["id"]]["y"])
                )
                travelled[agent["id"]] = travelled.get(agent["id"], 0.0) + step
    assert travelled["agent-coder"] > 20
    assert any(
        agent["id"] == "agent-coder" and math.hypot(agent["vx"], agent["vy"]) > 10
        for frame in frames
        for agent in frame["agents"]
    )
    assert any(agent["dock"] for frame in frames for agent in frame["agents"])
    assert any(frame["badges"] for frame in frames)


async def _assert_export_is_a_valid_project(
    harness: Gateway, session_id: str, final: SwarmSnapshot
) -> None:
    url = f"{SESSIONS}/{session_id}/export"
    owner_copy = await harness.http.get(url, params={"format": "zip"}, headers=harness.auth())
    viewer_copy = await harness.http.get(url, headers=harness.auth(VIEWER))
    assert owner_copy.status_code == 200, owner_copy.text
    assert owner_copy.headers["content-type"] == "application/zip"
    assert owner_copy.content == viewer_copy.content

    files = unzip(owner_copy.content)
    assert f"Bundle state hash: `{final.bundle_hash}`" in files["README.md"]
    assert python_problems(files) == []
    assert sum(path.startswith("backend/") and path.endswith(".py") for path in files) >= 10
    assert frontend_problems(files) == []

    bundle = final.bundle_state
    for model in bundle.backend.models:
        assert f"class {model.name}(" in files["backend/app/models.py"]
        assert f"export interface {model.name} {{" in files["frontend/src/api/types.ts"]
    client = files["frontend/src/api/client.ts"]
    for endpoint in bundle.backend.endpoints:
        assert f'"{endpoint.method.value}", `{endpoint.path}`' in client
    app = files["frontend/src/App.tsx"]
    imported = re.findall(r'from "\./components/(\w+)"', app)
    assert len(imported) == len(bundle.frontend.components)


@deadline(60)
async def test_the_same_prompt_on_the_same_start_yields_byte_identical_exports() -> None:
    """The mock is deterministic and the wall clock is held, so a run is a pure function."""
    digests: list[str] = []
    for _ in range(2):
        async with gateway() as harness:
            session_id = await harness.create_session(
                OWNER, bundle=_starting_bundle(), prompt=PROMPT
            )
            runtime = harness.service._sessions[session_id]
            assert runtime.run is not None
            await runtime.run
            export = await harness.http.get(
                f"{SESSIONS}/{session_id}/export", headers=harness.auth()
            )
            assert export.status_code == 200, export.text
            digests.append(hashlib.sha256(export.content).hexdigest())
    assert digests[0] == digests[1]


def _starting_bundle() -> dict[str, Any]:
    return {
        "project_id": "saas-landing",
        "frontend": {"site_id": "site", "title": "SaaS", "theme": "dark", "components": []},
        "backend": {"service_name": "api", "database_dialect": "SQLITE"},
    }


@deadline(10)
async def test_the_viewer_page_is_served_alongside() -> None:
    async with gateway() as harness:
        session_id = await harness.create_session()
        page = await harness.http.get(f"/canvas/{session_id}")
    assert page.status_code == 200
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    assert "default-src 'none'" in page.headers["content-security-policy"]
