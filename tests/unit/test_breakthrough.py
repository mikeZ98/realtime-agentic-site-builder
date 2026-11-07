"""Reality forking, time travel, the human override, cost telemetry and the invariant monitor."""

import asyncio
import copy
from typing import Any

import pytest
from e2e_support import (
    OWNER,
    SESSIONS,
    VIEWER,
    Frame,
    Gateway,
    Socket,
    gateway,
    settings,
)
from pydantic import SecretStr

from realtime_agentic_site_builder.agents.provider import MockLLMProvider
from realtime_agentic_site_builder.canvas.compositor import (
    CanvasCompositor,
    canvas_delta,
    participant,
    thought_entry,
)
from realtime_agentic_site_builder.config import ModelPrice, Settings
from realtime_agentic_site_builder.core.hub import SwarmSessionHub, ThoughtEvent
from realtime_agentic_site_builder.core.invariants import verify_timeline
from realtime_agentic_site_builder.core.lock_manager import (
    AsyncLockManager,
    LockEvent,
)
from realtime_agentic_site_builder.core.patch_engine import bundle_size, replay_operations
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    ComponentType,
)
from realtime_agentic_site_builder.domain.errors import FrameNotRetainedError
from realtime_agentic_site_builder.domain.swarm import (
    AgentRole,
    AgentThoughtStream,
    LockState,
    PatchOperation,
    ReplayTimeline,
    StreamDelta,
)
from realtime_agentic_site_builder.gateway.connection import HUMAN_OVERRIDE_PRIORITY

DOCUMENT = "frontend:document"


def component(component_id: str, y: int) -> dict[str, Any]:
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.HERO,
        title=component_id.title(),
        code="export default function S() { return <section />; }",
        box=BoundingBox(x=0, y=y, width=400, height=100),
    ).model_dump(mode="json")


def bundle() -> dict[str, Any]:
    return {
        "project_id": "fork",
        "frontend": {
            "site_id": "s",
            "title": "Start",
            "theme": "dark",
            "components": [component("hero", 0), component("footer", 200)],
        },
        "backend": {"service_name": "api", "database_dialect": "SQLITE"},
    }


async def reply(socket: Socket, request_id: str) -> Frame:
    while (frame := await socket.next_of("ack", "error"))["request_id"] != request_id:
        pass
    return frame


async def edited(harness: Gateway, titles: list[str]) -> tuple[str, Socket]:
    """Open a session and commit one title change per entry, returning its owner's socket."""
    session_id = await harness.create_session(OWNER, bundle=bundle())
    socket = harness.socket(session_id)
    await socket.connect()
    socket.send({"type": "lock.acquire", "target_id": DOCUMENT, "request_id": "lock"})
    await reply(socket, "lock")
    for index, title in enumerate(titles):
        socket.send(
            {
                "type": "delta",
                "request_id": f"d{index}",
                "target_domain": "frontend",
                "target_id": DOCUMENT,
                "patches": [{"op": "replace", "path": "/frontend/title", "value": title}],
            }
        )
        assert (await reply(socket, f"d{index}"))["kind"] == "ack"
    return session_id, socket


# --- reality forking ------------------------------------------------------------------------


async def test_a_fork_branches_history_at_a_frame_into_a_session_of_its_own() -> None:
    config = settings(ws_max_deltas_per_second=100)
    async with gateway(config) as harness:
        source_id, socket = await edited(harness, ["One", "Two", "Three", "Four"])
        response = await harness.http.post(
            f"{SESSIONS}/{source_id}/fork", params={"from_seq": 2}, headers=harness.auth(VIEWER)
        )
        assert response.status_code == 201, response.text
        body = response.json()
        fork_id = body["session_id"]
        assert response.headers["location"] == f"{SESSIONS}/{fork_id}"
        assert body["run_started"] is False
        assert body["snapshot"]["frame_seq"] == 2
        assert body["snapshot"]["bundle_state"]["frontend"]["title"] == "Two"

        hub = harness.service.hub
        source, fork = hub.timeline(source_id), hub.timeline(fork_id)
        assert [frame.frame_seq for frame in fork.snapshots] == [0, 1, 2]
        assert [frame.bundle_hash for frame in fork.snapshots] == [
            frame.bundle_hash for frame in source.snapshots[:3]
        ]
        assert {delta.session_id for delta in fork.deltas} == {fork_id}
        assert verify_timeline(fork).holds
        assert harness.service._sessions[fork_id].owner == VIEWER, "the fork is the forker's"

        # The branches are independent from here on.
        assert (await reply_after_edit(socket, "Five", "d5"))["frame_seq"] == 5
        fork_owner = harness.socket(fork_id, VIEWER)
        await fork_owner.connect()
        fork_owner.send({"type": "lock.acquire", "target_id": DOCUMENT, "request_id": "lock"})
        await reply(fork_owner, "lock")
        assert (await reply_after_edit(fork_owner, "Alternate", "a"))["frame_seq"] == 3
        assert hub.latest_snapshot(source_id).bundle_state.frontend.title == "Five"
        assert hub.latest_snapshot(fork_id).bundle_state.frontend.title == "Alternate"
        await socket.drop()
        await fork_owner.drop()


async def reply_after_edit(socket: Socket, title: str, request_id: str) -> Frame:
    socket.send(
        {
            "type": "delta",
            "request_id": request_id,
            "target_domain": "frontend",
            "target_id": DOCUMENT,
            "patches": [{"op": "replace", "path": "/frontend/title", "value": title}],
        }
    )
    return await reply(socket, request_id)


async def test_a_fork_shares_its_frames_with_its_source_instead_of_copying_them() -> None:
    """Snapshots are immutable, so a branch costs a reference per frame, whatever their size."""
    secret = SecretStr("x" * 40)
    hub = SwarmSessionHub(Settings(gateway_secret=secret))
    await hub.create_session("source", FullstackProjectBundle.model_validate(bundle()))
    await hub.lock_manager("source").acquire(DOCUMENT, "agent-coder", 60, hub.now_ms())
    for index in range(5):
        await hub.apply_delta(
            "source",
            StreamDelta.model_validate(
                {
                    "session_id": "source",
                    "agent_id": "agent-coder",
                    "delta_seq": index,
                    "target_domain": "frontend",
                    "target_id": DOCUMENT,
                    "patches": [{"op": "replace", "path": "/frontend/title", "value": f"T{index}"}],
                    "emitted_at_ms": 0,
                }
            ),
        )
    await hub.fork_session("source", "fork", 4)
    source, fork = hub.timeline("source"), hub.timeline("fork")
    assert all(a is b for a, b in zip(fork.snapshots, source.snapshots, strict=False))
    assert all(a.patches is b.patches for a, b in zip(fork.deltas, source.deltas, strict=False)), (
        "only the address on each delta is new"
    )
    assert (
        hub.history_bytes("fork")
        == sum(hub._sessions["source"].frame_bytes) - (hub._sessions["source"].frame_bytes[-1])
    )
    await hub.shutdown()


async def test_forking_at_the_first_frame_held_inherits_no_deltas() -> None:
    async with gateway() as harness:
        source_id, socket = await edited(harness, ["One"])
        response = await harness.http.post(
            f"{SESSIONS}/{source_id}/fork", params={"from_seq": 0}, headers=harness.auth()
        )
        fork = harness.service.hub.timeline(response.json()["session_id"])
        assert (len(fork.snapshots), len(fork.deltas)) == (1, 0)
        await socket.drop()


@pytest.mark.parametrize("frame_seq", [3, 99])
async def test_a_frame_outside_what_is_held_cannot_be_forked(frame_seq: int) -> None:
    async with gateway() as harness:
        source_id, socket = await edited(harness, ["One", "Two"])
        response = await harness.http.post(
            f"{SESSIONS}/{source_id}/fork",
            params={"from_seq": frame_seq},
            headers=harness.auth(),
        )
        assert response.status_code == 404
        problem = response.json()
        assert problem["type"].endswith("/frame-not-retained")
        assert (problem["frame_seq"], problem["earliest"], problem["latest"]) == (frame_seq, 0, 2)
        await socket.drop()


async def test_a_trimmed_frame_cannot_be_forked_either() -> None:
    async with gateway(settings(replay_history_bytes=1)) as harness:
        source_id, socket = await edited(harness, ["One", "Two", "Three"])
        response = await harness.http.post(
            f"{SESSIONS}/{source_id}/fork", params={"from_seq": 0}, headers=harness.auth()
        )
        assert response.status_code == 404
        assert response.json()["earliest"] == 2
        await socket.drop()


async def test_a_fork_of_a_trimmed_window_starts_at_its_oldest_frame_and_ends_where_asked() -> None:
    sizes = [bundle_size(frame.bundle_state) for frame in (await recorded(list("ABCD"))).snapshots]
    hub = SwarmSessionHub(
        Settings(gateway_secret=SecretStr("x" * 40), replay_history_bytes=sum(sizes[-3:]))
    )
    await commit_titles(hub, list("ABCD"))
    assert [frame.frame_seq for frame in hub.timeline("s1").snapshots] == [2, 3, 4]
    await hub.fork_session("s1", "fork", 3)
    fork = hub.timeline("fork")
    assert [frame.frame_seq for frame in fork.snapshots] == [2, 3]
    assert [delta.delta_seq for delta in fork.deltas] == [2]
    await hub.shutdown()


async def test_forking_needs_a_token_and_counts_against_the_forkers_session_limit() -> None:
    async with gateway(settings(gateway_max_sessions_per_user=1)) as harness:
        source_id = await harness.create_session(OWNER, bundle=bundle())
        url = f"{SESSIONS}/{source_id}/fork"
        assert (await harness.http.post(url, params={"from_seq": 0})).status_code == 401
        refused = await harness.http.post(url, params={"from_seq": 0}, headers=harness.auth())
        assert refused.status_code == 429, "the owner already has their one session"
        assert (await harness.http.post(url, headers=harness.auth(VIEWER))).status_code == 422
        missing = await harness.http.post(
            f"{SESSIONS}/nope/fork", params={"from_seq": 0}, headers=harness.auth(VIEWER)
        )
        assert missing.status_code == 404


async def test_a_session_id_cannot_be_forked_into_twice() -> None:
    hub = SwarmSessionHub(Settings(gateway_secret=SecretStr("x" * 40)))
    await hub.create_session("source", FullstackProjectBundle.model_validate(bundle()))
    await hub.fork_session("source", "fork", 0)
    with pytest.raises(Exception, match="already exists"):
        await hub.fork_session("source", "fork", 0)
    with pytest.raises(FrameNotRetainedError):
        await hub.fork_session("source", "other", 1)
    await hub.shutdown()


# --- time travel ----------------------------------------------------------------------------


def frontend_at(history: dict[str, Any], frame_seq: int) -> dict[str, Any]:
    """Rebuild the layout at a frame the way the viewer's scrubber does: patch the base forward.

    Nothing but RFC 6902 runs here, as in the page: each frontend delta's patches, then the
    corrections that carry the server's stamping and normalization.
    """
    document = {"frontend": copy.deepcopy(history["base_frontend"])}
    for entry in history["entries"]:
        if entry["frame_seq"] > frame_seq:
            break
        delta = StreamDelta.model_validate(entry["delta"])
        patches = delta.patches if delta.target_domain == "frontend" else ()
        fixes = [PatchOperation.model_validate(fix) for fix in entry["corrections"]]
        replay_operations(document, (*patches, *fixes))
    rebuilt: dict[str, Any] = document["frontend"]
    return rebuilt


async def test_history_lets_a_client_rebuild_the_layout_at_every_frame_held() -> None:
    config = settings(ws_max_deltas_per_second=100)
    async with gateway(config) as harness:
        session_id, socket = await edited(harness, ["One", "Two"])
        socket.send({"type": "lock.acquire", "target_id": "footer", "request_id": "footer"})
        await reply(socket, "footer")
        socket.send(
            {
                "type": "delta",
                "request_id": "move",
                "target_domain": "frontend",
                "target_id": "footer",
                # Committed as "Bye & thanks...": the server unescapes and NFKC-normalizes it.
                "patches": [
                    {
                        "op": "replace",
                        "path": "/frontend/components/1/title",
                        "value": "Bye &amp; thanks\u2026",
                    }
                ],
            }
        )
        await reply(socket, "move")
        response = await harness.http.get(
            f"{SESSIONS}/{session_id}/history", headers=harness.auth(VIEWER)
        )
        history = response.json()
        timeline = harness.service.hub.timeline(session_id)
        assert history["base_frame_seq"] == 0
        assert [entry["frame_seq"] for entry in history["entries"]] == [1, 2, 3]
        for frame in timeline.snapshots:
            expected = frame.bundle_state.frontend.model_dump(mode="json")
            assert frontend_at(history, frame.frame_seq) == expected
        assert timeline.snapshots[-1].bundle_state.frontend.components[1].title == "Bye & thanks..."
        assert {fix["path"] for fix in history["entries"][-1]["corrections"]} == {
            "/frontend/components/1/title",
            "/frontend/components/1/ast_version",
        }, "the checksum covers code and box, so a new title leaves it alone"
        assert [entry["bundle_hash"] for entry in history["entries"]] == [
            frame.bundle_hash for frame in timeline.snapshots[1:]
        ]
        assert history["entries"][-1]["locks"][0]["locked_by_agent_id"] == "user:alice"
        await socket.drop()


# --- the human override ---------------------------------------------------------------------


async def test_the_owner_can_take_a_lock_from_any_agent_and_everyone_is_told() -> None:
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, bundle=bundle())
        hub = harness.service.hub
        manager = hub.lock_manager(session_id)
        await manager.acquire("hero", "agent-architect", 90, hub.now_ms())
        owner = harness.socket(session_id)
        viewer = harness.socket(session_id, VIEWER, query="view=canvas")
        await owner.connect()
        await viewer.connect()

        owner.send({"type": "lock.override", "target_id": "hero", "request_id": "stop"})
        ack = await reply(owner, "stop")
        assert ack["lock"]["state"] == "ACQUIRED"
        assert ack["lock"]["priority"] == HUMAN_OVERRIDE_PRIORITY == 95
        assert manager.live_holder("hero", hub.now_ms()) == "user:alice"

        kinds: set[str] = set()
        while kinds != {"PREEMPTED", "OVERRIDDEN"}:
            event = (await owner.next_of("lock"))["event"]
            kinds.add(event["kind"])
            if event["kind"] == "PREEMPTED":
                assert event["lock"]["locked_by_agent_id"] == "agent-architect"
                assert event["problem"]["status"] == 409
        alert = await viewer.next_of("canvas.override")
        assert alert["target_id"] == "hero"
        assert alert["by"]["label"] == "user:alice"
        await owner.drop()
        await viewer.drop()


async def test_a_viewer_cannot_override_and_an_override_of_nothing_is_a_plain_lock() -> None:
    async with gateway(settings(ws_max_locks_per_user=1)) as harness:
        session_id = await harness.create_session(OWNER, bundle=bundle())
        viewer = harness.socket(session_id, VIEWER)
        await viewer.connect()
        viewer.send({"type": "lock.override", "target_id": "hero", "request_id": "v"})
        assert (await reply(viewer, "v"))["problem"]["status"] == 403
        owner = harness.socket(session_id)
        await owner.connect()
        owner.send({"type": "lock.override", "target_id": "hero", "request_id": "free"})
        assert (await reply(owner, "free"))["lock"]["state"] == "ACQUIRED"
        owner.send({"type": "lock.override", "target_id": "footer", "request_id": "quota"})
        assert (await reply(owner, "quota"))["problem"]["status"] == 429, "quota still applies"
        await viewer.drop()
        await owner.drop()


async def lock_events(manager_events: list[LockEvent]) -> list[tuple[str, str]]:
    return [(event.kind.value, event.lock.locked_by_agent_id) for event in manager_events]


async def test_override_only_takes_from_a_lower_priority_live_holder() -> None:
    events: list[LockEvent] = []
    now = [1000]
    manager = AsyncLockManager(lease_ms=100, clock=lambda: now[0], on_event=events.append)
    await manager.acquire("t", "agent-a", 95, now[0])
    equal = await manager.override("t", "user:alice", 95, now[0])
    assert equal.state is LockState.WAITING, "an equal priority is not outranked"
    await manager.release("t", "user:alice")
    renewed = await manager.override("t", "agent-a", 95, now[0] + 10)
    assert (renewed.state, renewed.acquired_at_ms) == (LockState.ACQUIRED, now[0] + 10)
    now[0] += 1000
    lapsed = await manager.override("t", "user:alice", 95, now[0])
    assert lapsed.state is LockState.ACQUIRED
    assert ("OVERRIDDEN", "user:alice") not in await lock_events(events), (
        "a lapsed lease is reclaimed, not overridden"
    )


async def test_an_overridden_agent_keeps_its_other_locks_and_its_queue_goes_on() -> None:
    events: list[LockEvent] = []
    manager = AsyncLockManager(clock=lambda: 0, on_event=events.append)
    await manager.acquire("hero", "agent-coder", 60, 0)
    await manager.acquire("footer", "agent-coder", 60, 0)
    await manager.acquire("hero", "agent-styler", 40, 0)
    await manager.override("hero", "user:alice", 95, 0)
    assert manager.live_holder("footer", 0) == "agent-coder"
    assert manager.live_holder("hero", 0) == "user:alice"
    await manager.release("hero", "user:alice")
    assert manager.live_holder("hero", 0) == "agent-styler"
    assert await lock_events(events) == [
        ("PREEMPTED", "agent-coder"),
        ("OVERRIDDEN", "user:alice"),
        ("PROMOTED", "agent-styler"),
    ]


async def test_an_override_during_a_run_is_survived_by_the_swarm() -> None:
    """The agent finds its hold gone at its next commit, and the run still finishes."""
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, prompt="Build it")
        hub = harness.service.hub
        owner = harness.socket(session_id)
        await owner.connect()
        manager = hub.lock_manager(session_id)
        async with asyncio.timeout(5):
            while not any(lock.state is LockState.ACQUIRED for lock in manager.active_locks()):
                await asyncio.sleep(0)
        target = next(lock.target_id for lock in manager.active_locks())
        owner.send({"type": "lock.override", "target_id": target, "request_id": "stop"})
        await reply(owner, "stop")
        runtime = harness.service._sessions[session_id]
        if runtime.run is not None:
            await runtime.run
        assert verify_timeline(hub.timeline(session_id)).holds
        await owner.drop()


# --- the canvas stream ----------------------------------------------------------------------


async def test_canvas_viewers_get_the_layout_and_then_every_commit_as_a_patch() -> None:
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, bundle=bundle())
        viewer = harness.socket(session_id, VIEWER, query="view=canvas")
        hello = await viewer.connect()
        assert hello["frame_seq"] == 0
        assert hello["frontend"]["title"] == "Start"
        owner = harness.socket(session_id)
        await owner.connect()
        owner.send({"type": "lock.acquire", "target_id": DOCUMENT, "request_id": "l"})
        await reply(owner, "l")
        await reply_after_edit(owner, "Live", "d")
        delta = await viewer.next_of("canvas.delta")
        assert (delta["frame_seq"], delta["agent_id"]) == (1, "user:alice")
        assert delta["target_id"] == DOCUMENT
        assert delta["color"] == participant("user:alice").color
        assert delta["patches"] == [{"op": "replace", "path": "/frontend/title", "value": "Live"}]
        frame = await viewer.next_of("canvas")
        if "scene" in frame:
            assert frame["scene"]["overlays"][0]["component_type"] == "HERO"
        await owner.drop()
        await viewer.drop()


async def test_backend_commits_reach_the_timeline_without_their_patches() -> None:
    hub = SwarmSessionHub(Settings(gateway_secret=SecretStr("x" * 40)))
    await hub.create_session("s1", FullstackProjectBundle.model_validate(bundle()))
    listener = hub.subscribe("s1", lossless=True)
    await hub.lock_manager("s1").acquire("backend:document", "agent-backend", 70, hub.now_ms())
    await hub.apply_delta(
        "s1",
        StreamDelta.model_validate(
            {
                "session_id": "s1",
                "agent_id": "agent-backend",
                "delta_seq": 0,
                "target_domain": "backend",
                "target_id": "backend:document",
                "patches": [{"op": "replace", "path": "/backend/service_name", "value": "new"}],
                "emitted_at_ms": 0,
            }
        ),
    )
    event = listener.poll()
    sent = canvas_delta(event)  # type: ignore[arg-type]
    assert (sent.target_domain, sent.patches, sent.frame_seq) == ("backend", (), 1)
    assert (sent.target_id, sent.color) == ("backend:document", participant("agent-backend").color)
    await hub.shutdown()


def thought(tool: str, args: dict[str, Any] | None, **extra: Any) -> ThoughtEvent:
    return ThoughtEvent(
        origin="w",
        session_id="s",
        seq=0,
        thought=AgentThoughtStream(
            agent_id="agent-critic",
            role=AgentRole.CRITIC_LINTER,
            thought="t",
            tool_name=tool,
            tool_args=args,
            timestamp_ms=1,
            **extra,
        ),
    )


@pytest.mark.parametrize(
    ("event", "fields"),
    [
        (thought("graph.enter", {"node": "critic"}), {"node": "critic", "approved": None}),
        (thought("graph.exit", {"node": "styler"}), {"node": "styler", "approved": None}),
        (thought("critic.verdict", {"approved": False}), {"node": None, "approved": False}),
        (thought("critic.verdict", {"approved": "no"}), {"node": None, "approved": None}),
        (thought("graph.enter", {"node": 3}), {"node": None, "approved": None}),
        (
            thought("llm.generate", {"node": "x"}, tokens_used=120, cost_usd=0.0012),
            {
                "node": None,
                "tokens": 120,
                "cost_usd": 0.0012,
            },
        ),
        (thought("lock.wait", None), {"node": None, "tokens": 0, "cost_usd": None}),
    ],
)
def test_feed_entries_carry_what_the_hud_reads(event: ThoughtEvent, fields: dict[str, Any]) -> None:
    entry = thought_entry(event).model_dump()
    assert {key: entry[key] for key in fields} == fields


def test_a_price_turns_token_counts_into_dollars() -> None:
    price = ModelPrice(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0)
    assert price.cost_usd(1_000_000, 0) == 3.0
    assert price.cost_usd(2_000, 400) == pytest.approx(0.012)


async def test_a_run_reports_cost_only_for_models_that_have_a_price() -> None:
    priced = settings(
        model_prices={"mock": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}}
    )
    for config, expect_cost in ((priced, True), (settings(), False)):
        async with gateway(config, provider=MockLLMProvider()) as harness:
            session_id = await harness.create_session(OWNER, prompt="Build it")
            runtime = harness.service._sessions[session_id]
            assert runtime.run is not None
            await runtime.run
            generated = [
                event.thought
                for event in harness.service.hub.thought_backlog(session_id).events
                if event.thought.tool_name == "llm.generate"
            ]
            assert generated
            assert all((t.cost_usd is not None) is expect_cost for t in generated)
            if expect_cost:
                assert all(
                    t.cost_usd == pytest.approx(t.tokens_used * 1.5e-6, rel=0.5) for t in generated
                )


def test_overlays_tell_the_dom_mirror_what_kind_of_section_to_draw() -> None:
    compositor = CanvasCompositor(FullstackProjectBundle.model_validate(bundle()))
    frame = compositor.compose(0.0, (), 0)
    assert [overlay.component_type for overlay in frame.scene.overlays] == ["HERO", "HERO"]


# --- the invariant monitor ------------------------------------------------------------------


async def test_a_real_swarm_run_satisfies_every_invariant_over_http() -> None:
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, prompt="Build a landing page")
        runtime = harness.service._sessions[session_id]
        assert runtime.run is not None
        await runtime.run
        response = await harness.http.get(
            f"{SESSIONS}/{session_id}/verify", headers=harness.auth(VIEWER)
        )
    report = response.json()
    assert response.status_code == 200
    assert report["holds"] is True
    assert {item["name"] for item in report["invariants"]} == {
        "log-complete",
        "frames-contiguous",
        "deltas-ordered",
        "mutual-exclusion",
        "commits-locked",
        "leases-live",
        "replay-exact",
    }
    assert all(item["checked"] >= 1 for item in report["invariants"])


async def test_verification_shares_the_heavy_work_limit_and_needs_a_token() -> None:
    async with gateway(settings(gateway_max_concurrent_exports=1)) as harness:
        session_id = await harness.create_session(OWNER)
        url = f"{SESSIONS}/{session_id}/verify"
        assert (await harness.http.get(url)).status_code == 401
        harness.service._exports_in_flight = 1
        busy = await harness.http.get(url, headers=harness.auth())
        harness.service._exports_in_flight = 0
        assert busy.status_code == 429


async def recorded(titles: list[str]) -> ReplayTimeline:
    hub = SwarmSessionHub(Settings(gateway_secret=SecretStr("x" * 40)))
    await commit_titles(hub, titles)
    timeline = hub.timeline("s1")
    await hub.shutdown()
    return timeline


async def commit_titles(hub: SwarmSessionHub, titles: list[str]) -> None:
    await hub.create_session("s1", FullstackProjectBundle.model_validate(bundle()))
    await hub.lock_manager("s1").acquire(DOCUMENT, "agent-coder", 60, hub.now_ms())
    for index, title in enumerate(titles):
        await hub.apply_delta(
            "s1",
            StreamDelta.model_validate(
                {
                    "session_id": "s1",
                    "agent_id": "agent-coder",
                    "delta_seq": index,
                    "target_domain": "frontend",
                    "target_id": DOCUMENT,
                    "patches": [{"op": "replace", "path": "/frontend/title", "value": title}],
                    "emitted_at_ms": 0,
                }
            ),
        )


def tampered(timeline: ReplayTimeline, **changes: Any) -> ReplayTimeline:
    """Build a log that skips validation, as a buggy component upstream could have produced."""
    return ReplayTimeline.model_construct(
        session_id=timeline.session_id,
        deltas=changes.get("deltas", timeline.deltas),
        snapshots=changes.get("snapshots", timeline.snapshots),
    )


def violated(timeline: ReplayTimeline) -> dict[str, tuple[str, ...]]:
    report = verify_timeline(timeline)
    return {item.name: item.violations for item in report.invariants if not item.holds}


async def test_a_clean_log_holds_every_invariant() -> None:
    report = verify_timeline(await recorded(["A", "B", "C"]))
    assert report.holds
    assert (report.first_frame, report.last_frame) == (0, 3)


async def test_a_gap_in_frames_and_reordered_deltas_are_named() -> None:
    timeline = await recorded(["A", "B", "C"])
    gap = timeline.snapshots[2].model_copy(update={"frame_seq": 7})
    snapshots = (*timeline.snapshots[:2], gap, timeline.snapshots[3])
    deltas = (timeline.deltas[1], timeline.deltas[0], timeline.deltas[2])
    found = violated(tampered(timeline, snapshots=snapshots, deltas=deltas))
    assert found["frames-contiguous"] == ("frame 7 follows frame 1", "frame 3 follows frame 7")
    assert found["deltas-ordered"] == ("delta 0 follows delta 1",)
    assert "replay-exact" in found, "reordered deltas no longer replay to the recorded state"


async def test_a_repeated_delta_number_and_a_truncated_log_are_named() -> None:
    timeline = await recorded(["A", "B", "C"])
    repeated = timeline.deltas[2].model_copy(update={"delta_seq": 1})
    found = violated(tampered(timeline, deltas=(*timeline.deltas[:2], repeated)))
    assert found["deltas-ordered"] == ("delta 1 follows delta 1",)

    truncated = violated(tampered(timeline, deltas=timeline.deltas[:2]))
    assert truncated == {"log-complete": ("3 frames follow the first, but 2 deltas were recorded",)}
    surplus = violated(tampered(timeline, snapshots=timeline.snapshots[:3]))
    assert surplus["log-complete"] == ("2 frames follow the first, but 3 deltas were recorded",)


async def test_holding_some_other_element_does_not_cover_a_commit() -> None:
    timeline = await recorded(["A"])
    elsewhere = timeline.snapshots[1].active_locks[0].model_copy(update={"target_id": "hero"})
    frame = timeline.snapshots[1].model_copy(update={"active_locks": (elsewhere,)})
    found = violated(tampered(timeline, snapshots=(timeline.snapshots[0], frame)))
    assert found["commits-locked"] == (
        f"frame 1: agent-coder committed to {DOCUMENT} without holding it",
    )


async def test_two_holders_of_one_element_are_named() -> None:
    timeline = await recorded(["A"])
    rival = (
        timeline.snapshots[1]
        .active_locks[0]
        .model_copy(update={"locked_by_agent_id": "agent-rival"})
    )
    frame = timeline.snapshots[1].model_copy(
        update={"active_locks": (*timeline.snapshots[1].active_locks, rival)}
    )
    found = violated(tampered(timeline, snapshots=(timeline.snapshots[0], frame)))
    assert found["mutual-exclusion"] == (f"frame 1 records 2 holders of {DOCUMENT}",)


async def test_an_unlocked_commit_and_one_on_a_lapsed_lease_are_named() -> None:
    timeline = await recorded(["A", "B"])
    unlocked = timeline.snapshots[1].model_copy(update={"active_locks": ()})
    lapsed = timeline.snapshots[2].model_copy(
        update={"timestamp_ms": timeline.snapshots[2].active_locks[0].expires_at_ms}
    )
    found = violated(tampered(timeline, snapshots=(timeline.snapshots[0], unlocked, lapsed)))
    assert found["commits-locked"] == (
        f"frame 1: agent-coder committed to {DOCUMENT} without holding it",
    )
    assert found["leases-live"] == (
        f"frame 2: agent-coder committed to {DOCUMENT} on a lapsed lease",
    )


async def test_a_divergent_or_unappliable_log_is_named() -> None:
    timeline = await recorded(["A", "B"])
    wrong = timeline.snapshots[1].model_copy(update={"bundle_hash": "0" * 64})
    found = violated(
        tampered(timeline, snapshots=(timeline.snapshots[0], wrong, timeline.snapshots[2]))
    )
    assert found["replay-exact"] == ("frame 1: replay reaches a different state",)

    base = timeline.snapshots[0].model_copy(update={"bundle_hash": "0" * 64})
    found = violated(tampered(timeline, snapshots=(base, *timeline.snapshots[1:])))
    assert found["replay-exact"] == ("frame 0 does not match its own hash",)

    broken = timeline.deltas[0].model_copy(
        update={
            "patches": (
                timeline.deltas[0].patches[0].model_copy(update={"path": "/frontend/missing/x"}),
            )
        }
    )
    found = violated(tampered(timeline, deltas=(broken, timeline.deltas[1])))
    assert found["replay-exact"][0].startswith("frame 1: delta 0 no longer applies")


async def test_violations_are_reported_in_bounded_number() -> None:
    timeline = await recorded([f"T{index}" for index in range(8)])
    frames = tuple(frame.model_copy(update={"active_locks": ()}) for frame in timeline.snapshots)
    found = violated(tampered(timeline, snapshots=frames))
    assert len(found["commits-locked"]) == 5
