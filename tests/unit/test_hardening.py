"""Each test here reproduces a failure found in the hardening audit."""

import asyncio
import json
import logging
import re
import threading
import tracemalloc
from collections.abc import Mapping
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
from starlette.types import Message, Receive, Scope, Send

from realtime_agentic_site_builder.core.hub import (
    RedisEventBus,
    SwarmSessionHub,
    ThoughtEvent,
    session_channel,
)
from realtime_agentic_site_builder.core.patch_engine import PatchEngine, bundle_size
from realtime_agentic_site_builder.domain.backend import (
    FullstackProjectBundle,
)
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    ComponentType,
)
from realtime_agentic_site_builder.domain.errors import BundleTooLargeError
from realtime_agentic_site_builder.domain.swarm import (
    AgentRole,
    AgentThoughtStream,
    PatchOperation,
    StreamDelta,
)
from realtime_agentic_site_builder.gateway.observability import (
    RequestIdMiddleware,
    install_log_correlation,
)
from realtime_agentic_site_builder.synthesis.exporter import export_bundle

DOCUMENT = "frontend:document"


def component(component_id: str, y: int, code_bytes: int = 0) -> dict[str, Any]:
    code = f"export default function S() {{ return <p>{'x' * code_bytes}</p>; }}"
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.FEATURES,
        title=component_id,
        code=code,
        box=BoundingBox(x=0, y=y, width=100, height=50),
    ).model_dump(mode="json")


def bundle(*components: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": "hardening",
        "frontend": {"site_id": "s", "title": "T", "theme": "dark", "components": list(components)},
        "backend": {"service_name": "api", "database_dialect": "SQLITE"},
    }


async def owned(harness: Gateway, body: dict[str, Any] | None = None) -> tuple[str, Socket]:
    session_id = await harness.create_session(OWNER, bundle=body or bundle(component("hero", 0)))
    socket = harness.socket(session_id)
    await socket.connect()
    return session_id, socket


async def reply(socket: Socket, request_id: str) -> Frame:
    while (frame := await socket.next_of("ack", "error"))["request_id"] != request_id:
        pass
    return frame


async def lock(socket: Socket, target_id: str, request_id: str) -> Frame:
    socket.send({"type": "lock.acquire", "target_id": target_id, "request_id": request_id})
    return await reply(socket, request_id)


async def set_title(socket: Socket, title: str, request_id: str) -> Frame:
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


# --- a hostile message cannot kill a connection ---------------------------------------------


@pytest.mark.parametrize("user_id", [OWNER, VIEWER])
async def test_nesting_deep_enough_to_exhaust_the_stack_is_an_error_not_a_crash(
    user_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """It fits within the message size limit, and used to close the socket with 1011."""
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, bundle=bundle(component("hero", 0)))
        socket = harness.socket(session_id, user_id)
        await socket.connect()
        nested = '{"type":"ping","request_id":"deep","p":' + "[" * 60_000 + "]" * 60_000 + "}"
        assert len(nested) < harness.service.settings.ws_max_message_bytes
        with caplog.at_level(logging.ERROR):
            socket._to_app.put_nowait({"type": "websocket.receive", "text": nested})
            refused = await socket.next_of("error")
        assert refused["problem"]["status"] == 422
        assert refused["request_id"] is None, "the id cannot be recovered from such a message"
        socket.send({"type": "ping", "request_id": "alive"})
        assert (await socket.next_of("pong"))["request_id"] == "alive"
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
        await socket.drop()


# --- memory and size are bounded ------------------------------------------------------------


async def test_an_oversized_starting_bundle_is_refused_with_its_size() -> None:
    config = settings(max_bundle_bytes=20_000, max_request_bytes=1_000_000)
    async with gateway(config) as harness:
        response = await harness.http.post(
            SESSIONS,
            json={"bundle": bundle(component("hero", 0, code_bytes=30_000))},
            headers=harness.auth(),
        )
    assert response.status_code == 413
    problem = response.json()
    assert problem["type"].endswith("/bundle-too-large")
    assert problem["limit_bytes"] == 20_000
    assert problem["size_bytes"] > 20_000


async def test_deltas_cannot_grow_a_bundle_past_its_limit() -> None:
    """Adding components used to grow a bundle, and the cost of every later commit, forever."""
    async with gateway(settings(max_bundle_bytes=40_000)) as harness:
        session_id, socket = await owned(harness)
        hub = harness.service.hub
        before = hub.latest_snapshot(session_id)
        refused = None
        for index in range(10):
            target = f"pad-{index}"
            await lock(socket, target, f"l{index}")
            socket.send(
                {
                    "type": "delta",
                    "request_id": f"g{index}",
                    "target_domain": "frontend",
                    "target_id": target,
                    "patches": [
                        {
                            "op": "add",
                            "path": "/frontend/components/-",
                            "value": component(target, 100 + index * 60, code_bytes=9_000),
                        }
                    ],
                }
            )
            frame = await reply(socket, f"g{index}")
            if frame["kind"] == "error":
                refused = frame
                break
        assert refused is not None, "the limit must stop growth"
        assert refused["problem"]["status"] == 413
        assert refused["problem"]["type"].endswith("/bundle-too-large")
        latest = hub.latest_snapshot(session_id)
        assert latest.frame_seq == before.frame_seq + index, "the refused delta changed nothing"
        assert bundle_size(latest.bundle_state) <= 40_000
        await socket.drop()


def test_the_engine_reports_size_and_hash_from_one_serialization() -> None:
    start = FullstackProjectBundle.model_validate(bundle(component("hero", 0)))
    delta = StreamDelta(
        session_id="s",
        agent_id="a",
        delta_seq=0,
        target_domain="frontend",
        target_id=DOCUMENT,
        patches=(
            PatchOperation.model_validate(
                {"op": "replace", "path": "/frontend/title", "value": "New"}
            ),
        ),
        emitted_at_ms=0,
    )
    result = PatchEngine().apply(start, delta)
    assert result.state_hash == result.bundle.state_hash()
    assert result.size_bytes == bundle_size(result.bundle)
    assert PatchEngine().apply_patch(start, delta) == (result.bundle, result.state_hash)
    with pytest.raises(BundleTooLargeError):
        PatchEngine(max_bundle_bytes=result.size_bytes - 1).apply(start, delta)


async def test_the_replay_log_forgets_its_oldest_frames_once_over_budget() -> None:
    """Every frame holds a whole bundle: 200 edits on a 1 MB bundle used to retain 200 MB."""
    body = bundle(component("hero", 0, code_bytes=20_000))
    frame_bytes = bundle_size(FullstackProjectBundle.model_validate(body))
    budget = frame_bytes * 5
    config = settings(replay_history_bytes=budget, ws_max_deltas_per_second=1000)
    async with gateway(config) as harness:
        session_id, socket = await owned(harness, body)
        hub = harness.service.hub
        await lock(socket, DOCUMENT, "lock")
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        for index in range(60):
            assert (await set_title(socket, f"Title {index:03}", f"d{index}"))["kind"] == "ack"
        retained = tracemalloc.get_traced_memory()[0] - baseline
        tracemalloc.stop()

        assert hub.history_bytes(session_id) <= budget
        assert retained < frame_bytes * 15, "memory follows the budget, not the number of edits"
        timeline = hub.timeline(session_id)
        assert len(timeline.snapshots) == len(timeline.deltas) + 1 <= 5
        assert timeline.snapshots[0].frame_seq > 0
        assert timeline.snapshots[-1].frame_seq == 60
        engine = PatchEngine()
        state = timeline.snapshots[0].bundle_state
        for delta, snapshot in zip(timeline.deltas, timeline.snapshots[1:], strict=True):
            state, state_hash = engine.apply_patch(state, delta)
            assert state_hash == snapshot.bundle_hash, "the window is still a valid replay"

        # Sequence numbers keep counting across what was forgotten.
        assert (await set_title(socket, "Next", "next"))["frame_seq"] == 61
        await socket.drop()


async def test_the_newest_frame_is_kept_even_when_it_alone_exceeds_the_budget() -> None:
    async with gateway(settings(replay_history_bytes=1)) as harness:
        session_id, socket = await owned(harness)
        await lock(socket, DOCUMENT, "lock")
        for index in range(3):
            await set_title(socket, f"T{index}", f"d{index}")
        timeline = harness.service.hub.timeline(session_id)
        assert [frame.frame_seq for frame in timeline.snapshots] == [2, 3]
        await socket.drop()


async def test_a_replica_sizes_and_bounds_the_frames_it_adopts() -> None:
    from realtime_agentic_site_builder.core.hub import InMemoryEventBus

    bus = InMemoryEventBus()
    config = settings(replay_history_bytes=1)
    writer = SwarmSessionHub(config, bus=bus)
    replica = SwarmSessionHub(config, bus=bus)
    start = FullstackProjectBundle.model_validate(bundle(component("hero", 0)))
    for hub in (writer, replica):
        await hub.create_session("s1", start)
    manager = writer.lock_manager("s1")
    await manager.acquire(DOCUMENT, "agent-coder", 60, writer.now_ms())
    for index in range(3):
        await writer.apply_delta(
            "s1",
            StreamDelta(
                session_id="s1",
                agent_id="agent-coder",
                delta_seq=index,
                target_domain="frontend",
                target_id=DOCUMENT,
                patches=(
                    PatchOperation.model_validate(
                        {"op": "replace", "path": "/frontend/title", "value": f"T{index}"}
                    ),
                ),
                emitted_at_ms=1,
            ),
        )
    timeline = replica.timeline("s1")
    assert [frame.frame_seq for frame in timeline.snapshots] == [2, 3]
    assert replica.history_bytes("s1") == writer.history_bytes("s1")
    await writer.shutdown()
    await replica.shutdown()


# --- one tenant cannot take what the gateway shares out -------------------------------------


async def test_one_person_cannot_take_every_session_the_gateway_allows() -> None:
    config = settings(gateway_max_sessions=10, gateway_max_sessions_per_user=2)
    async with gateway(config) as harness:
        for _ in range(2):
            await harness.create_session("mallory")
        refused = await harness.http.post(SESSIONS, headers=harness.auth("mallory"))
        assert refused.status_code == 429
        assert refused.json()["type"].endswith("/capacity-exceeded")
        assert "limit per person" in refused.json()["detail"]
        assert (
            await harness.http.post(SESSIONS, headers=harness.auth("victim"))
        ).status_code == 201
        assert harness.service.metrics.refusals["sessions_per_user"] == 1


async def test_one_person_cannot_open_unlimited_sockets() -> None:
    async with gateway(settings(ws_max_connections_per_user=2)) as harness:
        session_id = await harness.create_session(OWNER)
        first, second = harness.socket(session_id, "mallory"), harness.socket(session_id, "mallory")
        await first.connect()
        await second.connect()
        third = harness.socket(session_id, "mallory")
        third._to_app.put_nowait({"type": "websocket.connect"})
        third.task = asyncio.create_task(
            harness.app(third._scope, third._to_app.get, third._send)  # type: ignore[arg-type]
        )
        answer = await asyncio.wait_for(third.from_app.get(), 2)
        assert (answer["type"], answer["code"]) == ("websocket.close", 1008)
        assert answer["reason"] == "Too many connections"
        other = harness.socket(session_id, "victim")
        await other.connect()

        await first.drop()
        again = harness.socket(session_id, "mallory")
        await again.connect(), "closing one frees its place"
        for socket in (second, other, again):
            await socket.drop()
        assert harness.service._sockets_per_user == {}


async def test_a_failure_while_setting_up_a_socket_leaves_no_listener_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER)

        def broken(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("setup failed")

        monkeypatch.setattr(harness.service, "_connect", broken)
        socket = harness.socket(session_id)
        socket._to_app.put_nowait({"type": "websocket.connect"})
        socket.task = asyncio.create_task(
            harness.app(socket._scope, socket._to_app.get, socket._send)  # type: ignore[arg-type]
        )
        await asyncio.wait_for(asyncio.gather(socket.task, return_exceptions=True), 2)
        assert harness.service.hub._sessions[session_id].listeners == set()
        assert harness.service._sockets_per_user == {}


async def test_edits_are_budgeted_per_person_across_their_connections() -> None:
    now = [0]
    async with gateway(settings(ws_max_deltas_per_second=2), clock=lambda: now[0]) as harness:
        session_id, first = await owned(harness)
        second = harness.socket(session_id)
        await second.connect()
        await lock(first, DOCUMENT, "lock")
        assert (await set_title(first, "One", "a"))["kind"] == "ack"
        assert (await set_title(second, "Two", "b"))["kind"] == "ack"
        refused = await set_title(first, "Three", "c")
        assert refused["problem"]["status"] == 429, "the second tab spent from the same budget"
        now[0] += 500
        assert (await set_title(second, "Four", "d"))["kind"] == "ack"
        await first.drop()
        await second.drop()
        assert harness.service._delta_budgets == {}, "the budget goes with the last connection"


async def test_one_person_cannot_hoard_locks_and_leaving_releases_even_queued_ones() -> None:
    async with gateway(settings(ws_max_locks_per_user=2)) as harness:
        session_id, socket = await owned(harness)
        hub = harness.service.hub
        manager = hub.lock_manager(session_id)
        await manager.acquire("contested", "agent-architect", 90, hub.now_ms())
        assert (await lock(socket, "ghost-1", "a"))["lock"]["state"] == "ACQUIRED"
        assert (await lock(socket, "contested", "b"))["lock"]["state"] == "WAITING"
        refused = await lock(socket, "ghost-2", "c")
        assert refused["problem"]["status"] == 429
        assert (await lock(socket, "ghost-1", "renew"))["kind"] == "ack", "renewing is free"
        socket.send({"type": "lock.release", "target_id": "ghost-1", "request_id": "free"})
        await reply(socket, "free")
        assert (await lock(socket, "ghost-2", "d"))["kind"] == "ack"
        await socket.drop()
        assert {lock.locked_by_agent_id for lock in manager.active_locks()} == {"agent-architect"}


async def test_exports_beyond_what_the_thread_pool_absorbs_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, finish = threading.Event(), threading.Event()

    def slow(*args: Any) -> Any:
        started.set()
        finish.wait(5)
        return export_bundle(*args)

    monkeypatch.setattr("realtime_agentic_site_builder.gateway.service.export_bundle", slow)
    async with gateway(settings(gateway_max_concurrent_exports=1)) as harness:
        session_id = await harness.create_session(OWNER)
        url = f"{SESSIONS}/{session_id}/export"
        first = asyncio.create_task(harness.http.get(url, headers=harness.auth()))
        await asyncio.to_thread(started.wait, 5)
        busy = await harness.http.get(url, headers=harness.auth())
        finish.set()
        assert busy.status_code == 429
        assert "busy" in busy.json()["detail"]
        assert (await first).status_code == 200
        assert (await harness.http.get(url, headers=harness.auth())).status_code == 200
        assert harness.service.metrics.exports == {"ok": 2}


async def test_an_export_that_cannot_be_built_is_counted_as_rejected() -> None:
    hero = ComponentNode.create(
        id="hero",
        component_type=ComponentType.HERO,
        title="Hero",
        code="<Hero />",
        box=BoundingBox(x=0, y=0, width=100, height=50),
    ).model_dump(mode="json")
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, bundle=bundle(hero))
        response = await harness.http.get(f"{SESSIONS}/{session_id}/export", headers=harness.auth())
        assert response.status_code == 422
        assert harness.service.metrics.exports == {"rejected": 1}
        assert harness.service._exports_in_flight == 0


# --- the event bus keeps going --------------------------------------------------------------


class _FlakyPubSub:
    def __init__(self) -> None:
        self.inbox: asyncio.Queue[Mapping[str, Any] | Exception] = asyncio.Queue()

    async def subscribe(self, *_channels: str) -> None:
        return None

    async def unsubscribe(self, *_channels: str) -> None:
        return None

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0
    ) -> Mapping[str, Any] | None:
        assert ignore_subscribe_messages
        try:
            item = await asyncio.wait_for(self.inbox.get(), timeout or 0.001)
        except TimeoutError:
            return None
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        return None


class _FlakyRedis:
    def __init__(self) -> None:
        self.pubsub_client = _FlakyPubSub()

    async def publish(self, _channel: str, _message: str) -> int:
        return 0

    def pubsub(self) -> _FlakyPubSub:
        return self.pubsub_client

    async def aclose(self) -> None:
        return None


def thought_message(seq: int) -> dict[str, Any]:
    event = ThoughtEvent(
        origin="elsewhere",
        session_id="s1",
        seq=seq,
        thought=AgentThoughtStream(
            agent_id="agent-coder", role=AgentRole.FRONTEND_CODER, thought="Hi", timestamp_ms=1
        ),
    )
    channel = session_channel("s1").encode()
    return {"type": "message", "channel": channel, "data": event.model_dump_json().encode()}


async def test_the_bus_reader_survives_an_unexpected_error_and_a_failing_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reader is a task nobody awaits: an uncaught error used to end delivery silently."""
    client = _FlakyRedis()
    bus = RedisEventBus(client, retry_delay_s=0)
    received: list[int] = []

    def failing(_event: Any) -> None:
        raise RuntimeError("handler bug")

    def recording(event: Any) -> None:
        received.append(event.seq)

    channel = session_channel("s1")
    await bus.subscribe(channel, failing)
    await bus.subscribe(channel, recording)
    with caplog.at_level(logging.ERROR):
        client.pubsub_client.inbox.put_nowait(RuntimeError("unexpected"))
        client.pubsub_client.inbox.put_nowait(thought_message(0))
        client.pubsub_client.inbox.put_nowait(thought_message(1))
        async with asyncio.timeout(2):
            while len(received) < 2:
                await asyncio.sleep(0.001)
    assert received == [0, 1]
    messages = [record.getMessage() for record in caplog.records]
    assert any("unexpected error" in message for message in messages)
    assert sum("event handler failed" in message for message in messages) == 2
    await bus.close()


# --- observability --------------------------------------------------------------------------


async def test_every_response_carries_a_request_id_and_problems_carry_it_as_trace_id() -> None:
    async with gateway() as harness:
        supplied = await harness.http.get(
            f"{SESSIONS}/missing", headers={**harness.auth(), "X-Request-ID": "client-42"}
        )
        generated = await harness.http.get(f"{SESSIONS}/missing", headers=harness.auth())
        forged = await harness.http.get(
            f"{SESSIONS}/missing",
            headers={**harness.auth(), "X-Request-ID": "bad id\\nINFO forged log line"},
        )
        refused_body = await harness.http.post(
            SESSIONS, content=b"x" * 11_000_000, headers=harness.auth()
        )
    assert supplied.headers["x-request-id"] == supplied.json()["trace_id"] == "client-42"
    assert re.fullmatch(r"[0-9a-f]{32}", generated.headers["x-request-id"])
    assert generated.json()["trace_id"] == generated.headers["x-request-id"]
    assert forged.headers["x-request-id"] != "bad id\\nINFO forged log line"
    assert refused_body.status_code == 413
    assert refused_body.json()["trace_id"] == refused_body.headers["x-request-id"]


async def test_log_lines_from_a_socket_carry_the_id_of_the_handshake_that_opened_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with gateway(settings(ws_catchup_deadline_ms=30)) as harness:
        session_id = await harness.create_session(OWNER)
        stall = asyncio.Event()
        sent = 0

        async def gate(_message: Message) -> None:
            nonlocal sent
            sent += 1
            if sent > 1:
                await stall.wait()

        socket = harness.socket(session_id, send_gate=gate)
        socket._scope["headers"] = [(b"x-request-id", b"handshake-7")]
        with caplog.at_level(logging.WARNING):
            await socket.connect()
            assert (await socket.closed())[0] == 1008
        stall.set()
        evicted = [r for r in caplog.records if "slow consumer" in r.getMessage()]
        assert evicted
        assert evicted[0].request_id == "handshake-7"  # type: ignore[attr-defined]
        assert harness.service.metrics.closes["Slow Consumer"] == 1


def test_log_correlation_is_installed_once_and_marks_records_outside_requests() -> None:
    install_log_correlation()
    factory = logging.getLogRecordFactory()
    install_log_correlation()
    assert logging.getLogRecordFactory() is factory
    record = logging.getLogger("x").makeRecord("x", logging.INFO, "f", 1, "m", (), None)
    assert record.request_id == "-"  # type: ignore[attr-defined]


async def test_request_ids_leave_other_scopes_alone() -> None:
    seen: list[str] = []

    async def inner(scope: Scope, _receive: Receive, _send: Send) -> None:
        seen.append(scope["type"])

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(_message: Message) -> None:
        return None

    await RequestIdMiddleware(inner)({"type": "lifespan"}, receive, send)
    assert seen == ["lifespan"]


async def test_health_needs_no_token_and_metrics_need_one() -> None:
    async with gateway(settings(gateway_max_sessions_per_user=1)) as harness:
        session_id = await harness.create_session(OWNER)
        await harness.http.post(SESSIONS, headers=harness.auth())
        health = await harness.http.get("/healthz")
        anonymous = await harness.http.get("/metrics")
        socket = harness.socket(session_id)
        await socket.connect()
        await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        await socket.closed()
        metrics = await harness.http.get("/metrics", headers=harness.auth())
    assert (health.status_code, health.json()) == (200, {"status": "ok"})
    assert anonymous.status_code == 401
    assert metrics.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = metrics.text
    for line in (
        "# TYPE rasb_sessions gauge",
        "rasb_sessions 0",
        "rasb_connections 0",
        "rasb_runs_active 0",
        "rasb_exports_in_flight 0",
        "rasb_replay_history_bytes 0",
        "# TYPE rasb_ws_closed_total counter",
        'rasb_ws_closed_total{reason="Session closed"} 1',
        'rasb_refused_total{limit="sessions_per_user"} 1',
    ):
        assert line in text.splitlines(), line
    assert json.dumps(text)  # The exposition is plain text a scraper can read as is.
