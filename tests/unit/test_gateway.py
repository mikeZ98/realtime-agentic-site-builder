import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel, SecretStr, ValidationError
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from realtime_agentic_site_builder.agents.provider import (
    LLMClient,
    LLMRequest,
    LLMResult,
    MockLLMProvider,
)
from realtime_agentic_site_builder.agents.schemas import SwarmPlan
from realtime_agentic_site_builder.config import Settings, get_settings
from realtime_agentic_site_builder.core.hub import (
    InMemoryEventBus,
    PresenceEvent,
    SwarmSessionHub,
    ThoughtEvent,
    session_channel,
)
from realtime_agentic_site_builder.domain.backend import (
    ApiEndpointNode,
    BackendBlueprint,
    DatabaseDialect,
    DataModelSchema,
    FullstackProjectBundle,
    HttpMethod,
)
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    ComponentType,
    SiteBlueprint,
)
from realtime_agentic_site_builder.domain.errors import (
    PROBLEM_JSON_MEDIA_TYPE,
    AuthenticationRequiredError,
    GatewayConfigurationError,
    LLMConfigurationError,
    LLMProviderError,
    SessionNotFoundError,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentRole,
    AgentThoughtStream,
    LockState,
    StreamDelta,
)
from realtime_agentic_site_builder.gateway.auth import (
    SUBPROTOCOL,
    TOKEN_SUBPROTOCOL_PREFIX,
    Identity,
    TokenAuthenticator,
    token_from_header,
    token_from_subprotocols,
)
from realtime_agentic_site_builder.gateway.protocol import client_request_id
from realtime_agentic_site_builder.gateway.service import GatewayService, create_app

SECRET = SecretStr("gateway-signing-key-for-unit-tests-only")
WAIT_S = 2.0
ALICE = "alice"
BOB = "bob"
SESSIONS = "/api/v1/sessions"
Message = dict[str, Any]


def make_settings(**overrides: Any) -> Settings:
    return Settings(**{"gateway_secret": SECRET, "broadcast_fps": 240, **overrides})


def make_thought(text: str = "Thinking") -> AgentThoughtStream:
    return AgentThoughtStream(
        agent_id="agent-coder",
        role=AgentRole.FRONTEND_CODER,
        thought=text,
        tool_name="llm.generate",
        timestamp_ms=1,
    )


def site_bundle(*components: ComponentNode) -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id="project-1",
        frontend=SiteBlueprint(site_id="site-1", title="Acme", theme="dark", components=components),
        backend=BackendBlueprint(service_name="api", database_dialect=DatabaseDialect.SQLITE),
    )


def hero() -> ComponentNode:
    return ComponentNode.create(
        id="hero",
        component_type=ComponentType.HERO,
        title="Welcome",
        code="<Hero />",
        box=BoundingBox(x=0, y=0, width=100, height=50),
    )


def bundle_json(*components: ComponentNode) -> dict[str, Any]:
    return site_bundle(*components).model_dump(mode="json")


@dataclass
class Harness:
    app: FastAPI
    service: GatewayService
    http: httpx.AsyncClient

    def token(self, user_id: str = ALICE, ttl_seconds: int = 3600) -> str:
        return self.service.authenticator.issue(user_id, ttl_seconds)

    def auth(self, user_id: str = ALICE) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(user_id)}"}

    async def create_session(self, *, user_id: str = ALICE, **body: Any) -> str:
        response = await self.http.post(SESSIONS, json=body, headers=self.auth(user_id))
        assert response.status_code == 201, response.text
        session_id: str = response.json()["session_id"]
        return session_id

    def socket(
        self,
        session_id: str,
        *,
        user_id: str = ALICE,
        last_seen: int | None = None,
        send_gate: Callable[[Message], Awaitable[None]] | None = None,
    ) -> "SocketDriver":
        query = "" if last_seen is None else f"last_seen_thought_seq={last_seen}"
        return SocketDriver(
            self.app,
            f"/ws/sessions/{session_id}",
            query=query,
            subprotocols=[SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}{self.token(user_id)}"],
            send_gate=send_gate,
        )


@asynccontextmanager
async def running(
    settings: Settings | None = None, *, provider: Any = None, **options: Any
) -> AsyncIterator[Harness]:
    app = create_app(settings or make_settings(), provider=provider or MockLLMProvider(), **options)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://gateway") as http,
    ):
        yield Harness(app, app.state.gateway, http)


class SocketDriver:
    """Speaks ASGI to the application directly, in this event loop, with no socket involved.

    That leaves everything a real client controls in the test's hands, above all when ``send``
    returns, which is how a stalled TCP connection looks from inside the server.
    """

    def __init__(
        self,
        app: FastAPI,
        path: str,
        *,
        query: str = "",
        subprotocols: list[str] | None = None,
        headers: list[tuple[bytes, bytes]] | None = None,
        send_gate: Callable[[Message], Awaitable[None]] | None = None,
    ) -> None:
        self._app = app
        self._scope: dict[str, Any] = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "http_version": "1.1",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": query.encode(),
            "headers": headers or [],
            "subprotocols": subprotocols or [],
            "client": ("client", 1),
            "server": ("gateway", 80),
        }
        self._to_app: asyncio.Queue[Message] = asyncio.Queue()
        self.from_app: asyncio.Queue[Message] = asyncio.Queue()
        self._send_gate = send_gate
        self.accepted_subprotocol: str | None = None
        self.task: asyncio.Task[None] | None = None

    async def _receive(self) -> Message:
        return await self._to_app.get()

    async def _send(self, message: Message) -> None:
        if self._send_gate is not None and message["type"] == "websocket.send":
            await self._send_gate(message)
        await self.from_app.put(message)

    async def _serve(self) -> None:
        await self._app(self._scope, cast("Any", self._receive), cast("Any", self._send))

    async def open(self) -> Message:
        """Start the handshake, and return the server's answer to it."""
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(self._serve())
        answer = await asyncio.wait_for(self.from_app.get(), WAIT_S)
        self.accepted_subprotocol = answer.get("subprotocol")
        return answer

    async def connect(self) -> dict[str, Any]:
        answer = await self.open()
        assert answer["type"] == "websocket.accept", answer
        return await self.receive()

    async def next_message(self) -> Message:
        return await asyncio.wait_for(self.from_app.get(), WAIT_S)

    async def receive(self) -> dict[str, Any]:
        message = await self.next_message()
        assert message["type"] == "websocket.send", message
        parsed: dict[str, Any] = json.loads(message["text"])
        return parsed

    async def receive_kind(self, *kinds: str) -> dict[str, Any]:
        """Skip the steady stream of cursor frames, up to the next frame of a wanted kind."""
        while True:
            frame = await self.receive()
            if frame["kind"] in kinds:
                return frame

    def send(self, payload: dict[str, Any]) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    def send_raw(self, **message: Any) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", **message})

    async def closed(self) -> tuple[int, str]:
        """Wait for the server to close, skipping what it sends first."""
        while True:
            message = await self.next_message()
            if message["type"] == "websocket.close":
                return message["code"], message.get("reason", "")

    async def finished(self) -> None:
        assert self.task is not None
        await asyncio.wait_for(self.task, WAIT_S)

    async def disconnect(self) -> None:
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1000})
        await self.finished()


async def wait_until(condition: Callable[[], Any], timeout: float = WAIT_S) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout)


# --- tokens ---------------------------------------------------------------------------------


def encode_claims(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def authenticator(now_ms: int = 1_000_000) -> TokenAuthenticator:
    return TokenAuthenticator(SECRET, clock=lambda: now_ms)


def test_token_roundtrip_yields_a_namespaced_identity() -> None:
    identity = authenticator().verify(authenticator().issue(ALICE, 60))
    assert identity == Identity(user_id=ALICE, expires_at_s=1060)
    assert identity.agent_id == "user:alice"


def test_people_can_never_share_an_agents_identity() -> None:
    identity = authenticator().verify(authenticator().issue("agent-coder", 60))
    assert identity.agent_id == "user:agent-coder"


@pytest.mark.parametrize(
    "mangle",
    [
        lambda token: token[:-2],
        lambda token: token + "x",
        lambda token: token.replace(".", ""),
        lambda _token: "",
        lambda _token: "garbage",
        lambda _token: ".",
        lambda token: "e30." + token.split(".")[1],
    ],
)
def test_tampered_tokens_are_rejected(mangle: Callable[[str], str]) -> None:
    with pytest.raises(AuthenticationRequiredError, match="not valid"):
        authenticator().verify(mangle(authenticator().issue(ALICE)))


def test_token_from_another_secret_is_rejected() -> None:
    other = TokenAuthenticator(SecretStr("a-completely-different-signing-key-value"))
    with pytest.raises(AuthenticationRequiredError, match="not valid"):
        authenticator().verify(other.issue(ALICE))


def test_expired_token_is_rejected_at_the_exact_second() -> None:
    token = authenticator(1_000_000).issue(ALICE, 60)
    assert authenticator(1_059_999).verify(token).user_id == ALICE
    with pytest.raises(AuthenticationRequiredError, match="expired"):
        authenticator(1_060_000).verify(token)


@pytest.mark.parametrize(
    "claims",
    [
        b"not json",
        b"[]",
        b"{}",
        b'{"sub":"alice"}',
        b'{"exp":9999999999}',
        b'{"sub":"bad id!","exp":9999999999}',
        b'{"sub":5,"exp":9999999999}',
    ],
)
def test_correctly_signed_nonsense_is_still_rejected(claims: bytes) -> None:
    subject = authenticator()
    payload = encode_claims(claims)
    token = f"{payload}.{subject._signature(payload)}"
    with pytest.raises(AuthenticationRequiredError, match="not valid"):
        subject.verify(token)


@pytest.mark.parametrize("expiry", ['"soon"', "1.5", "null"])
def test_correctly_signed_bad_expiry_is_rejected(expiry: str) -> None:
    subject = authenticator()
    payload = encode_claims(f'{{"sub":"alice","exp":{expiry}}}'.encode())
    with pytest.raises(AuthenticationRequiredError, match="expired"):
        subject.verify(f"{payload}.{subject._signature(payload)}")


@pytest.mark.parametrize("user_id", ["bad id!", "", "x" * 121])
def test_issuing_rejects_unusable_user_ids(user_id: str) -> None:
    with pytest.raises(ValueError):
        authenticator().issue(user_id)


@pytest.mark.parametrize(
    "header, expected",
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("BEARER   abc  ", "abc"),
        ("Basic abc", None),
        ("Bearer", None),
        ("Bearer   ", None),
        ("", None),
        (None, None),
    ],
)
def test_token_from_header(header: str | None, expected: str | None) -> None:
    assert token_from_header(header) == expected


def test_token_from_subprotocols() -> None:
    assert token_from_subprotocols(["rasb.v1", "rasb.token.abc"]) == "abc"
    assert token_from_subprotocols(["rasb.v1"]) is None
    assert token_from_subprotocols([]) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"request_id": "r1"}, "r1"),
        ({"request_id": 5}, None),
        ({"request_id": "x" * 65}, None),
        ({}, None),
        ([1], None),
    ],
)
def test_request_id_extraction_trusts_nothing(raw: Any, expected: str | None) -> None:
    assert client_request_id(raw) == expected


# --- settings and startup -------------------------------------------------------------------


def test_short_gateway_secret_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(gateway_secret=SecretStr("short"))


async def test_gateway_refuses_to_start_without_a_secret() -> None:
    app = create_app(Settings(), provider=MockLLMProvider())
    with pytest.raises(GatewayConfigurationError, match="RASB_GATEWAY_SECRET"):
        async with app.router.lifespan_context(app):
            pass


async def test_gateway_refuses_to_start_without_model_keys() -> None:
    app = create_app(make_settings())
    with pytest.raises(LLMConfigurationError):
        async with app.router.lifespan_context(app):
            pass


async def test_gateway_builds_and_closes_its_own_provider() -> None:
    settings = make_settings(anthropic_api_key=SecretStr("k"), openrouter_api_key=SecretStr("k"))
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        service: GatewayService = app.state.gateway
        assert service.provider is not None


def test_create_app_falls_back_to_process_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("RASB_MAX_REQUEST_BYTES", "1234")
    assert isinstance(create_app(), FastAPI)
    get_settings.cache_clear()


async def test_openapi_document_generates_for_every_route() -> None:
    async with running() as harness:
        response = await harness.http.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {SESSIONS, f"{SESSIONS}/{{session_id}}", f"{SESSIONS}/{{session_id}}/timeline"} <= set(
        paths
    )


# --- background automation ------------------------------------------------------------------


async def test_shutdown_leaves_no_task_behind() -> None:
    before = asyncio.all_tasks()
    async with running() as harness:
        await harness.create_session(prompt="Build it")
        await harness.create_session()
    await asyncio.sleep(0)
    assert asyncio.all_tasks() - before == set()


async def test_sweeper_expires_idle_sessions_stops_their_tickers_and_ends_their_sockets() -> None:
    clock_now = [1_000_000_000]
    async with running(
        make_settings(session_ttl_seconds=10, gateway_sweep_interval_seconds=0.005),
        hub_clock=lambda: clock_now[0],
    ) as harness:
        session_id = await harness.create_session()
        ticker = harness.service._sessions[session_id].ticker
        assert ticker.running
        socket = harness.socket(session_id)
        await socket.connect()

        clock_now[0] += 10_000

        assert await socket.closed() == (1000, "Session closed")
        await wait_until(lambda: session_id not in harness.service._sessions)
        assert not ticker.running
        await socket.finished()


async def test_ticker_starts_with_a_session_and_stops_when_it_is_deleted() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        ticker = harness.service._sessions[session_id].ticker
        await wait_until(lambda: ticker.ticks > 0)
        await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert not ticker.running


async def test_a_failed_sweep_is_logged_and_sweeping_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async with running(make_settings(gateway_sweep_interval_seconds=0.001)) as harness:
        original = harness.service.hub.sweep

        async def flaky(now_ms: int | None = None) -> tuple[str, ...]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("sweep bug")
            return await original(now_ms)

        harness.service.hub.sweep = flaky  # type: ignore[method-assign]
        with caplog.at_level(logging.ERROR):
            await wait_until(lambda: calls >= 3)
    assert "session sweep failed" in caplog.text


async def test_sweeper_tolerates_a_session_it_does_not_track() -> None:
    async with running() as harness:
        await harness.service._retire("never-registered")


async def test_shutdown_drains_buffered_events_then_closes_going_away() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        await harness.service.hub.broadcast_thought(session_id, make_thought("last words"))
        connection = next(iter(harness.service._connections))
        connection.drain()
        assert (await socket.receive_kind("thought"))["thought"]["thought"] == "last words"
        assert await socket.closed() == (1001, "Server shutting down")
        await socket.finished()


async def test_real_shutdown_delivers_what_is_buffered_then_closes_going_away() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        await harness.service.hub.broadcast_thought(session_id, make_thought("final words"))
    # The lifespan has now exited, so this is what really reached the client on the way down.
    received = []
    while not socket.from_app.empty():
        received.append(socket.from_app.get_nowait())
    assert received[-1]["type"] == "websocket.close"
    assert received[-1]["code"] == 1001
    sent = [json.loads(item["text"]) for item in received[:-1]]
    assert "final words" in [
        frame["thought"]["thought"] for frame in sent if frame["kind"] == "thought"
    ]


async def test_a_stalled_client_cannot_hold_up_shutdown() -> None:
    hang = asyncio.Event()

    async def stall(_message: Message) -> None:
        await hang.wait()

    settings = make_settings(gateway_drain_timeout_seconds=0.05, ws_catchup_deadline_ms=60_000)
    async with running(settings) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=stall)
        await socket.open()
        shutdown_started = asyncio.get_running_loop().time()
    assert asyncio.get_running_loop().time() - shutdown_started < 3
    assert socket.task is not None
    socket.task.cancel()
    await asyncio.gather(socket.task, return_exceptions=True)


# --- REST: sessions -------------------------------------------------------------------------


async def test_create_session_without_a_body_gets_an_empty_site() -> None:
    async with running() as harness:
        response = await harness.http.post(SESSIONS, headers=harness.auth())
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"].startswith("s-")
    assert response.headers["location"] == f"{SESSIONS}/{body['session_id']}"
    assert body["run_started"] is False
    assert body["snapshot"]["frame_seq"] == 0
    assert body["snapshot"]["bundle_state"]["frontend"]["components"] == []


async def test_create_session_uses_a_supplied_bundle() -> None:
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"bundle": bundle_json(hero())}, headers=harness.auth()
        )
    state = response.json()["snapshot"]["bundle_state"]
    assert [component["id"] for component in state["frontend"]["components"]] == ["hero"]


async def test_session_ids_are_chosen_by_the_server() -> None:
    async with running() as harness:
        first = await harness.create_session()
        second = await harness.create_session()
        spoofed = await harness.http.post(
            SESSIONS, json={"session_id": "mine"}, headers=harness.auth()
        )
    assert first != second
    assert spoofed.status_code == 422


async def test_create_session_with_a_prompt_starts_a_run() -> None:
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"prompt": "Build a landing page"}, headers=harness.auth()
        )
        assert response.json()["run_started"] is True
        session_id = response.json()["session_id"]
        await wait_until(lambda: harness.service._sessions[session_id].run is None, 5)
        snapshot = (
            await harness.http.get(f"{SESSIONS}/{session_id}", headers=harness.auth())
        ).json()
    assert snapshot["bundle_state"]["frontend"]["components"]
    assert snapshot["frame_seq"] > 0


async def test_get_session_returns_the_latest_snapshot() -> None:
    async with running() as harness:
        session_id = await harness.create_session(bundle=bundle_json(hero()))
        response = await harness.http.get(f"{SESSIONS}/{session_id}", headers=harness.auth(BOB))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json()["bundle_hash"] == site_bundle(hero()).state_hash()


async def test_delete_session_closes_it_for_good() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        response = await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert response.status_code == 204
        assert response.content == b""
        gone = await harness.http.get(f"{SESSIONS}/{session_id}", headers=harness.auth())
        again = await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
    assert gone.status_code == 404
    assert again.status_code == 404


async def test_only_the_creator_can_delete_a_session() -> None:
    async with running() as harness:
        session_id = await harness.create_session(user_id=ALICE)
        response = await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth(BOB))
        still_there = await harness.http.get(f"{SESSIONS}/{session_id}", headers=harness.auth(BOB))
    assert response.status_code == 403
    assert response.json()["type"].endswith("/permission-denied")
    assert still_there.status_code == 200


async def test_deleting_a_session_cancels_its_run() -> None:
    release = asyncio.Event()

    async def hold(_role: AgentRole, _request: LLMRequest[Any]) -> None:
        await release.wait()

    async with running(provider=HookedProvider(MockLLMProvider(), hold)) as harness:
        response = await harness.http.post(
            SESSIONS, json={"prompt": "Build it"}, headers=harness.auth()
        )
        session_id = response.json()["session_id"]
        run = harness.service._sessions[session_id].run
        assert run is not None
        await asyncio.sleep(0)
        await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert run.cancelled()


async def test_session_limit_is_enforced() -> None:
    async with running(make_settings(gateway_max_sessions=1)) as harness:
        await harness.create_session()
        response = await harness.http.post(SESSIONS, headers=harness.auth())
    assert response.status_code == 429
    assert response.json()["type"].endswith("/capacity-exceeded")


# --- REST: timeline -------------------------------------------------------------------------


async def test_timeline_returns_state_history_and_thought_backlog() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        hub = harness.service.hub
        for index in range(3):
            await hub.broadcast_thought(session_id, make_thought(f"t{index}"))
        response = await harness.http.get(
            f"{SESSIONS}/{session_id}/timeline", headers=harness.auth()
        )
    assert response.status_code == 200
    body = response.json()
    assert body["timeline"]["session_id"] == session_id
    assert body["timeline"]["snapshots"][0]["frame_seq"] == 0
    assert [event["seq"] for event in body["thoughts"]] == [0, 1, 2]
    assert body["thought_gap"] is False
    assert body["latest_thought_seq"] == 2


async def test_timeline_of_a_full_run_is_complete() -> None:
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"prompt": "Build it"}, headers=harness.auth()
        )
        session_id = response.json()["session_id"]
        await wait_until(lambda: harness.service._sessions[session_id].run is None, 5)
        body = (
            await harness.http.get(f"{SESSIONS}/{session_id}/timeline", headers=harness.auth())
        ).json()
    assert body["timeline"]["deltas"]
    assert len(body["timeline"]["snapshots"]) == len(body["timeline"]["deltas"]) + 1
    assert body["thoughts"][-1]["thought"]["tool_name"] == "run.completed"
    assert body["thoughts"][-1]["thought"]["tool_args"]["completed"] is True


async def test_timeline_reports_a_truncated_backlog() -> None:
    async with running(make_settings(thought_backlog_size=2)) as harness:
        session_id = await harness.create_session()
        for index in range(5):
            await harness.service.hub.broadcast_thought(session_id, make_thought(f"t{index}"))
        body = (
            await harness.http.get(f"{SESSIONS}/{session_id}/timeline", headers=harness.auth())
        ).json()
    assert [event["seq"] for event in body["thoughts"]] == [3, 4]
    assert body["thought_gap"] is True
    assert body["latest_thought_seq"] == 4


async def test_timeline_of_an_unknown_session_is_not_found() -> None:
    async with running() as harness:
        response = await harness.http.get(f"{SESSIONS}/missing/timeline", headers=harness.auth())
    assert response.status_code == 404


# --- REST: authentication and errors --------------------------------------------------------


def assert_problem(response: httpx.Response, status: int) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_JSON_MEDIA_TYPE
    problem: dict[str, Any] = response.json()
    assert problem["status"] == status
    assert problem["title"]
    return problem


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", SESSIONS),
        ("GET", f"{SESSIONS}/any"),
        ("GET", f"{SESSIONS}/any/timeline"),
        ("DELETE", f"{SESSIONS}/any"),
    ],
)
async def test_every_route_requires_a_token(method: str, path: str) -> None:
    async with running() as harness:
        response = await harness.http.request(method, path)
    problem = assert_problem(response, 401)
    assert problem["type"].endswith("/authentication-required")
    assert response.headers["www-authenticate"] == "Bearer"


async def test_bad_token_is_rejected() -> None:
    async with running() as harness:
        response = await harness.http.get(
            f"{SESSIONS}/any", headers={"Authorization": "Bearer forged.token"}
        )
    assert_problem(response, 401)


async def test_authentication_is_checked_before_the_body_is_judged() -> None:
    async with running() as harness:
        response = await harness.http.post(SESSIONS, content=b"{not json")
    assert_problem(response, 401)


async def test_validation_failure_maps_to_a_problem_without_echoing_the_input() -> None:
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS,
            json={"prompt": "ok", "undeclared": "do-not-echo-me"},
            headers=harness.auth(),
        )
    problem = assert_problem(response, 422)
    assert problem["errors"][0]["type"] == "extra_forbidden"
    assert problem["instance"] == SESSIONS
    assert "do-not-echo-me" not in response.text


async def test_malformed_json_maps_to_a_problem() -> None:
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS,
            content=b"{not json",
            headers={**harness.auth(), "Content-Type": "application/json"},
        )
    assert assert_problem(response, 422)["errors"][0]["type"] == "json_invalid"


@pytest.mark.parametrize("prompt", ["", "   ", "x" * 4097])
async def test_unusable_prompts_are_rejected(prompt: str) -> None:
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"prompt": prompt}, headers=harness.auth()
        )
    assert_problem(response, 422)


async def test_invalid_bundle_maps_to_a_problem() -> None:
    bundle = bundle_json(hero())
    bundle["frontend"]["components"][0]["state_checksum"] = "0" * 64
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"bundle": bundle}, headers=harness.auth()
        )
    assert_problem(response, 422)


async def test_domain_errors_raised_during_validation_keep_their_own_status() -> None:
    bundle = bundle_json(hero())
    bundle["frontend"]["components"][0]["connected_endpoint_id"] = "api_endpoint_get_none"
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"bundle": bundle}, headers=harness.auth()
        )
    problem = assert_problem(response, 400)
    assert problem["type"].endswith("/endpoint-binding-not-found")
    assert problem["instance"] == SESSIONS


async def test_uncompilable_handler_in_a_bundle_maps_to_its_problem() -> None:
    model = DataModelSchema.model_validate(
        {"name": "Lead", "fields": {"email": {"python_type": "EmailStr"}}, "table_name": "leads"}
    )
    endpoint = ApiEndpointNode.create(
        path="/api/v1/leads",
        method=HttpMethod.POST,
        summary="Create",
        handler_code="async def handler() -> None:\n    return None",
        request_model_ref="Lead",
    )
    bundle = site_bundle().model_copy(
        update={
            "backend": BackendBlueprint(
                service_name="api",
                database_dialect=DatabaseDialect.SQLITE,
                models=(model,),
                endpoints=(endpoint,),
            )
        }
    )
    payload = bundle.model_dump(mode="json")
    payload["backend"]["endpoints"][0]["handler_code"] = "async def broken(:\n    pass"
    async with running() as harness:
        response = await harness.http.post(
            SESSIONS, json={"bundle": payload}, headers=harness.auth()
        )
    assert assert_problem(response, 422)["type"].endswith("/ast-syntax-compilation-failure")


async def test_unknown_session_maps_to_a_problem() -> None:
    async with running() as harness:
        response = await harness.http.get(f"{SESSIONS}/missing", headers=harness.auth())
    problem = assert_problem(response, 404)
    assert problem["type"].endswith("/session-not-found")
    assert problem["instance"] == f"{SESSIONS}/missing"


async def test_unknown_route_and_wrong_method_are_problems_too() -> None:
    async with running() as harness:
        missing = await harness.http.get("/nope")
        wrong = await harness.http.put(SESSIONS)
    assert assert_problem(missing, 404)["title"] == "Not Found"
    assert_problem(wrong, 405)
    assert "POST" in wrong.headers["allow"]


async def test_unexpected_failure_reveals_nothing() -> None:
    async with running() as harness:

        def explode(_session_id: str) -> None:
            raise RuntimeError("database password is hunter2")

        cast("Any", harness.service.hub).latest_snapshot = explode
        response = await harness.http.get(f"{SESSIONS}/any", headers=harness.auth())
    assert_problem(response, 500)
    assert "hunter2" not in response.text


async def test_oversized_body_is_refused_from_its_declared_length() -> None:
    async with running(make_settings(max_request_bytes=100)) as harness:
        response = await harness.http.post(SESSIONS, content=b"x" * 101, headers=harness.auth())
    assert_problem(response, 413)


async def test_declared_oversize_is_refused_without_reading_any_of_the_body() -> None:
    async with running(make_settings(max_request_bytes=100)) as harness:
        messages: list[Message] = []

        async def receive() -> Message:
            raise AssertionError("the body must not be read")

        async def send(message: Message) -> None:
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": SESSIONS,
            "raw_path": SESSIONS.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-length", b"101")],
            "client": ("client", 1),
            "server": ("gateway", 80),
        }
        await harness.app(scope, cast("Any", receive), cast("Any", send))
    assert messages[0]["status"] == 413


async def test_oversized_body_is_refused_while_streaming_when_undeclared() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(5):
            yield b"x" * 60

    async with running(make_settings(max_request_bytes=100)) as harness:
        response = await harness.http.post(SESSIONS, content=chunks(), headers=harness.auth())
    assert_problem(response, 413)


async def test_body_at_the_limit_is_read_normally() -> None:
    body = json.dumps({"prompt": "x"}).encode()
    async with running(make_settings(max_request_bytes=len(body))) as harness:
        response = await harness.http.post(
            SESSIONS,
            content=body,
            headers={**harness.auth(), "Content-Type": "application/json"},
        )
    assert response.status_code == 201


async def test_non_numeric_content_length_falls_back_to_counting() -> None:
    async with running(make_settings(max_request_bytes=100)) as harness:
        scope_headers = [(b"content-length", b"banana"), (b"host", b"gateway")]
        messages: list[Message] = []

        async def receive() -> Message:
            return {"type": "http.request", "body": b"x" * 101, "more_body": False}

        async def send(message: Message) -> None:
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": SESSIONS,
            "raw_path": SESSIONS.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [
                *scope_headers,
                (b"authorization", f"Bearer {harness.token()}".encode()),
                (b"content-type", b"application/json"),
            ],
            "client": ("client", 1),
            "server": ("gateway", 80),
        }
        await harness.app(scope, receive, cast("Any", send))
    assert messages[0]["status"] == 413


# --- socket: handshake and identity ---------------------------------------------------------


async def test_socket_without_a_token_is_refused_before_it_is_accepted() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = SocketDriver(harness.app, f"/ws/sessions/{session_id}")
        answer = await socket.open()
    assert answer["type"] == "websocket.close"
    assert answer["code"] == 1008


async def test_socket_with_a_forged_token_is_refused() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = SocketDriver(
            harness.app,
            f"/ws/sessions/{session_id}",
            subprotocols=[SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}forged"],
        )
        answer = await socket.open()
    assert answer["type"] == "websocket.close"


async def test_browser_style_handshake_echoes_only_the_protocol_never_the_token() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        assert socket.accepted_subprotocol == SUBPROTOCOL
        await socket.disconnect()


async def test_header_token_works_for_non_browser_clients() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = SocketDriver(
            harness.app,
            f"/ws/sessions/{session_id}",
            headers=[(b"authorization", f"Bearer {harness.token(BOB)}".encode())],
        )
        sync = await socket.connect()
        assert socket.accepted_subprotocol is None
        assert sync["user_id"] == BOB
        await socket.disconnect()


async def test_socket_for_an_unknown_session_gets_a_distinct_close_code() -> None:
    async with running() as harness:
        socket = harness.socket("missing")
        assert (await socket.open())["type"] == "websocket.accept"
        assert await socket.closed() == (4404, "Session not found")


async def test_invalid_handshake_parameter_is_refused() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = SocketDriver(
            harness.app,
            f"/ws/sessions/{session_id}",
            query="last_seen_thought_seq=-1",
            subprotocols=[SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}{harness.token()}"],
        )
        answer = await socket.open()
    assert answer["type"] == "websocket.close"
    assert answer["code"] == 1008


async def test_first_frame_is_a_snapshot_with_the_verified_identity() -> None:
    async with running() as harness:
        session_id = await harness.create_session(bundle=bundle_json(hero()))
        socket = harness.socket(session_id, user_id=BOB)
        sync = await socket.connect()
        assert sync["kind"] == "sync"
        assert sync["session_id"] == session_id
        assert sync["user_id"] == BOB
        assert sync["agent_id"] == "user:bob"
        assert sync["snapshot"]["bundle_hash"] == site_bundle(hero()).state_hash()
        assert sync["latest_thought_seq"] is None
        assert sync["replayed_thoughts"] == 0
        assert sync["thought_gap"] is False
        await socket.disconnect()


@pytest.mark.parametrize(
    "spoof",
    [
        {"type": "cursor", "x": 1, "y": 2, "user_id": "victim"},
        {"type": "cursor", "x": 1, "y": 2, "agent_id": "agent-architect"},
        {
            "type": "delta",
            "agent_id": "agent-architect",
            "target_domain": "frontend",
            "target_id": "hero",
            "patches": [{"op": "test", "path": "/frontend/title", "value": "Acme"}],
        },
        {
            "type": "delta",
            "session_id": "other",
            "target_domain": "frontend",
            "target_id": "hero",
            "patches": [{"op": "test", "path": "/frontend/title", "value": "Acme"}],
        },
        {
            "type": "delta",
            "delta_seq": 99,
            "target_domain": "frontend",
            "target_id": "hero",
            "patches": [{"op": "test", "path": "/frontend/title", "value": "Acme"}],
        },
        {"type": "lock.acquire", "target_id": "hero", "priority": 100},
        {"type": "lock.acquire", "target_id": "hero", "agent_id": "agent-architect"},
        {"type": "prompt", "text": "x", "user_id": "victim"},
    ],
)
async def test_a_client_cannot_claim_an_identity_or_a_privilege(spoof: dict[str, Any]) -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({**spoof, "request_id": "r1"})
        frame = await socket.receive_kind("error")
        assert frame["request_id"] == "r1"
        assert frame["problem"]["status"] == 422
        assert harness.service.hub.lock_manager(session_id).active_locks() == ()
        await socket.disconnect()


# --- socket: streaming ----------------------------------------------------------------------


async def test_cursor_and_thought_events_stream_after_the_snapshot() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()

        cursor = await socket.receive_kind("cursor")
        assert cursor["frame"]["agent_id"].startswith("agent-")

        await harness.service.hub.broadcast_thought(session_id, make_thought("live"))
        thought = await socket.receive_kind("thought")
        assert thought["seq"] == 0
        assert thought["thought"]["thought"] == "live"
        assert thought["thought"]["tool_name"] == "llm.generate"
        await socket.disconnect()


async def test_ping_gets_a_pong_with_its_correlation_id() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "ping", "request_id": "beat-1"})
        pong = await socket.receive_kind("pong")
        assert pong["request_id"] == "beat-1"
        assert pong["server_time_ms"] > 0
        await socket.disconnect()


async def test_heartbeat_keeps_a_quiet_session_from_expiring() -> None:
    clock_now = [1_000_000_000]
    async with running(
        make_settings(session_ttl_seconds=10), hub_clock=lambda: clock_now[0]
    ) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        clock_now[0] += 9_000
        socket.send({"type": "ping"})
        await socket.receive_kind("pong")
        clock_now[0] += 9_000
        assert await harness.service.hub.sweep() == ()
        await socket.disconnect()


async def test_presence_reaches_others_under_the_verified_identity_only() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        alice = harness.socket(session_id, user_id=ALICE)
        bob = harness.socket(session_id, user_id=BOB)
        await alice.connect()
        await bob.connect()

        alice.send({"type": "cursor", "x": 120, "y": 340})
        presence = await bob.receive_kind("presence")
        assert (presence["user_id"], presence["x"], presence["y"]) == (ALICE, 120, 340)
        assert presence["active"] is True

        alice.send({"type": "select", "target_id": "hero"})
        selected = await bob.receive_kind("presence")
        assert selected["selected_id"] == "hero"
        assert (selected["x"], selected["y"]) == (120, 340), "selecting does not move the cursor"

        await alice.disconnect()
        left = await bob.receive_kind("presence")
        assert left["user_id"] == ALICE
        assert left["active"] is False
        await bob.disconnect()


async def test_moving_the_cursor_keeps_the_selection() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        alice = harness.socket(session_id, user_id=ALICE)
        bob = harness.socket(session_id, user_id=BOB)
        await alice.connect()
        await bob.connect()
        alice.send({"type": "select", "target_id": "hero"})
        await bob.receive_kind("presence")
        alice.send({"type": "cursor", "x": 7, "y": 8})
        moved = await bob.receive_kind("presence")
        assert (moved["x"], moved["selected_id"]) == (7, "hero")
        await alice.disconnect()
        await bob.disconnect()


async def test_a_reply_wakes_the_connection_when_nothing_else_is_happening() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await harness.service._sessions[session_id].ticker.stop()
        socket = harness.socket(session_id)
        await socket.connect()
        # With the ticker stopped, only the reply itself can get the egress loop moving again.
        await asyncio.sleep(0.02)
        while not socket.from_app.empty():
            socket.from_app.get_nowait()
        await asyncio.sleep(0)
        socket.send({"type": "ping", "request_id": "quiet"})
        assert (await socket.receive_kind("pong"))["request_id"] == "quiet"
        await socket.disconnect()


async def test_people_do_not_appear_as_agents_in_snapshots() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "cursor", "x": 1, "y": 1, "request_id": "c"})
        await socket.receive_kind("ack")
        socket.send({"type": "lock.acquire", "target_id": "hero", "request_id": "l"})
        await socket.receive_kind("ack")
        await socket.disconnect()
        hub = harness.service.hub
        assert all(not a.startswith("user:") for a in hub._require(session_id).agents)


async def test_cursors_are_acknowledged_only_when_asked() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "cursor", "x": 1, "y": 1})
        socket.send({"type": "select"})
        socket.send({"type": "ping", "request_id": "fence"})
        kinds = []
        while (frame := await socket.receive())["kind"] != "pong":
            kinds.append(frame["kind"])
        assert "ack" not in kinds
        socket.send({"type": "cursor", "x": 2, "y": 2, "request_id": "wanted"})
        assert (await socket.receive_kind("ack"))["request_id"] == "wanted"
        await socket.disconnect()


# --- socket: malformed input ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, request_id",
    [
        ("{not json", None),
        ('{"type":"unknown","request_id":"r9"}', "r9"),
        ('{"type":"cursor"}', None),
        ('{"type":"delta","target_domain":"frontend","target_id":"x","patches":[]}', None),
        ("[1,2,3]", None),
    ],
)
async def test_malformed_message_gets_an_error_and_the_connection_survives(
    text: str, request_id: str | None
) -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send_raw(text=text)
        frame = await socket.receive_kind("error")
        assert frame["problem"]["status"] == 422
        assert frame["request_id"] == request_id
        socket.send({"type": "ping"})
        await socket.receive_kind("pong")
        await socket.disconnect()


async def test_binary_frames_are_refused() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send_raw(bytes=b"\x00\x01")
        assert "Binary" in (await socket.receive_kind("error"))["problem"]["detail"]
        await socket.disconnect()


async def test_oversized_message_closes_with_message_too_big() -> None:
    async with running(make_settings(ws_max_message_bytes=100)) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send_raw(text="x" * 101)
        assert await socket.closed() == (1009, "Message too big")


async def test_size_limit_counts_bytes_not_characters() -> None:
    async with running(make_settings(ws_max_message_bytes=100)) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send_raw(text="é" * 60)
        assert (await socket.closed())[0] == 1009


async def test_message_flood_closes_the_connection() -> None:
    clock_now = [0]
    async with running(
        make_settings(ws_max_messages_per_second=5, broadcast_fps=1),
        clock=lambda: clock_now[0],
    ) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        for _ in range(6):
            socket.send({"type": "ping"})
        assert await socket.closed() == (1008, "Rate limit exceeded")


async def test_rate_limit_refills_over_time() -> None:
    clock_now = [0]
    async with running(
        make_settings(ws_max_messages_per_second=5, broadcast_fps=1),
        clock=lambda: clock_now[0],
    ) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        for _ in range(3):
            for _ in range(5):
                socket.send({"type": "ping"})
            for _ in range(5):
                await socket.receive_kind("pong")
            clock_now[0] += 1000
        await socket.disconnect()


async def test_a_client_that_floods_without_reading_replies_is_cut_off() -> None:
    hang = asyncio.Event()
    allowed = [1]

    async def stall_after_sync(_message: Message) -> None:
        if allowed[0] > 0:
            allowed[0] -= 1
            return
        await hang.wait()

    settings = make_settings(
        ws_max_messages_per_second=100_000, ws_catchup_deadline_ms=60_000, broadcast_fps=1
    )
    async with running(settings) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=stall_after_sync)
        await socket.connect()
        for _ in range(400):
            socket.send({"type": "ping"})
        await wait_until(lambda: not harness.service._connections, 3)
    hang.set()


async def test_silent_client_is_disconnected_at_the_heartbeat_timeout() -> None:
    async with running(make_settings(ws_heartbeat_timeout_seconds=0.02)) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        assert await socket.closed() == (1008, "Heartbeat timeout")


async def test_connection_ends_when_its_token_expires() -> None:
    clock_now = [1_000_000_000]
    async with running(
        make_settings(ws_heartbeat_timeout_seconds=60), hub_clock=lambda: clock_now[0]
    ) as harness:
        session_id = await harness.create_session()
        short = harness.token(ttl_seconds=1)
        socket = SocketDriver(
            harness.app,
            f"/ws/sessions/{session_id}",
            subprotocols=[SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}{short}"],
        )
        await socket.connect()
        clock_now[0] += 1_001
        # The expiry deadline was computed on the last receive, so a message prompts a re-check.
        socket.send({"type": "ping"})
        assert await socket.closed() == (1008, "Token expired")


async def test_unexpected_failure_closes_with_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()

        def explode(_session_id: str) -> None:
            raise RuntimeError("bug with a secret")

        cast("Any", harness.service.hub).touch = explode
        socket.send({"type": "ping"})
        with caplog.at_level(logging.ERROR):
            code, reason = await socket.closed()
    assert (code, reason) == (1011, "Internal error")
    assert "connection failed" in caplog.text


# --- socket: locks and deltas ---------------------------------------------------------------


def set_title(title: str) -> dict[str, Any]:
    return {
        "type": "delta",
        "request_id": "d1",
        "target_domain": "frontend",
        "target_id": "frontend:document",
        "patches": [{"op": "replace", "path": "/frontend/title", "value": title}],
    }


async def test_delta_without_a_lock_is_rejected_with_a_conflict() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send(set_title("Hijacked"))
        frame = await socket.receive_kind("error")
        assert frame["request_id"] == "d1"
        assert frame["problem"]["status"] == 409
        assert harness.service.hub.latest_snapshot(session_id).frame_seq == 0
        await socket.disconnect()


async def test_locked_delta_commits_under_the_verified_identity() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, user_id=BOB)
        await socket.connect()
        socket.send({"type": "lock.acquire", "target_id": "frontend:document", "request_id": "l"})
        ack = await socket.receive_kind("ack")
        assert ack["lock"]["state"] == "ACQUIRED"
        assert ack["lock"]["locked_by_agent_id"] == "user:bob"
        assert ack["lock"]["priority"] == 80, "priority is the server's choice"

        socket.send(set_title("Renamed"))
        ack = await socket.receive_kind("ack")
        assert (ack["request_id"], ack["frame_seq"]) == ("d1", 1)
        state = await socket.receive_kind("state")
        assert state["delta"]["agent_id"] == "user:bob"
        assert state["snapshot"]["bundle_state"]["frontend"]["title"] == "Renamed"
        timeline = harness.service.hub.timeline(session_id)
        assert [delta.agent_id for delta in timeline.deltas] == ["user:bob"]
        await socket.disconnect()


async def test_lock_held_by_someone_else_queues_the_request() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        alice = harness.socket(session_id, user_id=ALICE)
        bob = harness.socket(session_id, user_id=BOB)
        await alice.connect()
        await bob.connect()
        alice.send({"type": "lock.acquire", "target_id": "frontend:document"})
        await alice.receive_kind("ack")
        bob.send({"type": "lock.acquire", "target_id": "frontend:document"})
        assert (await bob.receive_kind("ack"))["lock"]["state"] == LockState.WAITING.value
        bob.send(set_title("Too soon"))
        assert (await bob.receive_kind("error"))["problem"]["status"] == 409
        await alice.disconnect()
        await bob.disconnect()


async def test_lock_release_frees_it_for_the_next_in_line() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        alice = harness.socket(session_id, user_id=ALICE)
        bob = harness.socket(session_id, user_id=BOB)
        await alice.connect()
        await bob.connect()
        alice.send({"type": "lock.acquire", "target_id": "hero"})
        await alice.receive_kind("ack")
        bob.send({"type": "lock.acquire", "target_id": "hero"})
        await bob.receive_kind("ack")
        alice.send({"type": "lock.release", "target_id": "hero", "request_id": "rel"})
        assert (await alice.receive_kind("ack"))["request_id"] == "rel"
        notice = await bob.receive_kind("lock")
        assert notice["event"]["kind"] == "PROMOTED"
        assert notice["event"]["lock"]["locked_by_agent_id"] == "user:bob"
        await alice.disconnect()
        await bob.disconnect()


async def test_locks_are_released_when_the_last_connection_of_a_user_leaves() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        manager = harness.service.hub.lock_manager(session_id)
        first_tab = harness.socket(session_id)
        second_tab = harness.socket(session_id)
        await first_tab.connect()
        await second_tab.connect()
        first_tab.send({"type": "lock.acquire", "target_id": "hero"})
        await first_tab.receive_kind("ack")

        await first_tab.disconnect()
        assert manager.live_holder("hero", harness.service.hub.now_ms()) == "user:alice"

        await second_tab.disconnect()
        assert manager.active_locks() == ()


async def test_cleanup_survives_the_session_ending_first() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "lock.acquire", "target_id": "hero"})
        await socket.receive_kind("ack")
        await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        assert await socket.closed() == (1000, "Session closed")
        await socket.finished()


async def test_intent_for_a_session_that_just_ended_is_an_error_not_a_crash() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        hub = harness.service.hub

        def gone(_session_id: str) -> None:
            raise SessionNotFoundError(session_id=session_id)

        cast("Any", hub).touch = gone
        socket.send({"type": "ping", "request_id": "late"})
        frame = await socket.receive_kind("error")
        assert frame["problem"]["status"] == 404
        await socket.disconnect()


# --- socket: prompts ------------------------------------------------------------------------


Hook = Callable[[AgentRole, LLMRequest[Any]], Awaitable[None]]


class _HookedClient:
    def __init__(self, inner: LLMClient, role: AgentRole, hook: Hook) -> None:
        self._inner = inner
        self._role = role
        self._hook = hook

    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        await self._hook(self._role, request)
        return await self._inner.generate_structured(request)


class HookedProvider:
    def __init__(self, inner: MockLLMProvider, hook: Hook) -> None:
        self._inner = inner
        self._hook = hook

    def client_for(self, role: AgentRole) -> LLMClient:
        return _HookedClient(self._inner.client_for(role), role, self._hook)


async def drain_thoughts_until(socket: SocketDriver, tool_name: str) -> list[dict[str, Any]]:
    seen = []
    while True:
        frame = await socket.receive_kind("thought")
        seen.append(frame)
        if frame["thought"]["tool_name"] == tool_name:
            return seen


async def test_prompt_runs_the_swarm_and_every_thought_arrives_in_order() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "prompt", "text": "Build a landing page", "request_id": "p"})
        assert (await socket.receive_kind("ack"))["request_id"] == "p"

        thoughts = await drain_thoughts_until(socket, "run.completed")

        assert [frame["seq"] for frame in thoughts] == list(range(len(thoughts)))
        assert thoughts[-1]["thought"]["tool_args"]["completed"] is True
        tools = {frame["thought"]["tool_name"] for frame in thoughts}
        assert {"graph.enter", "llm.generate", "hub.apply_delta"} <= tools
        await socket.disconnect()


async def test_swarm_cursors_come_from_the_simulator_during_a_run() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await wait_until(lambda: harness.service._sessions[session_id].ticker.ticks > 0)
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "prompt", "text": "Build it"})
        await socket.receive_kind("ack")
        while True:
            frame = await socket.receive()
            assert frame["kind"] != "cursor" or (frame["frame"]["x"], frame["frame"]["y"]) != (0, 0)
            if frame["kind"] == "thought" and frame["thought"]["tool_name"] == "run.completed":
                break
        await socket.disconnect()


async def test_second_prompt_during_a_run_is_refused() -> None:
    release = asyncio.Event()

    async def hold(_role: AgentRole, _request: LLMRequest[Any]) -> None:
        await release.wait()

    async with running(provider=HookedProvider(MockLLMProvider(), hold)) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "prompt", "text": "First"})
        await socket.receive_kind("ack")
        socket.send({"type": "prompt", "text": "Second", "request_id": "two"})
        frame = await socket.receive_kind("error")
        assert frame["request_id"] == "two"
        assert frame["problem"]["type"].endswith("/run-already-active")
        release.set()
        await drain_thoughts_until(socket, "run.completed")
        socket.send({"type": "prompt", "text": "Third", "request_id": "three"})
        assert (await socket.receive_kind("ack"))["request_id"] == "three"
        await drain_thoughts_until(socket, "run.completed")
        await socket.disconnect()


async def test_concurrent_run_limit_protects_the_model_budget() -> None:
    release = asyncio.Event()

    async def hold(_role: AgentRole, _request: LLMRequest[Any]) -> None:
        await release.wait()

    settings = make_settings(gateway_max_concurrent_runs=1)
    async with running(settings, provider=HookedProvider(MockLLMProvider(), hold)) as harness:
        await harness.create_session(prompt="First")
        response = await harness.http.post(
            SESSIONS, json={"prompt": "Second"}, headers=harness.auth()
        )
        assert_problem(response, 429)
        release.set()


async def test_model_failure_is_reported_to_clients() -> None:
    provider = MockLLMProvider({AgentRole.ARCHITECT: [LLMProviderError("overloaded", model="m")]})
    async with running(provider=provider) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "prompt", "text": "Build it"})
        thoughts = await drain_thoughts_until(socket, "run.failed")
        assert "overloaded" in thoughts[-1]["thought"]["thought"]
        assert thoughts[-1]["thought"]["tool_args"]["status"] == 502
        assert harness.service._sessions[session_id].run is None
        await socket.disconnect()


async def test_unexpected_run_failure_tells_clients_nothing_specific(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = MockLLMProvider({AgentRole.ARCHITECT: [RuntimeError("internal secret detail")]})
    async with running(provider=provider) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        socket.send({"type": "prompt", "text": "Build it"})
        with caplog.at_level(logging.ERROR):
            thoughts = await drain_thoughts_until(socket, "run.failed")
        assert "internal secret" not in json.dumps(thoughts[-1])
        assert thoughts[-1]["thought"]["tool_args"]["status"] == 500
        await socket.disconnect()
    assert "swarm run failed" in caplog.text


async def test_run_ending_with_its_session_is_not_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def close_session_first(_role: AgentRole, _request: LLMRequest[Any]) -> None:
        await harness.service.hub.close_session(session_id)

    provider = HookedProvider(
        MockLLMProvider({AgentRole.ARCHITECT: [SwarmPlan()]}), close_session_first
    )
    async with running(provider=provider) as harness:
        session_id = await harness.create_session()
        runtime = harness.service._sessions[session_id]
        with caplog.at_level(logging.INFO):
            harness.service.start_run(session_id, "Build it")
            await wait_until(lambda: runtime.run is None)
    assert "ended during its run" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_announcing_a_run_after_its_session_ended_is_harmless() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await harness.http.delete(f"{SESSIONS}/{session_id}", headers=harness.auth())
        await harness.service._announce_run(session_id, "run.failed", "late", {})


# --- reconnection ---------------------------------------------------------------------------


async def publish_thoughts(harness: Harness, session_id: str, count: int) -> None:
    for index in range(count):
        await harness.service.hub.broadcast_thought(session_id, make_thought(f"t{index}"))


async def test_reconnect_replays_missed_thoughts_contiguously_then_goes_live() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 10)

        socket = harness.socket(session_id, last_seen=4)
        sync = await socket.connect()
        assert sync["replayed_thoughts"] == 5
        assert sync["thought_gap"] is False
        assert sync["latest_thought_seq"] == 9

        replayed = [await socket.receive() for _ in range(5)]
        assert [frame["kind"] for frame in replayed] == ["thought"] * 5
        assert [frame["seq"] for frame in replayed] == [5, 6, 7, 8, 9]

        await harness.service.hub.broadcast_thought(session_id, make_thought("live"))
        live = await socket.receive_kind("thought")
        assert live["seq"] == 10
        assert live["thought"]["thought"] == "live"
        await socket.disconnect()


async def test_up_to_date_client_gets_no_replay() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 3)
        socket = harness.socket(session_id, last_seen=2)
        sync = await socket.connect()
        assert (sync["replayed_thoughts"], sync["thought_gap"]) == (0, False)
        await socket.disconnect()


async def test_fresh_client_gets_no_replay_and_no_false_gap() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 3)
        socket = harness.socket(session_id)
        sync = await socket.connect()
        assert sync["replayed_thoughts"] == 0
        assert sync["thought_gap"] is False
        assert sync["latest_thought_seq"] == 2
        await socket.disconnect()


async def test_fresh_client_sees_no_gap_even_after_the_ring_has_evicted() -> None:
    async with running(make_settings(thought_backlog_size=2)) as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 5)
        socket = harness.socket(session_id)
        sync = await socket.connect()
        assert sync["thought_gap"] is False, "it never asked for what was evicted"
        assert sync["replayed_thoughts"] == 0
        await socket.disconnect()


async def test_replay_older_than_the_ring_reports_the_gap() -> None:
    async with running(make_settings(thought_backlog_size=3)) as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 10)
        socket = harness.socket(session_id, last_seen=1)
        sync = await socket.connect()
        assert sync["thought_gap"] is True
        assert sync["replayed_thoughts"] == 3
        assert [(await socket.receive())["seq"] for _ in range(3)] == [7, 8, 9]
        await socket.disconnect()


async def test_client_claiming_to_be_ahead_of_the_log_is_reset() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 3)
        socket = harness.socket(session_id, last_seen=999)
        sync = await socket.connect()
        assert sync["thought_gap"] is True
        assert sync["replayed_thoughts"] == 3
        assert [(await socket.receive())["seq"] for _ in range(3)] == [0, 1, 2]
        await socket.disconnect()


async def test_thoughts_published_during_a_replay_are_neither_lost_nor_repeated() -> None:
    release = asyncio.Event()
    sends = 0

    async def pause_mid_burst(_message: Message) -> None:
        nonlocal sends
        sends += 1
        if sends == 3:
            await release.wait()

    settings = make_settings(broadcast_fps=1, ws_catchup_deadline_ms=60_000)
    async with running(settings) as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 5)
        socket = harness.socket(session_id, last_seen=0, send_gate=pause_mid_burst)
        await socket.connect()
        await wait_until(lambda: sends == 3)
        await publish_thoughts(harness, session_id, 3)
        release.set()
        seqs = [(await socket.receive_kind("thought"))["seq"] for _ in range(7)]
        assert seqs == [1, 2, 3, 4, 5, 6, 7]
        await socket.disconnect()


async def test_full_disconnect_and_reconnect_cycle_during_a_run_misses_nothing() -> None:
    async with running() as harness:
        session_id = await harness.create_session()
        first = harness.socket(session_id)
        await first.connect()
        first.send({"type": "prompt", "text": "Build it"})
        await first.receive_kind("ack")
        seen = [(await first.receive_kind("thought"))["seq"] for _ in range(3)]
        await first.disconnect()

        await wait_until(lambda: harness.service._sessions[session_id].run is None, 5)

        second = harness.socket(session_id, last_seen=seen[-1])
        sync = await second.connect()
        assert sync["thought_gap"] is False
        for _ in range(sync["replayed_thoughts"]):
            seen.append((await second.receive())["seq"])
        assert seen == list(range(sync["latest_thought_seq"] + 1))
        await second.disconnect()


# --- backpressure ---------------------------------------------------------------------------


class Pipe:
    """Stands in for a TCP connection: sends complete only as the client reads."""

    def __init__(self, *, allow: int = 0) -> None:
        self.allow = allow
        self.released = asyncio.Event()

    async def gate(self, _message: Message) -> None:
        if self.allow > 0:
            self.allow -= 1
            return
        await self.released.wait()


async def test_stalled_client_is_evicted_with_policy_violation_slow_consumer() -> None:
    pipe = Pipe(allow=1)
    async with running(make_settings(ws_catchup_deadline_ms=30)) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=pipe.gate)
        await socket.connect()
        await harness.service.hub.broadcast_thought(session_id, make_thought())
        assert await socket.closed() == (1008, "Slow Consumer")
        await socket.finished()
        assert not harness.service._connections


async def test_eviction_during_the_replay_burst_also_counts() -> None:
    pipe = Pipe(allow=1)
    async with running(make_settings(ws_catchup_deadline_ms=30)) as harness:
        session_id = await harness.create_session()
        await publish_thoughts(harness, session_id, 5)
        socket = harness.socket(session_id, last_seen=0, send_gate=pipe.gate)
        await socket.open()
        assert await socket.closed() == (1008, "Slow Consumer")


async def test_client_draining_too_slowly_is_evicted_though_every_send_succeeds() -> None:
    """Each send beats any per-send timeout, but the client would fall behind forever."""
    settings = make_settings(ws_catchup_deadline_ms=120, broadcast_fps=1)

    async def slow(_message: Message) -> None:
        await asyncio.sleep(0.02)

    async with running(settings) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=slow)
        await socket.connect()
        await publish_thoughts(harness, session_id, 500)
        assert await socket.closed() == (1008, "Slow Consumer")


async def test_slow_client_within_the_deadline_loses_no_state_or_thought() -> None:
    settings = make_settings(ws_catchup_deadline_ms=10_000, broadcast_fps=240)

    async def slow(_message: Message) -> None:
        await asyncio.sleep(0.001)

    async with running(settings) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=slow)
        await socket.connect()
        manager = harness.service.hub.lock_manager(session_id)
        await manager.acquire("frontend:document", "writer", 50, harness.service.hub.now_ms())
        count = 400
        for seq in range(count):
            await harness.service.hub.broadcast_thought(session_id, make_thought(f"t{seq}"))
            delta = {
                "session_id": session_id,
                "agent_id": "writer",
                "delta_seq": seq,
                "target_domain": "frontend",
                "target_id": "frontend:document",
                "patches": [{"op": "replace", "path": "/frontend/title", "value": f"T{seq}"}],
                "emitted_at_ms": 0,
            }
            await harness.service.hub.apply_delta(session_id, StreamDelta.model_validate(delta))

        thoughts: list[int] = []
        states: list[int] = []
        while len(thoughts) < count or len(states) < count:
            frame = await socket.receive()
            if frame["kind"] == "thought":
                thoughts.append(frame["seq"])
            elif frame["kind"] == "state":
                states.append(frame["delta"]["delta_seq"])
        assert thoughts == list(range(count))
        assert states == list(range(count))
        await socket.disconnect()


async def test_slow_clients_cursor_frames_conflate_to_the_latest() -> None:
    release = asyncio.Event()
    sent: list[dict[str, Any]] = []
    armed = [False]

    async def gate(message: Message) -> None:
        sent.append(json.loads(message["text"]))
        if armed[0]:
            await release.wait()

    settings = make_settings(ws_catchup_deadline_ms=60_000, broadcast_fps=240)
    async with running(settings) as harness:
        session_id = await harness.create_session()
        await harness.service._sessions[session_id].ticker.stop()
        socket = harness.socket(session_id, send_gate=gate)
        await socket.connect()
        hub = harness.service.hub
        armed[0] = True

        await hub.broadcast_presence(session_id, BOB, x=0, y=0)
        await wait_until(lambda: len(sent) >= 2)
        # The connection is now stuck sending. These should collapse into only the last.
        for x in range(1, 200):
            await hub.broadcast_presence(session_id, BOB, x=x, y=0)
            await asyncio.sleep(0)
        await asyncio.sleep(0.02)
        armed[0] = False
        release.set()

        positions: list[int] = []
        while not positions or positions[-1] != 199:
            frame = await socket.receive()
            if frame["kind"] == "presence":
                positions.append(frame["x"])
        assert len(positions) < 20, "older frames should have been dropped"
        await socket.disconnect()


async def test_healthy_client_outlives_the_deadline_many_times_over() -> None:
    settings = make_settings(ws_catchup_deadline_ms=250)
    async with running(settings) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id)
        await socket.connect()
        for index in range(3):
            await harness.service.hub.broadcast_thought(session_id, make_thought(f"t{index}"))
            await socket.receive_kind("thought")
            await asyncio.sleep(0.3)
        assert harness.service._connections
        await socket.disconnect()


async def test_burst_after_a_long_idle_spell_is_not_mistaken_for_falling_behind() -> None:
    async with running(make_settings(ws_catchup_deadline_ms=250, broadcast_fps=1)) as harness:
        session_id = await harness.create_session()
        await harness.service._sessions[session_id].ticker.stop()
        socket = harness.socket(session_id)
        await socket.connect()
        await asyncio.sleep(0.3)
        await publish_thoughts(harness, session_id, 5)
        assert [(await socket.receive_kind("thought"))["seq"] for _ in range(5)] == list(range(5))
        assert harness.service._connections
        await socket.disconnect()


async def test_steady_stream_that_never_lets_the_client_go_idle_is_not_eviction() -> None:
    """A cursor is always waiting here, so only sending them can show the client is keeping up."""

    async def pace(_message: Message) -> None:
        await asyncio.sleep(0.002)

    async with running(make_settings(ws_catchup_deadline_ms=250, broadcast_fps=1)) as harness:
        session_id = await harness.create_session()
        await harness.service._sessions[session_id].ticker.stop()
        socket = harness.socket(session_id, send_gate=pace)
        await socket.connect()
        listener = next(iter(harness.service.hub._require(session_id).listeners))
        finish_at = asyncio.get_running_loop().time() + 0.6
        x = 0
        while asyncio.get_running_loop().time() < finish_at:
            x += 1
            listener.offer(PresenceEvent(origin="w", session_id=session_id, user_id=BOB, x=x % 100))
            await asyncio.sleep(0)
            assert harness.service._connections, "a client keeping up was evicted"
        await socket.disconnect()


async def test_close_to_a_stalled_peer_cannot_hang_the_connection() -> None:
    forever = asyncio.Event()
    allow = [1]

    async def gate(_message: Message) -> None:
        if allow[0] > 0:
            allow[0] -= 1
            return
        await forever.wait()

    async with running(make_settings(ws_catchup_deadline_ms=20)) as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=gate)

        original_close = socket._send

        async def stuck_close(message: Message) -> None:
            if message["type"] == "websocket.close":
                await forever.wait()
            await original_close(message)

        socket._send = stuck_close  # type: ignore[method-assign]
        await socket.connect()
        await harness.service.hub.broadcast_thought(session_id, make_thought())
        await socket.finished()
        assert not harness.service._connections


@pytest.mark.parametrize("failure", [RuntimeError("gone"), WebSocketDisconnect(1006)])
async def test_client_vanishing_mid_send_ends_quietly(failure: Exception) -> None:
    fail = [False]

    async def gate(_message: Message) -> None:
        if fail[0]:
            raise failure

    async with running() as harness:
        session_id = await harness.create_session()
        socket = harness.socket(session_id, send_gate=gate)
        await socket.connect()
        fail[0] = True
        await harness.service.hub.broadcast_thought(session_id, make_thought())
        await socket.finished()
        assert not harness.service._connections


async def test_many_clients_receive_the_same_stream_independently() -> None:
    settings = make_settings(broadcast_fps=60)
    async with running(settings) as harness:
        session_id = await harness.create_session()
        sockets = [harness.socket(session_id, user_id=f"user-{i}") for i in range(25)]
        for socket in sockets:
            await socket.connect()
        await publish_thoughts(harness, session_id, 50)
        for socket in sockets:
            seqs = [(await socket.receive_kind("thought"))["seq"] for _ in range(50)]
            assert seqs == list(range(50))
        await asyncio.gather(*(socket.disconnect() for socket in sockets))


# --- a real websocket stack -----------------------------------------------------------------


def test_starlette_test_client_completes_a_full_session() -> None:
    """Goes through Starlette's own client, to check this all fits the real ASGI handshake."""
    app = create_app(make_settings(), provider=MockLLMProvider(), bus=InMemoryEventBus())
    with TestClient(app) as client:
        service: GatewayService = cast("Any", client.app).state.gateway
        token = service.authenticator.issue(ALICE)
        created = client.post(SESSIONS, headers={"Authorization": f"Bearer {token}"})
        session_id = created.json()["session_id"]

        with client.websocket_connect(
            f"/ws/sessions/{session_id}",
            subprotocols=[SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}{token}"],
        ) as socket:
            assert socket.accepted_subprotocol == SUBPROTOCOL
            assert socket.receive_json()["kind"] == "sync"
            socket.send_json({"type": "ping", "request_id": "p"})
            while (frame := socket.receive_json())["kind"] != "pong":
                assert frame["kind"] in {"cursor", "presence"}
            assert frame["request_id"] == "p"

        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(f"/ws/sessions/{session_id}"),
        ):
            pass

        deleted = client.delete(
            f"{SESSIONS}/{session_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert deleted.status_code == 204


# --- hub support for the gateway ------------------------------------------------------------


async def test_replica_keeps_a_backlog_of_thoughts_for_its_own_reconnecting_clients() -> None:
    bus = InMemoryEventBus()
    settings = make_settings()
    writer, replica = SwarmSessionHub(settings, bus=bus), SwarmSessionHub(settings, bus=bus)
    for hub in (writer, replica):
        await hub.create_session("s1", site_bundle())
    for index in range(3):
        await writer.broadcast_thought("s1", make_thought(f"t{index}"))
    await wait_until(lambda: replica.thought_backlog("s1").latest_seq == 2)

    backlog = replica.thought_backlog("s1", 0)
    assert [event.seq for event in backlog.events] == [1, 2]
    assert backlog.gap is False
    await writer.shutdown()
    await replica.shutdown()


async def test_thoughts_pub_sub_lost_on_the_way_show_up_as_a_gap() -> None:
    bus = InMemoryEventBus()
    settings = make_settings()
    hub = SwarmSessionHub(settings, bus=bus)
    await hub.create_session("s1", site_bundle())
    channel = session_channel("s1")
    for seq in (0, 1, 4):
        event = ThoughtEvent(origin="other", session_id="s1", seq=seq, thought=make_thought())
        await bus.publish(channel, event)
    backlog = hub.thought_backlog("s1", 0)
    assert [event.seq for event in backlog.events] == [1, 4]
    assert backlog.gap is True
    assert hub.thought_backlog("s1").gap is True
    assert hub.thought_backlog("s1", 4).gap is False
    await hub.shutdown()


async def test_backlog_of_a_session_with_no_thoughts_is_empty_and_whole() -> None:
    hub = SwarmSessionHub(make_settings(), bus=InMemoryEventBus())
    await hub.create_session("s1", site_bundle())
    backlog = hub.thought_backlog("s1")
    assert (backlog.events, backlog.gap, backlog.latest_seq) == ((), False, None)
    await hub.shutdown()


async def test_presence_counts_as_session_activity() -> None:
    clock_now = [0]
    hub = SwarmSessionHub(
        make_settings(session_ttl_seconds=1), bus=InMemoryEventBus(), clock=lambda: clock_now[0]
    )
    await hub.create_session("s1", site_bundle())
    clock_now[0] = 900
    await hub.broadcast_presence("s1", BOB, x=1, y=1)
    clock_now[0] = 1500
    assert await hub.sweep() == ()
    await hub.shutdown()


async def test_presence_replicates_through_json_between_workers() -> None:
    bus = InMemoryEventBus()
    settings = make_settings()
    writer, replica = SwarmSessionHub(settings, bus=bus), SwarmSessionHub(settings, bus=bus)
    for hub in (writer, replica):
        await hub.create_session("s1", site_bundle())
    remote = replica.subscribe("s1")
    await writer.broadcast_presence("s1", BOB, x=3, y=4, selected_id="hero")
    event = await asyncio.wait_for(anext(remote), WAIT_S)
    assert isinstance(event, PresenceEvent)
    assert PresenceEvent.model_validate_json(event.model_dump_json()) == event
    assert (event.x, event.selected_id) == (3, "hero")
    await writer.shutdown()
    await replica.shutdown()


async def test_lossless_listener_keeps_all_state_where_a_bounded_one_drops_it() -> None:
    hub = SwarmSessionHub(make_settings(), bus=InMemoryEventBus(), listener_capacity=1)
    await hub.create_session("s1", site_bundle())
    bounded, lossless = hub.subscribe("s1"), hub.subscribe("s1", lossless=True)
    manager = hub.lock_manager("s1")
    await manager.acquire("frontend:document", "w", 50, hub.now_ms())
    for seq in range(5):
        await hub.apply_delta(
            "s1",
            StreamDelta.model_validate(
                {
                    "session_id": "s1",
                    "agent_id": "w",
                    "delta_seq": seq,
                    "target_domain": "frontend",
                    "target_id": "frontend:document",
                    "patches": [{"op": "replace", "path": "/frontend/title", "value": f"T{seq}"}],
                    "emitted_at_ms": 0,
                }
            ),
        )
    assert bounded.dropped == 4
    assert lossless.dropped == 0
    assert lossless.lossless_pending == 5
    await hub.shutdown()
