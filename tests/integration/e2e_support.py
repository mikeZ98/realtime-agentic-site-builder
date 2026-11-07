"""The harness shared by the end-to-end suites: a real application, driven entirely in memory.

Requests go through ``httpx.ASGITransport`` and sockets through raw ASGI messages, so nothing ever
opens a network socket, and both run in the test's own event loop. The only clock that moves on its
own is the monotonic one pacing frames. The wall clock behind tokens, leases and timestamps is held
still, so no lease can lapse mid-test and every timestamp in a bundle is reproducible.
"""

import ast
import asyncio
import functools
import io
import json
import posixpath
import re
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final, cast

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, SecretStr

from realtime_agentic_site_builder.agents.provider import (
    LLMClient,
    LLMRequest,
    LLMResult,
    MockLLMProvider,
)
from realtime_agentic_site_builder.agents.tsx import find_tsx_issue
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import InMemoryEventBus
from realtime_agentic_site_builder.domain.swarm import AgentRole
from realtime_agentic_site_builder.gateway.auth import SUBPROTOCOL, TOKEN_SUBPROTOCOL_PREFIX
from realtime_agentic_site_builder.gateway.service import GatewayService, create_app

FIXED_NOW_MS: Final = 1_800_000_000_000
WAIT_S: Final = 5.0
SESSIONS: Final = "/api/v1/sessions"
OWNER: Final = "alice"
VIEWER: Final = "bob"
Message = dict[str, Any]
Frame = dict[str, Any]


def deadline[**P](
    seconds: float,
) -> Callable[[Callable[P, Coroutine[Any, Any, None]]], Callable[P, Coroutine[Any, Any, None]]]:
    """Fail a scenario that runs too long, instead of letting it hang the whole suite."""

    def decorate(
        scenario: Callable[P, Coroutine[Any, Any, None]],
    ) -> Callable[P, Coroutine[Any, Any, None]]:
        @functools.wraps(scenario)
        async def bounded(*args: P.args, **kwargs: P.kwargs) -> None:
            async with asyncio.timeout(seconds):
                await scenario(*args, **kwargs)

        return bounded

    return decorate


def settings(**overrides: Any) -> Settings:
    return Settings(
        **{
            "gateway_secret": SecretStr("end-to-end-signing-key-for-hermetic-tests"),
            "broadcast_fps": 60,
            **overrides,
        }
    )


class GatedProvider:
    """The deterministic mock, with every model call waiting for the gate and then a latency.

    The bare mock answers instantly, so a whole run is over within a few frames. A real model takes
    seconds per call. The latency changes when things happen, never what happens.
    """

    def __init__(self, *, latency_s: float = 0.0, open_gate: bool = False) -> None:
        self.inner = MockLLMProvider()
        self.gate = asyncio.Event()
        if open_gate:
            self.gate.set()
        self.latency_s = latency_s
        self.calls = 0

    def client_for(self, role: AgentRole) -> LLMClient:
        return _GatedClient(self, role)


class _GatedClient:
    def __init__(self, provider: GatedProvider, role: AgentRole) -> None:
        self._provider = provider
        self._client = provider.inner.client_for(role)

    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        self._provider.calls += 1
        await self._provider.gate.wait()
        await asyncio.sleep(self._provider.latency_s)
        return await self._client.generate_structured(request)


@dataclass
class Gateway:
    app: FastAPI
    service: GatewayService
    http: httpx.AsyncClient
    bus: InMemoryEventBus

    def token(self, user_id: str = OWNER) -> str:
        return self.service.authenticator.issue(user_id)

    def auth(self, user_id: str = OWNER) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(user_id)}"}

    async def create_session(self, user_id: str = OWNER, **body: Any) -> str:
        response = await self.http.post(SESSIONS, json=body, headers=self.auth(user_id))
        assert response.status_code == 201, response.text
        session_id: str = response.json()["session_id"]
        return session_id

    def socket(
        self,
        session_id: str,
        user_id: str = OWNER,
        *,
        query: str = "",
        send_gate: Callable[[Message], Awaitable[None]] | None = None,
    ) -> "Socket":
        return Socket(
            self.app,
            f"/ws/sessions/{session_id}",
            query,
            [SUBPROTOCOL, f"{TOKEN_SUBPROTOCOL_PREFIX}{self.token(user_id)}"],
            send_gate,
        )


@asynccontextmanager
async def gateway(
    config: Settings | None = None, *, provider: Any = None
) -> AsyncIterator[Gateway]:
    bus = InMemoryEventBus()
    app = create_app(
        config or settings(),
        provider=provider or MockLLMProvider(),
        bus=bus,
        hub_clock=lambda: FIXED_NOW_MS,
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://gateway") as http,
    ):
        yield Gateway(app, app.state.gateway, http, bus)


class Socket:
    """A WebSocket client speaking ASGI to the app. The test decides when each ``send`` returns."""

    def __init__(
        self,
        app: FastAPI,
        path: str,
        query: str,
        subprotocols: list[str],
        send_gate: Callable[[Message], Awaitable[None]] | None,
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
            "headers": [],
            "subprotocols": subprotocols,
            "client": ("client", 1),
            "server": ("gateway", 80),
        }
        self._to_app: asyncio.Queue[Message] = asyncio.Queue()
        self.from_app: asyncio.Queue[Message] = asyncio.Queue()
        self._send_gate = send_gate
        self.task: asyncio.Task[None] | None = None

    async def _send(self, message: Message) -> None:
        if self._send_gate is not None and message["type"] == "websocket.send":
            await self._send_gate(message)
        await self.from_app.put(message)

    async def connect(self) -> Frame:
        self._to_app.put_nowait({"type": "websocket.connect"})
        self.task = asyncio.create_task(
            self._app(self._scope, cast("Any", self._to_app.get), cast("Any", self._send))
        )
        answer = await asyncio.wait_for(self.from_app.get(), WAIT_S)
        assert answer["type"] == "websocket.accept", answer
        assert answer["subprotocol"] == SUBPROTOCOL
        return await self.receive()

    async def receive(self) -> Frame:
        message = await asyncio.wait_for(self.from_app.get(), WAIT_S)
        assert message["type"] == "websocket.send", message
        frame: Frame = json.loads(message["text"])
        return frame

    async def next_of(self, *kinds: str) -> Frame:
        while (frame := await self.receive())["kind"] not in kinds:
            pass
        return frame

    def send(self, payload: dict[str, Any]) -> None:
        self._to_app.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    async def closed(self) -> tuple[int, str]:
        while True:
            message = await asyncio.wait_for(self.from_app.get(), WAIT_S)
            if message["type"] == "websocket.close":
                return message["code"], message.get("reason", "")

    async def finished(self) -> None:
        assert self.task is not None
        await asyncio.wait_for(self.task, WAIT_S)

    async def drop(self) -> None:
        """Vanish the way a lost network does: no close handshake, just a disconnect."""
        self._to_app.put_nowait({"type": "websocket.disconnect", "code": 1006})
        await self.finished()


def unzip(content: bytes) -> dict[str, str]:
    """Read an exported archive in memory, keyed by path below its single top-level directory."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        roots = {name.split("/", 1)[0] for name in names}
        assert len(roots) == 1, roots
        return {name.split("/", 1)[1]: archive.read(name).decode() for name in names}


_SPECIFIER: Final = re.compile(
    r"""^\s*(?:import|export)\b[^'";]*?\bfrom\s*['"]([^'"]+)['"]""", re.M
)


def frontend_problems(files: dict[str, str]) -> list[str]:
    """Check what can be checked without a JavaScript toolchain, which a hermetic run cannot use.

    Every TS and TSX file must be structurally balanced, every relative import must resolve to a
    file in the archive, every bare import must be a declared dependency, and the configuration
    must be valid JSON. What a compiler adds beyond this is type checking.
    """
    problems: list[str] = []
    package = json.loads(files["frontend/package.json"])
    declared = set(package["dependencies"]) | set(package["devDependencies"])
    json.loads(files["frontend/tsconfig.json"])
    for path, text in files.items():
        if not path.startswith("frontend/src/") or not path.endswith((".ts", ".tsx")):
            continue
        issue = find_tsx_issue(text)
        if issue is not None:
            problems.append(f"{path}: {issue}")
        for specifier in _SPECIFIER.findall(text):
            if specifier.startswith("."):
                target = posixpath.normpath(posixpath.join(posixpath.dirname(path), specifier))
                if not any(f"{target}{suffix}" in files for suffix in ("", ".ts", ".tsx")):
                    problems.append(f"{path}: {specifier} does not resolve")
            elif specifier.split("/")[0] not in declared:
                problems.append(f"{path}: {specifier} is not a dependency")
    return problems


def python_problems(files: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for path, text in files.items():
        if path.endswith(".py"):
            try:
                compile(ast.parse(text, filename=path), path, "exec", dont_inherit=True)
            except SyntaxError as error:
                problems.append(f"{path}: {error.msg}")
    return problems
