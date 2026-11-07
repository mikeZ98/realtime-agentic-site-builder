"""The FastAPI application: lifecycle, background automation and REST and WebSocket ingress."""

import asyncio
import contextlib
import logging
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Annotated, Any, Final, Literal, cast

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from realtime_agentic_site_builder.agents.graph import create_swarm_graph
from realtime_agentic_site_builder.agents.identity import IDENTITIES
from realtime_agentic_site_builder.agents.provider import LLMProvider, TieredLLMProvider
from realtime_agentic_site_builder.agents.state import initial_state
from realtime_agentic_site_builder.canvas.compositor import (
    MAX_THOUGHT_HISTORY,
    CanvasCompositor,
    thought_entry,
)
from realtime_agentic_site_builder.canvas.viewer import viewer_response
from realtime_agentic_site_builder.config import Settings, get_settings
from realtime_agentic_site_builder.core.clock import Clock, system_now_ms
from realtime_agentic_site_builder.core.hub import (
    EventBus,
    SessionListener,
    SwarmSessionHub,
    ThoughtBacklog,
    ThoughtEvent,
)
from realtime_agentic_site_builder.core.invariants import InvariantReport, verify_timeline
from realtime_agentic_site_builder.domain.backend import (
    BackendBlueprint,
    DatabaseDialect,
    FullstackProjectBundle,
)
from realtime_agentic_site_builder.domain.base import MAX_TEXT_LENGTH, DomainModel
from realtime_agentic_site_builder.domain.blueprint import SiteBlueprint
from realtime_agentic_site_builder.domain.errors import (
    PROBLEM_JSON_MEDIA_TYPE,
    AuthenticationRequiredError,
    CapacityExceededError,
    GatewayConfigurationError,
    PermissionDeniedError,
    ProblemDetails,
    ProjectSynthesisError,
    RunAlreadyActiveError,
    SessionNotFoundError,
    SiteBuilderError,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    ReplayTimeline,
    StreamDelta,
    SwarmSnapshot,
)
from realtime_agentic_site_builder.gateway.auth import (
    SUBPROTOCOL,
    Identity,
    TokenAuthenticator,
    token_from_header,
    token_from_subprotocols,
)
from realtime_agentic_site_builder.gateway.connection import (
    WS_POLICY_VIOLATION,
    WS_SESSION_NOT_FOUND,
    CanvasConnection,
    ClientConnection,
)
from realtime_agentic_site_builder.gateway.observability import (
    GatewayMetrics,
    RequestIdMiddleware,
    current_request_id,
    install_log_correlation,
)
from realtime_agentic_site_builder.gateway.protocol import CanvasHello, PromptText, SyncFrame
from realtime_agentic_site_builder.simulation.controller import CursorSimulator
from realtime_agentic_site_builder.simulation.ticker import SimulationTicker, monotonic_ms
from realtime_agentic_site_builder.synthesis.exporter import (
    ArchiveFormat,
    ProjectArchive,
    export_bundle,
)

logger = logging.getLogger(__name__)

JSON_MEDIA_TYPE: Final = "application/json"
SessionView = Literal["canvas"]


class CreateSessionRequest(DomainModel):
    prompt: PromptText | None = None
    bundle: FullstackProjectBundle | None = None


class CreateSessionResponse(DomainModel):
    session_id: str
    snapshot: SwarmSnapshot
    run_started: bool


class HistoryEntry(DomainModel):
    frame_seq: int
    timestamp_ms: int
    bundle_hash: str
    delta: StreamDelta
    locks: tuple[ElementLock, ...]


class HistoryResponse(DomainModel):
    """Enough to rebuild the layout at every frame held, without a full bundle per frame."""

    session_id: str
    base_frame_seq: int
    base_bundle_hash: str
    base_frontend: SiteBlueprint
    base_locks: tuple[ElementLock, ...]
    entries: tuple[HistoryEntry, ...]


def history_of(timeline: ReplayTimeline) -> HistoryResponse:
    base = timeline.snapshots[0]
    return HistoryResponse(
        session_id=timeline.session_id,
        base_frame_seq=base.frame_seq,
        base_bundle_hash=base.bundle_hash,
        base_frontend=base.bundle_state.frontend,
        base_locks=base.active_locks,
        entries=tuple(
            HistoryEntry(
                frame_seq=frame.frame_seq,
                timestamp_ms=frame.timestamp_ms,
                bundle_hash=frame.bundle_hash,
                delta=delta,
                locks=frame.active_locks,
            )
            for delta, frame in zip(timeline.deltas, timeline.snapshots[1:], strict=True)
        ),
    )


class TimelineResponse(DomainModel):
    timeline: ReplayTimeline
    thoughts: tuple[ThoughtEvent, ...]
    thought_gap: bool
    latest_thought_seq: int | None


@dataclass(slots=True)
class _SessionRuntime:
    owner: str
    simulator: CursorSimulator
    ticker: SimulationTicker
    run: asyncio.Task[None] | None = None
    connections: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _Budget:
    """A token bucket, so a burst is allowed but a sustained rate is not."""

    tokens: float
    refilled_at_ms: int


class GatewayService:
    """What the routes act on. It exists once per application, for the span of its lifespan."""

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        authenticator: TokenAuthenticator,
        *,
        bus: EventBus | None,
        hub_clock: Clock,
        clock: Clock,
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self.hub = SwarmSessionHub(settings, bus=bus, clock=hub_clock)
        self.settings = settings
        self.provider = provider
        self.authenticator = authenticator
        self.clock = clock
        self._sleep = sleep
        self._sessions: dict[str, _SessionRuntime] = {}
        self._connections: set[ClientConnection] = set()
        self._sockets_per_user: Counter[str] = Counter()
        self._delta_budgets: dict[str, _Budget] = {}
        self._exports_in_flight = 0
        self.metrics = GatewayMetrics()
        self._sweeper: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop(self) -> None:
        """Drain clients, then stop the work behind them, then close the bus last."""
        for connection in tuple(self._connections):
            connection.drain()
        deadline = self.settings.gateway_drain_timeout_seconds
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(deadline):
                while self._connections:
                    await asyncio.sleep(0.01)
        await self._cancel(self._sweeper)
        self._sweeper = None
        for runtime in self._sessions.values():
            await self._cancel(runtime.run)
            await runtime.ticker.stop()
        await self.hub.sweep()
        self._sessions.clear()
        await self.hub.shutdown()

    @staticmethod
    async def _cancel(task: asyncio.Task[Any] | None) -> None:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _sweep_forever(self) -> None:
        while True:
            await self._sleep(self.settings.gateway_sweep_interval_seconds)
            try:
                for session_id in await self.hub.sweep():
                    await self._retire(session_id)
            except Exception:
                # One bad sweep must not end expiry for every session from now on.
                logger.exception("session sweep failed")

    async def _retire(self, session_id: str) -> None:
        runtime = self._sessions.pop(session_id, None)
        if runtime is not None:
            await self._cancel(runtime.run)
            await runtime.ticker.stop()

    async def create_session(
        self, identity: Identity, bundle: FullstackProjectBundle | None, prompt: str | None
    ) -> CreateSessionResponse:
        self._admit_session(identity)
        # Generated here, so a session's id is not something anyone can guess or choose.
        session_id = f"s-{uuid.uuid4().hex}"
        snapshot = await self.hub.create_session(session_id, bundle or _empty_bundle(session_id))
        self._start_runtime(identity, session_id)
        if prompt is not None:
            self.start_run(session_id, prompt)
        return CreateSessionResponse(
            session_id=session_id, snapshot=snapshot, run_started=prompt is not None
        )

    def _admit_session(self, identity: Identity) -> None:
        if len(self._sessions) >= self.settings.gateway_max_sessions:
            self._refuse("sessions")
            raise CapacityExceededError("The gateway is at its session limit.")
        owned = sum(runtime.owner == identity.user_id for runtime in self._sessions.values())
        if owned >= self.settings.gateway_max_sessions_per_user:
            # Without this, one tenant could take every session the gateway allows.
            self._refuse("sessions_per_user")
            raise CapacityExceededError(
                f"You already have {owned} sessions open, which is the limit per person."
            )

    def _start_runtime(self, identity: Identity, session_id: str) -> None:
        simulator = CursorSimulator()
        ticker = SimulationTicker(self.hub, session_id, simulator)
        self._sessions[session_id] = _SessionRuntime(identity.user_id, simulator, ticker)
        ticker.start()

    def _runtime(self, session_id: str) -> _SessionRuntime:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            raise SessionNotFoundError(session_id=session_id)
        return runtime

    def require_owner(self, session_id: str, identity: Identity) -> None:
        if self._runtime(session_id).owner != identity.user_id:
            raise PermissionDeniedError(
                "Only the session's creator can do this. Others can watch and show presence."
            )

    async def close_session(self, identity: Identity, session_id: str) -> None:
        self.require_owner(session_id, identity)
        await self._retire(session_id)
        await self.hub.close_session(session_id)

    def start_run(self, session_id: str, prompt: str) -> None:
        """Start the swarm on a prompt. A run costs real money, so both limits are hard ones."""
        runtime = self._runtime(session_id)
        if runtime.run is not None:
            raise RunAlreadyActiveError("A run is already in progress in this session.")
        active = self.active_runs()
        if active >= self.settings.gateway_max_concurrent_runs:
            self._refuse("concurrent_runs")
            raise CapacityExceededError("The gateway is at its limit of concurrent runs.")
        runtime.run = asyncio.create_task(self._run_swarm(session_id, runtime, prompt))

    async def _run_swarm(self, session_id: str, runtime: _SessionRuntime, prompt: str) -> None:
        try:
            graph = create_swarm_graph(self.hub, self.provider, cursors=runtime.simulator)
            bundle = self.hub.latest_snapshot(session_id).bundle_state
            state = await graph.ainvoke(initial_state(session_id, prompt, bundle))
            await self._announce_run(
                session_id,
                "run.completed",
                "The run finished.",
                {"completed": state["completed"], "iteration": state["iteration"]},
            )
        except SessionNotFoundError:
            logger.info("session %s ended during its run", session_id)
        except SiteBuilderError as error:
            await self._announce_run(
                session_id, "run.failed", error.detail, {"status": error.status}
            )
        except Exception:
            # Never repeat an unexpected error's text to clients, since it can carry anything.
            logger.exception("swarm run failed in session %s", session_id)
            await self._announce_run(
                session_id, "run.failed", "The run failed unexpectedly.", {"status": 500}
            )
        finally:
            runtime.run = None

    async def _announce_run(
        self, session_id: str, tool_name: str, text: str, tool_args: dict[str, Any]
    ) -> None:
        identity = IDENTITIES[AgentRole.ARCHITECT]
        thought = AgentThoughtStream(
            agent_id=identity.agent_id,
            role=identity.role,
            thought=text[:MAX_TEXT_LENGTH],
            tool_name=tool_name,
            tool_args=tool_args,
            timestamp_ms=self.hub.now_ms(),
        )
        with contextlib.suppress(SessionNotFoundError):
            await self.hub.broadcast_thought(session_id, thought)

    def _refuse(self, limit: str) -> None:
        self.metrics.refusals[limit] += 1

    def admit_socket(self, identity: Identity) -> None:
        """Count a socket against its owner's limit. Pair every success with ``release_socket``.

        Counted before the handshake completes and with no await in between, so a burst of
        simultaneous connections cannot all slip under the limit together.
        """
        if self._sockets_per_user[identity.user_id] >= self.settings.ws_max_connections_per_user:
            self._refuse("connections_per_user")
            raise CapacityExceededError("Too many open connections for one person.")
        self._sockets_per_user[identity.user_id] += 1

    def release_socket(self, identity: Identity) -> None:
        self._sockets_per_user[identity.user_id] -= 1
        if self._sockets_per_user[identity.user_id] <= 0:
            del self._sockets_per_user[identity.user_id]
            self._delta_budgets.pop(identity.user_id, None)

    def spend_delta(self, identity: Identity) -> None:
        """Charge one human edit against a budget shared by all of that person's connections.

        Each edit re-serializes and re-validates the whole bundle on the event loop, so a person
        committing as fast as messages are allowed would slow every session on this worker.
        """
        rate = self.settings.ws_max_deltas_per_second
        now_ms = self.clock()
        budget = self._delta_budgets.setdefault(identity.user_id, _Budget(float(rate), now_ms))
        elapsed_s = max(now_ms - budget.refilled_at_ms, 0) / 1000
        budget.tokens = min(budget.tokens + elapsed_s * rate, float(rate))
        budget.refilled_at_ms = now_ms
        if budget.tokens < 1:
            self._refuse("deltas_per_second")
            raise CapacityExceededError(f"Edits are limited to {rate} per second per person.")
        budget.tokens -= 1

    def check_lock_quota(self, session_id: str, identity: Identity, target_id: str) -> None:
        """Bound the locks one person holds or waits for, which every snapshot then carries."""
        own = {
            lock.target_id
            for lock in self.hub.lock_manager(session_id).active_locks()
            if lock.locked_by_agent_id == identity.agent_id
        }
        if target_id not in own and len(own) >= self.settings.ws_max_locks_per_user:
            self._refuse("locks_per_user")
            raise CapacityExceededError(
                f"You already hold or wait for {len(own)} locks, which is the limit per person."
            )

    async def _heavy[T](self, work: Callable[[], T]) -> T:
        """Run CPU-bound work off the event loop, refusing what the thread pool cannot absorb."""
        if self._exports_in_flight >= self.settings.gateway_max_concurrent_exports:
            self._refuse("concurrent_exports")
            raise CapacityExceededError("The gateway is busy with other exports. Retry shortly.")
        self._exports_in_flight += 1
        try:
            return await asyncio.to_thread(work)
        finally:
            self._exports_in_flight -= 1

    async def export(self, session_id: str, archive_format: ArchiveFormat) -> ProjectArchive:
        bundle = self.hub.latest_snapshot(session_id).bundle_state
        try:
            archive = await self._heavy(lambda: export_bundle(bundle, archive_format))
        except ProjectSynthesisError:
            # A refusal for being busy is counted among refusals, not as a project that failed.
            self.metrics.exports["rejected"] += 1
            raise
        self.metrics.exports["ok"] += 1
        return archive

    async def verify(self, session_id: str) -> InvariantReport:
        """Check the session's invariants over the history it still holds."""
        timeline = self.hub.timeline(session_id)
        return await self._heavy(lambda: verify_timeline(timeline))

    async def fork_session(
        self, identity: Identity, source_id: str, frame_seq: int
    ) -> CreateSessionResponse:
        """Branch another session at one of its frames, into a new session this person owns."""
        self._admit_session(identity)
        session_id = f"s-{uuid.uuid4().hex}"
        snapshot = await self.hub.fork_session(source_id, session_id, frame_seq)
        self._start_runtime(identity, session_id)
        return CreateSessionResponse(session_id=session_id, snapshot=snapshot, run_started=False)

    def gauges(self) -> tuple[tuple[str, str, float], ...]:
        history = sum(self.hub.history_bytes(session_id) for session_id in self._sessions)
        return (
            ("rasb_sessions", "Open sessions.", len(self._sessions)),
            ("rasb_connections", "Open sockets.", len(self._connections)),
            ("rasb_runs_active", "Swarm runs in progress.", self.active_runs()),
            ("rasb_exports_in_flight", "Exports being built.", self._exports_in_flight),
            ("rasb_replay_history_bytes", "Bytes held by replay logs.", history),
        )

    def active_runs(self) -> int:
        return sum(runtime.run is not None for runtime in self._sessions.values())

    def _canvas_connection(
        self,
        websocket: WebSocket,
        identity: Identity,
        session_id: str,
        runtime: _SessionRuntime,
        listener: SessionListener,
        snapshot: SwarmSnapshot,
        backlog: ThoughtBacklog,
        last_seen: int | None,
    ) -> CanvasConnection:
        """Build a canvas client from a settled picture of the session. This never suspends.

        A reconnecting viewer gets everything after what it last saw. A new one gets recent history.
        """
        if last_seen is None:
            replay, gap = backlog.events[-MAX_THOUGHT_HISTORY:], False
        else:
            replay, gap = backlog.events, backlog.gap
        now_ms = self.clock()
        compositor = CanvasCompositor(snapshot.bundle_state)
        for other in self._connections:
            presence = other.presence() if other.session_id == session_id else None
            if presence is not None:
                compositor.place_human(*presence, now_ms)
        hello = CanvasHello(
            session_id=session_id,
            user_id=identity.user_id,
            agent_id=identity.agent_id,
            is_owner=runtime.owner == identity.user_id,
            latest_thought_seq=backlog.latest_seq,
            replayed_thoughts=len(replay),
            thought_gap=gap,
            frame_seq=snapshot.frame_seq,
            frontend=snapshot.bundle_state.frontend,
        )
        return CanvasConnection(
            self,
            websocket,
            identity,
            session_id,
            listener,
            (
                hello.model_dump_json(),
                *(thought_entry(event).model_dump_json() for event in replay),
            ),
            compositor,
        )

    async def serve(
        self,
        websocket: WebSocket,
        identity: Identity,
        session_id: str,
        last_seen: int | None,
        view: SessionView | None = None,
    ) -> None:
        """Serve one client. Everything up to the first await below it happens atomically.

        Subscribing, reading the backlog and taking the snapshot have no suspension point between
        them, so no thought or state can land in a gap, or be seen twice, across the switch from
        replay to the live stream.
        """
        try:
            runtime = self._runtime(session_id)
            listener = self.hub.subscribe(session_id, lossless=True)
            snapshot = self.hub.latest_snapshot(session_id)
            backlog = self.hub.thought_backlog(session_id, last_seen)
        except SessionNotFoundError:
            await websocket.close(WS_SESSION_NOT_FOUND, "Session not found")
            return
        try:
            connection = self._connect(
                websocket,
                identity,
                session_id,
                runtime,
                listener,
                snapshot,
                backlog,
                last_seen,
                view,
            )
        except BaseException:
            # A lossless listener nobody reads would hold every thought from now on.
            listener.close()
            raise
        self._connections.add(connection)
        runtime.connections[identity.user_id] = runtime.connections.get(identity.user_id, 0) + 1
        try:
            await connection.run()
        finally:
            self._connections.discard(connection)
            # Someone with a second tab open has not left.
            if runtime.connections[identity.user_id] == 1:
                await connection.announce_departure()
            await self._release_on_last_departure(session_id, runtime, identity)

    def _connect(
        self,
        websocket: WebSocket,
        identity: Identity,
        session_id: str,
        runtime: _SessionRuntime,
        listener: SessionListener,
        snapshot: SwarmSnapshot,
        backlog: ThoughtBacklog,
        last_seen: int | None,
        view: SessionView | None,
    ) -> ClientConnection:
        if view == "canvas":
            return self._canvas_connection(
                websocket, identity, session_id, runtime, listener, snapshot, backlog, last_seen
            )
        replay = backlog.events if last_seen is not None else ()
        sync = SyncFrame(
            session_id=session_id,
            user_id=identity.user_id,
            agent_id=identity.agent_id,
            snapshot=snapshot,
            latest_thought_seq=backlog.latest_seq,
            replayed_thoughts=len(replay),
            thought_gap=backlog.gap if last_seen is not None else False,
        )
        frames = (sync.model_dump_json(), *(event.model_dump_json() for event in replay))
        return ClientConnection(self, websocket, identity, session_id, listener, frames)

    async def _release_on_last_departure(
        self, session_id: str, runtime: _SessionRuntime, identity: Identity
    ) -> None:
        """A second tab may still be working under these locks, so only the last one frees them."""
        remaining = runtime.connections[identity.user_id] - 1
        runtime.connections[identity.user_id] = remaining
        if remaining > 0:
            return
        del runtime.connections[identity.user_id]
        # Asked of the lock manager, which is the truth, rather than tracked alongside it.
        with contextlib.suppress(SessionNotFoundError):
            manager = self.hub.lock_manager(session_id)
            targets = {
                lock.target_id
                for lock in manager.active_locks()
                if lock.locked_by_agent_id == identity.agent_id
            }
            for target_id in sorted(targets):
                await manager.release(target_id, identity.agent_id)


def _empty_bundle(session_id: str) -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id=session_id,
        frontend=SiteBlueprint(site_id=session_id, title="Untitled site", theme="minimal"),
        backend=BackendBlueprint(service_name=session_id, database_dialect=DatabaseDialect.SQLITE),
    )


class BodyLimitMiddleware:
    """Enforced ahead of FastAPI, which reads a whole body before any dependency can reject it."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self._app = app
        self._limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self._limit:
            response = _problem_response(
                ProblemDetails(
                    status=413, title="Request too large", detail="The body is too large."
                )
            )
            await response(scope, receive, send)
            return
        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self._limit:
                # FastAPI turns anything but an HTTPException raised here into a misleading 400.
                raise HTTPException(status_code=413, detail="The body is too large.")
            return message

        await self._app(scope, counting_receive, send)


def _problem_response(problem: ProblemDetails, headers: dict[str, str] | None = None) -> Response:
    body = problem.to_json_dict()
    request_id = current_request_id()
    if request_id is not None:
        body["trace_id"] = request_id
    return JSONResponse(
        body,
        status_code=problem.status or 500,
        media_type=PROBLEM_JSON_MEDIA_TYPE,
        headers=headers,
    )


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SiteBuilderError)
    async def _domain_error(request: Request, error: SiteBuilderError) -> Response:
        problem = error.to_problem().model_copy(update={"instance": request.url.path})
        unauthorized = isinstance(error, AuthenticationRequiredError)
        return _problem_response(problem, {"WWW-Authenticate": "Bearer"} if unauthorized else None)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, error: RequestValidationError) -> Response:
        # The fields are picked one by one, so no part of the rejected input is echoed back.
        details = [
            {"loc": [str(part) for part in item["loc"]], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        problem = ProblemDetails.model_validate(
            {
                "title": "Invalid request",
                "status": 422,
                "detail": "The request failed validation.",
                "instance": request.url.path,
                "errors": details,
            }
        )
        return _problem_response(problem)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, error: HTTPException) -> Response:
        problem = ProblemDetails(
            title=HTTPStatus(error.status_code).phrase,
            status=error.status_code,
            detail=str(error.detail),
            instance=request.url.path,
        )
        return _problem_response(problem, dict(error.headers or {}))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, _error: Exception) -> Response:
        problem = ProblemDetails(
            title="Internal server error",
            status=500,
            detail="The server failed to handle this request.",
            instance=request.url.path,
        )
        return _problem_response(problem)

    @app.exception_handler(WebSocketRequestValidationError)
    async def _invalid_socket_request(
        websocket: WebSocket, _error: WebSocketRequestValidationError
    ) -> None:
        await websocket.close(WS_POLICY_VIOLATION, "Invalid connection parameters")


def _service_of(request: Request) -> GatewayService:
    service: GatewayService = request.app.state.gateway
    return service


def _identity(
    service: Annotated[GatewayService, Depends(_service_of)],
    authorization: Annotated[str | None, Header()] = None,
) -> Identity:
    token = token_from_header(authorization)
    if token is None:
        raise AuthenticationRequiredError("A bearer token is required.")
    return service.authenticator.verify(token)


Service = Annotated[GatewayService, Depends(_service_of)]
CurrentUser = Annotated[Identity, Depends(_identity)]
router = APIRouter()


@router.post("/api/v1/sessions", status_code=201, responses={201: {"model": CreateSessionResponse}})
async def create_session(
    service: Service, identity: CurrentUser, body: CreateSessionRequest | None = None
) -> Response:
    request = body or CreateSessionRequest()
    created = await service.create_session(identity, request.bundle, request.prompt)
    return Response(
        created.model_dump_json(),
        status_code=201,
        media_type=JSON_MEDIA_TYPE,
        headers={"Location": f"/api/v1/sessions/{created.session_id}"},
    )


@router.get(
    "/api/v1/sessions/{session_id}",
    responses={200: {"model": SwarmSnapshot}},
    dependencies=[Depends(_identity)],
)
async def get_session(session_id: str, service: Service) -> Response:
    snapshot = service.hub.latest_snapshot(session_id)
    return Response(snapshot.model_dump_json(), media_type=JSON_MEDIA_TYPE)


@router.get(
    "/api/v1/sessions/{session_id}/timeline",
    responses={200: {"model": TimelineResponse}},
    dependencies=[Depends(_identity)],
)
async def get_timeline(session_id: str, service: Service) -> Response:
    backlog = service.hub.thought_backlog(session_id)
    response = TimelineResponse(
        timeline=service.hub.timeline(session_id),
        thoughts=backlog.events,
        thought_gap=backlog.gap,
        latest_thought_seq=backlog.latest_seq,
    )
    # A long session's log holds a full bundle per frame, which is too much to encode on the loop.
    return Response(await asyncio.to_thread(response.model_dump_json), media_type=JSON_MEDIA_TYPE)


@router.get(
    "/api/v1/sessions/{session_id}/export",
    response_class=Response,
    responses={
        200: {
            "description": "The project as a reproducible archive.",
            "content": {"application/zip": {}, "application/gzip": {}},
        }
    },
    dependencies=[Depends(_identity)],
)
async def export_session(
    session_id: str,
    service: Service,
    archive_format: Annotated[ArchiveFormat, Query(alias="format")] = ArchiveFormat.ZIP,
) -> Response:
    # Finished before the response starts, so a bundle that cannot be exported still gets a problem
    # document. Compiling and compressing a large project is too much work for the event loop.
    archive = await service.export(session_id, archive_format)
    return StreamingResponse(
        archive.chunks(),
        media_type=archive.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{archive.filename}"',
            "Content-Length": str(len(archive.content)),
        },
    )


@router.post(
    "/api/v1/sessions/{session_id}/fork",
    status_code=201,
    responses={201: {"model": CreateSessionResponse}},
)
async def fork_session(
    session_id: str,
    service: Service,
    identity: CurrentUser,
    from_seq: Annotated[int, Query(ge=0)],
) -> Response:
    """Branch the session at frame ``from_seq`` into a new one, owned by whoever asked."""
    forked = await service.fork_session(identity, session_id, from_seq)
    return Response(
        forked.model_dump_json(),
        status_code=201,
        media_type=JSON_MEDIA_TYPE,
        headers={"Location": f"/api/v1/sessions/{forked.session_id}"},
    )


@router.get(
    "/api/v1/sessions/{session_id}/history",
    responses={200: {"model": HistoryResponse}},
    dependencies=[Depends(_identity)],
)
async def get_history(session_id: str, service: Service) -> Response:
    """The replay window in its compact form: one base layout, then each delta and its locks."""
    timeline = service.hub.timeline(session_id)
    return Response(
        await asyncio.to_thread(lambda: history_of(timeline).model_dump_json()),
        media_type=JSON_MEDIA_TYPE,
    )


@router.get(
    "/api/v1/sessions/{session_id}/verify",
    responses={200: {"model": InvariantReport}},
    dependencies=[Depends(_identity)],
)
async def verify_session(session_id: str, service: Service) -> Response:
    report = await service.verify(session_id)
    return Response(report.model_dump_json(), media_type=JSON_MEDIA_TYPE)


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, service: Service, identity: CurrentUser) -> Response:
    await service.close_session(identity, session_id)
    return Response(status_code=204)


@router.get("/healthz", include_in_schema=False)
async def healthz() -> Response:
    """Liveness for orchestrators. It needs no token and says nothing about any session."""
    return Response('{"status":"ok"}', media_type=JSON_MEDIA_TYPE)


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
    dependencies=[Depends(_identity)],
)
async def metrics(service: Service) -> Response:
    return PlainTextResponse(
        service.metrics.render(service.gauges()),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/canvas/{session_id}", response_class=HTMLResponse, include_in_schema=False)
async def canvas_viewer() -> Response:
    """Serve the viewer, which authenticates itself on the socket. A page load cannot send a header.

    The page is the same static bytes for every session and caller, and it reflects nothing from
    the request, so serving it reveals nothing, including whether the session exists.
    """
    return viewer_response()


@router.websocket("/ws/sessions/{session_id}")
async def session_socket(
    websocket: WebSocket,
    session_id: str,
    last_seen_thought_seq: Annotated[int | None, Query(ge=0)] = None,
    view: SessionView | None = None,
) -> None:
    service: GatewayService = websocket.app.state.gateway
    subprotocols = websocket.scope.get("subprotocols", [])
    token = token_from_subprotocols(subprotocols) or token_from_header(
        websocket.headers.get("authorization")
    )
    try:
        if token is None:
            raise AuthenticationRequiredError("A bearer token is required.")
        identity = service.authenticator.verify(token)
    except AuthenticationRequiredError:
        # Closing before accepting makes the server answer the handshake with a plain 403.
        await websocket.close(WS_POLICY_VIOLATION, "Authentication required")
        return
    try:
        service.admit_socket(identity)
    except CapacityExceededError:
        await websocket.close(WS_POLICY_VIOLATION, "Too many connections")
        return
    try:
        await websocket.accept(subprotocol=SUBPROTOCOL if SUBPROTOCOL in subprotocols else None)
        await service.serve(websocket, identity, session_id, last_seen_thought_seq, view)
    finally:
        service.release_socket(identity)


def create_app(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    bus: EventBus | None = None,
    hub_clock: Clock | None = None,
    clock: Clock = monotonic_ms,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> FastAPI:
    """Build the application. Run it with ``uvicorn ...gateway.service:create_app --factory``."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        secret = resolved.gateway_secret
        if secret is None:
            raise GatewayConfigurationError(
                "RASB_GATEWAY_SECRET is required: unauthenticated clients could spend model budget."
            )
        owned_provider = TieredLLMProvider(resolved) if provider is None else None
        now_ms = hub_clock or system_now_ms
        service = GatewayService(
            resolved,
            provider or cast("LLMProvider", owned_provider),
            TokenAuthenticator(secret, clock=now_ms),
            bus=bus,
            hub_clock=now_ms,
            clock=clock,
            sleep=sleep,
        )
        app.state.gateway = service
        service.start()
        try:
            yield
        finally:
            await service.stop()
            if owned_provider is not None:
                await owned_provider.aclose()

    install_log_correlation()
    app = FastAPI(title="Realtime Agentic Site Builder", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, limit=resolved.max_request_bytes)
    # Added last, so it runs first: every response, including a refused body, gets an id.
    app.add_middleware(RequestIdMiddleware)
    _install_error_handlers(app)
    app.include_router(router)
    return app
