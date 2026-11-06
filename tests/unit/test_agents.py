import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

import anthropic
import httpx
import httpx2
import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from realtime_agentic_site_builder.agents.graph import (
    BACKEND,
    CODER,
    create_swarm_graph,
    route_after_critic,
)
from realtime_agentic_site_builder.agents.nodes import backend_node, coder_node, critic_node
from realtime_agentic_site_builder.agents.provider import (
    MAX_VALIDATION_ATTEMPTS,
    OPENROUTER_MAX_RETRIES,
    AnthropicLLMClient,
    LLMClient,
    LLMRequest,
    LLMResult,
    MockLLMProvider,
    OpenRouterLLMClient,
    TieredLLMProvider,
    TokenUsage,
)
from realtime_agentic_site_builder.agents.runtime import (
    IDENTITIES,
    LOCK_POLL_S,
    NodeContext,
    SwarmRuntime,
    wireframe_code,
)
from realtime_agentic_site_builder.agents.schemas import (
    ClassFix,
    CoderOutput,
    ComponentSpec,
    CriticIssue,
    CriticVerdict,
    EndpointSpec,
    HandlerOutput,
    ModelFieldSpec,
    ModelSpec,
    StyleOutput,
    SwarmPlan,
)
from realtime_agentic_site_builder.agents.state import AgentWorkflowState, initial_state
from realtime_agentic_site_builder.agents.tsx import find_tsx_issue
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import (
    CursorEvent,
    InMemoryEventBus,
    SwarmSessionHub,
)
from realtime_agentic_site_builder.core.patch_engine import (
    BACKEND_DOCUMENT_LOCK,
    FRONTEND_DOCUMENT_LOCK,
)
from realtime_agentic_site_builder.domain.backend import (
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
    ElementLockConflictError,
    LLMConfigurationError,
    LLMProviderError,
    LockAcquisitionTimeoutError,
    SessionNotFoundError,
)
from realtime_agentic_site_builder.domain.swarm import (
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    ElementLock,
    LockState,
    StreamDelta,
    SwarmSnapshot,
)

SESSION = "session-1"
PROMPT = "Build a landing page with a contact form"
ENDPOINT_ID = "api_endpoint_post_leads"
BROKEN_TSX = "export default function Hero() {\n  return <div>;\n}"
GOOD_TSX = "export default function Hero() {\n  return <div>Hi</div>;\n}"
GOOD_HANDLER = "async def handler() -> None:\n    return None"
HERO_ROLE = AgentRole.FRONTEND_CODER


class FakeClock:
    def __init__(self, now: int = 1_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class RecordingHub(SwarmSessionHub):
    """Sees every frame and commit before the real hub, including who held the lock at the time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.frames: list[AgentCursorFrame] = []
        self.commits: list[tuple[StreamDelta, str | None]] = []

    async def broadcast_frame(self, session_id: str, frame: AgentCursorFrame) -> None:
        self.frames.append(frame)
        await super().broadcast_frame(session_id, frame)

    async def apply_delta(self, session_id: str, delta: StreamDelta) -> SwarmSnapshot:
        holder = self.lock_manager(session_id).live_holder(delta.target_id, self.now_ms())
        self.commits.append((delta, holder))
        return await super().apply_delta(session_id, delta)

    def entered(self) -> list[str]:
        return [
            str(frame.current_thought.tool_args["node"])
            for frame in self.frames
            if frame.current_thought is not None
            and frame.current_thought.tool_name == "graph.enter"
            and frame.current_thought.tool_args is not None
        ]


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
    """Runs a callback at the moment of each model call, while the node's locks are held."""

    def __init__(self, inner: MockLLMProvider, hook: Hook) -> None:
        self.inner = inner
        self._hook = hook

    def client_for(self, role: AgentRole) -> LLMClient:
        return _HookedClient(self.inner.client_for(role), role, self._hook)


def empty_bundle(
    components: tuple[ComponentNode, ...] = (),
    models: tuple[DataModelSchema, ...] = (),
) -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id="project-1",
        frontend=SiteBlueprint(site_id="site-1", title="Acme", theme="dark", components=components),
        backend=BackendBlueprint(
            service_name="acme-api", database_dialect=DatabaseDialect.SQLITE, models=models
        ),
    )


def make_settings(**overrides: Any) -> Settings:
    return Settings(**{"broadcast_fps": 240, **overrides})


async def make_hub(
    bundle: FullstackProjectBundle | None = None,
    *,
    clock: FakeClock | None = None,
    **settings: Any,
) -> RecordingHub:
    kwargs: dict[str, Any] = {"clock": clock} if clock is not None else {}
    hub = RecordingHub(make_settings(**settings), bus=InMemoryEventBus(), **kwargs)
    await hub.create_session(SESSION, bundle or empty_bundle())
    return hub


async def run_swarm(
    hub: SwarmSessionHub, provider: Any, bundle: FullstackProjectBundle | None = None
) -> AgentWorkflowState:
    graph = create_swarm_graph(hub, provider)
    current = hub.latest_snapshot(SESSION).bundle_state if bundle is None else bundle
    return cast("AgentWorkflowState", await graph.ainvoke(initial_state(SESSION, PROMPT, current)))


def component(state: AgentWorkflowState, component_id: str) -> ComponentNode:
    return next(c for c in state["current_bundle"].frontend.components if c.id == component_id)


def make_context(
    hub: SwarmSessionHub, provider: Any = None, *, role: AgentRole = AgentRole.CRITIC_LINTER
) -> NodeContext:
    state = initial_state(SESSION, PROMPT, hub.latest_snapshot(SESSION).bundle_state)
    return NodeContext(SwarmRuntime(hub, provider or MockLLMProvider()), state, role)


def thoughts_with(hub: RecordingHub, tool_name: str) -> list[Any]:
    return [
        frame.current_thought
        for frame in hub.frames
        if frame.current_thought is not None and frame.current_thought.tool_name == tool_name
    ]


# --- settings and errors --------------------------------------------------------------------


def test_default_tiers_use_the_blueprints_models() -> None:
    settings = Settings()
    assert settings.architect_model == "anthropic:claude-sonnet-5-5"
    assert settings.coder_model == "openrouter:deepseek/deepseek-v4.1-flash"
    assert settings.stylist_model == "anthropic:claude-haiku-5-5"
    assert settings.critic_model == "openrouter:openai/gpt-4o-mini"


def test_tier_routes_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RASB_CODER_MODEL", "openrouter:z-ai/glm-5.3-flash")
    monkeypatch.setenv("RASB_AGENT_MAX_ITERATIONS", "7")
    settings = Settings()
    assert settings.coder_model == "openrouter:z-ai/glm-5.3-flash"
    assert settings.agent_max_iterations == 7


@pytest.mark.parametrize("route", ["claude-sonnet-5-5", "bedrock:x", "anthropic:", "anthropic:a b"])
def test_malformed_routes_are_rejected(route: str) -> None:
    with pytest.raises(ValidationError):
        Settings(architect_model=route)


def test_keys_stay_out_of_repr() -> None:
    settings = make_settings(anthropic_api_key=SecretStr("shh"))
    assert "shh" not in repr(settings)


def test_provider_errors_render_as_problems() -> None:
    problem = LLMProviderError("HTTP 500", model="m").to_problem().to_json_dict()
    assert problem["status"] == 502
    assert problem["model"] == "m"
    assert LLMConfigurationError("no key").to_problem().to_json_dict()["status"] == 500


# --- tiered routing -------------------------------------------------------------------------


def test_every_role_routes_to_its_tier() -> None:
    provider = TieredLLMProvider(
        make_settings(anthropic_api_key=SecretStr("k"), openrouter_api_key=SecretStr("k"))
    )
    routes = {role: provider.client_for(role) for role in AgentRole}
    assert isinstance(routes[AgentRole.ARCHITECT], AnthropicLLMClient)
    assert isinstance(routes[AgentRole.STYLER], AnthropicLLMClient)
    assert routes[AgentRole.STYLER] is routes[AgentRole.COPYWRITER]
    assert isinstance(routes[AgentRole.FRONTEND_CODER], OpenRouterLLMClient)
    assert routes[AgentRole.FRONTEND_CODER] is routes[AgentRole.BACKEND_ENGINEER]
    assert isinstance(routes[AgentRole.CRITIC_LINTER], OpenRouterLLMClient)
    assert routes[AgentRole.FRONTEND_CODER] is not routes[AgentRole.CRITIC_LINTER]
    assert routes[AgentRole.ARCHITECT] is not routes[AgentRole.STYLER]


async def test_clients_are_released_on_close() -> None:
    http = httpx.AsyncClient()
    sdk = anthropic.AsyncAnthropic(api_key="k")
    provider = TieredLLMProvider(
        make_settings(openrouter_api_key=SecretStr("k")), anthropic_client=sdk, http_client=http
    )
    await provider.aclose()
    assert http.is_closed
    assert sdk.is_closed()


async def test_close_tolerates_a_single_endpoint_setup() -> None:
    only_openrouter = make_settings(
        openrouter_api_key=SecretStr("k"),
        architect_model="openrouter:anthropic/claude-sonnet-5.5",
        stylist_model="openrouter:anthropic/claude-haiku-5.5",
    )
    await TieredLLMProvider(only_openrouter).aclose()
    only_anthropic = make_settings(
        anthropic_api_key=SecretStr("k"),
        coder_model="anthropic:claude-haiku-5-5",
        critic_model="anthropic:claude-haiku-5-5",
    )
    await TieredLLMProvider(only_anthropic).aclose()


@pytest.mark.parametrize(
    "settings, variable",
    [
        ({"openrouter_api_key": SecretStr("k")}, "RASB_ANTHROPIC_API_KEY"),
        ({"anthropic_api_key": SecretStr("k")}, "RASB_OPENROUTER_API_KEY"),
    ],
)
def test_missing_key_fails_at_startup_and_names_the_variable(
    settings: dict[str, Any], variable: str
) -> None:
    with pytest.raises(LLMConfigurationError, match=variable):
        TieredLLMProvider(make_settings(**settings))


# --- anthropic client -----------------------------------------------------------------------


def anthropic_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> Any:
    return anthropic.AsyncAnthropic(
        api_key="k",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def anthropic_reply(*content: dict[str, Any], stop_reason: str = "end_turn") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": list(content),
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 3, "output_tokens": 4},
        },
    )


def text_block(payload: BaseModel | str) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else payload.model_dump_json()
    return {"type": "text", "text": text}


def request_for[T: BaseModel](schema: type[T]) -> LLMRequest[T]:
    return LLMRequest(role=HERO_ROLE, system="sys", prompt="do it", output_schema=schema)


@pytest.mark.parametrize(
    "model, effort, has_fallback",
    [
        ("claude-sonnet-5-5", "high", True),
        ("claude-haiku-5-5", "medium", False),
        ("claude-custom", None, False),
    ],
)
async def test_anthropic_wire_format(model: str, effort: str | None, has_fallback: bool) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return anthropic_reply(text_block(CoderOutput(code=GOOD_TSX)))

    client = AnthropicLLMClient(anthropic_client(handler), model)
    result = await client.generate_structured(request_for(CoderOutput))

    assert result.output.code == GOOD_TSX
    assert result.usage == TokenUsage(input_tokens=3, output_tokens=4)
    assert result.model == model
    body = json.loads(seen[0].content)
    assert body["model"] == model
    assert body["system"] == "sys"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"].get("effort") == effort
    assert "tool_choice" not in body, "forced tool use is rejected by the current models"
    assert ("fallbacks" in body) is has_fallback
    betas = seen[0].headers.get("anthropic-beta", "")
    assert ("server-side-fallback-2026-07-01" in betas) is has_fallback


async def test_anthropic_refusal_is_an_error() -> None:
    client = AnthropicLLMClient(
        anthropic_client(lambda _r: anthropic_reply(stop_reason="refusal")), "m"
    )
    with pytest.raises(LLMProviderError, match="refused"):
        await client.generate_structured(request_for(CoderOutput))


async def test_anthropic_discards_text_from_before_a_fallback() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return anthropic_reply(
            text_block('{"code": "partial'),
            {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
            text_block(CoderOutput(code=GOOD_TSX)),
        )

    client = AnthropicLLMClient(anthropic_client(handler), "m")
    assert (await client.generate_structured(request_for(CoderOutput))).output.code == GOOD_TSX


async def test_anthropic_ignores_thinking_blocks() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return anthropic_reply(
            {"type": "thinking", "thinking": "", "signature": "s"},
            text_block(CoderOutput(code=GOOD_TSX)),
        )

    client = AnthropicLLMClient(anthropic_client(handler), "m")
    assert (await client.generate_structured(request_for(CoderOutput))).output.code == GOOD_TSX


async def test_anthropic_api_failure_names_only_the_error_type() -> None:
    client = AnthropicLLMClient(
        anthropic_client(lambda _r: httpx2.Response(500, json={"error": {"message": "secret"}})),
        "m",
    )
    with pytest.raises(LLMProviderError) as caught:
        await client.generate_structured(request_for(CoderOutput))
    assert "secret" not in caught.value.detail
    assert isinstance(caught.value.__cause__, anthropic.APIError)


async def test_invalid_reply_is_retried_with_the_error_and_usage_totalled() -> None:
    bodies: list[dict[str, Any]] = []
    replies = [text_block('{"code": ""}'), text_block(CoderOutput(code=GOOD_TSX))]

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return anthropic_reply(replies.pop(0))

    client = AnthropicLLMClient(anthropic_client(handler), "m")
    result = await client.generate_structured(request_for(CoderOutput))

    assert result.output.code == GOOD_TSX
    assert result.usage.total_tokens == 14
    assert bodies[0]["messages"][0]["content"] == "do it"
    assert "previous reply was invalid" in bodies[1]["messages"][0]["content"]
    assert "code" in bodies[1]["messages"][0]["content"]


async def test_reply_that_never_validates_fails_after_its_attempts() -> None:
    calls = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return anthropic_reply(text_block("not json"))

    client = AnthropicLLMClient(anthropic_client(handler), "m")
    with pytest.raises(LLMProviderError, match="failed validation"):
        await client.generate_structured(request_for(CoderOutput))
    assert calls == MAX_VALIDATION_ATTEMPTS


# --- openrouter client ----------------------------------------------------------------------


def openrouter_reply(payload: BaseModel) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": payload.model_dump_json()}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        },
    )


def openrouter_client(
    handler: Callable[[httpx.Request], httpx.Response], pauses: list[float] | None = None
) -> OpenRouterLLMClient:
    async def record(seconds: float) -> None:
        if pauses is not None:
            pauses.append(seconds)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterLLMClient(http, "vendor/model", "key-1", sleep=record)


async def test_openrouter_wire_format() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return openrouter_reply(CoderOutput(code=GOOD_TSX))

    result = await openrouter_client(handler).generate_structured(request_for(CoderOutput))

    assert result.output.code == GOOD_TSX
    assert result.usage == TokenUsage(input_tokens=5, output_tokens=6)
    assert seen[0].headers["authorization"] == "Bearer key-1"
    body = json.loads(seen[0].content)
    assert body["model"] == "vendor/model"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"]["json_schema"]["name"] == "CoderOutput"
    assert body["provider"] == {"require_parameters": True}


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_openrouter_retries_transient_status(status: int) -> None:
    replies = [httpx.Response(status), openrouter_reply(CoderOutput(code=GOOD_TSX))]
    pauses: list[float] = []
    client = openrouter_client(lambda _r: replies.pop(0), pauses)
    assert (await client.generate_structured(request_for(CoderOutput))).output.code == GOOD_TSX
    assert pauses == [1.0]


async def test_openrouter_backs_off_then_gives_up() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    pauses: list[float] = []
    with pytest.raises(LLMProviderError, match="HTTP 429"):
        await openrouter_client(handler, pauses).generate_structured(request_for(CoderOutput))
    assert calls == OPENROUTER_MAX_RETRIES + 1
    assert pauses == [1.0, 2.0]


async def test_openrouter_does_not_retry_a_client_error() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    with pytest.raises(LLMProviderError, match="HTTP 401"):
        await openrouter_client(handler).generate_structured(request_for(CoderOutput))
    assert calls == 1


async def test_openrouter_transport_failure_is_retried_and_its_message_withheld() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("detail with a secret")

    with pytest.raises(LLMProviderError, match="ConnectError") as caught:
        await openrouter_client(handler).generate_structured(request_for(CoderOutput))
    assert "secret" not in caught.value.detail
    assert isinstance(caught.value.__cause__, httpx.ConnectError)


@pytest.mark.parametrize(
    "reply, message",
    [
        (httpx.Response(200, content=b"<html>"), "not JSON"),
        (httpx.Response(200, json=[1]), "returned an error"),
        (httpx.Response(200, json={"error": {"code": 502}}), "returned an error"),
        (httpx.Response(200, json={"choices": []}), "malformed"),
        (httpx.Response(200, json={"choices": [{}]}), "malformed"),
        (httpx.Response(200, json={"choices": None}), "malformed"),
        (
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": "many"},
                },
            ),
            "malformed",
        ),
    ],
)
async def test_openrouter_rejects_unusable_responses(reply: httpx.Response, message: str) -> None:
    client = openrouter_client(lambda _r: reply)
    with pytest.raises(LLMProviderError, match=message):
        await client.generate_structured(request_for(CoderOutput))


async def test_openrouter_tolerates_missing_usage() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": CoderOutput(code=GOOD_TSX).model_dump_json()}}]}
        return httpx.Response(200, json=body)

    result = await openrouter_client(handler).generate_structured(request_for(CoderOutput))
    assert result.usage.total_tokens == 0


# --- mock provider --------------------------------------------------------------------------


async def test_mock_is_deterministic_and_records_calls() -> None:
    first, second = MockLLMProvider(), MockLLMProvider()
    client = first.client_for(AgentRole.ARCHITECT)
    plan = await client.generate_structured(request_for(SwarmPlan))
    again = await second.client_for(AgentRole.ARCHITECT).generate_structured(request_for(SwarmPlan))
    assert plan.output == again.output
    assert plan.usage == again.usage
    assert plan.usage.total_tokens > 0
    assert [call.output_schema for call in first.calls] == [SwarmPlan]


async def test_mock_plays_its_script_then_falls_back_to_defaults() -> None:
    provider = MockLLMProvider(
        {HERO_ROLE: [CoderOutput(code=BROKEN_TSX), LLMProviderError("down", model="mock")]}
    )
    client = provider.client_for(HERO_ROLE)
    assert (await client.generate_structured(request_for(CoderOutput))).output.code == BROKEN_TSX
    with pytest.raises(LLMProviderError, match="down"):
        await client.generate_structured(request_for(CoderOutput))
    assert (await client.generate_structured(request_for(CoderOutput))).output.code != BROKEN_TSX


async def test_mock_rejects_a_script_of_the_wrong_schema() -> None:
    provider = MockLLMProvider({HERO_ROLE: [HandlerOutput(handler_code=GOOD_HANDLER)]})
    with pytest.raises(LLMProviderError, match="scripted a HandlerOutput"):
        await provider.client_for(HERO_ROLE).generate_structured(request_for(CoderOutput))


# --- plan validation ------------------------------------------------------------------------


def contact_spec(path: str | None = "/api/v1/leads", **overrides: Any) -> ComponentSpec:
    return ComponentSpec.model_validate(
        {
            "id": "contact",
            "component_type": ComponentType.CONTACT_FORM,
            "title": "Contact",
            "brief": "Form",
            "binds_path": path,
            **overrides,
        }
    )


def lead_model(python_type: str = "EmailStr") -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "name": "Lead",
            "table_name": "leads",
            "fields": [{"name": "email", "python_type": python_type}],
        }
    )


def post_endpoint(
    method: HttpMethod = HttpMethod.POST, request_model_ref: str | None = "Lead"
) -> EndpointSpec:
    return EndpointSpec(
        path="/api/v1/leads",
        method=method,
        summary="Create",
        brief="Store it",
        request_model_ref=request_model_ref,
    )


def test_plan_accepts_a_consistent_contact_form() -> None:
    plan = SwarmPlan(
        components=(contact_spec(),), models=(lead_model(),), endpoints=(post_endpoint(),)
    )
    assert plan.components[0].endpoint_id == ENDPOINT_ID


@pytest.mark.parametrize(
    "components, models, endpoints, message",
    [
        ((contact_spec(), contact_spec()), (), (), "duplicate component ids"),
        ((), (lead_model(), lead_model()), (), "duplicate model names"),
        ((), (), (post_endpoint(), post_endpoint()), "duplicate endpoints"),
        ((contact_spec(),), (), (), "unplanned endpoint"),
        ((contact_spec(None),), (), (), "must bind to a POST"),
        (
            (contact_spec(binds_method=HttpMethod.GET),),
            (),
            (post_endpoint(HttpMethod.GET),),
            "must bind to a POST",
        ),
        ((contact_spec(),), (), (post_endpoint(request_model_ref=None),), "request model"),
        ((contact_spec(),), (), (post_endpoint(),), "request model"),
        ((contact_spec(),), (lead_model("str"),), (post_endpoint(),), "request model"),
    ],
)
def test_plan_rejects_inconsistent_references(
    components: tuple[ComponentSpec, ...],
    models: tuple[ModelSpec, ...],
    endpoints: tuple[EndpointSpec, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SwarmPlan(components=components, models=models, endpoints=endpoints)


def test_model_spec_rejects_duplicate_fields_and_converts_to_the_domain() -> None:
    with pytest.raises(ValidationError, match="duplicate fields"):
        ModelSpec(
            name="Lead",
            table_name="leads",
            fields=(
                ModelFieldSpec(name="a", python_type="str"),
                ModelFieldSpec(name="a", python_type="int"),
            ),
        )
    domain = lead_model().to_domain()
    assert domain.has_email_field
    assert domain.fields["email"].python_type == "EmailStr"


def test_every_output_schema_is_accepted_by_the_structured_output_transform() -> None:
    for schema in (SwarmPlan, CoderOutput, HandlerOutput, StyleOutput, CriticVerdict):
        assert anthropic.transform_schema(schema)["type"] == "object"


# --- tsx checker ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "",
        "<div/>",
        "<div></div>",
        "<></>",
        "const a = <><b/></>;",
        "return <section className='p-8' id=\"x\">Hi</section>;",
        '<a href="x>y">t</a>',
        "<a b={() => <c d={1}/>}>t</a>",
        "<p>don't panic, it's fine</p>",
        "<p>a > b</p>",
        "<p>{items.map((i) => <li key={i}>{i}</li>)}</p>",
        "<p>{/* a comment with </div> */}</p>",
        "const ok = a < b && c > d;",
        "const ok = 1<2;",
        "const [v, set] = useState<string>('');",
        "const m: Map<string, number> = new Map();",
        "function f<T>(x: T) { return x; }",
        "const f = <T,>(x: T) => x;",
        "const f = <T extends object>(x: T) => x;",
        "const x = cond ? <a/> : <b/>;",
        "const x = cond && <a/>;",
        "export default <Foo.Bar a='1'/>;",
        "// just a comment with <div>",
        "// comment\nconst a = 1;",
        "/* block <div> */ const a = 1;",
        "const s = 'it\\'s';",
        'const s = "a\\"b";',
        "const t = `plain <div> text`;",
        "const t = `a ${b + `c ${d}`} e`;",
        "const t = `escaped \\` tick`;",
        "const a = [1, {b: (2)}];",
    ],
)
def test_tsx_checker_accepts_balanced_code(code: str) -> None:
    assert find_tsx_issue(code) is None


@pytest.mark.parametrize(
    "code, line, message",
    [
        ("<div>", 1, "<div> is never closed"),
        ("const a = 1;\n<div>\n", 2, "<div> is never closed"),
        ("<div><span></div>", 1, "<span> is closed by </div>"),
        ("<>a</div>", 1, "<> is closed by </div>"),
        ("<div>a</>", 1, "<div> is closed by </>"),
        ("return </div>;", 1, "closing tag has no matching"),
        ("<div>a</div x>", 1, "closing tag is malformed"),
        ("<div>a</div", 1, "closing tag is malformed"),
        ("<div", 1, "<div> is never closed"),
        ("<div @a>", 1, "unexpected @"),
        ('<div a="b>', 1, "attribute value is never closed"),
        ("<div a={1>", 1, "{ is never closed"),
        ("<div>{a</div>", 1, "{ is never closed"),
        ("function f() {", 1, "{ is never closed"),
        ("(", 1, "( is never closed"),
        (")", 1, "unexpected )"),
        ("(]", 1, "unexpected ]"),
        ("{)", 1, "unexpected )"),
        ("const a = 'b;", 1, "string opened with ' is never closed"),
        ('const a = "b\n";', 1, 'string opened with " is never closed'),
        ("const a = 'b\\", 1, "string opened with ' is never closed"),
        ("/* never closed", 1, "block comment is never closed"),
        ("const t = `a", 1, "template literal is never closed"),
        ("const t = `a ${b`", 1, "template literal is never closed"),
    ],
)
def test_tsx_checker_reports_structural_problems(code: str, line: int, message: str) -> None:
    issue = find_tsx_issue(code)
    assert issue is not None
    assert issue.line == line
    assert message in issue.message
    assert str(issue) == f"line {line}: {issue.message}"


def test_tsx_checker_survives_hostile_nesting_without_recursing() -> None:
    assert find_tsx_issue("(" * 100_000 + ")" * 100_000) is None
    assert find_tsx_issue("<a>" * 20_000 + "</a>" * 20_000) is None


# --- state and routing ----------------------------------------------------------------------


def test_initial_state_starts_empty() -> None:
    bundle = empty_bundle()
    state = initial_state(SESSION, PROMPT, bundle)
    assert state["current_bundle"] is bundle
    assert state["active_locks"] == ()
    assert state["pending_deltas"] == []
    assert state["agent_thoughts"] == []
    assert state["critic_feedback"] == []
    assert state["iteration"] == 0
    assert state["completed"] is False


@pytest.mark.parametrize(
    "frontend, backend, expected",
    [
        (("hero",), (), [CODER]),
        ((), (ENDPOINT_ID,), [BACKEND]),
        (("hero",), (ENDPOINT_ID,), [CODER, BACKEND]),
        ((), (), "__end__"),
    ],
)
def test_router_sends_rejections_to_the_agent_that_owns_them(
    frontend: tuple[str, ...], backend: tuple[str, ...], expected: Any
) -> None:
    state = initial_state(SESSION, PROMPT, empty_bundle())
    state["retry_frontend"] = frontend
    state["retry_backend"] = backend
    assert route_after_critic(state) == expected


# --- workflow: the happy path ---------------------------------------------------------------


async def test_clean_run_visits_each_node_once_and_completes() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider())

    entered = hub.entered()
    assert entered[0] == "planner"
    assert sorted(entered[1:3]) == ["backend", "coder"]
    assert entered[3:] == ["styler", "critic"]
    assert state["completed"] is True
    assert state["iteration"] == 1
    assert state["critic_feedback"] == []
    assert state["retry_frontend"] == ()
    assert state["retry_backend"] == ()
    await hub.shutdown()


async def test_clean_run_builds_a_bound_contact_form() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider())

    form = component(state, "contact")
    assert form.component_type is ComponentType.CONTACT_FORM
    assert form.connected_endpoint_id == ENDPOINT_ID
    assert "<section" in form.code
    assert "wireframe" not in form.code
    assert [model.name for model in state["current_bundle"].backend.models] == ["Lead"]
    assert [endpoint.id for endpoint in state["current_bundle"].backend.endpoints] == [ENDPOINT_ID]
    assert state["current_bundle"].frontend.title == "Generated site"
    assert state["current_bundle"] == hub.latest_snapshot(SESSION).bundle_state
    await hub.shutdown()


async def test_wireframes_are_stacked_in_reading_order() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider())
    hero, form = state["current_bundle"].frontend.components
    assert hero.box.y == 0
    assert form.box.y == hero.box.bottom
    await hub.shutdown()


async def test_run_is_replayable_from_its_timeline() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider())
    timeline = hub.timeline(SESSION)
    assert sorted(d.delta_seq for d in state["pending_deltas"]) == [
        d.delta_seq for d in timeline.deltas
    ]
    assert {d.agent_id for d in timeline.deltas} <= {i.agent_id for i in IDENTITIES.values()}
    await hub.shutdown()


async def test_planner_keeps_what_the_session_already_has() -> None:
    hero = ComponentNode.create(
        id="hero",
        component_type=ComponentType.HERO,
        title="Kept",
        code=GOOD_TSX,
        box=BoundingBox(x=0, y=0, width=100, height=70),
    )
    model = lead_model().to_domain()
    hub = await make_hub(empty_bundle((hero,), (model,)))
    state = await run_swarm(hub, MockLLMProvider())

    assert component(state, "hero").code == GOOD_TSX
    assert component(state, "hero").title == "Kept"
    assert component(state, "contact").box.y == 70
    assert len(state["current_bundle"].backend.models) == 1
    await hub.shutdown()


async def test_plan_with_nothing_new_commits_nothing() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider({AgentRole.ARCHITECT: [SwarmPlan()]}))
    assert state["pending_deltas"] == []
    assert state["completed"] is True
    await hub.shutdown()


async def test_rerun_on_a_finished_site_skips_endpoints_that_exist() -> None:
    hub = await make_hub()
    provider = MockLLMProvider()
    await run_swarm(hub, provider)
    before = len(provider.calls)
    state = await run_swarm(hub, provider)
    schemas = [call.output_schema for call in provider.calls[before:]]
    assert HandlerOutput not in schemas
    assert CoderOutput not in schemas
    assert state["completed"] is True
    assert "contact" not in {delta.target_id for delta in state["pending_deltas"]}
    await hub.shutdown()


# --- workflow: locks ------------------------------------------------------------------------


async def test_every_commit_is_made_under_its_agents_lock() -> None:
    hub = await make_hub()
    await run_swarm(hub, MockLLMProvider())
    assert hub.commits
    for delta, holder in hub.commits:
        assert holder == delta.agent_id, delta.target_id
    await hub.shutdown()


async def test_locks_are_held_during_generation_and_released_after() -> None:
    hub = await make_hub()
    held_during_call: dict[AgentRole, set[str]] = {}
    targets = (FRONTEND_DOCUMENT_LOCK, BACKEND_DOCUMENT_LOCK, "hero", "contact", ENDPOINT_ID)

    async def probe(role: AgentRole, _request: LLMRequest[Any]) -> None:
        manager = hub.lock_manager(SESSION)
        agent = IDENTITIES[role].agent_id
        held_during_call.setdefault(role, set()).update(
            target for target in targets if manager.live_holder(target, hub.now_ms()) == agent
        )

    state = await run_swarm(hub, HookedProvider(MockLLMProvider(), probe))

    assert held_during_call[AgentRole.ARCHITECT] == {FRONTEND_DOCUMENT_LOCK, BACKEND_DOCUMENT_LOCK}
    assert held_during_call[AgentRole.FRONTEND_CODER] == {"hero", "contact"}
    assert held_during_call[AgentRole.BACKEND_ENGINEER] == {ENDPOINT_ID}
    assert FRONTEND_DOCUMENT_LOCK in held_during_call[AgentRole.STYLER]
    assert held_during_call[AgentRole.CRITIC_LINTER] == set()
    assert hub.lock_manager(SESSION).active_locks() == ()
    assert state["active_locks"] == ()
    await hub.shutdown()


async def test_no_background_tasks_outlive_the_run() -> None:
    hub = await make_hub()
    before = asyncio.all_tasks()
    await run_swarm(hub, MockLLMProvider())
    assert asyncio.all_tasks() - before == set()
    await hub.shutdown()


async def test_coder_and_backend_contend_for_a_form_without_losing_either_write() -> None:
    """The coder holds the form's lock through generation, so binding it has to wait."""
    hub = await make_hub()
    coder_holds_form = asyncio.Event()
    backend_blocked = asyncio.Event()

    async def force_contention(role: AgentRole, request: LLMRequest[Any]) -> None:
        if role is AgentRole.BACKEND_ENGINEER:
            await asyncio.wait_for(coder_holds_form.wait(), 5)
        if role is AgentRole.FRONTEND_CODER and "contact" in request.prompt:
            coder_holds_form.set()
            await asyncio.wait_for(backend_blocked.wait(), 5)

    original = hub.broadcast_frame

    async def watch(session_id: str, frame: AgentCursorFrame) -> None:
        if frame.current_action is AgentAction.BLOCKED_WAITING:
            backend_blocked.set()
        await original(session_id, frame)

    hub.broadcast_frame = watch  # type: ignore[method-assign]
    state = await run_swarm(hub, HookedProvider(MockLLMProvider(), force_contention))

    form = component(state, "contact")
    assert "<section" in form.code
    assert form.connected_endpoint_id == ENDPOINT_ID
    assert form.component_type is ComponentType.CONTACT_FORM
    assert state["completed"] is True
    waits = thoughts_with(hub, "lock.wait")
    assert [t.agent_id for t in waits] == [IDENTITIES[AgentRole.BACKEND_ENGINEER].agent_id]
    assert waits[0].tool_args["target_id"] == "contact"
    await hub.shutdown()


async def test_lease_lost_during_generation_becomes_feedback_and_a_retry() -> None:
    hub = await make_hub()
    stolen = False

    async def steal(role: AgentRole, _request: LLMRequest[Any]) -> None:
        nonlocal stolen
        if role is AgentRole.FRONTEND_CODER and not stolen:
            stolen = True
            await hub.lock_manager(SESSION).release("hero", IDENTITIES[role].agent_id)

    state = await run_swarm(hub, HookedProvider(MockLLMProvider(), steal))

    assert any("component hero" in line and "conflict" in line for line in state["critic_feedback"])
    assert state["completed"] is True
    assert state["iteration"] == 2
    assert "<section" in component(state, "hero").code
    await hub.shutdown()


async def test_failed_planner_releases_its_locks_and_fails_the_run() -> None:
    hub = await make_hub()
    provider = MockLLMProvider({AgentRole.ARCHITECT: [LLMProviderError("down", model="m")]})
    with pytest.raises(LLMProviderError, match="down"):
        await run_swarm(hub, provider)
    assert hub.lock_manager(SESSION).active_locks() == ()
    last = hub.frames[-1]
    assert last.current_action is AgentAction.IDLE
    await hub.shutdown()


async def test_closing_the_session_during_a_failure_keeps_the_original_error() -> None:
    hub = await make_hub()

    async def close_it(_role: AgentRole, _request: LLMRequest[Any]) -> None:
        await hub.close_session(SESSION)

    provider = MockLLMProvider({AgentRole.ARCHITECT: [LLMProviderError("down", model="m")]})
    with pytest.raises(LLMProviderError, match="down"):
        await run_swarm(hub, HookedProvider(provider, close_it))
    await hub.shutdown()


async def test_session_closed_mid_run_reports_the_real_failure() -> None:
    hub = await make_hub()

    async def close_it(_role: AgentRole, _request: LLMRequest[Any]) -> None:
        await hub.close_session(SESSION)

    with pytest.raises(SessionNotFoundError):
        await run_swarm(hub, HookedProvider(MockLLMProvider(), close_it))
    await hub.shutdown()


async def test_missing_session_fails_immediately() -> None:
    hub = RecordingHub(make_settings(), bus=InMemoryEventBus())
    with pytest.raises(SessionNotFoundError):
        await run_swarm(hub, MockLLMProvider(), empty_bundle())
    await hub.shutdown()


# --- lock scope -----------------------------------------------------------------------------


async def test_scope_acquires_in_sorted_order_and_releases_on_failure() -> None:
    hub = await make_hub()
    manager = hub.lock_manager(SESSION)
    context = make_context(hub, role=AgentRole.STYLER)
    agent = context.identity.agent_id
    with pytest.raises(RuntimeError, match="boom"):
        async with context.locks() as locks:
            await locks.acquire("b", "a")
            assert [lock.target_id for lock in manager.active_locks()] == ["a", "b"]
            assert locks.lock_on("a") is not None
            assert locks.lock_on("missing") is None
            await locks.acquire("a")
            assert manager.live_holder("a", hub.now_ms()) == agent
            raise RuntimeError("boom")
    assert manager.active_locks() == ()
    await hub.shutdown()


async def test_scope_takes_locks_in_sorted_order_so_agents_cannot_form_a_cycle() -> None:
    hub = await make_hub()
    manager = hub.lock_manager(SESSION)
    order: list[str] = []
    original = manager.acquire

    async def record(target_id: str, agent_id: str, priority: int, now_ms: int) -> ElementLock:
        order.append(target_id)
        return await original(target_id, agent_id, priority, now_ms)

    manager.acquire = record  # type: ignore[method-assign]
    async with make_context(hub).locks() as locks:
        await locks.acquire("c", "a", "b")
    assert order == ["a", "b", "c"]
    await hub.shutdown()


async def test_scope_gives_up_after_the_wait_limit_and_withdraws_its_request() -> None:
    clock = FakeClock(0)
    hub = await make_hub(clock=clock, agent_lock_wait_ms=100, lock_lease_ms=30_000)
    manager = hub.lock_manager(SESSION)
    await manager.acquire("hero", "someone-else", 90, 0)

    idle = asyncio.Event()

    async def advance(seconds: float) -> None:
        if seconds != LOCK_POLL_S:
            await idle.wait()  # The heartbeat stays out of this test's clock.
        clock.now += 60

    runtime = SwarmRuntime(hub, MockLLMProvider(), sleep=advance)
    state = initial_state(SESSION, PROMPT, empty_bundle())
    context = NodeContext(runtime, state, AgentRole.STYLER)
    with pytest.raises(LockAcquisitionTimeoutError) as caught:
        async with context.locks() as locks:
            await locks.acquire("hero")
    assert caught.value.extensions["timeout_ms"] == 100
    assert [lock.locked_by_agent_id for lock in manager.active_locks()] == ["someone-else"]
    assert len(thoughts_with(hub, "lock.wait")) == 1
    await hub.shutdown()


async def test_scope_announces_a_wait_once_however_long_it_lasts() -> None:
    hub = await make_hub()
    manager = hub.lock_manager(SESSION)
    await manager.acquire("hero", "someone-else", 90, hub.now_ms())
    polls = 0
    idle = asyncio.Event()

    async def release_on_third_poll(seconds: float) -> None:
        nonlocal polls
        if seconds != LOCK_POLL_S:
            await idle.wait()
        polls += 1
        if polls == 3:
            await manager.release("hero", "someone-else")

    runtime = SwarmRuntime(hub, MockLLMProvider(), sleep=release_on_third_poll)
    state = initial_state(SESSION, PROMPT, empty_bundle())
    context = NodeContext(runtime, state, AgentRole.STYLER)
    async with context.locks() as locks:
        await locks.acquire("hero")
        assert locks.lock_on("hero") is not None
    assert polls == 3
    assert len(thoughts_with(hub, "lock.wait")) == 1
    await hub.shutdown()


async def test_scope_reports_being_evicted_by_deadlock_resolution() -> None:
    hub = await make_hub()
    manager = hub.lock_manager(SESSION)
    now = hub.now_ms()
    context = make_context(hub, role=AgentRole.CRITIC_LINTER)
    await manager.acquire("t2", "rival", 100, now)
    with pytest.raises(ElementLockConflictError):
        async with context.locks() as locks:
            await locks.acquire("t1")
            await manager.acquire("t1", "rival", 100, now)
            await locks.acquire("t2")
    assert manager.live_holder("t1", hub.now_ms()) == "rival"
    await hub.shutdown()


class Heartbeat:
    """A sleep that only returns when the test allows one renewal round."""

    def __init__(self) -> None:
        self.allow = asyncio.Semaphore(0)
        self.waiting = asyncio.Event()

    async def sleep(self, _seconds: float) -> None:
        self.waiting.set()
        await self.allow.acquire()

    async def beat(self) -> None:
        self.waiting.clear()
        self.allow.release()
        await asyncio.wait_for(self.waiting.wait(), 2)


async def test_heartbeat_keeps_a_lease_alive_through_slow_generation() -> None:
    clock = FakeClock(0)
    hub = await make_hub(clock=clock, lock_lease_ms=3_000)
    heartbeat = Heartbeat()
    state = initial_state(SESSION, PROMPT, empty_bundle())
    runtime = SwarmRuntime(hub, MockLLMProvider(), sleep=heartbeat.sleep)
    context = NodeContext(runtime, state, AgentRole.STYLER)
    agent = context.identity.agent_id
    manager = hub.lock_manager(SESSION)

    async with context.locks() as locks:
        await locks.acquire("hero")
        await asyncio.wait_for(heartbeat.waiting.wait(), 2)
        for now in (2_000, 4_000, 6_000, 8_000):
            clock.now = now
            await heartbeat.beat()
            assert manager.live_holder("hero", now) == agent
        renewed = locks.lock_on("hero")
        assert renewed is not None
        assert renewed.acquired_at_ms == 8_000
    await hub.shutdown()


async def test_heartbeat_does_not_take_back_a_lock_that_was_lost() -> None:
    clock = FakeClock(0)
    hub = await make_hub(clock=clock, lock_lease_ms=3_000)
    heartbeat = Heartbeat()
    state = initial_state(SESSION, PROMPT, empty_bundle())
    runtime = SwarmRuntime(hub, MockLLMProvider(), sleep=heartbeat.sleep)
    context = NodeContext(runtime, state, AgentRole.STYLER)
    manager = hub.lock_manager(SESSION)

    async with context.locks() as locks:
        await locks.acquire("hero")
        await asyncio.wait_for(heartbeat.waiting.wait(), 2)
        await manager.release("hero", context.identity.agent_id)
        await manager.acquire("hero", "new-owner", 50, 0)
        await heartbeat.beat()
        assert locks.lock_on("hero") is None
        assert manager.live_holder("hero", 0) == "new-owner"
        waiting = [lock for lock in manager.active_locks() if lock.state is LockState.WAITING]
        assert waiting == [], "a lost hold must not be requested again"
    assert manager.live_holder("hero", 0) == "new-owner"
    await hub.shutdown()


# --- workflow: critic loop ------------------------------------------------------------------


async def test_syntax_failure_routes_back_to_the_coder_only() -> None:
    hub = await make_hub()
    provider = MockLLMProvider({AgentRole.FRONTEND_CODER: [CoderOutput(code=BROKEN_TSX)]})
    state = await run_swarm(hub, provider)

    assert state["completed"] is True
    assert state["iteration"] == 2
    entered = hub.entered()
    assert entered.count("planner") == 1
    assert entered.count(CODER) == 2
    assert entered.count(BACKEND) == 1
    assert entered.count("styler") == 2
    assert entered.count("critic") == 2
    assert entered[-3:] == [CODER, "styler", "critic"]
    assert any(
        "hero" in line and "<div> is never closed" in line for line in state["critic_feedback"]
    )
    assert find_tsx_issue(component(state, "hero").code) is None
    await hub.shutdown()


async def test_retry_regenerates_only_the_rejected_component_and_briefs_it() -> None:
    hub = await make_hub()
    provider = MockLLMProvider({AgentRole.FRONTEND_CODER: [CoderOutput(code=BROKEN_TSX)]})
    await run_swarm(hub, provider)

    coder_calls = [call for call in provider.calls if call.output_schema is CoderOutput]
    assert len(coder_calls) == 3
    retry = coder_calls[-1]
    assert "Component id: hero" in retry.prompt
    assert "previous attempt was rejected" in retry.prompt
    assert "<div> is never closed" in retry.prompt
    assert BROKEN_TSX in retry.prompt
    await hub.shutdown()


async def test_uncompilable_handler_routes_back_to_the_backend_only() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {AgentRole.BACKEND_ENGINEER: [HandlerOutput(handler_code="async def broken(:\n    pass")]}
    )
    state = await run_swarm(hub, provider)

    assert state["completed"] is True
    assert state["iteration"] == 2
    entered = hub.entered()
    assert entered.count(BACKEND) == 2
    assert entered.count(CODER) == 1
    assert any("failed to compile" in line for line in state["critic_feedback"])
    assert not any("binding of" in line for line in state["critic_feedback"])
    assert component(state, "contact").connected_endpoint_id == ENDPOINT_ID
    retry = [c for c in provider.calls if c.output_schema is HandlerOutput][-1]
    assert "previous attempt was rejected" in retry.prompt
    await hub.shutdown()


async def test_handler_without_an_async_function_is_rejected_with_its_reason() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {AgentRole.BACKEND_ENGINEER: [HandlerOutput(handler_code="def sync_handler(): pass")]}
    )
    state = await run_swarm(hub, provider)
    assert any("async function" in line for line in state["critic_feedback"])
    assert state["completed"] is True
    await hub.shutdown()


async def test_failures_on_both_sides_retry_both_and_style_once() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {
            AgentRole.FRONTEND_CODER: [CoderOutput(code=BROKEN_TSX)],
            AgentRole.BACKEND_ENGINEER: [HandlerOutput(handler_code="not python (")],
        }
    )
    state = await run_swarm(hub, provider)
    entered = hub.entered()
    assert entered.count(CODER) == 2
    assert entered.count(BACKEND) == 2
    assert entered.count("styler") == 2
    assert state["completed"] is True
    await hub.shutdown()


@pytest.mark.parametrize("max_iterations", [1, 3, 20])
async def test_loop_stops_at_the_iteration_limit_and_reports_failure(max_iterations: int) -> None:
    hub = await make_hub(agent_max_iterations=max_iterations)
    bad = [CoderOutput(code=BROKEN_TSX) for _ in range(2 * max_iterations)]
    provider = MockLLMProvider({AgentRole.FRONTEND_CODER: bad})
    state = await run_swarm(hub, provider)

    assert state["completed"] is False
    assert state["iteration"] == max_iterations
    assert state["retry_frontend"] == ()
    assert any(f"gave up after {max_iterations}" in line for line in state["critic_feedback"])
    await hub.shutdown()


async def test_model_failure_on_one_component_does_not_stop_the_others() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {AgentRole.FRONTEND_CODER: [LLMProviderError("overloaded", model="m")]}
    )
    state = await run_swarm(hub, provider)
    assert any("overloaded" in line for line in state["critic_feedback"])
    assert any("still a wireframe" in line for line in state["critic_feedback"])
    assert state["completed"] is True
    await hub.shutdown()


async def test_vanished_component_becomes_feedback_rather_than_a_crash() -> None:
    hub = await make_hub()
    state = initial_state(SESSION, PROMPT, empty_bundle())
    state["iteration"] = 1
    state["retry_frontend"] = ("gone",)
    context = NodeContext(SwarmRuntime(hub, MockLLMProvider()), state, AgentRole.FRONTEND_CODER)
    await coder_node(state, context)
    assert "component gone no longer exists" in context.feedback[0]
    assert hub.lock_manager(SESSION).active_locks() == ()
    await hub.shutdown()


def retry_plan() -> SwarmPlan:
    other = EndpointSpec(
        path="/api/v1/other", method=HttpMethod.GET, summary="Other", brief="Read it"
    )
    return SwarmPlan(
        components=(contact_spec(),),
        models=(lead_model(),),
        endpoints=(post_endpoint(), other),
    )


async def test_backend_retry_regenerates_only_the_rejected_handler() -> None:
    hub = await make_hub()
    provider = MockLLMProvider()
    state = initial_state(SESSION, PROMPT, empty_bundle())
    state["plan"] = retry_plan()
    state["iteration"] = 1
    state["retry_backend"] = ("api_endpoint_get_other",)
    context = NodeContext(SwarmRuntime(hub, provider), state, AgentRole.BACKEND_ENGINEER)
    await backend_node(state, context)
    assert len(provider.calls) == 1
    assert "/api/v1/other" in provider.calls[0].prompt
    await hub.shutdown()


async def test_backend_retry_for_an_unbound_form_does_not_regenerate_handlers() -> None:
    hub = await make_hub()
    provider = MockLLMProvider()
    state = initial_state(SESSION, PROMPT, empty_bundle())
    state["plan"] = retry_plan()
    state["iteration"] = 1
    state["retry_backend"] = ("contact",)
    context = NodeContext(SwarmRuntime(hub, provider), state, AgentRole.BACKEND_ENGINEER)
    await backend_node(state, context)
    assert provider.calls == []
    await hub.shutdown()


async def test_model_critic_rejection_retries_components_and_endpoints() -> None:
    hub = await make_hub()
    verdict = CriticVerdict(
        approved=False,
        issues=(
            CriticIssue(target_id="hero", message="missing alt text"),
            CriticIssue(target_id=ENDPOINT_ID, message="does not validate input"),
        ),
    )
    provider = MockLLMProvider(
        {
            AgentRole.CRITIC_LINTER: [verdict],
            AgentRole.BACKEND_ENGINEER: [
                HandlerOutput(handler_code=GOOD_HANDLER),
                HandlerOutput(handler_code="async def better() -> None:\n    return None"),
            ],
        }
    )
    state = await run_swarm(hub, provider)

    assert state["completed"] is True
    assert state["iteration"] == 2
    assert any("missing alt text" in line for line in state["critic_feedback"])
    endpoint = state["current_bundle"].backend.endpoints[0]
    assert "better" in endpoint.handler_code
    assert len(state["current_bundle"].backend.endpoints) == 1
    await hub.shutdown()


async def test_model_critic_advice_without_a_real_target_does_not_burn_an_iteration() -> None:
    hub = await make_hub()
    verdict = CriticVerdict(
        approved=False, issues=(CriticIssue(target_id="the-vibe", message="feels off"),)
    )
    state = await run_swarm(hub, MockLLMProvider({AgentRole.CRITIC_LINTER: [verdict]}))
    assert state["completed"] is True
    assert state["iteration"] == 1
    assert any("advisory: feels off" in line for line in state["critic_feedback"])
    await hub.shutdown()


async def test_model_critic_cannot_overrule_a_structural_failure() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {
            AgentRole.FRONTEND_CODER: [CoderOutput(code=BROKEN_TSX)],
            AgentRole.CRITIC_LINTER: [CriticVerdict(approved=True)],
        }
    )
    state = await run_swarm(hub, provider)
    assert state["iteration"] == 2
    assert find_tsx_issue(component(state, "hero").code) is None
    await hub.shutdown()


async def test_critic_reports_a_form_bound_to_the_wrong_thing_or_not_at_all() -> None:
    unbound = ComponentNode.create(
        id="contact",
        component_type=ComponentType.CUSTOM_DYNAMIC,
        title="Contact",
        code=GOOD_TSX,
        box=BoundingBox(x=0, y=0, width=100, height=50),
    )
    hub = await make_hub(empty_bundle((unbound,)))
    plan = SwarmPlan(
        components=(
            contact_spec(),
            ComponentSpec.model_validate(
                {
                    "id": "absent",
                    "component_type": ComponentType.HERO,
                    "title": "Absent",
                    "brief": "Never added",
                }
            ),
        ),
        models=(lead_model(),),
        endpoints=(post_endpoint(),),
    )
    state = initial_state(SESSION, PROMPT, empty_bundle())
    state["plan"] = plan
    context = NodeContext(SwarmRuntime(hub, MockLLMProvider()), state, AgentRole.CRITIC_LINTER)
    result = await critic_node(state, context)

    assert result["completed"] is False
    assert set(result["retry_backend"]) == {"contact", ENDPOINT_ID}
    assert result["retry_frontend"] == ()
    await hub.shutdown()


async def test_critic_flags_a_wireframe_left_behind() -> None:
    wireframe = ComponentNode.create(
        id="hero",
        component_type=ComponentType.HERO,
        title="Hero",
        code=wireframe_code("hero"),
        box=BoundingBox(x=0, y=0, width=100, height=50),
    )
    hub = await make_hub(empty_bundle((wireframe,)))
    context = make_context(hub)
    result = await critic_node(context.state, context)
    assert result["retry_frontend"] == ("hero",)
    assert "still a wireframe" in context.feedback[0]
    await hub.shutdown()


# --- workflow: styler -----------------------------------------------------------------------


async def test_styler_applies_theme_and_class_fixes() -> None:
    hub = await make_hub()
    style = StyleOutput(
        theme="cyberpunk",
        fixes=(
            ClassFix(component_id="hero", find="p-8", replace="p-4 md:p-8"),
            ClassFix(component_id="hero", find="not-present", replace="x"),
            ClassFix(component_id="unknown", find="p-8", replace="x"),
        ),
    )
    state = await run_swarm(hub, MockLLMProvider({AgentRole.STYLER: [style]}))
    assert state["current_bundle"].frontend.theme == "cyberpunk"
    assert "p-4 md:p-8" in component(state, "hero").code
    assert 'className="p-8"' in component(state, "contact").code
    assert state["completed"] is True
    await hub.shutdown()


async def test_styler_is_skipped_when_a_retry_touches_only_the_backend() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {AgentRole.BACKEND_ENGINEER: [HandlerOutput(handler_code="not python (")]}
    )
    await run_swarm(hub, provider)
    assert [c.output_schema for c in provider.calls].count(StyleOutput) == 1
    await hub.shutdown()


async def test_styler_on_retry_looks_only_at_the_rejected_component() -> None:
    hub = await make_hub()
    provider = MockLLMProvider({AgentRole.FRONTEND_CODER: [CoderOutput(code=BROKEN_TSX)]})
    await run_swarm(hub, provider)
    styles = [c for c in provider.calls if c.output_schema is StyleOutput]
    assert len(styles) == 2
    assert "Component contact" in styles[0].prompt
    assert "Component hero" in styles[1].prompt
    assert "Component contact" not in styles[1].prompt
    await hub.shutdown()


async def test_styler_ignores_a_fix_for_a_component_it_did_not_lock() -> None:
    """On a retry only the rejected component is locked. A fix elsewhere would be a 409."""
    hub = await make_hub()
    outside = StyleOutput(fixes=(ClassFix(component_id="contact", find="p-8", replace="p-1"),))
    provider = MockLLMProvider(
        {
            AgentRole.FRONTEND_CODER: [CoderOutput(code=BROKEN_TSX)],
            AgentRole.STYLER: [StyleOutput(), outside],
        }
    )
    state = await run_swarm(hub, provider)
    assert 'className="p-8"' in component(state, "contact").code
    assert not any("styling" in line for line in state["critic_feedback"])
    await hub.shutdown()


async def test_styler_commits_nothing_when_no_fix_applies() -> None:
    hub = await make_hub()
    style = StyleOutput(fixes=(ClassFix(component_id="hero", find="not-present", replace="x"),))
    state = await run_swarm(hub, MockLLMProvider({AgentRole.STYLER: [style]}))
    styler = IDENTITIES[AgentRole.STYLER].agent_id
    assert [d for d in state["pending_deltas"] if d.agent_id == styler] == []
    await hub.shutdown()


async def test_styler_leaves_a_wireframe_for_the_coder() -> None:
    hub = await make_hub()
    provider = MockLLMProvider(
        {AgentRole.FRONTEND_CODER: [LLMProviderError("overloaded", model="m")]}
    )
    await run_swarm(hub, provider)
    first_style = next(c for c in provider.calls if c.output_schema is StyleOutput)
    assert "Component hero" not in first_style.prompt
    assert "Component contact" in first_style.prompt
    await hub.shutdown()


async def test_styler_breaking_the_code_is_caught_and_repaired() -> None:
    hub = await make_hub()
    style = StyleOutput(fixes=(ClassFix(component_id="hero", find="</section>", replace=""),))
    state = await run_swarm(hub, MockLLMProvider({AgentRole.STYLER: [style]}))
    assert state["iteration"] == 2
    assert find_tsx_issue(component(state, "hero").code) is None
    await hub.shutdown()


async def test_styler_rejected_patch_becomes_feedback() -> None:
    hub = await make_hub()
    whole = (
        'export default function Section() {\n  return <section className="p-8">Hi</section>;\n}'
    )
    style = StyleOutput(fixes=(ClassFix(component_id="hero", find=whole, replace=""),))
    state = await run_swarm(hub, MockLLMProvider({AgentRole.STYLER: [style]}))
    assert any("styling" in line for line in state["critic_feedback"])
    await hub.shutdown()


async def test_styler_reports_when_there_is_nothing_to_change() -> None:
    hub = await make_hub()
    await run_swarm(hub, MockLLMProvider())
    assert any(
        frame.current_thought is not None
        and frame.current_thought.thought == "No style changes were needed."
        for frame in hub.frames
    )
    await hub.shutdown()


# --- workflow: telemetry --------------------------------------------------------------------


async def test_every_transition_is_broadcast_with_its_badge() -> None:
    hub = await make_hub()
    await run_swarm(hub, MockLLMProvider())

    enters = thoughts_with(hub, "graph.enter")
    exits = thoughts_with(hub, "graph.exit")
    assert [t.tool_args["node"] for t in enters].count("planner") == 1
    assert len(enters) == len(exits) == 5
    assert all(t.tool_args["iteration"] == 0 for t in enters)
    assert {t.role for t in enters} == {
        AgentRole.ARCHITECT,
        AgentRole.FRONTEND_CODER,
        AgentRole.BACKEND_ENGINEER,
        AgentRole.STYLER,
        AgentRole.CRITIC_LINTER,
    }
    await hub.shutdown()


async def test_model_calls_carry_a_badge_with_usage_and_latency() -> None:
    hub = await make_hub()
    await run_swarm(hub, MockLLMProvider())
    badges = thoughts_with(hub, "llm.generate")
    assert {t.role for t in badges} == {
        AgentRole.ARCHITECT,
        AgentRole.FRONTEND_CODER,
        AgentRole.BACKEND_ENGINEER,
        AgentRole.STYLER,
        AgentRole.CRITIC_LINTER,
    }
    for badge in badges:
        assert badge.tool_args["model"] == "mock"
        assert badge.tool_args["schema"]
        assert badge.tokens_used > 0
        assert badge.execution_latency_ms > 0
    planner = next(t for t in badges if t.role is AgentRole.ARCHITECT)
    assert planner.thought == "A landing page with a working contact form."
    assert planner.tool_args["schema"] == "SwarmPlan"
    await hub.shutdown()


async def test_commits_and_locks_carry_badges_and_verdict_is_reported() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider())
    patches = thoughts_with(hub, "hub.apply_delta")
    assert len(patches) == len(state["pending_deltas"])
    assert all(t.tool_args["operations"] >= 1 for t in patches)
    assert {t.tool_args["target_id"] for t in thoughts_with(hub, "lock.acquire")} == {
        "hero",
        "contact",
        ENDPOINT_ID,
    }
    verdict = thoughts_with(hub, "critic.verdict")
    assert verdict[-1].tool_args == {"approved": True, "iteration": 1}
    await hub.shutdown()


async def test_state_collects_the_same_thoughts_the_hub_saw() -> None:
    hub = await make_hub()
    state = await run_swarm(hub, MockLLMProvider())
    seen = sorted(
        (f.current_thought.agent_id, f.current_thought.thought)
        for f in hub.frames
        if f.current_thought is not None
    )
    collected = sorted((t.agent_id, t.thought) for t in state["agent_thoughts"])
    assert collected == seen
    await hub.shutdown()


async def test_frames_identify_their_agent_and_track_the_element_in_hand() -> None:
    hub = await make_hub()
    await run_swarm(hub, MockLLMProvider())
    for frame in hub.frames:
        identity = IDENTITIES[frame.role]
        assert frame.agent_id == identity.agent_id
        assert frame.name == identity.name
        assert frame.color == identity.color
        assert frame.current_thought is not None
        assert frame.current_thought.agent_id == frame.agent_id
    editing = [f for f in hub.frames if f.current_action is AgentAction.EDITING]
    assert editing
    assert all(f.y > 0 for f in editing), "the cursor should sit on the component being written"
    assert all(f.active_lock is not None for f in editing)
    await hub.shutdown()


async def test_listeners_receive_cursor_events_during_a_run() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    await run_swarm(hub, MockLLMProvider())
    received: list[Any] = []
    while True:
        try:
            received.append(await asyncio.wait_for(anext(listener), 0.2))
        except TimeoutError:
            break
    cursors = [event for event in received if isinstance(event, CursorEvent)]
    assert cursors
    assert all(event.frame.current_thought is not None for event in cursors)
    await hub.shutdown()


@pytest.mark.parametrize(
    "rationale, expected",
    [
        ("Plain reasoning.", "Plain reasoning."),
        ("Safe <script>alert(1)</script>text", "Safe text"),
        ("<script>alert(1)</script>", "Architect is working."),
        ("", "Architect finished thinking."),
    ],
)
async def test_model_rationale_is_sanitized_before_broadcast(rationale: str, expected: str) -> None:
    hub = await make_hub()
    plan = SwarmPlan(rationale=rationale)
    await run_swarm(hub, MockLLMProvider({AgentRole.ARCHITECT: [plan]}))
    badge = next(t for t in thoughts_with(hub, "llm.generate") if t.role is AgentRole.ARCHITECT)
    assert badge.thought == expected
    await hub.shutdown()


async def test_overlong_rationale_is_truncated_rather_than_fatal() -> None:
    hub = await make_hub()
    plan = SwarmPlan(rationale="x" * 100_000)
    state = await run_swarm(hub, MockLLMProvider({AgentRole.ARCHITECT: [plan]}))
    badge = next(t for t in thoughts_with(hub, "llm.generate") if t.role is AgentRole.ARCHITECT)
    assert len(badge.thought) == 4096
    assert state["completed"] is True
    await hub.shutdown()


# --- hub additions --------------------------------------------------------------------------


async def test_reserved_sequence_numbers_are_unique_and_follow_commits() -> None:
    hub = await make_hub()
    assert [hub.reserve_delta_seq(SESSION) for _ in range(3)] == [0, 1, 2]
    delta = StreamDelta.model_validate(
        {
            "session_id": SESSION,
            "agent_id": "a",
            "delta_seq": 9,
            "target_domain": "frontend",
            "target_id": "x",
            "patches": [{"op": "test", "path": "/frontend/title", "value": "Acme"}],
            "emitted_at_ms": 0,
        }
    )
    manager = hub.lock_manager(SESSION)
    for target in ("x", FRONTEND_DOCUMENT_LOCK):
        await manager.acquire(target, "a", 50, hub.now_ms())
    await hub.apply_delta(SESSION, delta)
    assert hub.reserve_delta_seq(SESSION) == 10
    await hub.shutdown()


async def test_hub_exposes_time_settings_and_latest_state() -> None:
    clock = FakeClock(55)
    hub = await make_hub(clock=clock, agent_max_iterations=9)
    assert hub.now_ms() == 55
    assert hub.settings.agent_max_iterations == 9
    assert hub.latest_snapshot(SESSION).frame_seq == 0
    await hub.shutdown()
