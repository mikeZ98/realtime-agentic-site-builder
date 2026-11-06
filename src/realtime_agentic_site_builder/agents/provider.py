"""Tiered model routing: each agent role maps to a model, which maps to a vendor endpoint.

Claude models go through the official SDK and every other model through OpenRouter's HTTP API. Both
ask for JSON matching a schema and validate the reply here, so a client only ever returns a checked
object. A reply that fails validation is retried with the error attached.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

import anthropic
import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from realtime_agentic_site_builder.agents.schemas import (
    AgentOutput,
    CoderOutput,
    ComponentSpec,
    CriticVerdict,
    EndpointSpec,
    HandlerOutput,
    ModelFieldSpec,
    ModelSpec,
    StyleOutput,
    SwarmPlan,
)
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.domain.backend import HttpMethod
from realtime_agentic_site_builder.domain.base import DomainModel, NonNegativeInt
from realtime_agentic_site_builder.domain.blueprint import ComponentType
from realtime_agentic_site_builder.domain.errors import LLMConfigurationError, LLMProviderError
from realtime_agentic_site_builder.domain.swarm import AgentRole

DEFAULT_MAX_TOKENS: Final = 16_000
MAX_VALIDATION_ATTEMPTS: Final = 2
OPENROUTER_MAX_RETRIES: Final = 2
OPENROUTER_RETRY_DELAY_S: Final = 1.0
OPENROUTER_URL: Final = "https://openrouter.ai/api/v1/chat/completions"
_RETRYABLE_STATUS: Final = frozenset({408, 409, 429})
_REFUSAL_STOP: Final = "refusal"
# Only Sonnet takes this form of server-side fallback. Haiku has none, so it must not be sent there.
_SERVER_FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"
_SERVER_FALLBACK_MODELS: Final = frozenset({"claude-sonnet-5-5"})

# Depth trades against cost per tier. Haiku defaults to medium, so it is set rather than implied.
_EFFORT_BY_MODEL: Final = {"claude-sonnet-5-5": "high", "claude-haiku-5-5": "medium"}

_TIER_SETTING: Final[Mapping[AgentRole, str]] = {
    AgentRole.ARCHITECT: "architect_model",
    AgentRole.FRONTEND_CODER: "coder_model",
    AgentRole.BACKEND_ENGINEER: "coder_model",
    AgentRole.DATABASE_ARCHITECT: "coder_model",
    AgentRole.STYLER: "stylist_model",
    AgentRole.COPYWRITER: "stylist_model",
    AgentRole.CRITIC_LINTER: "critic_model",
    AgentRole.OPTIMIZER: "critic_model",
}


class TokenUsage(DomainModel):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMRequest[T: BaseModel]:
    role: AgentRole
    system: str
    prompt: str
    output_schema: type[T]
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class LLMResult[T: BaseModel]:
    output: T
    usage: TokenUsage
    model: str
    latency_ms: float


class LLMClient(Protocol):
    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]: ...


class LLMProvider(Protocol):
    def client_for(self, role: AgentRole) -> LLMClient: ...


Transport = Callable[[LLMRequest[Any], str], Awaitable[tuple[str, TokenUsage]]]


async def _generate_validated[T: BaseModel](
    request: LLMRequest[T], model: str, transport: Transport
) -> LLMResult[T]:
    """Call the model until its reply validates, totalling usage across every attempt."""
    started = time.perf_counter()
    usage = TokenUsage()
    prompt = request.prompt
    problem = ""
    for _ in range(MAX_VALIDATION_ATTEMPTS):
        text, attempt_usage = await transport(request, prompt)
        usage += attempt_usage
        try:
            output = request.output_schema.model_validate_json(text)
        except ValidationError as error:
            first = error.errors(include_url=False, include_context=False, include_input=False)[0]
            problem = f"{'.'.join(str(part) for part in first['loc'])}: {first['msg']}"
            prompt = f"{request.prompt}\n\nYour previous reply was invalid ({problem}). Try again."
        else:
            latency_ms = (time.perf_counter() - started) * 1000
            return LLMResult(output=output, usage=usage, model=model, latency_ms=latency_ms)
    raise LLMProviderError(f"reply failed validation ({problem})", model=model)


class AnthropicLLMClient:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        return await _generate_validated(request, self._model, self._call)

    async def _call(self, request: LLMRequest[Any], prompt: str) -> tuple[str, TokenUsage]:
        output_config: dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(request.output_schema),
            }
        }
        effort = _EFFORT_BY_MODEL.get(self._model)
        if effort is not None:
            output_config["effort"] = effort
        extra: dict[str, Any] = {}
        if self._model in _SERVER_FALLBACK_MODELS:
            extra = {"betas": [_SERVER_FALLBACK_BETA], "fallbacks": "default"}
        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": prompt}],
                output_config=cast("Any", output_config),
                **extra,
            )
        except anthropic.APIError as error:
            raise LLMProviderError(type(error).__name__, model=self._model) from error
        if response.stop_reason == _REFUSAL_STOP:
            raise LLMProviderError("the request was refused", model=self._model)
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
        )
        return _text_after_last_fallback(response.content), usage


def _text_after_last_fallback(content: Sequence[Any]) -> str:
    """Join the reply's text. Text before a fallback block came from the model that declined."""
    chunks: list[str] = []
    for block in content:
        if block.type == "fallback":
            chunks.clear()
        elif block.type == "text":
            chunks.append(block.text)
    return "".join(chunks)


class OpenRouterLLMClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        model: str,
        api_key: str,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = http
        self._model = model
        self._api_key = api_key
        self._sleep = sleep

    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        return await _generate_validated(request, self._model, self._call)

    async def _call(self, request: LLMRequest[Any], prompt: str) -> tuple[str, TokenUsage]:
        body = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema.__name__,
                    "schema": request.output_schema.model_json_schema(),
                },
            },
            # Without this, routing may pick a host that silently ignores the schema.
            "provider": {"require_parameters": True},
        }
        payload = await self._post(body)
        try:
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            return str(text), TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        except (KeyError, IndexError, TypeError, ValidationError) as error:
            raise LLMProviderError("the response was malformed", model=self._model) from error

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        attempt = 0
        while True:
            cause: httpx.HTTPError | None = None
            try:
                response = await self._http.post(OPENROUTER_URL, json=body, headers=headers)
            except httpx.HTTPError as error:
                # Only the error type: a transport error's message is not ours to repeat.
                failure, cause = type(error).__name__, error
            else:
                status = response.status_code
                if status < 400:
                    return self._decode(response)
                failure = f"HTTP {status}"
                if status not in _RETRYABLE_STATUS and status < 500:
                    raise LLMProviderError(failure, model=self._model)
            if attempt == OPENROUTER_MAX_RETRIES:
                raise LLMProviderError(failure, model=self._model) from cause
            await self._sleep(OPENROUTER_RETRY_DELAY_S * 2**attempt)
            attempt += 1

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise LLMProviderError("the response was not JSON", model=self._model) from error
        # OpenRouter reports upstream failures inside a 200 response.
        if not isinstance(payload, dict) or "error" in payload:
            raise LLMProviderError("the provider returned an error", model=self._model)
        return payload


class TieredLLMProvider:
    """Builds a client per configured route, failing at startup if a key is missing."""

    def __init__(
        self,
        settings: Settings,
        *,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._anthropic = anthropic_client
        self._http = http_client
        self._clients: dict[str, LLMClient] = {}
        self._routes = {
            role: str(getattr(settings, setting)) for role, setting in _TIER_SETTING.items()
        }
        for route in sorted(set(self._routes.values())):
            self._clients[route] = self._build_client(route, settings)

    def client_for(self, role: AgentRole) -> LLMClient:
        return self._clients[self._routes[role]]

    def _build_client(self, route: str, settings: Settings) -> LLMClient:
        endpoint, model = route.split(":", 1)
        if endpoint == "anthropic":
            if self._anthropic is None:
                self._anthropic = anthropic.AsyncAnthropic(
                    api_key=_require_key(
                        settings.anthropic_api_key, "RASB_ANTHROPIC_API_KEY", route
                    ),
                    timeout=settings.llm_timeout_seconds,
                )
            return AnthropicLLMClient(self._anthropic, model)
        key = _require_key(settings.openrouter_api_key, "RASB_OPENROUTER_API_KEY", route)
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
        return OpenRouterLLMClient(self._http, model, key)

    async def aclose(self) -> None:
        if self._anthropic is not None:
            await self._anthropic.close()
        if self._http is not None:
            await self._http.aclose()


def _require_key(secret: SecretStr | None, variable: str, route: str) -> str:
    if secret is None:
        raise LLMConfigurationError(f"{variable} is required by the model route {route}.")
    return secret.get_secret_value()


def _default_plan() -> SwarmPlan:
    lead = ModelSpec(
        name="Lead",
        table_name="leads",
        fields=(ModelFieldSpec(name="email", python_type="EmailStr", indexed=True),),
    )
    endpoint = EndpointSpec(
        path="/api/v1/leads",
        method=HttpMethod.POST,
        summary="Create a lead",
        brief="Store a submitted contact form.",
        request_model_ref="Lead",
    )
    return SwarmPlan(
        rationale="A landing page with a working contact form.",
        site_title="Generated site",
        theme="minimal",
        components=(
            ComponentSpec(
                id="hero",
                component_type=ComponentType.HERO,
                title="Welcome",
                brief="Headline and call to action.",
            ),
            ComponentSpec(
                id="contact",
                component_type=ComponentType.CONTACT_FORM,
                title="Contact us",
                brief="Collect an email address.",
                binds_path="/api/v1/leads",
            ),
        ),
        models=(lead,),
        endpoints=(endpoint,),
    )


_DEFAULT_RESPONSES: Final[Mapping[type[AgentOutput], Callable[[], AgentOutput]]] = {
    SwarmPlan: _default_plan,
    CoderOutput: lambda: CoderOutput(
        rationale="Rendered the section.",
        code=(
            "export default function Section() {\n"
            '  return <section className="p-8">Hi</section>;\n'
            "}"
        ),
    ),
    HandlerOutput: lambda: HandlerOutput(
        rationale="Implemented the handler.",
        handler_code="async def handler() -> dict[str, str]:\n    return {'status': 'ok'}",
    ),
    StyleOutput: lambda: StyleOutput(rationale="The palette already works."),
    CriticVerdict: lambda: CriticVerdict(rationale="Nothing to flag.", approved=True),
}


class MockLLMProvider:
    """Deterministic and network-free, for tests and offline runs.

    ``script`` queues responses per role, consumed in order. A queued exception is raised. Once a
    queue runs out, each schema has a canned default, so a test scripts only what it cares about.
    """

    def __init__(
        self, script: Mapping[AgentRole, Sequence[BaseModel | Exception]] | None = None
    ) -> None:
        self._script = {role: list(items) for role, items in (script or {}).items()}
        self.calls: list[LLMRequest[Any]] = []

    def client_for(self, role: AgentRole) -> LLMClient:
        return _MockClient(self, role)

    def _next_response(self, role: AgentRole, request: LLMRequest[Any]) -> BaseModel:
        self.calls.append(request)
        queue = self._script.get(role)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return _DEFAULT_RESPONSES[cast("type[AgentOutput]", request.output_schema)]()


class _MockClient:
    def __init__(self, provider: MockLLMProvider, role: AgentRole) -> None:
        self._provider = provider
        self._role = role

    async def generate_structured[T: BaseModel](self, request: LLMRequest[T]) -> LLMResult[T]:
        output = self._provider._next_response(self._role, request)
        if not isinstance(output, request.output_schema):
            raise LLMProviderError(
                f"scripted a {type(output).__name__} for a {request.output_schema.__name__}",
                model="mock",
            )
        usage = TokenUsage(
            input_tokens=len(request.prompt) // 4,
            output_tokens=len(output.model_dump_json()) // 4,
        )
        return LLMResult(output=output, usage=usage, model="mock", latency_ms=1.0)
