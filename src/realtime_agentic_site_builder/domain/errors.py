"""RFC 9457 problem details and the domain error taxonomy.

Domain errors deliberately do not derive from ``ValueError``: raised inside a Pydantic validator
they propagate unwrapped, so they reach the HTTP layer with their own status instead of a generic
422 validation failure.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, ClassVar, Final, Self
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

PROBLEM_JSON_MEDIA_TYPE: Final = "application/problem+json"
PROBLEM_TYPE_BASE: Final = "https://github.com/mikeZ98/realtime-agentic-site-builder/problems/"
MAX_SNIPPET_LENGTH: Final = 2000

_EXTENSION_NAME: Final = TypeAdapter(Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{2,}$")])
_EXTENSION_VALUE: Final = TypeAdapter(JsonValue)


class ProblemDetails(BaseModel):
    """RFC 9457 problem object. Undeclared members are extension attributes, kept at top level."""

    model_config = ConfigDict(frozen=True, extra="allow", str_strip_whitespace=True)

    type: Annotated[str, Field(pattern=r"^\S+$")] = "about:blank"
    title: str | None = None
    status: Annotated[int, Field(strict=True, ge=100, le=599)] | None = None
    detail: str | None = None
    instance: Annotated[str, Field(pattern=r"^\S+$")] | None = None

    @model_validator(mode="after")
    def _validate_extensions(self) -> Self:
        # RFC 9457 section 3.2 asks for these names so problems can also be rendered as XML.
        for name, value in (self.model_extra or {}).items():
            _EXTENSION_NAME.validate_python(name)
            _EXTENSION_VALUE.validate_python(value)
        return self

    def to_json_dict(self) -> dict[str, object]:
        """Return the response body, omitting absent members as RFC 9457 requires."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


class SiteBuilderError(Exception):
    """Root of the domain error hierarchy. Every error renders as an RFC 9457 problem."""

    status: ClassVar[int] = 500
    title: ClassVar[str] = "Site builder failure"
    slug: ClassVar[str] = "site-builder-failure"

    def __init__(
        self,
        detail: str,
        *,
        instance: str | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.instance = instance
        self.extensions: dict[str, JsonValue] = dict(extensions or {})

    def to_problem(self) -> ProblemDetails:
        return ProblemDetails.model_validate(
            {
                "type": f"{PROBLEM_TYPE_BASE}{self.slug}",
                "title": self.title,
                "status": self.status,
                "detail": self.detail,
                "instance": self.instance,
                **self.extensions,
            }
        )


class ElementLockConflictError(SiteBuilderError):
    status: ClassVar[int] = 409
    title: ClassVar[str] = "Element lock conflict"
    slug: ClassVar[str] = "element-lock-conflict"

    def __init__(
        self,
        *,
        competing_agent_ids: Sequence[str],
        resource_uri: str,
        instance: str | None = None,
    ) -> None:
        agents = sorted(set(competing_agent_ids))
        super().__init__(
            f"Agents {', '.join(agents)} are in conflict over {resource_uri}.",
            instance=instance,
            extensions={"competing_agent_ids": list(agents), "resource_uri": resource_uri},
        )


class ASTSyntaxCompilationError(SiteBuilderError):
    status: ClassVar[int] = 422
    title: ClassVar[str] = "AST syntax compilation failure"
    slug: ClassVar[str] = "ast-syntax-compilation-failure"

    def __init__(
        self,
        reason: str,
        *,
        line_number: int,
        syntax_token: str | None,
        code_snippet: str,
        instance: str | None = None,
    ) -> None:
        super().__init__(
            f"Code failed to compile at line {line_number}: {reason}.",
            instance=instance,
            extensions={
                "line_number": line_number,
                "syntax_token": syntax_token,
                "code_snippet": code_snippet[:MAX_SNIPPET_LENGTH],
            },
        )

    @classmethod
    def from_syntax_error(cls, error: SyntaxError, code: str) -> Self:
        token = None
        if error.text and error.offset:
            end = error.end_offset - 1 if error.end_offset else None
            token = error.text[error.offset - 1 : end].strip() or None
        return cls(error.msg, line_number=error.lineno or 1, syntax_token=token, code_snippet=code)


class EndpointBindingNotFoundError(SiteBuilderError):
    status: ClassVar[int] = 400
    title: ClassVar[str] = "Endpoint binding not found"
    slug: ClassVar[str] = "endpoint-binding-not-found"

    def __init__(
        self, *, connected_endpoint_id: str, component_id: str, instance: str | None = None
    ) -> None:
        super().__init__(
            f"Component {component_id} is bound to unknown endpoint {connected_endpoint_id}.",
            instance=instance,
            extensions={
                "connected_endpoint_id": connected_endpoint_id,
                "component_id": component_id,
            },
        )


class ReplayStateDivergenceError(SiteBuilderError):
    status: ClassVar[int] = 500
    title: ClassVar[str] = "Replay state divergence"
    slug: ClassVar[str] = "replay-state-divergence"

    def __init__(
        self, *, frame_seq: int, expected_hash: str, computed_hash: str, instance: str | None = None
    ) -> None:
        super().__init__(
            f"Replay frame {frame_seq} diverged from its recorded state hash.",
            instance=instance,
            extensions={
                "frame_seq": frame_seq,
                "expected_hash": expected_hash,
                "computed_hash": computed_hash,
            },
        )


class LockAcquisitionTimeoutError(SiteBuilderError):
    status: ClassVar[int] = 408
    title: ClassVar[str] = "Lock acquisition timeout"
    slug: ClassVar[str] = "lock-acquisition-timeout"

    def __init__(
        self, *, agent_id: str, target_id: str, timeout_ms: int, instance: str | None = None
    ) -> None:
        super().__init__(
            f"Agent {agent_id} did not acquire {target_id} within {timeout_ms} ms.",
            instance=instance,
            extensions={"agent_id": agent_id, "target_id": target_id, "timeout_ms": timeout_ms},
        )


class SessionNotFoundError(SiteBuilderError):
    status: ClassVar[int] = 404
    title: ClassVar[str] = "Session not found"
    slug: ClassVar[str] = "session-not-found"

    def __init__(self, *, session_id: str, instance: str | None = None) -> None:
        super().__init__(
            f"Session {session_id} does not exist or has been closed.",
            instance=instance,
            extensions={"session_id": session_id},
        )


class SessionAlreadyExistsError(SiteBuilderError):
    status: ClassVar[int] = 409
    title: ClassVar[str] = "Session already exists"
    slug: ClassVar[str] = "session-already-exists"

    def __init__(self, *, session_id: str, instance: str | None = None) -> None:
        super().__init__(
            f"Session {session_id} already exists.",
            instance=instance,
            extensions={"session_id": session_id},
        )


class DeltaRejectedError(SiteBuilderError):
    """A delta that cannot join the session's log, as opposed to one whose patch fails to apply."""

    status: ClassVar[int] = 409
    title: ClassVar[str] = "Delta rejected"
    slug: ClassVar[str] = "delta-rejected"

    def __init__(
        self, reason: str, *, session_id: str, delta_seq: int, instance: str | None = None
    ) -> None:
        super().__init__(
            f"Delta {delta_seq} was rejected by session {session_id}: {reason}.",
            instance=instance,
            extensions={"session_id": session_id, "delta_seq": delta_seq},
        )


class LLMProviderError(SiteBuilderError):
    """A model call failed, refused, or kept returning output that fails validation."""

    status: ClassVar[int] = 502
    title: ClassVar[str] = "LLM provider failure"
    slug: ClassVar[str] = "llm-provider-failure"

    def __init__(self, reason: str, *, model: str, instance: str | None = None) -> None:
        super().__init__(
            f"Model {model} failed: {reason}.", instance=instance, extensions={"model": model}
        )


class LLMConfigurationError(SiteBuilderError):
    status: ClassVar[int] = 500
    title: ClassVar[str] = "LLM configuration failure"
    slug: ClassVar[str] = "llm-configuration-failure"


def element_resource_uri(target_id: str) -> str:
    """Return the URI identifying a lockable element in conflict diagnostics."""
    return f"urn:site-builder:element:{quote(target_id, safe='')}"
