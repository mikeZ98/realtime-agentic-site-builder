"""Fullstack synthesis contracts binding UI components to generated API endpoints."""

import ast
import keyword
import re
import warnings
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    SerializerFunctionWrapHandler,
    StrictBool,
    field_serializer,
    field_validator,
    model_validator,
)

from realtime_agentic_site_builder.domain.base import (
    CodePayload,
    DomainModel,
    FrozenMap,
    Identifier,
    SanitizedTitle,
    canonical_json,
    require_unique,
    sha256_hex,
)
from realtime_agentic_site_builder.domain.blueprint import (
    ComponentNode,
    ComponentType,
    SiteBlueprint,
)
from realtime_agentic_site_builder.domain.errors import (
    ASTSyntaxCompilationError,
    EndpointBindingNotFoundError,
)

API_PREFIX: Final = "/api/v1/"
EMAIL_COMPATIBLE_TYPE: Final = "EmailStr"

_PATH_SEGMENT: Final = r"(?:[a-z0-9_-]+|\{[a-z_][a-z0-9_]*\})"
ApiPath = Annotated[
    str,
    Field(
        max_length=512, pattern=rf"^{re.escape(API_PREFIX)}{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})*$"
    ),
]
ModelName = Annotated[str, Field(max_length=128, pattern=r"^[A-Z][a-zA-Z0-9]+$")]
PythonType = Literal["str", "int", "float", "bool", "EmailStr", "datetime", "UUID"]


class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class DatabaseDialect(StrEnum):
    SQLITE = "SQLITE"
    POSTGRESQL = "POSTGRESQL"


def _require_non_keyword(name: str) -> str:
    if keyword.iskeyword(name):
        raise ValueError(f"field name {name!r} is a Python keyword")
    return name


# These strings are fed into code generation, so they are limited to plain identifiers.
FieldName = Annotated[
    str, Field(max_length=128, pattern=r"^[a-z_][a-z0-9_]*$"), AfterValidator(_require_non_keyword)
]
DottedCallable = Annotated[str, Field(max_length=256, pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")]


class FieldDefinition(DomainModel):
    python_type: PythonType
    nullable: StrictBool = False
    indexed: StrictBool = False
    default_factory: DottedCallable | None = None


class DataModelSchema(DomainModel):
    name: ModelName
    fields: Annotated[Mapping[FieldName, FieldDefinition], Field(min_length=1)]
    table_name: Annotated[str, Field(max_length=128, pattern=r"^[a-z][a-z0-9_]*$")]

    @field_validator("fields", mode="after")
    @classmethod
    def _freeze_fields(cls, fields: Mapping[str, FieldDefinition]) -> Mapping[str, FieldDefinition]:
        return FrozenMap(fields)

    @field_serializer("fields", mode="wrap")
    def _serialize_fields(
        self, fields: Mapping[str, FieldDefinition], handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        serialized: dict[str, Any] = handler(dict(fields))
        return serialized

    @property
    def has_email_field(self) -> bool:
        return any(field.python_type == EMAIL_COMPATIBLE_TYPE for field in self.fields.values())


def endpoint_id_for(method: HttpMethod, path: str) -> str:
    """Derive the deterministic id ``api_endpoint_{method}_{path_slug}``."""
    slug = re.sub(r"[^a-z0-9]+", "_", path.removeprefix(API_PREFIX)).strip("_")
    return f"api_endpoint_{method.value.lower()}_{slug}"


class ApiEndpointNode(DomainModel):
    id: Identifier
    path: ApiPath
    method: HttpMethod
    summary: SanitizedTitle
    request_model_ref: ModelName | None = None
    response_model_ref: ModelName | None = None
    handler_code: CodePayload

    @field_validator("handler_code", mode="after")
    @classmethod
    def _require_async_handler(cls, code: str) -> str:
        """Parse without executing, and require a top-level ``async def``."""
        try:
            # Parsing untrusted snippets must not spill their SyntaxWarnings into the server's logs.
            with warnings.catch_warnings(action="ignore", category=SyntaxWarning):
                module = ast.parse(code)
        except SyntaxError as error:
            raise ASTSyntaxCompilationError.from_syntax_error(error, code) from error
        except (RecursionError, MemoryError) as error:
            raise ASTSyntaxCompilationError(
                "code is too deeply nested to parse",
                line_number=1,
                syntax_token=None,
                code_snippet=code,
            ) from error
        if not any(isinstance(node, ast.AsyncFunctionDef) for node in module.body):
            raise ValueError("handler_code must define a top-level async function")
        return code

    @model_validator(mode="after")
    def _require_deterministic_id(self) -> Self:
        expected = endpoint_id_for(self.method, self.path)
        if self.id != expected:
            raise ValueError(f"endpoint id must be {expected!r}, got {self.id!r}")
        return self

    @classmethod
    def create(
        cls,
        *,
        path: str,
        method: HttpMethod,
        summary: str,
        handler_code: str,
        request_model_ref: str | None = None,
        response_model_ref: str | None = None,
    ) -> Self:
        """Build an endpoint, deriving its id from the method and path."""
        return cls(
            id=endpoint_id_for(method, path),
            path=path,
            method=method,
            summary=summary,
            request_model_ref=request_model_ref,
            response_model_ref=response_model_ref,
            handler_code=handler_code,
        )


class BackendBlueprint(DomainModel):
    service_name: Identifier
    database_dialect: DatabaseDialect
    models: tuple[DataModelSchema, ...] = ()
    endpoints: tuple[ApiEndpointNode, ...] = ()

    @model_validator(mode="after")
    def _validate_references(self) -> Self:
        require_unique((model.name for model in self.models), "model names")
        require_unique((model.table_name for model in self.models), "table names")
        require_unique((endpoint.id for endpoint in self.endpoints), "endpoint ids")
        known_models = {model.name for model in self.models}
        for endpoint in self.endpoints:
            for ref in (endpoint.request_model_ref, endpoint.response_model_ref):
                if ref is not None and ref not in known_models:
                    raise ValueError(f"endpoint {endpoint.id} references unknown model {ref}")
        return self


class FullstackProjectBundle(DomainModel):
    project_id: Identifier
    frontend: SiteBlueprint
    backend: BackendBlueprint

    @model_validator(mode="after")
    def _validate_bindings(self) -> Self:
        endpoints = {endpoint.id: endpoint for endpoint in self.backend.endpoints}
        models = {model.name: model for model in self.backend.models}
        for component in self.frontend.components:
            endpoint = None
            if component.connected_endpoint_id is not None:
                endpoint = endpoints.get(component.connected_endpoint_id)
                if endpoint is None:
                    raise EndpointBindingNotFoundError(
                        connected_endpoint_id=component.connected_endpoint_id,
                        component_id=component.id,
                    )
            if component.component_type is ComponentType.CONTACT_FORM:
                _require_contact_binding(component, endpoint, models)
        return self

    def state_hash(self) -> str:
        """Hash the canonical JSON form, which is stable across roundtrips and key order."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


def _require_contact_binding(
    component: ComponentNode,
    endpoint: ApiEndpointNode | None,
    models: Mapping[str, DataModelSchema],
) -> None:
    if endpoint is None or endpoint.method is not HttpMethod.POST:
        raise ValueError(f"contact form {component.id} must bind to a POST endpoint")
    request_model = models.get(endpoint.request_model_ref or "")
    if request_model is None or not request_model.has_email_field:
        raise ValueError(
            f"contact form {component.id} must bind to a request model with an "
            f"{EMAIL_COMPATIBLE_TYPE} field"
        )
