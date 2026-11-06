"""What the models are asked to return. Everything is validated before any of it is acted on."""

from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from realtime_agentic_site_builder.domain.backend import (
    EMAIL_COMPATIBLE_TYPE,
    ApiPath,
    DataModelSchema,
    FieldName,
    HttpMethod,
    ModelName,
    PythonType,
    endpoint_id_for,
)
from realtime_agentic_site_builder.domain.base import (
    CodePayload,
    Identifier,
    SanitizedText,
    SanitizedTitle,
    require_unique,
)
from realtime_agentic_site_builder.domain.blueprint import ComponentId, ComponentType, Theme

MIN_COMPONENT_HEIGHT: Final = 40
MAX_COMPONENT_HEIGHT: Final = 4000
DEFAULT_COMPONENT_HEIGHT: Final = 400


class AgentOutput(BaseModel):
    """Base for model replies. ``rationale`` feeds the live thought stream."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    rationale: str = ""


class ComponentSpec(AgentOutput):
    id: ComponentId
    component_type: ComponentType
    title: SanitizedTitle
    brief: SanitizedText
    height: Annotated[int, Field(ge=MIN_COMPONENT_HEIGHT, le=MAX_COMPONENT_HEIGHT)] = (
        DEFAULT_COMPONENT_HEIGHT
    )
    binds_path: ApiPath | None = None
    binds_method: HttpMethod = HttpMethod.POST

    @property
    def endpoint_id(self) -> str | None:
        return (
            None if self.binds_path is None else endpoint_id_for(self.binds_method, self.binds_path)
        )


class ModelFieldSpec(AgentOutput):
    name: FieldName
    python_type: PythonType
    nullable: StrictBool = False
    indexed: StrictBool = False


class ModelSpec(AgentOutput):
    """A flat field list, because structured outputs cannot express a free-form mapping."""

    name: ModelName
    table_name: Annotated[str, Field(max_length=128, pattern=r"^[a-z][a-z0-9_]*$")]
    fields: Annotated[tuple[ModelFieldSpec, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _require_unique_fields(self) -> Self:
        require_unique((field.name for field in self.fields), f"fields on model {self.name}")
        return self

    def to_domain(self) -> DataModelSchema:
        return DataModelSchema.model_validate(
            {
                "name": self.name,
                "table_name": self.table_name,
                "fields": {
                    field.name: {
                        "python_type": field.python_type,
                        "nullable": field.nullable,
                        "indexed": field.indexed,
                    }
                    for field in self.fields
                },
            }
        )


class EndpointSpec(AgentOutput):
    path: ApiPath
    method: HttpMethod
    summary: SanitizedTitle
    brief: SanitizedText
    request_model_ref: ModelName | None = None
    response_model_ref: ModelName | None = None

    @property
    def endpoint_id(self) -> str:
        return endpoint_id_for(self.method, self.path)


class SwarmPlan(AgentOutput):
    """The architect's blueprint. Layout comes from component order, not from the model."""

    site_title: SanitizedTitle | None = None
    theme: Theme | None = None
    components: tuple[ComponentSpec, ...] = ()
    models: tuple[ModelSpec, ...] = ()
    endpoints: tuple[EndpointSpec, ...] = ()

    @model_validator(mode="after")
    def _require_consistent_references(self) -> Self:
        require_unique((component.id for component in self.components), "component ids")
        require_unique((model.name for model in self.models), "model names")
        require_unique((endpoint.endpoint_id for endpoint in self.endpoints), "endpoints")
        endpoints = {endpoint.endpoint_id: endpoint for endpoint in self.endpoints}
        models = {model.name: model for model in self.models}
        for component in self.components:
            endpoint = None
            if component.endpoint_id is not None:
                endpoint = endpoints.get(component.endpoint_id)
                if endpoint is None:
                    raise ValueError(f"component {component.id} binds to an unplanned endpoint")
            if component.component_type is ComponentType.CONTACT_FORM:
                _require_contact_endpoint(component, endpoint, models)
        return self


def _require_contact_endpoint(
    component: ComponentSpec, endpoint: EndpointSpec | None, models: dict[str, ModelSpec]
) -> None:
    """Reject a plan the bundle's contact form rule would only refuse much later, at bind time."""
    if endpoint is None or endpoint.method is not HttpMethod.POST:
        raise ValueError(f"contact form {component.id} must bind to a POST endpoint")
    request_model = models.get(endpoint.request_model_ref or "")
    if request_model is None or not any(
        field.python_type == EMAIL_COMPATIBLE_TYPE for field in request_model.fields
    ):
        raise ValueError(
            f"contact form {component.id} needs a request model with an "
            f"{EMAIL_COMPATIBLE_TYPE} field"
        )


class CoderOutput(AgentOutput):
    code: CodePayload


class HandlerOutput(AgentOutput):
    handler_code: CodePayload


class ClassFix(AgentOutput):
    component_id: ComponentId
    find: Annotated[str, Field(min_length=1)]
    replace: str


class StyleOutput(AgentOutput):
    theme: Theme | None = None
    fixes: tuple[ClassFix, ...] = ()


class CriticIssue(AgentOutput):
    target_id: Identifier
    message: str


class CriticVerdict(AgentOutput):
    approved: StrictBool
    issues: tuple[CriticIssue, ...] = ()
