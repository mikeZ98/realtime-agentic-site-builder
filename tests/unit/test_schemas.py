import copy
import json
import pickle  # nosec B403
import warnings
from typing import Any, cast

import pytest
from pydantic import ValidationError

from realtime_agentic_site_builder.domain.backend import (
    ApiEndpointNode,
    BackendBlueprint,
    DatabaseDialect,
    DataModelSchema,
    FieldDefinition,
    FullstackProjectBundle,
    HttpMethod,
    endpoint_id_for,
)
from realtime_agentic_site_builder.domain.base import (
    MAX_JSON_DEPTH,
    MAX_TITLE_LENGTH,
    FrozenMap,
    canonical_json,
    freeze_json,
    require_unique,
    sanitize_text,
    sha256_hex,
    thaw_json,
)
from realtime_agentic_site_builder.domain.blueprint import (
    BoundingBox,
    ComponentNode,
    ComponentType,
    SiteBlueprint,
    compute_state_checksum,
    normalize_code,
)
from realtime_agentic_site_builder.domain.errors import (
    MAX_SNIPPET_LENGTH,
    PROBLEM_JSON_MEDIA_TYPE,
    PROBLEM_TYPE_BASE,
    ASTSyntaxCompilationError,
    ElementLockConflictError,
    EndpointBindingNotFoundError,
    LockAcquisitionTimeoutError,
    ProblemDetails,
    ReplayStateDivergenceError,
    SiteBuilderError,
    element_resource_uri,
)
from realtime_agentic_site_builder.domain.swarm import (
    DEFAULT_LOCK_TIMEOUT_MS,
    MAX_LOCK_TIMEOUT_MS,
    VIRTUAL_VIEWPORT_HEIGHT,
    VIRTUAL_VIEWPORT_WIDTH,
    AgentAction,
    AgentCursorFrame,
    AgentRole,
    AgentThoughtStream,
    ElementLock,
    JsonPatchOp,
    LockState,
    PatchOperation,
    ReplayTimeline,
    StreamDelta,
    SwarmSnapshot,
    arbitrate_lock,
    find_wait_cycle,
    resolve_deadlock,
)

HANDLER = "async def handler() -> None:\n    return None"
# Digests pinned from the canonical encoding, so a change to hashing or serialization that would
# invalidate recorded replays fails here even though it would still roundtrip consistently.
GOLDEN_STATE_CHECKSUM = "36ca8030c8aabdc3a74a91be52b1470f90150e8394fffe0a27e4a2c29cf878d6"
GOLDEN_BUNDLE_HASH = "b86288b37cd33ef0c8219d01bd64820b1dde699b521c8e03152543b01ee58306"


def make_box(x: int = 0, y: int = 0, width: int = 100, height: int = 50) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def make_component(
    component_id: str = "hero",
    *,
    box: BoundingBox | None = None,
    component_type: ComponentType = ComponentType.HERO,
    connected_endpoint_id: str | None = None,
) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=component_type,
        title="Welcome",
        code="<Hero />",
        box=box or make_box(),
        connected_endpoint_id=connected_endpoint_id,
    )


def make_site(*components: ComponentNode) -> SiteBlueprint:
    return SiteBlueprint(site_id="site-1", title="Acme", theme="dark", components=components)


def make_model(name: str = "Lead", python_type: str = "EmailStr") -> DataModelSchema:
    return DataModelSchema.model_validate(
        {
            "name": name,
            "fields": {
                "contact": {"python_type": python_type, "indexed": True},
                "id": {"python_type": "UUID", "default_factory": "uuid.uuid4"},
            },
            "table_name": name.lower(),
        }
    )


def make_endpoint(
    path: str = "/api/v1/leads",
    method: HttpMethod = HttpMethod.POST,
    request_model_ref: str | None = "Lead",
    response_model_ref: str | None = None,
) -> ApiEndpointNode:
    return ApiEndpointNode.create(
        path=path,
        method=method,
        summary="Create a lead",
        handler_code=HANDLER,
        request_model_ref=request_model_ref,
        response_model_ref=response_model_ref,
    )


def make_backend(
    *endpoints: ApiEndpointNode, models: tuple[DataModelSchema, ...] | None = None
) -> BackendBlueprint:
    return BackendBlueprint(
        service_name="acme-api",
        database_dialect=DatabaseDialect.POSTGRESQL,
        models=(make_model(),) if models is None else models,
        endpoints=endpoints,
    )


def make_contact_form(connected_endpoint_id: str | None) -> ComponentNode:
    return make_component(
        "contact",
        component_type=ComponentType.CONTACT_FORM,
        connected_endpoint_id=connected_endpoint_id,
    )


def make_bundle() -> FullstackProjectBundle:
    endpoint = make_endpoint()
    return FullstackProjectBundle(
        project_id="project-1",
        frontend=make_site(make_contact_form(endpoint.id)),
        backend=make_backend(endpoint),
    )


def make_lock(
    target_id: str,
    agent_id: str,
    state: LockState = LockState.ACQUIRED,
    *,
    priority: int = 50,
    acquired_at_ms: int = 1000,
    lamport_ts: int = 0,
) -> ElementLock:
    return ElementLock(
        target_id=target_id,
        locked_by_agent_id=agent_id,
        state=state,
        priority=priority,
        acquired_at_ms=acquired_at_ms,
        lamport_ts=lamport_ts,
    )


def make_thought(
    agent_id: str = "agent-a", role: AgentRole = AgentRole.STYLER
) -> AgentThoughtStream:
    return AgentThoughtStream(
        agent_id=agent_id, role=role, thought="Checking contrast", timestamp_ms=1000
    )


def make_cursor(
    agent_id: str = "agent-a",
    *,
    x: int = 10,
    y: int = 20,
    thought: AgentThoughtStream | None = None,
    lock: ElementLock | None = None,
) -> AgentCursorFrame:
    return AgentCursorFrame(
        agent_id=agent_id,
        name="Styler",
        role=AgentRole.STYLER,
        color="#FF00aa",
        x=x,
        y=y,
        current_action=AgentAction.EDITING,
        current_thought=thought,
        active_lock=lock,
    )


def make_snapshot(frame_seq: int = 0, timestamp_ms: int = 1000) -> SwarmSnapshot:
    lock = make_lock("contact", "agent-a")
    return SwarmSnapshot.capture(
        frame_seq=frame_seq,
        step_name="SKELETON_INITIALIZED",
        timestamp_ms=timestamp_ms,
        bundle_state=make_bundle(),
        active_agents=(make_cursor(thought=make_thought(), lock=lock),),
        active_locks=(lock,),
    )


def make_delta(delta_seq: int = 0, session_id: str = "session-1") -> StreamDelta:
    return StreamDelta(
        session_id=session_id,
        agent_id="agent-1",
        delta_seq=delta_seq,
        target_domain="frontend",
        target_id="hero",
        patches=(PatchOperation(op=JsonPatchOp.REMOVE, path="/components/0"),),
        emitted_at_ms=1000,
    )


# --- Immutability and closed schemas


@pytest.mark.parametrize(
    ("instance", "attribute"),
    [
        (make_box(), "x"),
        (make_component(), "title"),
        (make_site(), "title"),
        (make_model(), "name"),
        (make_endpoint(), "path"),
        (make_backend(), "service_name"),
        (make_bundle(), "project_id"),
        (make_thought(), "thought"),
        (make_lock("a", "b"), "state"),
        (make_cursor(), "x"),
        (PatchOperation(op=JsonPatchOp.REMOVE, path="/a"), "path"),
        (make_delta(), "delta_seq"),
        (make_snapshot(), "bundle_hash"),
        (ProblemDetails(title="x"), "title"),
    ],
)
def test_models_reject_attribute_assignment(instance: object, attribute: str) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        setattr(instance, attribute, 1)


def test_mapping_fields_cannot_be_mutated_in_place() -> None:
    schema = make_model()
    fields = cast("dict[str, FieldDefinition]", schema.fields)
    with pytest.raises(TypeError):
        fields["injected"] = FieldDefinition(python_type="str")

    thought = AgentThoughtStream(
        agent_id="a",
        role=AgentRole.STYLER,
        thought="x",
        tool_name="ast_parser",
        tool_args={"nested": {"items": [1]}},
        timestamp_ms=1,
    )
    assert thought.tool_args is not None
    nested = cast("dict[str, Any]", thought.tool_args["nested"])
    with pytest.raises(TypeError):
        nested["items"] = []
    assert nested["items"] == (1,)


def test_frozen_models_are_hashable_and_equal_by_value() -> None:
    assert hash(make_bundle()) == hash(make_bundle())
    assert len({make_snapshot(), make_snapshot()}) == 1
    assert make_snapshot() == make_snapshot()


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (BoundingBox, {"x": 0, "y": 0, "width": 1, "height": 1, "depth": 1}),
        (FieldDefinition, {"python_type": "str", "unique": True}),
        (ElementLock, {"target_id": "a", "owner": "b"}),
        (PatchOperation, {"op": "remove", "path": "/a", "note": "x"}),
    ],
)
def test_undeclared_fields_are_rejected(model: Any, payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="(?i)extra"):
        model.model_validate(payload)


def test_strings_are_stripped() -> None:
    site = SiteBlueprint.model_validate({"site_id": "  s1  ", "title": " T ", "theme": "dark"})
    assert (site.site_id, site.title) == ("s1", "T")


# --- Bounding boxes


@pytest.mark.parametrize(
    "overrides",
    [
        {"x": -1},
        {"y": -1},
        {"width": -1},
        {"height": -10},
        {"width": 0},
        {"height": 0},
        {"x": 1.5},
        {"x": "1"},
        {"x": True},
        {"x": 2**53},
    ],
)
def test_invalid_bounding_boxes_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BoundingBox.model_validate({"x": 0, "y": 0, "width": 10, "height": 10, **overrides})


def test_bounding_box_geometry() -> None:
    box = make_box(x=10, y=20, width=30, height=40)
    assert (box.right, box.bottom) == (40, 60)
    assert box.overlaps(make_box(x=39, y=59, width=5, height=5))
    assert not box.overlaps(make_box(x=40, y=20))
    assert not box.overlaps(make_box(x=10, y=60))
    assert not box.overlaps(make_box(x=0, y=20, width=10))
    assert not box.overlaps(make_box(x=10, y=0, height=20))


# --- Component nodes


@pytest.mark.parametrize("component_id", ["Hero", "hero_1", "hero 1", "", "héro"])
def test_invalid_component_ids_are_rejected(component_id: str) -> None:
    with pytest.raises(ValidationError):
        make_component(component_id)


@pytest.mark.parametrize("component_id", ["hero-1", "018f4e2a-7b1c-7def-8abc-0123456789ab"])
def test_slugs_and_uuid7_are_valid_component_ids(component_id: str) -> None:
    assert make_component(component_id).id == component_id


def test_ast_version_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ComponentNode.create(
            id="hero",
            component_type=ComponentType.HERO,
            title="T",
            code="<Hero />",
            box=make_box(),
            ast_version=0,
        )


def test_state_checksum_matches_independent_digest() -> None:
    assert compute_state_checksum("<Hero />", make_box()) == GOLDEN_STATE_CHECKSUM
    assert make_component().state_checksum == GOLDEN_STATE_CHECKSUM


def test_state_checksum_ignores_formatting_noise_only() -> None:
    box = make_box()
    baseline = compute_state_checksum("<A />\n<B />", box)
    assert normalize_code("  <A />  \r\n<B />\t\r\n\n") == "<A />\n<B />"
    assert compute_state_checksum("<A />  \r<B />\n", box) == baseline
    assert compute_state_checksum("<A />\n  <B />", box) != baseline
    assert compute_state_checksum("<A />\n<B />", make_box(x=1)) != baseline


@pytest.mark.parametrize(
    "tamper", [{"code": "<Evil />"}, {"box": {"x": 1, "y": 0, "width": 100, "height": 50}}]
)
def test_component_drift_is_detected(tamper: dict[str, object]) -> None:
    payload = {**make_component().model_dump(mode="json"), **tamper}
    with pytest.raises(ValidationError, match="state_checksum drift"):
        ComponentNode.model_validate(payload)


# --- Text sanitization


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<script>alert(1)</script>Hello", "Hello"),
        ("Hi<script>alert(1)", "Hi"),
        ("<style>b{}</style>Hi", "Hi"),
        ("Hello<img src=x onerror=alert(1)>", "Hello"),
        ("&lt;script&gt;alert(1)&lt;/script&gt;Hi", "Hi"),
        ("&amp;lt;b&amp;gt;Hi", "Hi"),
        ("java\x00script:alert(1)", "alert(1)"),
        ("JaVaScRiPt : alert(1)", "alert(1)"),
        ("data:text/html,hello", ",hello"),
        ('x" onmouseover="alert(1)', 'x" "alert(1)'),
        ("＜script＞x＜/script＞Hi", "Hi"),
        ("Pri‮ce​", "Price"),
        ("Line\nbreak\tok", "Line break ok"),
        ("contrast < 4.5", "contrast < 4.5"),
        ("Terms & Conditions", "Terms & Conditions"),
    ],
)
def test_sanitize_text(raw: str, expected: str) -> None:
    cleaned = sanitize_text(raw)
    assert cleaned == expected
    assert sanitize_text(cleaned) == cleaned


@pytest.mark.parametrize(
    "title", ["   ", "\x00\x01", "<script>alert(1)</script>", "<scr<script>ipt>x</script>Hi"]
)
def test_titles_with_nothing_left_after_sanitizing_are_rejected(title: str) -> None:
    with pytest.raises(ValidationError):
        ComponentNode.create(
            id="hero",
            component_type=ComponentType.HERO,
            title=title,
            code="<Hero />",
            box=make_box(),
        )


@pytest.mark.parametrize(
    "crafted",
    [
        "&" + "amp;" * 1000 + "lt;x",
        "java" * 100 + "script:" * 100,
    ],
)
def test_text_needing_many_sanitizer_passes_is_rejected(crafted: str) -> None:
    with pytest.raises(ValueError, match="too deeply obfuscated"):
        sanitize_text(crafted)


def test_model_reports_over_obfuscated_text_as_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="too deeply obfuscated"):
        SiteBlueprint(site_id="s", title="&" + "amp;" * 50 + "lt;x", theme="light")


def test_title_is_sanitized_on_the_model() -> None:
    site = SiteBlueprint(site_id="s", title="Acme<script>x</script>", theme="light")
    assert site.title == "Acme"


def test_title_expanding_past_its_limit_is_rejected_rather_than_breaking_reparse() -> None:
    with pytest.raises(ValidationError, match="after sanitization"):
        SiteBlueprint(site_id="s", title="ﷺ" * MAX_TITLE_LENGTH, theme="light")


# --- Site blueprint


def test_component_ids_must_be_unique() -> None:
    first = make_component("same", box=make_box(y=0))
    second = make_component("same", box=make_box(y=50))
    with pytest.raises(ValidationError, match="duplicate component ids: same"):
        make_site(first, second)


def test_vertical_stack_and_grid_rows_are_valid() -> None:
    site = make_site(
        make_component("nav", box=make_box(width=300, height=50)),
        make_component("left", box=make_box(y=50, width=100)),
        make_component("right", box=make_box(x=100, y=50, width=200)),
        make_component("foot", box=make_box(y=100, width=300)),
    )
    assert [component.id for component in site.components] == ["nav", "left", "right", "foot"]


def test_empty_site_is_valid_as_an_initial_skeleton() -> None:
    assert make_site().components == ()


@pytest.mark.parametrize(
    ("first", "second"),
    [(make_box(y=50), make_box(y=0)), (make_box(x=100, y=0), make_box(x=0, y=0))],
)
def test_components_out_of_reading_order_are_rejected(
    first: BoundingBox, second: BoundingBox
) -> None:
    with pytest.raises(ValidationError, match="out of order"):
        make_site(make_component("a", box=first), make_component("b", box=second))


def test_overlapping_components_are_rejected() -> None:
    with pytest.raises(ValidationError, match="components a and b overlap"):
        make_site(
            make_component("a", box=make_box(height=50)),
            make_component("b", box=make_box(y=49, height=50)),
        )


def test_overlap_is_found_past_a_disjoint_neighbour() -> None:
    with pytest.raises(ValidationError, match="components tall and c overlap"):
        make_site(
            make_component("tall", box=make_box(width=100, height=500)),
            make_component("side", box=make_box(x=200, y=0, width=50, height=50)),
            make_component("c", box=make_box(x=50, y=100, width=10, height=10)),
        )


# --- Backend contracts


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "lead", "fields": {"a": {"python_type": "str"}}, "table_name": "leads"},
        {"name": "L", "fields": {"a": {"python_type": "str"}}, "table_name": "leads"},
        {"name": "Lead", "fields": {}, "table_name": "leads"},
        {"name": "Lead", "fields": {"Bad-Name": {"python_type": "str"}}, "table_name": "leads"},
        {"name": "Lead", "fields": {"class": {"python_type": "str"}}, "table_name": "leads"},
        {"name": "Lead", "fields": {"a": {"python_type": "bytes"}}, "table_name": "leads"},
        {"name": "Lead", "fields": {"a": {"python_type": "str"}}, "table_name": "Leads"},
        {"name": "Lead", "fields": {"a": {"python_type": "str"}}, "table_name": "leads; DROP"},
        {
            "name": "Lead",
            "fields": {"a": {"python_type": "str", "default_factory": "os.system('x')"}},
            "table_name": "leads",
        },
        {
            "name": "Lead",
            "fields": {"a": {"python_type": "str", "nullable": "yes"}},
            "table_name": "leads",
        },
    ],
)
def test_invalid_data_models_are_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DataModelSchema.model_validate(payload)


def test_data_model_reports_email_compatible_fields() -> None:
    assert make_model().has_email_field
    assert not make_model(python_type="str").has_email_field


@pytest.mark.parametrize(
    ("method", "path", "expected"),
    [
        (HttpMethod.GET, "/api/v1/leads", "api_endpoint_get_leads"),
        (HttpMethod.PATCH, "/api/v1/leads/{lead_id}", "api_endpoint_patch_leads_lead_id"),
        (HttpMethod.DELETE, "/api/v1/a-b/c", "api_endpoint_delete_a_b_c"),
    ],
)
def test_endpoint_ids_are_deterministic(method: HttpMethod, path: str, expected: str) -> None:
    assert endpoint_id_for(method, path) == expected
    assert make_endpoint(path, method, None).id == expected


def test_endpoint_id_must_match_method_and_path() -> None:
    payload = {**make_endpoint().model_dump(mode="json"), "id": "api_endpoint_get_leads"}
    with pytest.raises(ValidationError, match="endpoint id must be"):
        ApiEndpointNode.model_validate(payload)


@pytest.mark.parametrize(
    "path",
    [
        "/api/v2/leads",
        "/api/v1/",
        "/api/v1",
        "/leads",
        "/api/v1/../admin",
        "/api/v1//leads",
        "/api/v1/leads/",
        "/api/v1/leads?x=1",
        "/api/v1/Leads",
        "/api/v1/le%2fads",
    ],
)
def test_unsafe_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValidationError):
        make_endpoint(path)


def test_handler_with_invalid_syntax_raises_compilation_error() -> None:
    code = "async def broken(:\n    pass"
    with pytest.raises(ASTSyntaxCompilationError) as caught:
        ApiEndpointNode.create(
            path="/api/v1/leads", method=HttpMethod.GET, summary="S", handler_code=code
        )
    problem = caught.value.to_problem().to_json_dict()
    assert problem["status"] == 422
    assert problem["line_number"] == 1
    assert problem["syntax_token"] == ":"
    assert problem["code_snippet"] == code


def test_handler_too_deep_to_parse_raises_compilation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overflow(_source: str) -> None:
        raise RecursionError

    monkeypatch.setattr("ast.parse", overflow)
    with pytest.raises(ASTSyntaxCompilationError, match="too deeply nested"):
        make_endpoint()


@pytest.mark.parametrize("code", ["def handler():\n    return 1", "x = 1"])
def test_handler_must_define_an_async_function(code: str) -> None:
    with pytest.raises(ValidationError, match="top-level async function"):
        ApiEndpointNode.create(
            path="/api/v1/leads", method=HttpMethod.GET, summary="S", handler_code=code
        )


def test_handler_parsing_does_not_emit_warnings_or_execute_code() -> None:
    code = 'raise SystemExit(1)\nasync def handler():\n    return "\\d"'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        endpoint = ApiEndpointNode.create(
            path="/api/v1/leads", method=HttpMethod.GET, summary="S", handler_code=code
        )
    assert endpoint.handler_code == code


def test_compilation_error_handles_sparse_syntax_errors() -> None:
    bare = ASTSyntaxCompilationError.from_syntax_error(SyntaxError("bad"), "x")
    assert bare.extensions["syntax_token"] is None
    assert bare.extensions["line_number"] == 1

    open_ended = SyntaxError("bad", ("f", 3, 2, "a b c\n", None, None))
    error = ASTSyntaxCompilationError.from_syntax_error(open_ended, "x")
    assert error.extensions["syntax_token"] == "b c"
    assert error.extensions["line_number"] == 3


def test_compilation_error_truncates_long_snippets() -> None:
    error = ASTSyntaxCompilationError(
        "bad", line_number=1, syntax_token=None, code_snippet="x" * (MAX_SNIPPET_LENGTH + 1)
    )
    assert error.extensions["code_snippet"] == "x" * MAX_SNIPPET_LENGTH


@pytest.mark.parametrize(
    ("models", "endpoints", "message"),
    [
        ((make_model(), make_model()), (), "duplicate model names: Lead"),
        ((make_model("Lead"), make_model("LEAD")), (), None),
        (
            (make_model(),),
            (make_endpoint(), make_endpoint()),
            "duplicate endpoint ids: api_endpoint_post_leads",
        ),
        (
            (make_model(),),
            (make_endpoint("/api/v1/a-b"), make_endpoint("/api/v1/a_b")),
            "duplicate endpoint ids: api_endpoint_post_a_b",
        ),
        ((make_model(),), (make_endpoint(request_model_ref="Nope"),), "unknown model Nope"),
        ((make_model(),), (make_endpoint(response_model_ref="Nope"),), "unknown model Nope"),
    ],
)
def test_backend_rejects_inconsistent_definitions(
    models: tuple[DataModelSchema, ...],
    endpoints: tuple[ApiEndpointNode, ...],
    message: str | None,
) -> None:
    with pytest.raises(ValidationError, match=message or "duplicate table names: lead"):
        make_backend(*endpoints, models=models)


# --- Cross-domain bindings


def test_valid_bindings_pass() -> None:
    endpoint = make_endpoint()
    plain = make_endpoint("/api/v1/stats", HttpMethod.GET, None, "Lead")
    bundle = FullstackProjectBundle(
        project_id="p",
        frontend=make_site(
            make_contact_form(endpoint.id),
            make_component("stats", box=make_box(y=50), connected_endpoint_id=plain.id),
            make_component("hero", box=make_box(y=100)),
        ),
        backend=make_backend(endpoint, plain),
    )
    assert len(bundle.frontend.components) == 3


def test_dangling_endpoint_reference_raises_binding_error() -> None:
    with pytest.raises(EndpointBindingNotFoundError) as caught:
        FullstackProjectBundle(
            project_id="p",
            frontend=make_site(make_component(connected_endpoint_id="api_endpoint_get_gone")),
            backend=make_backend(),
        )
    problem = caught.value.to_problem().to_json_dict()
    assert problem["status"] == 400
    assert problem["connected_endpoint_id"] == "api_endpoint_get_gone"
    assert problem["component_id"] == "hero"


@pytest.mark.parametrize(
    ("endpoint", "models", "message"),
    [
        (None, (make_model(),), "must bind to a POST endpoint"),
        (
            make_endpoint(method=HttpMethod.PUT),
            (make_model(),),
            "must bind to a POST endpoint",
        ),
        (make_endpoint(request_model_ref=None), (make_model(),), "EmailStr field"),
        (make_endpoint(), (make_model(python_type="str"),), "EmailStr field"),
    ],
)
def test_contact_form_requires_post_with_email_field(
    endpoint: ApiEndpointNode | None, models: tuple[DataModelSchema, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        FullstackProjectBundle(
            project_id="p",
            frontend=make_site(make_contact_form(endpoint.id if endpoint else None)),
            backend=make_backend(*(() if endpoint is None else (endpoint,)), models=models),
        )


# --- Thought stream and cursors


def test_thought_stream_records_tool_telemetry() -> None:
    thought = AgentThoughtStream(
        agent_id="agent-a",
        role=AgentRole.BACKEND_ENGINEER,
        thought="Compiling <script>x</script>schema",
        tool_name="compile_pydantic_schema",
        tool_args={"model": "Lead", "retries": [1, 2.5, None, True]},
        tokens_used=120,
        execution_latency_ms=12.5,
        timestamp_ms=1000,
    )
    assert thought.thought == "Compiling schema"
    assert AgentThoughtStream.model_validate_json(thought.model_dump_json()) == thought
    dumped = json.loads(thought.model_dump_json())
    assert dumped["tool_args"] == {"model": "Lead", "retries": [1, 2.5, None, True]}


@pytest.mark.parametrize(
    "overrides",
    [
        {"tokens_used": -1},
        {"execution_latency_ms": -0.1},
        {"execution_latency_ms": float("inf")},
        {"execution_latency_ms": float("nan")},
        {"timestamp_ms": -1},
        {"tool_name": "bad name"},
        {"tool_args": {"a": 1}},
        {"tool_name": "t", "tool_args": {"a": object()}},
        {"tool_name": "t", "tool_args": {"a": float("nan")}},
        {"tool_name": "t", "tool_args": {"a": {1: "non-string key"}}},
    ],
)
def test_invalid_telemetry_is_rejected(overrides: dict[str, object]) -> None:
    payload = {
        "agent_id": "a",
        "role": "STYLER",
        "thought": "x",
        "timestamp_ms": 1,
        **overrides,
    }
    with pytest.raises(ValidationError):
        AgentThoughtStream.model_validate(payload)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (10, 20, (10, 20)),
        (-5, -5, (0, 0)),
        (10**6, 10**6, (VIRTUAL_VIEWPORT_WIDTH, VIRTUAL_VIEWPORT_HEIGHT)),
    ],
)
def test_cursor_coordinates_are_clamped(x: int, y: int, expected: tuple[int, int]) -> None:
    cursor = make_cursor(x=x, y=y)
    assert (cursor.x, cursor.y) == expected
    assert AgentCursorFrame.model_validate_json(cursor.model_dump_json()) == cursor


@pytest.mark.parametrize(
    "overrides",
    [{"color": "red"}, {"color": "#12345"}, {"color": "#GGGGGG"}, {"x": True}, {"y": "1"}],
)
def test_invalid_cursor_frames_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AgentCursorFrame.model_validate(
            {
                "agent_id": "a",
                "name": "n",
                "role": "STYLER",
                "color": "#000000",
                "x": 0,
                "y": 0,
                **overrides,
            }
        )


@pytest.mark.parametrize(
    "thought",
    [make_thought(agent_id="agent-b"), make_thought(role=AgentRole.ARCHITECT)],
)
def test_cursor_rejects_thoughts_of_other_agents(thought: AgentThoughtStream) -> None:
    with pytest.raises(ValidationError, match="current_thought does not belong"):
        make_cursor(thought=thought)


def test_cursor_rejects_locks_of_other_agents() -> None:
    with pytest.raises(ValidationError, match="active_lock does not belong"):
        make_cursor(lock=make_lock("hero", "agent-b"))


# --- Lock fields and arbitration


def test_lock_timeout_bounds() -> None:
    assert make_lock("a", "b").timeout_ms == DEFAULT_LOCK_TIMEOUT_MS == 5000
    assert MAX_LOCK_TIMEOUT_MS == 30000
    base = make_lock("a", "b").model_dump(mode="json")
    assert ElementLock.model_validate({**base, "timeout_ms": 30000}).timeout_ms == 30000
    for timeout_ms in (0, 30001):
        with pytest.raises(ValidationError):
            ElementLock.model_validate({**base, "timeout_ms": timeout_ms})


@pytest.mark.parametrize("priority", [0, 101, True, "50"])
def test_lock_priority_bounds(priority: object) -> None:
    base = make_lock("a", "b").model_dump(mode="json")
    with pytest.raises(ValidationError):
        ElementLock.model_validate({**base, "priority": priority})


def test_lock_expiry_boundary() -> None:
    lock = make_lock("a", "b", acquired_at_ms=1000)
    assert lock.expires_at_ms == 6000
    assert not lock.is_expired(5999)
    assert lock.is_expired(6000)


def test_lock_ranking_is_a_total_order() -> None:
    assert make_lock("t", "a", priority=60).outranks(make_lock("t", "b", priority=50))
    assert make_lock("t", "a", lamport_ts=1).outranks(make_lock("t", "b", lamport_ts=2))
    assert make_lock("t", "b").outranks(make_lock("t", "a"))
    assert not make_lock("t", "a").outranks(make_lock("t", "b"))


def test_higher_priority_does_not_preempt_a_live_holder_without_deadlock() -> None:
    held = make_lock("t", "low", priority=1)
    request = make_lock("t", "high", LockState.WAITING, priority=100)
    result = arbitrate_lock(held, request, now_ms=2000)
    assert result.held == held
    assert result.request == request


def test_expired_holder_is_released_to_the_request() -> None:
    held = make_lock("t", "old")
    request = make_lock("t", "new", LockState.WAITING, acquired_at_ms=5500)
    result = arbitrate_lock(held, request, now_ms=6000)
    assert result.held.state is LockState.TIMEOUT_RELEASED
    assert result.request.state is LockState.ACQUIRED
    assert result.request.acquired_at_ms == 6000


def test_request_is_granted_when_element_is_free() -> None:
    free = make_lock("t", "old", LockState.UNLOCKED)
    result = arbitrate_lock(free, make_lock("t", "new", LockState.WAITING), now_ms=2000)
    assert result.held == free
    assert result.request.state is LockState.ACQUIRED


def test_request_past_its_deadline_raises_timeout() -> None:
    held = make_lock("t", "holder", acquired_at_ms=5000)
    request = make_lock("t", "late", LockState.WAITING, acquired_at_ms=1000)
    with pytest.raises(LockAcquisitionTimeoutError) as caught:
        arbitrate_lock(held, request, now_ms=6000)
    problem = caught.value.to_problem().to_json_dict()
    assert problem["status"] == 408
    assert (problem["agent_id"], problem["target_id"], problem["timeout_ms"]) == ("late", "t", 5000)


@pytest.mark.parametrize(
    ("held", "contender", "message"),
    [
        (make_lock("t", "a"), make_lock("other", "b", LockState.WAITING), "same element"),
        (make_lock("t", "a"), make_lock("t", "a", LockState.WAITING), "its own lock"),
        (make_lock("t", "a"), make_lock("t", "b"), "WAITING state"),
    ],
)
def test_malformed_arbitration_is_rejected(
    held: ElementLock, contender: ElementLock, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        arbitrate_lock(held, contender, now_ms=1000)


# --- Deadlock detection and preemption


def crossed_locks(priority_a: int = 50, priority_b: int = 50, **kwargs: int) -> list[ElementLock]:
    """Agent a holds x and wants y while agent b holds y and wants x."""
    lamport_a = kwargs.get("lamport_a", 0)
    lamport_b = kwargs.get("lamport_b", 0)
    return [
        make_lock("x", "a", priority=priority_a, lamport_ts=lamport_a),
        make_lock("y", "a", LockState.WAITING, priority=priority_a, lamport_ts=lamport_a),
        make_lock("y", "b", priority=priority_b, lamport_ts=lamport_b),
        make_lock("x", "b", LockState.WAITING, priority=priority_b, lamport_ts=lamport_b),
    ]


def states_by_key(locks: tuple[ElementLock, ...]) -> dict[tuple[str, str], LockState]:
    return {(lock.locked_by_agent_id, lock.target_id): lock.state for lock in locks}


def test_two_agent_deadlock_preempts_the_lower_priority_agent() -> None:
    resolution = resolve_deadlock(crossed_locks(priority_a=80, priority_b=20), now_ms=2000)
    assert resolution is not None
    assert set(resolution.cycle) == {"a", "b"}
    assert resolution.victim_agent_id == "b"
    assert resolution.released_target_ids == ("y",)
    assert states_by_key(resolution.locks) == {
        ("a", "x"): LockState.ACQUIRED,
        ("a", "y"): LockState.ACQUIRED,
        ("b", "y"): LockState.CONFLICT_RESOLVED,
        ("b", "x"): LockState.CONFLICT_RESOLVED,
    }
    granted = next(
        lock
        for lock in resolution.locks
        if lock.locked_by_agent_id == "a" and lock.target_id == "y"
    )
    assert granted.acquired_at_ms == 2000
    assert find_wait_cycle(resolution.locks) == ()


def test_equal_priority_deadlock_preempts_the_younger_lamport_timestamp() -> None:
    resolution = resolve_deadlock(crossed_locks(lamport_a=9, lamport_b=3), now_ms=2000)
    assert resolution is not None
    assert resolution.victim_agent_id == "a"


def test_full_tie_is_broken_by_agent_id() -> None:
    resolution = resolve_deadlock(crossed_locks(), now_ms=2000)
    assert resolution is not None
    assert resolution.victim_agent_id == "a"


def test_deadlock_victim_is_independent_of_input_order() -> None:
    locks = crossed_locks(priority_a=80, priority_b=20)
    forward = resolve_deadlock(locks, now_ms=2000)
    backward = resolve_deadlock(list(reversed(locks)), now_ms=2000)
    assert forward is not None
    assert backward is not None
    assert forward.victim_agent_id == backward.victim_agent_id
    assert states_by_key(forward.locks) == states_by_key(backward.locks)


def test_three_agent_cycle_is_detected_and_broken() -> None:
    locks = [
        make_lock("x", "a", priority=30),
        make_lock("y", "a", LockState.WAITING, priority=30),
        make_lock("y", "b", priority=10),
        make_lock("z", "b", LockState.WAITING, priority=10),
        make_lock("z", "c", priority=20),
        make_lock("x", "c", LockState.WAITING, priority=20),
    ]
    assert find_wait_cycle(locks) == ("a", "b", "c")
    resolution = resolve_deadlock(locks, now_ms=2000)
    assert resolution is not None
    assert resolution.victim_agent_id == "b"
    assert find_wait_cycle(resolution.locks) == ()


def test_released_element_goes_to_the_highest_ranked_waiter() -> None:
    locks = [
        *crossed_locks(priority_a=80, priority_b=20),
        make_lock("y", "c", LockState.WAITING, priority=90),
    ]
    resolution = resolve_deadlock(locks, now_ms=2000)
    assert resolution is not None
    states = states_by_key(resolution.locks)
    assert states["c", "y"] is LockState.ACQUIRED
    assert states["a", "y"] is LockState.WAITING


def test_victim_gives_up_uncontended_locks_and_keeps_finished_ones() -> None:
    locks = [
        *crossed_locks(priority_a=80, priority_b=20),
        make_lock("idle", "b"),
        make_lock("old", "b", LockState.TIMEOUT_RELEASED),
    ]
    resolution = resolve_deadlock(locks, now_ms=2000)
    assert resolution is not None
    assert resolution.released_target_ids == ("idle", "y")
    states = states_by_key(resolution.locks)
    assert states["b", "idle"] is LockState.CONFLICT_RESOLVED
    assert states["b", "old"] is LockState.TIMEOUT_RELEASED


def test_independent_deadlocks_clear_one_call_at_a_time() -> None:
    other = [
        make_lock("p", "c", priority=10),
        make_lock("q", "c", LockState.WAITING, priority=10),
        make_lock("q", "d", priority=90),
        make_lock("p", "d", LockState.WAITING, priority=90),
    ]
    first = resolve_deadlock([*crossed_locks(priority_a=80, priority_b=20), *other], now_ms=2000)
    assert first is not None
    assert first.victim_agent_id == "b"
    second = resolve_deadlock(first.locks, now_ms=2000)
    assert second is not None
    assert second.victim_agent_id == "c"
    assert resolve_deadlock(second.locks, now_ms=2000) is None


@pytest.mark.parametrize(
    "locks",
    [
        [],
        [make_lock("x", "a")],
        [make_lock("x", "a"), make_lock("x", "b", LockState.WAITING)],
        [make_lock("x", "a", LockState.WAITING)],
        [make_lock("x", "a"), make_lock("x", "a", LockState.WAITING)],
        [
            make_lock("x", "c"),
            make_lock("x", "a", LockState.WAITING),
            make_lock("x", "b", LockState.WAITING),
        ],
        [
            make_lock("x", "b"),
            make_lock("y", "c"),
            make_lock("x", "a", LockState.WAITING),
            make_lock("y", "a", LockState.WAITING),
            make_lock("y", "b", LockState.WAITING),
        ],
    ],
)
def test_contention_without_a_cycle_is_not_a_deadlock(locks: list[ElementLock]) -> None:
    assert find_wait_cycle(locks) == ()
    assert resolve_deadlock(locks, now_ms=2000) is None


def test_long_wait_chains_do_not_exhaust_the_call_stack() -> None:
    agents = 5000
    chain = [make_lock(f"t{i}", f"a{i}") for i in range(agents)]
    chain += [make_lock(f"t{i + 1}", f"a{i}", LockState.WAITING) for i in range(agents - 1)]
    assert find_wait_cycle(chain) == ()

    ring = [*chain, make_lock("t0", f"a{agents - 1}", LockState.WAITING)]
    assert len(find_wait_cycle(ring)) == agents


def test_cycle_is_reported_without_the_agents_leading_into_it() -> None:
    locks = [
        make_lock("x", "b"),
        make_lock("y", "c"),
        make_lock("x", "a", LockState.WAITING),
        make_lock("y", "b", LockState.WAITING),
        make_lock("x", "c", LockState.WAITING),
    ]
    assert find_wait_cycle(locks) == ("b", "c")


def test_deadlock_resolution_renders_a_conflict_problem() -> None:
    resolution = resolve_deadlock(crossed_locks(priority_a=80, priority_b=20), now_ms=2000)
    assert resolution is not None
    problem = resolution.to_error().to_problem().to_json_dict()
    assert problem["status"] == 409
    assert problem["competing_agent_ids"] == ["a", "b"]
    assert problem["resource_uri"] == "urn:site-builder:element:y"


def test_two_holders_of_one_element_is_a_lock_collision() -> None:
    with pytest.raises(ElementLockConflictError) as caught:
        find_wait_cycle([make_lock("x", "a"), make_lock("x", "b")])
    problem = caught.value.to_problem().to_json_dict()
    assert problem["competing_agent_ids"] == ["a", "b"]
    assert problem["resource_uri"] == "urn:site-builder:element:x"


def test_element_resource_uri_is_percent_encoded() -> None:
    assert element_resource_uri("a/b c") == "urn:site-builder:element:a%2Fb%20c"


# --- RFC 6902 patches


@pytest.mark.parametrize(
    "document",
    [
        {"op": "add", "path": "/components/0/code", "value": "<A />"},
        {"op": "add", "path": "/a", "value": None},
        {"op": "replace", "path": "/a", "value": {"b": [1, {"c": None}]}},
        {"op": "test", "path": "", "value": 1.5},
        {"op": "remove", "path": "/a~1b/~0c"},
        {"op": "move", "path": "/a", "from": "/b"},
        {"op": "move", "path": "/a", "from": "/a"},
        {"op": "copy", "path": "/a/b", "from": "/a"},
        {"op": "add", "path": "/trailing space ", "value": 1},
    ],
)
def test_patches_roundtrip_to_the_exact_rfc6902_shape(document: dict[str, Any]) -> None:
    patch = PatchOperation.model_validate(document)
    assert patch.model_dump(mode="json") == document
    assert json.loads(patch.model_dump_json()) == document
    assert PatchOperation.model_validate_json(patch.model_dump_json()) == patch


def test_move_accepts_from_by_python_name() -> None:
    patch = PatchOperation(op=JsonPatchOp.MOVE, path="/a", from_="/b")
    assert patch.model_dump(mode="json")["from"] == "/b"


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"op": "add", "path": "/a"}, "'value' is required on 'add'"),
        ({"op": "replace", "path": "/a"}, "'value' is required on 'replace'"),
        ({"op": "test", "path": "/a"}, "'value' is required on 'test'"),
        ({"op": "remove", "path": "/a", "value": 1}, "'value' is not allowed on 'remove'"),
        ({"op": "move", "path": "/a", "value": 1, "from": "/b"}, "'value' is not allowed"),
        ({"op": "move", "path": "/a"}, "'from' is required on 'move'"),
        ({"op": "copy", "path": "/a"}, "'from' is required on 'copy'"),
        ({"op": "remove", "path": "/a", "from": "/b"}, "'from' is not allowed on 'remove'"),
        ({"op": "move", "path": "/a/b", "from": "/a"}, "own children"),
        ({"op": "move", "path": "/a", "from": ""}, "own children"),
        ({"op": "remove", "path": "no-slash"}, "pattern"),
        ({"op": "remove", "path": "/bad~2escape"}, "pattern"),
        ({"op": "remove", "path": "/dangling~"}, "pattern"),
        ({"op": "copy", "path": "/a", "from": "bad"}, "pattern"),
        ({"op": "merge", "path": "/a"}, "Input should be"),
    ],
)
def test_invalid_patches_are_rejected(document: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PatchOperation.model_validate(document)


def test_patch_values_must_be_json_compatible() -> None:
    with pytest.raises(ValidationError, match="not JSON-compatible"):
        PatchOperation(op=JsonPatchOp.ADD, path="/a", value=object())


@pytest.mark.parametrize(
    "overrides",
    [
        {"patches": []},
        {"delta_seq": -1},
        {"agent_id": "bad id!"},
        {"agent_id": ""},
        {"agent_id": None},
        {"target_domain": "database"},
        {"emitted_at_ms": -1},
    ],
)
def test_invalid_deltas_are_rejected(overrides: dict[str, object]) -> None:
    payload = {**make_delta().model_dump(mode="json"), **overrides}
    with pytest.raises(ValidationError):
        StreamDelta.model_validate(payload)


def test_delta_requires_an_author() -> None:
    payload = make_delta().model_dump(mode="json")
    del payload["agent_id"]
    with pytest.raises(ValidationError, match="agent_id"):
        StreamDelta.model_validate(payload)


def test_delta_author_is_normalized_to_match_lock_owners() -> None:
    payload = {**make_delta().model_dump(mode="json"), "agent_id": "  agent-1  "}
    assert StreamDelta.model_validate(payload).agent_id == "agent-1"


@pytest.mark.parametrize("agent_id", ["   ", "<script>", "a b", "a\nb", "x" * 129])
def test_delta_author_rejects_unsanitary_values(agent_id: str) -> None:
    payload = {**make_delta().model_dump(mode="json"), "agent_id": agent_id}
    with pytest.raises(ValidationError):
        StreamDelta.model_validate(payload)


# --- Replay protocol


def test_snapshot_rejects_a_mismatched_bundle_hash() -> None:
    payload = make_snapshot().model_dump(mode="json")
    payload["bundle_hash"] = "0" * 64
    with pytest.raises(ReplayStateDivergenceError) as caught:
        SwarmSnapshot.model_validate(payload)
    problem = caught.value.to_problem().to_json_dict()
    assert problem["status"] == 500
    assert problem["frame_seq"] == 0
    assert problem["expected_hash"] == "0" * 64
    assert problem["computed_hash"] == make_bundle().state_hash()


def test_snapshot_detects_a_tampered_bundle_with_a_valid_shape() -> None:
    payload = make_snapshot().model_dump(mode="json")
    payload["bundle_state"]["project_id"] = "project-2"
    with pytest.raises(ReplayStateDivergenceError):
        SwarmSnapshot.model_validate(payload)


def test_replayed_state_is_verified_against_the_frame() -> None:
    snapshot = make_snapshot()
    snapshot.verify_replayed_state(make_bundle())
    other = make_bundle().model_copy(update={"project_id": "project-2"})
    with pytest.raises(ReplayStateDivergenceError):
        snapshot.verify_replayed_state(other)


def test_snapshot_rejects_colliding_holders() -> None:
    with pytest.raises(ElementLockConflictError):
        SwarmSnapshot.capture(
            frame_seq=0,
            step_name="THEME_INJECTED",
            timestamp_ms=1,
            bundle_state=make_bundle(),
            active_locks=(make_lock("x", "a"), make_lock("x", "b")),
        )


def test_snapshot_allows_a_holder_with_waiters_and_released_locks() -> None:
    locks = (
        make_lock("x", "a"),
        make_lock("x", "b", LockState.WAITING),
        make_lock("x", "c", LockState.TIMEOUT_RELEASED),
    )
    snapshot = SwarmSnapshot.capture(
        frame_seq=0,
        step_name="BACKEND_SCAFFOLDED",
        timestamp_ms=1,
        bundle_state=make_bundle(),
        active_locks=locks,
    )
    assert snapshot.active_locks == locks


def test_snapshot_rejects_duplicate_agents() -> None:
    with pytest.raises(ValidationError, match="duplicate agent ids"):
        SwarmSnapshot.capture(
            frame_seq=0,
            step_name="STEP",
            timestamp_ms=1,
            bundle_state=make_bundle(),
            active_agents=(make_cursor(), make_cursor()),
        )


def test_snapshot_requires_agent_locks_to_be_listed() -> None:
    with pytest.raises(ValidationError, match="missing from active_locks"):
        SwarmSnapshot.capture(
            frame_seq=0,
            step_name="STEP",
            timestamp_ms=1,
            bundle_state=make_bundle(),
            active_agents=(make_cursor(lock=make_lock("x", "agent-a")),),
        )


@pytest.mark.parametrize("step_name", ["lowercase", "HAS SPACE", ""])
def test_snapshot_step_name_format(step_name: str) -> None:
    with pytest.raises(ValidationError):
        SwarmSnapshot.capture(
            frame_seq=0, step_name=step_name, timestamp_ms=1, bundle_state=make_bundle()
        )


def test_timeline_accepts_monotonic_history() -> None:
    timeline = ReplayTimeline(
        session_id="session-1",
        deltas=(make_delta(0), make_delta(1), make_delta(5)),
        snapshots=(make_snapshot(0, 1000), make_snapshot(1, 1000), make_snapshot(2, 2000)),
    )
    assert ReplayTimeline.model_validate_json(timeline.model_dump_json()) == timeline


@pytest.mark.parametrize(
    ("deltas", "snapshots", "message"),
    [
        ((make_delta(0, "session-2"),), (), "another session"),
        ((make_delta(1), make_delta(1)), (), "delta_seq 1 is not monotonic"),
        ((make_delta(2), make_delta(1)), (), "delta_seq 1 is not monotonic"),
        ((), (make_snapshot(1), make_snapshot(1)), "frame_seq 1 is not monotonic"),
        ((), (make_snapshot(0, 2000), make_snapshot(1, 1000)), "backwards in time"),
    ],
)
def test_timeline_rejects_out_of_order_history(
    deltas: tuple[StreamDelta, ...], snapshots: tuple[SwarmSnapshot, ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ReplayTimeline(session_id="session-1", deltas=deltas, snapshots=snapshots)


# --- Serialization determinism


def test_bundle_hash_matches_pinned_digest() -> None:
    assert make_bundle().state_hash() == GOLDEN_BUNDLE_HASH


def test_json_roundtrip_preserves_checksums_and_bundle_hash() -> None:
    snapshot = make_snapshot()
    encoded = snapshot.model_dump_json()
    restored = SwarmSnapshot.model_validate_json(encoded)

    assert restored == snapshot
    assert restored.model_dump_json() == encoded
    assert restored.bundle_hash == snapshot.bundle_hash == GOLDEN_BUNDLE_HASH
    assert restored.bundle_state.state_hash() == GOLDEN_BUNDLE_HASH
    assert (
        restored.bundle_state.frontend.components[0].state_checksum
        == snapshot.bundle_state.frontend.components[0].state_checksum
        == GOLDEN_STATE_CHECKSUM
    )


def test_python_mode_roundtrip_is_equivalent() -> None:
    snapshot = make_snapshot()
    assert SwarmSnapshot.model_validate(snapshot.model_dump()) == snapshot
    assert SwarmSnapshot.model_validate(snapshot.model_dump(mode="json")) == snapshot


def test_bundle_hash_is_independent_of_mapping_key_order() -> None:
    forward = {"a": {"python_type": "str"}, "b": {"python_type": "EmailStr"}}
    backward = dict(reversed(list(forward.items())))

    def bundle(fields: dict[str, dict[str, str]]) -> FullstackProjectBundle:
        model = DataModelSchema.model_validate(
            {"name": "Lead", "fields": fields, "table_name": "lead"}
        )
        return FullstackProjectBundle(
            project_id="p", frontend=make_site(), backend=make_backend(models=(model,))
        )

    assert list(bundle(forward).backend.models[0].fields) != list(
        bundle(backward).backend.models[0].fields
    )
    assert bundle(forward).state_hash() == bundle(backward).state_hash()


def test_any_bundle_change_changes_the_hash() -> None:
    baseline = make_bundle()
    renamed = baseline.model_copy(update={"project_id": "project-2"})
    assert renamed.state_hash() != baseline.state_hash()


@pytest.mark.parametrize("model", [SwarmSnapshot, ReplayTimeline, ProblemDetails])
@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_contracts_export_json_schema(model: Any, mode: str) -> None:
    assert model.model_json_schema(mode=mode)["type"] == "object"


# --- Shared primitives


def test_frozen_map_behaves_as_a_read_only_mapping() -> None:
    frozen = FrozenMap({"a": 1, "b": 2})
    assert frozen["a"] == 1
    assert list(frozen) == ["a", "b"]
    assert len(frozen) == 2
    assert frozen == {"a": 1, "b": 2}
    assert hash(frozen) == hash(FrozenMap({"b": 2, "a": 1}))
    assert repr(frozen) == "FrozenMap({'a': 1, 'b': 2})"


def test_freeze_and_thaw_are_inverse() -> None:
    document = {"a": [1, 2.5, "x", None, True, {"b": []}]}
    frozen = freeze_json(document)
    assert isinstance(frozen, FrozenMap)
    assert isinstance(frozen["a"], tuple)
    assert thaw_json(frozen) == document


@pytest.mark.parametrize(
    "value",
    ["\ud800", ["ok", "\ud800"], {"\ud800": 1}, {"nested": {"key": "\udfff"}}],
)
def test_freeze_rejects_text_that_could_not_be_serialized(value: object) -> None:
    with pytest.raises(ValueError, match="valid Unicode"):
        freeze_json(value)


def test_unserializable_values_fail_at_validation_rather_than_at_broadcast() -> None:
    with pytest.raises(ValidationError, match="valid Unicode"):
        PatchOperation(op=JsonPatchOp.ADD, path="/a", value="\ud800")
    with pytest.raises(ValidationError, match="valid Unicode"):
        AgentThoughtStream(
            agent_id="a",
            role=AgentRole.STYLER,
            thought="x",
            tool_name="t",
            tool_args={"a": "\ud800"},
            timestamp_ms=1,
        )


def test_factory_reports_invalid_text_as_a_validation_error() -> None:
    with pytest.raises(ValidationError):
        ComponentNode.create(
            id="hero",
            component_type=ComponentType.HERO,
            title="T",
            code="\ud800",
            box=make_box(),
        )


def test_models_survive_pickling_and_deep_copies() -> None:
    snapshot = make_snapshot()
    assert copy.deepcopy(snapshot) == snapshot
    assert pickle.loads(pickle.dumps(snapshot)) == snapshot  # nosec B301


def test_freeze_rejects_excessive_depth() -> None:
    value: object = 1
    for _ in range(MAX_JSON_DEPTH + 1):
        value = [value]
    with pytest.raises(ValueError, match="nested deeper"):
        freeze_json(value)


def test_canonical_json_is_byte_stable() -> None:
    assert canonical_json({"b": 1, "a": ["é", None]}) == '{"a":["é",null],"b":1}'
    with pytest.raises(ValueError, match="not JSON compliant"):
        canonical_json(float("nan"))
    assert sha256_hex("abc") == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_require_unique() -> None:
    require_unique(["a", "b"], "ids")
    with pytest.raises(ValueError, match="duplicate ids: a, b"):
        require_unique(["b", "a", "b", "a", "c"], "ids")


# --- RFC 9457


def test_problem_json_serialization_matches_the_specification() -> None:
    problem = ProblemDetails.model_validate(
        {
            "type": "https://example.com/probs/out-of-credit",
            "title": "You do not have enough credit.",
            "status": 403,
            "detail": "Your current balance is 30, but that costs 50.",
            "instance": "/account/12345/msgs/abc",
            "balance": 30,
            "accounts": ["/account/12345", "/account/67890"],
        }
    )
    expected = {
        "type": "https://example.com/probs/out-of-credit",
        "title": "You do not have enough credit.",
        "status": 403,
        "detail": "Your current balance is 30, but that costs 50.",
        "instance": "/account/12345/msgs/abc",
        "balance": 30,
        "accounts": ["/account/12345", "/account/67890"],
    }
    assert PROBLEM_JSON_MEDIA_TYPE == "application/problem+json"
    assert json.loads(problem.to_json()) == expected
    assert problem.to_json_dict() == expected
    assert ProblemDetails.model_validate_json(problem.to_json()) == problem


def test_problem_defaults_and_absent_members() -> None:
    assert ProblemDetails().to_json_dict() == {"type": "about:blank"}
    assert json.loads(ProblemDetails(status=404).to_json()) == {
        "type": "about:blank",
        "status": 404,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"status": 99},
        {"status": 600},
        {"status": "500"},
        {"type": "has space"},
        {"instance": "has space"},
        {"ab": 1},
        {"bad-name": 1},
        {"non_json": object()},
    ],
)
def test_invalid_problems_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProblemDetails.model_validate(payload)


@pytest.mark.parametrize(
    ("error", "status", "slug", "extensions"),
    [
        (
            SiteBuilderError("boom"),
            500,
            "site-builder-failure",
            {},
        ),
        (
            ElementLockConflictError(
                competing_agent_ids=("b", "a", "b"), resource_uri="urn:site-builder:element:x"
            ),
            409,
            "element-lock-conflict",
            {"competing_agent_ids": ["a", "b"], "resource_uri": "urn:site-builder:element:x"},
        ),
        (
            ASTSyntaxCompilationError(
                "bad token", line_number=4, syntax_token="@", code_snippet="a @ b"
            ),
            422,
            "ast-syntax-compilation-failure",
            {"line_number": 4, "syntax_token": "@", "code_snippet": "a @ b"},
        ),
        (
            EndpointBindingNotFoundError(connected_endpoint_id="gone", component_id="hero"),
            400,
            "endpoint-binding-not-found",
            {"connected_endpoint_id": "gone", "component_id": "hero"},
        ),
        (
            ReplayStateDivergenceError(frame_seq=3, expected_hash="a", computed_hash="b"),
            500,
            "replay-state-divergence",
            {"frame_seq": 3, "expected_hash": "a", "computed_hash": "b"},
        ),
        (
            LockAcquisitionTimeoutError(agent_id="a", target_id="x", timeout_ms=5000),
            408,
            "lock-acquisition-timeout",
            {"agent_id": "a", "target_id": "x", "timeout_ms": 5000},
        ),
    ],
)
def test_domain_errors_render_as_problem_details(
    error: SiteBuilderError, status: int, slug: str, extensions: dict[str, object]
) -> None:
    assert isinstance(error, SiteBuilderError)
    assert not isinstance(error, ValueError)
    assert str(error) == error.detail
    assert json.loads(error.to_problem().to_json()) == {
        "type": f"{PROBLEM_TYPE_BASE}{slug}",
        "title": error.title,
        "status": status,
        "detail": error.detail,
        **extensions,
    }


def test_domain_error_carries_instance() -> None:
    error = EndpointBindingNotFoundError(
        connected_endpoint_id="gone", component_id="hero", instance="/sessions/1"
    )
    assert error.to_problem().instance == "/sessions/1"
