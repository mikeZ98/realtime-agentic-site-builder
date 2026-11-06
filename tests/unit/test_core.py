import asyncio
import logging
import random
from collections.abc import Awaitable, Mapping
from typing import Any, cast

import pytest
from pydantic import RedisDsn, ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from realtime_agentic_site_builder.config import Settings, get_settings
from realtime_agentic_site_builder.core.clock import system_now_ms
from realtime_agentic_site_builder.core.hub import (
    CursorEvent,
    InMemoryEventBus,
    LockNotice,
    RedisEventBus,
    SessionListener,
    StateEvent,
    SwarmEvent,
    SwarmSessionHub,
    create_event_bus,
    session_channel,
)
from realtime_agentic_site_builder.core.lock_manager import (
    AsyncLockManager,
    LockEvent,
    LockEventKind,
)
from realtime_agentic_site_builder.core.patch_engine import (
    BACKEND_DOCUMENT_LOCK,
    FRONTEND_DOCUMENT_LOCK,
    PatchEngine,
    changed_elements,
    elements_touched,
    parse_pointer,
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
    compute_state_checksum,
)
from realtime_agentic_site_builder.domain.errors import (
    ASTSyntaxCompilationError,
    DeltaRejectedError,
    ElementLockConflictError,
    ReplayStateDivergenceError,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SiteBuilderError,
    element_resource_uri,
)
from realtime_agentic_site_builder.domain.swarm import (
    DEFAULT_LOCK_TIMEOUT_MS,
    MAX_LOCK_TIMEOUT_MS,
    AgentCursorFrame,
    AgentRole,
    ElementLock,
    LockState,
    PatchOperation,
    ReplayTimeline,
    StreamDelta,
    SwarmSnapshot,
    find_wait_cycle,
)

HANDLER = "async def handler() -> None:\n    return None"
SESSION = "session-1"
LEASE_MS = 5000
HERO_CODE = "/frontend/components/0/code"
AGENT = "agent-a"
WAIT_S = 2.0


class FakeClock:
    def __init__(self, now: int = 1_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def make_component(
    component_id: str = "hero", *, y: int = 0, code: str = "<Hero />"
) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.HERO,
        title="Welcome",
        code=code,
        box=BoundingBox(x=0, y=y, width=100, height=50),
    )


def make_bundle() -> FullstackProjectBundle:
    model = DataModelSchema.model_validate(
        {
            "name": "Lead",
            "fields": {"contact": {"python_type": "EmailStr", "indexed": True}},
            "table_name": "lead",
        }
    )
    endpoint = ApiEndpointNode.create(
        path="/api/v1/leads",
        method=HttpMethod.POST,
        summary="Create a lead",
        handler_code=HANDLER,
        request_model_ref="Lead",
    )
    return FullstackProjectBundle(
        project_id="project-1",
        frontend=SiteBlueprint(
            site_id="site-1",
            title="Acme",
            theme="dark",
            components=(make_component(), make_component("features", y=100)),
        ),
        backend=BackendBlueprint(
            service_name="acme-api",
            database_dialect=DatabaseDialect.POSTGRESQL,
            models=(model,),
            endpoints=(endpoint,),
        ),
    )


def make_delta(
    *patches: Mapping[str, Any],
    seq: int = 0,
    domain: str = "frontend",
    session_id: str = SESSION,
    agent: str = AGENT,
    target: str = "hero",
) -> StreamDelta:
    return StreamDelta.model_validate(
        {
            "session_id": session_id,
            "agent_id": agent,
            "delta_seq": seq,
            "target_domain": domain,
            "target_id": target,
            "patches": list(patches),
            "emitted_at_ms": 0,
        }
    )


def set_code(
    code: str, *, seq: int = 0, session_id: str = SESSION, agent: str = AGENT
) -> StreamDelta:
    return make_delta(
        {"op": "replace", "path": HERO_CODE, "value": code},
        seq=seq,
        session_id=session_id,
        agent=agent,
    )


def make_frame(
    agent_id: str = "agent-a", *, x: int = 0, active_lock: ElementLock | None = None
) -> AgentCursorFrame:
    return AgentCursorFrame(
        agent_id=agent_id,
        name="Agent",
        role=AgentRole.STYLER,
        color="#112233",
        x=x,
        y=0,
        active_lock=active_lock,
    )


def nested(depth: int) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for _ in range(depth):
        value = {"a": value}
    return value


def make_settings(**overrides: Any) -> Settings:
    return Settings(**{"broadcast_fps": 240, "lock_lease_ms": LEASE_MS, **overrides})


async def next_event(listener: SessionListener) -> SwarmEvent:
    return await asyncio.wait_for(anext(listener), WAIT_S)


async def assert_silent(listener: SessionListener) -> None:
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(anext(listener), 0.02)


def replay(timeline: ReplayTimeline) -> FullstackProjectBundle:
    """Rebuild the final state from the base frame and the delta log, checking every frame."""
    engine = PatchEngine()
    state = timeline.snapshots[0].bundle_state
    for delta, frame in zip(timeline.deltas, timeline.snapshots[1:], strict=True):
        state, _ = engine.apply_patch(state, delta)
        frame.verify_replayed_state(state)
    return state


# --- config ---------------------------------------------------------------------------------


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.redis_url is None
    assert settings.session_ttl_seconds == 3600
    assert settings.lock_lease_ms == 5000
    assert settings.max_patch_size_bytes == 65536
    assert settings.broadcast_fps == 30
    assert settings.log_level == "INFO"


def test_settings_read_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RASB_LOCK_LEASE_MS", "1234")
    monkeypatch.setenv("RASB_REDIS_URL", "redis://cache:6380/2")
    monkeypatch.setenv("LOCK_LEASE_MS", "9")
    settings = Settings()
    assert settings.lock_lease_ms == 1234
    assert str(settings.redis_url) == "redis://cache:6380/2"


def test_settings_treat_empty_environment_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RASB_REDIS_URL", "")
    assert Settings().redis_url is None


def test_settings_are_frozen() -> None:
    with pytest.raises(ValidationError):
        cast("Any", Settings()).lock_lease_ms = 1


def test_settings_hide_the_redis_password_from_repr() -> None:
    settings = make_settings(redis_url=RedisDsn("redis://user:hunter2hunter2@cache:6379/0"))
    assert "hunter2" not in repr(settings)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("lock_lease_ms", 0),
        ("lock_lease_ms", MAX_LOCK_TIMEOUT_MS + 1),
        ("broadcast_fps", 0),
        ("broadcast_fps", 241),
        ("session_ttl_seconds", 0),
        ("max_patch_size_bytes", 0),
        ("log_level", "LOUD"),
        ("redis_url", "http://cache:6379"),
        ("undeclared", 1),
    ],
)
def test_settings_reject_invalid_values(field_name: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: value})


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_system_clock_reports_milliseconds() -> None:
    assert system_now_ms() > 1_600_000_000_000


# --- patch engine: operations ---------------------------------------------------------------


def test_add_inserts_into_arrays_and_objects() -> None:
    component = make_component("cta", y=200).model_dump(mode="json")
    first = make_component("banner", y=0).model_dump(mode="json")
    patched, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta(
            {"op": "add", "path": "/frontend/components/-", "value": component},
            {"op": "remove", "path": "/frontend/components/0"},
            {"op": "add", "path": "/frontend/components/0", "value": first},
            {"op": "add", "path": "/frontend/title", "value": "Renamed"},
        ),
    )
    assert [c.id for c in patched.frontend.components] == ["banner", "features", "cta"]
    assert patched.frontend.title == "Renamed"


def test_remove_deletes_array_elements() -> None:
    patched, _ = PatchEngine().apply_patch(
        make_bundle(), make_delta({"op": "remove", "path": "/frontend/components/1"})
    )
    assert [c.id for c in patched.frontend.components] == ["hero"]


def test_replace_updates_value_and_preserves_position() -> None:
    patched, _ = PatchEngine().apply_patch(make_bundle(), set_code("<Hero v2 />"))
    assert patched.frontend.components[0].code == "<Hero v2 />"
    assert [c.id for c in patched.frontend.components] == ["hero", "features"]


def test_move_relocates_values_and_array_elements() -> None:
    second = DataModelSchema.model_validate(
        {
            "name": "Order",
            "fields": {"total": {"python_type": "int"}},
            "table_name": "order_rows",
        }
    ).model_dump(mode="json")
    patched, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta(
            {"op": "add", "path": "/backend/models/-", "value": second},
            {"op": "move", "from": "/backend/models/0", "path": "/backend/models/1"},
            domain="backend",
        ),
    )
    assert [model.name for model in patched.backend.models] == ["Order", "Lead"]

    renamed, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta(
            {"op": "add", "path": "/frontend/draft", "value": "Moved"},
            {"op": "move", "from": "/frontend/draft", "path": "/frontend/title"},
        ),
    )
    assert renamed.frontend.title == "Moved"


def test_copy_duplicates_without_aliasing() -> None:
    patched, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta(
            {"op": "copy", "from": "/frontend/components/0", "path": "/frontend/draft"},
            {"op": "replace", "path": "/frontend/draft/code", "value": "<Changed />"},
            {"op": "test", "path": "/frontend/components/0/code", "value": "<Hero />"},
            {"op": "remove", "path": "/frontend/draft"},
            {"op": "copy", "from": "/frontend/title", "path": "/frontend/components/0/title"},
        ),
    )
    assert patched.frontend.components[0].title == "Acme"
    assert patched.frontend.components[0].code == "<Hero />"


def test_full_lifecycle_in_a_single_delta() -> None:
    bundle = make_bundle()
    patched, state_hash = PatchEngine().apply_patch(
        bundle,
        make_delta(
            {"op": "test", "path": "/frontend/title", "value": "Acme"},
            {"op": "add", "path": "/frontend/draft", "value": {"title": "Draft"}},
            {"op": "copy", "from": "/frontend/draft/title", "path": "/frontend/title"},
            {"op": "replace", "path": "/frontend/draft/title", "value": "<Hero v3 />"},
            {"op": "move", "from": "/frontend/draft/title", "path": HERO_CODE},
            {"op": "remove", "path": "/frontend/draft"},
            {"op": "test", "path": "/frontend/title", "value": "Draft"},
        ),
    )
    assert patched.frontend.title == "Draft"
    assert patched.frontend.components[0].code == "<Hero v3 />"
    assert state_hash == patched.state_hash()
    assert state_hash != bundle.state_hash()


def test_pointer_decoding_applies_rfc6901_order() -> None:
    assert parse_pointer("") == ()
    assert parse_pointer("/") == ("",)
    assert parse_pointer("/a~1b/m~0n/~01") == ("a/b", "m~n", "~1")


@pytest.mark.parametrize("key", ["a~1b", "m~0n", "~01", "with space", ""])
def test_escaped_members_address_the_same_location(key: str) -> None:
    patched, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta(
            {"op": "add", "path": f"/frontend/{key}", "value": "Escaped"},
            {"op": "move", "from": f"/frontend/{key}", "path": "/frontend/title"},
        ),
    )
    assert patched.frontend.title == "Escaped"


@pytest.mark.parametrize(
    "path, value",
    [
        ("/frontend/components/0/box", {"height": 50, "width": 100, "x": 0, "y": 0}),
        ("/frontend/components/0/box/x", 0.0),
        ("/frontend/components/0/connected_endpoint_id", None),
        ("/backend/models/0/fields/contact/indexed", True),
    ],
)
def test_test_op_uses_json_equality(path: str, value: object) -> None:
    domain = path.split("/")[1]
    bundle = make_bundle()
    patched, state_hash = PatchEngine().apply_patch(
        bundle, make_delta({"op": "test", "path": path, "value": value}, domain=domain)
    )
    assert patched == bundle
    assert state_hash == bundle.state_hash()


# --- patch engine: atomic rejection ---------------------------------------------------------


REJECTED_PATCHES: list[tuple[str, dict[str, Any]]] = [
    ("missing member on remove", {"op": "remove", "path": "/frontend/nope"}),
    ("missing member on replace", {"op": "replace", "path": "/frontend/nope", "value": 1}),
    ("missing member on test", {"op": "test", "path": "/frontend/nope", "value": 1}),
    ("missing move source", {"op": "move", "from": "/frontend/nope", "path": "/frontend/title"}),
    ("missing copy source", {"op": "copy", "from": "/frontend/nope", "path": "/frontend/title"}),
    ("missing parent", {"op": "add", "path": "/frontend/nope/child", "value": 1}),
    ("through a scalar", {"op": "test", "path": "/frontend/title/x", "value": 1}),
    ("add below a scalar", {"op": "add", "path": "/frontend/title/x", "value": 1}),
    ("remove below a scalar", {"op": "remove", "path": "/frontend/title/x"}),
    ("leading zero", {"op": "remove", "path": "/frontend/components/00"}),
    ("signed index", {"op": "remove", "path": "/frontend/components/+1"}),
    ("negative index", {"op": "remove", "path": "/frontend/components/-1"}),
    ("non-ASCII digit", {"op": "remove", "path": "/frontend/components/١"}),
    ("underscored index", {"op": "remove", "path": "/frontend/components/1_0"}),
    ("named index", {"op": "add", "path": "/frontend/components/first", "value": {}}),
    ("index past the end", {"op": "add", "path": "/frontend/components/3", "value": {}}),
    ("removal at length", {"op": "remove", "path": "/frontend/components/2"}),
    ("oversized index", {"op": "remove", "path": f"/frontend/components/{'9' * 5000}"}),
    ("end marker on remove", {"op": "remove", "path": "/frontend/components/-"}),
    ("end marker on replace", {"op": "replace", "path": "/frontend/components/-", "value": {}}),
    ("end marker on test", {"op": "test", "path": "/frontend/components/-", "value": {}}),
    ("end marker mid-path", {"op": "add", "path": "/frontend/components/-/title", "value": "x"}),
    ("failed test", {"op": "test", "path": "/frontend/title", "value": "Other"}),
    (
        "bool is not int",
        {"op": "test", "path": "/frontend/components/0/ast_version", "value": True},
    ),
    ("int is not bool", {"op": "test", "path": "/frontend/title", "value": 0}),
    ("null is not absent", {"op": "test", "path": "/frontend/nope", "value": None}),
    (
        "array length differs",
        {"op": "test", "path": "/frontend/components", "value": []},
    ),
    (
        "object members differ",
        {"op": "test", "path": "/frontend/components/0/box", "value": {"x": 0}},
    ),
    (
        "object values differ",
        {
            "op": "test",
            "path": "/frontend/components/0/box",
            "value": {"x": 1, "y": 0, "width": 100, "height": 50},
        },
    ),
    ("root replace", {"op": "replace", "path": "", "value": {}}),
    ("other domain", {"op": "replace", "path": "/backend/service_name", "value": "x"}),
    ("sibling prefix", {"op": "add", "path": "/frontendish", "value": "x"}),
    ("top-level field", {"op": "replace", "path": "/project_id", "value": "x"}),
    ("copy across domains", {"op": "copy", "from": "/backend/service_name", "path": "/frontend/x"}),
    ("undeclared member", {"op": "add", "path": "/frontend/extra", "value": "x"}),
    ("wrong value type", {"op": "replace", "path": "/frontend/theme", "value": "neon"}),
    ("frontend scalar", {"op": "replace", "path": "/frontend", "value": "x"}),
    ("components scalar", {"op": "replace", "path": "/frontend/components", "value": "x"}),
    ("component scalar", {"op": "replace", "path": "/frontend/components/0", "value": "x"}),
    ("component id type", {"op": "replace", "path": "/frontend/components/0/id", "value": []}),
    ("missing code", {"op": "remove", "path": HERO_CODE}),
    ("malformed box", {"op": "replace", "path": "/frontend/components/0/box", "value": {"x": 1}}),
    ("overlap", {"op": "replace", "path": "/frontend/components/1/box/y", "value": 10}),
    (
        "forged checksum",
        {"op": "replace", "path": "/frontend/components/0/state_checksum", "value": "0" * 64},
    ),
    (
        "unbound endpoint",
        {
            "op": "replace",
            "path": "/frontend/components/0/connected_endpoint_id",
            "value": "api_endpoint_get_none",
        },
    ),
]


@pytest.mark.parametrize("label, patch", REJECTED_PATCHES, ids=[row[0] for row in REJECTED_PATCHES])
def test_invalid_operation_is_rejected_without_mutation(label: str, patch: dict[str, Any]) -> None:
    bundle = make_bundle()
    before = bundle.state_hash()
    # The leading operation succeeds, so a rollback failure would show up as a changed title.
    delta = make_delta({"op": "replace", "path": "/frontend/title", "value": "Partial"}, patch)
    with pytest.raises(ASTSyntaxCompilationError, match="Code failed to compile"):
        PatchEngine().apply_patch(bundle, delta)
    assert bundle.state_hash() == before, label
    assert bundle.frontend.title == "Acme"


def test_failure_reports_the_position_of_the_failing_operation() -> None:
    delta = make_delta(
        {"op": "test", "path": "/frontend/title", "value": "Acme"},
        {"op": "remove", "path": "/frontend/nope"},
        {"op": "add", "path": "/frontend/title", "value": "Never"},
    )
    with pytest.raises(ASTSyntaxCompilationError) as caught:
        PatchEngine().apply_patch(make_bundle(), delta)
    problem = caught.value.to_problem().to_json_dict()
    assert problem["status"] == 422
    assert problem["line_number"] == 2
    assert problem["syntax_token"] == "/frontend/nope"
    assert '"op":"remove"' in str(problem["code_snippet"])


def test_schema_failure_names_the_location_and_chains_its_cause() -> None:
    delta = make_delta({"op": "replace", "path": "/frontend/theme", "value": "neon"})
    with pytest.raises(ASTSyntaxCompilationError, match=r"frontend\.theme") as caught:
        PatchEngine().apply_patch(make_bundle(), delta)
    assert isinstance(caught.value.__cause__, ValidationError)
    problem = caught.value.to_problem().to_json_dict()
    assert problem["line_number"] == 1
    assert "syntax_token" not in problem


def test_whole_bundle_failure_reports_the_root_location() -> None:
    bundle = make_bundle()
    contact_form = make_component("contact", y=200).model_dump(mode="json")
    contact_form["component_type"] = "CONTACT_FORM"
    with pytest.raises(ASTSyntaxCompilationError, match=r"schema at root: .*contact form"):
        PatchEngine().apply_patch(
            bundle,
            make_delta({"op": "add", "path": "/frontend/components/-", "value": contact_form}),
        )


def test_uncompilable_handler_keeps_its_own_source_position() -> None:
    delta = make_delta(
        {
            "op": "replace",
            "path": "/backend/endpoints/0/handler_code",
            "value": "async def handler(:\n    pass",
        },
        domain="backend",
    )
    with pytest.raises(ASTSyntaxCompilationError) as caught:
        PatchEngine().apply_patch(make_bundle(), delta)
    assert caught.value.extensions["code_snippet"] == "async def handler(:\n    pass"


def test_patch_size_limit_is_inclusive() -> None:
    delta = set_code("<Hero v2 />")
    exact = len(b'[{"op":"replace","path":"/frontend/components/0/code","value":"<Hero v2 />"}]')
    PatchEngine(max_patch_size_bytes=exact).apply_patch(make_bundle(), delta)
    with pytest.raises(ASTSyntaxCompilationError, match="byte limit"):
        PatchEngine(max_patch_size_bytes=exact - 1).apply_patch(make_bundle(), delta)


def test_patch_size_counts_bytes_not_characters() -> None:
    delta = make_delta({"op": "test", "path": "/frontend/title", "value": "é" * 100})
    with pytest.raises(ASTSyntaxCompilationError, match="byte limit"):
        PatchEngine(max_patch_size_bytes=150).apply_patch(make_bundle(), delta)


def test_copy_amplification_is_bounded_by_the_patch_budget() -> None:
    """Without a budget, each of these copies doubles the document: 2**400 bytes by the end."""
    copies = [
        {"op": "copy", "from": "/frontend/x", "path": f"/frontend/x/{index}"}
        for index in range(400)
    ]
    delta = make_delta({"op": "add", "path": "/frontend/x", "value": {"seed": "a" * 512}}, *copies)
    with pytest.raises(ASTSyntaxCompilationError, match="copied data exceeds"):
        PatchEngine().apply_patch(make_bundle(), delta)


def test_document_too_deep_to_traverse_is_rejected() -> None:
    operations: list[dict[str, Any]] = [
        {"op": "add", "path": "/frontend/x", "value": nested(60)},
    ]
    for level in range(1, 30):
        pointer = "/frontend/x" + "/a" * (60 * level)
        operations.append({"op": "add", "path": pointer, "value": nested(60)})
    operations.append({"op": "copy", "from": "/frontend/x", "path": "/frontend/y"})
    engine = PatchEngine(max_patch_size_bytes=10_000_000)
    with pytest.raises(ASTSyntaxCompilationError, match="recursion"):
        engine.apply_patch(make_bundle(), make_delta(*operations))


# --- patch engine: versions and checksums ---------------------------------------------------


def test_change_bumps_version_and_recomputes_checksum() -> None:
    patched, _ = PatchEngine().apply_patch(make_bundle(), set_code("<Hero v2 />"))
    hero, features = patched.frontend.components
    assert hero.ast_version == 2
    assert hero.state_checksum == compute_state_checksum("<Hero v2 />", hero.box)
    assert features.ast_version == 1


def test_version_is_strictly_monotonic_across_deltas() -> None:
    engine = PatchEngine()
    bundle = make_bundle()
    versions = []
    for revision in range(5):
        bundle, _ = engine.apply_patch(bundle, set_code(f"<Hero v{revision} />"))
        versions.append(bundle.frontend.components[0].ast_version)
    assert versions == [2, 3, 4, 5, 6]


def test_non_code_change_bumps_version_but_keeps_checksum() -> None:
    bundle = make_bundle()
    patched, _ = PatchEngine().apply_patch(
        bundle, make_delta({"op": "replace", "path": "/frontend/components/0/title", "value": "Hi"})
    )
    assert patched.frontend.components[0].ast_version == 2
    assert (
        patched.frontend.components[0].state_checksum
        == bundle.frontend.components[0].state_checksum
    )


def test_no_op_delta_changes_nothing() -> None:
    bundle = make_bundle()
    patched, state_hash = PatchEngine().apply_patch(
        bundle, make_delta({"op": "replace", "path": HERO_CODE, "value": "<Hero />"})
    )
    assert patched == bundle
    assert state_hash == bundle.state_hash()


def test_patch_cannot_move_a_version_backwards() -> None:
    engine = PatchEngine()
    bundle, _ = engine.apply_patch(make_bundle(), set_code("<Hero v2 />"))
    version_pointer = "/frontend/components/0/ast_version"
    patched, _ = engine.apply_patch(
        bundle, make_delta({"op": "replace", "path": version_pointer, "value": 1})
    )
    assert patched.frontend.components[0].ast_version == 2

    rewritten = bundle.frontend.components[0].model_dump(mode="json")
    rewritten.update(code="<Hero v3 />", ast_version=1)
    rewritten["state_checksum"] = compute_state_checksum(
        "<Hero v3 />", bundle.frontend.components[0].box
    )
    whole, _ = engine.apply_patch(
        bundle, make_delta({"op": "replace", "path": "/frontend/components/0", "value": rewritten})
    )
    assert whole.frontend.components[0].ast_version == 3


def test_new_component_starts_at_version_one_whatever_it_claims() -> None:
    component = make_component("cta", y=200).model_dump(mode="json")
    component["ast_version"] = 99
    patched, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta({"op": "add", "path": "/frontend/components/-", "value": component}),
    )
    assert patched.frontend.components[-1].ast_version == 1


def test_omitted_engine_fields_are_derived() -> None:
    component = make_component("cta", y=200).model_dump(mode="json")
    del component["state_checksum"], component["ast_version"]
    patched, _ = PatchEngine().apply_patch(
        make_bundle(),
        make_delta({"op": "add", "path": "/frontend/components/-", "value": component}),
    )
    assert patched.frontend.components[-1].ast_version == 1
    assert (
        patched.frontend.components[-1].state_checksum
        == make_component("cta", y=200).state_checksum
    )


def test_correct_supplied_checksum_is_accepted() -> None:
    bundle = make_bundle()
    checksum = compute_state_checksum("<Hero v2 />", bundle.frontend.components[0].box)
    patched, _ = PatchEngine().apply_patch(
        bundle,
        make_delta(
            {"op": "replace", "path": HERO_CODE, "value": "<Hero v2 />"},
            {"op": "replace", "path": "/frontend/components/0/state_checksum", "value": checksum},
        ),
    )
    assert patched.frontend.components[0].state_checksum == checksum


def test_stale_checksum_from_an_earlier_revision_is_drift() -> None:
    engine = PatchEngine()
    original = make_bundle()
    stale = original.frontend.components[0].state_checksum
    bundle, _ = engine.apply_patch(original, set_code("<Hero v2 />"))
    bundle, _ = engine.apply_patch(bundle, set_code("<Hero v3 />"))
    with pytest.raises(ASTSyntaxCompilationError, match="state_checksum drift"):
        engine.apply_patch(
            bundle,
            make_delta(
                {"op": "replace", "path": "/frontend/components/0/state_checksum", "value": stale}
            ),
        )


def test_replaying_a_delta_is_deterministic_across_key_order() -> None:
    forward = {"x": 5, "y": 0, "width": 100, "height": 50}
    backward = dict(reversed(list(forward.items())))
    pointer = "/frontend/components/0/box"
    first = PatchEngine().apply_patch(
        make_bundle(), make_delta({"op": "replace", "path": pointer, "value": forward})
    )
    second = PatchEngine().apply_patch(
        make_bundle(), make_delta({"op": "replace", "path": pointer, "value": backward})
    )
    assert first == second


# --- patch engine: snapshots ----------------------------------------------------------------


def test_snapshot_is_deterministic_for_equal_inputs() -> None:
    lock = ElementLock(
        target_id="hero",
        locked_by_agent_id="agent-a",
        state=LockState.ACQUIRED,
        priority=50,
        acquired_at_ms=1,
    )
    agents = (make_frame(active_lock=lock),)
    first = PatchEngine().generate_snapshot(
        make_bundle(), 3, "PATCHING", agents, (lock,), timestamp_ms=42
    )
    second = PatchEngine().generate_snapshot(
        make_bundle(), 3, "PATCHING", agents, (lock,), timestamp_ms=42
    )
    assert first == second
    assert first.bundle_hash == make_bundle().state_hash()
    assert first.frame_seq == 3
    assert first.step_name == "PATCHING"
    assert first.active_agents == agents
    assert first.active_locks == (lock,)


def test_snapshot_takes_its_time_from_the_clock_by_default() -> None:
    snapshot = PatchEngine(clock=FakeClock(77)).generate_snapshot(make_bundle(), 0, "START", (), ())
    assert snapshot.timestamp_ms == 77


def test_snapshot_detects_a_diverged_replay() -> None:
    engine = PatchEngine()
    snapshot = engine.generate_snapshot(make_bundle(), 0, "START", (), ())
    patched, _ = engine.apply_patch(make_bundle(), set_code("<Hero v2 />"))
    with pytest.raises(ReplayStateDivergenceError):
        snapshot.verify_replayed_state(patched)


def test_snapshot_is_immutable() -> None:
    snapshot = PatchEngine().generate_snapshot(make_bundle(), 0, "START", (), ())
    with pytest.raises(ValidationError):
        cast("Any", snapshot).frame_seq = 1


# --- lock manager ---------------------------------------------------------------------------


def make_manager(
    clock: FakeClock | None = None,
) -> tuple[AsyncLockManager, list[LockEvent], FakeClock]:
    events: list[LockEvent] = []
    active_clock = clock or FakeClock(0)
    manager = AsyncLockManager(lease_ms=LEASE_MS, clock=active_clock, on_event=events.append)
    return manager, events, active_clock


def kinds(events: list[LockEvent]) -> list[tuple[LockEventKind, str, str]]:
    return [(e.kind, e.lock.target_id, e.lock.locked_by_agent_id) for e in events]


async def test_uncontested_acquire_grants_the_lease() -> None:
    manager, events, _ = make_manager()
    lock = await manager.acquire("hero", "a", 50, 100)
    assert lock.state is LockState.ACQUIRED
    assert lock.acquired_at_ms == 100
    assert lock.timeout_ms == LEASE_MS
    assert manager.active_locks() == (lock,)
    assert events == []


async def test_default_lease_and_event_sink_need_no_configuration() -> None:
    manager = AsyncLockManager()
    now_ms = system_now_ms()
    lock = await manager.acquire("hero", "a", 50, now_ms)
    await manager.acquire("hero", "b", 50, now_ms)
    assert lock.timeout_ms == DEFAULT_LOCK_TIMEOUT_MS
    assert await manager.release("hero", "a") is not None


async def test_holder_reacquiring_renews_its_lease() -> None:
    manager, _, _ = make_manager()
    first = await manager.acquire("hero", "a", 50, 0)
    second = await manager.acquire("hero", "a", 50, 4000)
    assert second.acquired_at_ms == 4000
    assert second.lamport_ts == first.lamport_ts
    assert await manager.sweep_expired(LEASE_MS) == ()


async def test_contender_queues_without_preempting_a_higher_rank_holder_rule() -> None:
    """Priority alone never takes a live lock away from its holder."""
    manager, _, _ = make_manager()
    held = await manager.acquire("hero", "a", 1, 0)
    request = await manager.acquire("hero", "b", 100, 0)
    assert request.state is LockState.WAITING
    assert manager.active_locks() == (held, request)


async def test_polling_a_queued_request_keeps_its_place() -> None:
    manager, _, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    first = await manager.acquire("hero", "b", 50, 10)
    assert await manager.acquire("hero", "b", 99, 20) == first


async def test_release_promotes_the_highest_priority_waiter() -> None:
    manager, events, clock = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "low", 10, 0)
    await manager.acquire("hero", "high", 90, 0)
    clock.now = 100
    promoted = await manager.release("hero", "a")
    assert promoted is not None
    assert promoted.locked_by_agent_id == "high"
    assert promoted.state is LockState.ACQUIRED
    assert promoted.acquired_at_ms == 100
    assert kinds(events) == [(LockEventKind.PROMOTED, "hero", "high")]


async def test_equal_priority_promotes_the_older_request() -> None:
    manager, _, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "z-first", 50, 0)
    await manager.acquire("hero", "b-second", 50, 0)
    promoted = await manager.release("hero", "a")
    assert promoted is not None
    assert promoted.locked_by_agent_id == "z-first"


async def test_release_without_waiters_frees_the_element() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    assert await manager.release("hero", "a") is None
    assert manager.active_locks() == ()
    assert events == []


async def test_only_the_holder_can_free_a_lock() -> None:
    manager, _, _ = make_manager()
    held = await manager.acquire("hero", "a", 50, 0)
    assert await manager.release("hero", "stranger") is None
    assert await manager.release("unknown", "a") is None
    assert manager.active_locks() == (held,)


async def test_queued_agent_can_withdraw_its_request() -> None:
    manager, events, _ = make_manager()
    held = await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "b", 50, 0)
    other = await manager.acquire("hero", "c", 50, 0)
    assert await manager.release("hero", "b") is None
    assert manager.active_locks() == (held, other)
    assert await manager.release("hero", "c") is None
    assert manager.active_locks() == (held,)
    assert events == []


async def test_deadlock_preempts_the_lowest_priority_agent() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("t1", "low", 10, 0)
    await manager.acquire("t2", "high", 90, 0)
    await manager.acquire("t2", "low", 10, 0)
    lock = await manager.acquire("t1", "high", 90, 0)

    assert lock.state is LockState.ACQUIRED
    assert find_wait_cycle(manager.active_locks()) == ()
    assert {(x.target_id, x.locked_by_agent_id) for x in manager.active_locks()} == {
        ("t1", "high"),
        ("t2", "high"),
    }
    assert kinds(events) == [
        (LockEventKind.PREEMPTED, "t1", "low"),
        (LockEventKind.PREEMPTED, "t2", "low"),
        (LockEventKind.PROMOTED, "t1", "high"),
    ]
    assert all(e.lock.state is LockState.CONFLICT_RESOLVED for e in events[:2])
    problem = events[0].problem
    assert problem is not None
    assert problem.status == 409
    assert problem.to_json_dict()["competing_agent_ids"] == ["high", "low"]


async def test_request_that_makes_its_agent_the_victim_says_so() -> None:
    manager, _, _ = make_manager()
    await manager.acquire("t1", "low", 10, 0)
    await manager.acquire("t2", "high", 90, 0)
    await manager.acquire("t1", "high", 90, 0)
    lock = await manager.acquire("t2", "low", 10, 0)
    assert lock.state is LockState.CONFLICT_RESOLVED
    assert lock.target_id == "t2"
    assert [x.locked_by_agent_id for x in manager.active_locks()] == ["high", "high"]


async def test_deadlock_resolution_leaves_unrelated_locks_alone() -> None:
    manager, events, _ = make_manager()
    bystander = await manager.acquire("other", "c", 50, 0)
    queued = await manager.acquire("other", "d", 50, 0)
    await manager.acquire("t1", "low", 10, 0)
    await manager.acquire("t2", "high", 90, 0)
    await manager.acquire("t2", "low", 10, 0)
    await manager.acquire("t1", "high", 90, 0)
    assert bystander in manager.active_locks()
    assert queued in manager.active_locks()
    assert all(e.lock.target_id != "other" for e in events)


async def test_long_cycle_evicts_one_agent_and_frees_all_it_held() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("t1", "a", 80, 0)
    await manager.acquire("t2", "b", 20, 0)
    await manager.acquire("t3", "c", 60, 0)
    await manager.acquire("unrelated", "b", 20, 0)
    await manager.acquire("t2", "a", 80, 0)
    await manager.acquire("t3", "b", 20, 0)
    await manager.acquire("t1", "c", 60, 0)

    assert {e.lock.locked_by_agent_id for e in events if e.kind is LockEventKind.PREEMPTED} == {"b"}
    remaining = {x.target_id: x for x in manager.active_locks()}
    assert "unrelated" not in remaining
    assert remaining["t2"].locked_by_agent_id == "a"
    assert remaining["t2"].state is LockState.ACQUIRED


async def test_promotion_that_completes_a_cycle_is_resolved_on_release() -> None:
    """Granting t1 to ``high`` makes ``low``, queued on t1, wait for it while it waits for t2."""
    manager, events, _ = make_manager()
    await manager.acquire("t1", "x", 50, 0)
    await manager.acquire("t2", "low", 10, 0)
    await manager.acquire("t2", "high", 90, 0)
    await manager.acquire("t1", "high", 90, 0)
    await manager.acquire("t1", "low", 10, 0)
    assert find_wait_cycle(manager.active_locks()) == ()

    new_holder = await manager.release("t1", "x")

    assert new_holder is not None
    assert new_holder.locked_by_agent_id == "high"
    assert find_wait_cycle(manager.active_locks()) == ()
    assert {e.lock.locked_by_agent_id for e in events if e.kind is LockEventKind.PREEMPTED} == {
        "low"
    }


async def test_promotion_that_completes_a_cycle_is_resolved_on_sweep() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("t1", "x", 50, 0)
    await manager.acquire("t2", "low", 10, 4000)
    await manager.acquire("t2", "high", 90, 4000)
    await manager.acquire("t1", "high", 90, 4000)
    await manager.acquire("t1", "low", 10, 4000)

    reclaimed = await manager.sweep_expired(LEASE_MS)

    assert [x.locked_by_agent_id for x in reclaimed] == ["x"]
    assert find_wait_cycle(manager.active_locks()) == ()
    assert LockEventKind.PREEMPTED in {e.kind for e in events}


async def test_lapsed_lease_is_not_a_reason_to_evict_anyone() -> None:
    """The graph c -> b -> a -> c looks deadlocked, but a's lease on t1 has already lapsed."""
    manager, events, _ = make_manager()
    await manager.acquire("t1", "a", 50, 0)
    await manager.acquire("t2", "b", 50, 4000)
    await manager.acquire("t3", "c", 50, 4000)
    await manager.acquire("t1", "b", 50, 4000)
    await manager.acquire("t3", "a", 50, 4000)
    lock = await manager.acquire("t2", "c", 50, LEASE_MS)
    assert lock.state is LockState.WAITING
    assert LockEventKind.PREEMPTED not in {e.kind for e in events}
    assert (LockEventKind.TIMED_OUT, "t1", "a") in kinds(events)
    assert (LockEventKind.PROMOTED, "t1", "b") in kinds(events)


async def test_sweep_reclaims_lapsed_lease_and_promotes() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "b", 50, 4000)
    assert await manager.sweep_expired(LEASE_MS - 1) == ()

    reclaimed = await manager.sweep_expired(LEASE_MS)

    assert [x.state for x in reclaimed] == [LockState.TIMEOUT_RELEASED]
    assert reclaimed[0].locked_by_agent_id == "a"
    holder = manager.active_locks()[0]
    assert holder.locked_by_agent_id == "b"
    assert holder.state is LockState.ACQUIRED
    assert holder.acquired_at_ms == LEASE_MS
    assert kinds(events) == [
        (LockEventKind.TIMED_OUT, "hero", "a"),
        (LockEventKind.PROMOTED, "hero", "b"),
    ]


async def test_timeout_notification_carries_the_domain_error() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.sweep_expired(LEASE_MS)
    problem = events[0].problem
    assert problem is not None
    document = problem.to_json_dict()
    assert document["status"] == 408
    assert str(document["type"]).endswith("/lock-acquisition-timeout")
    assert document["agent_id"] == "a"
    assert document["target_id"] == "hero"
    assert document["timeout_ms"] == LEASE_MS


async def test_sweep_expires_waiters_that_waited_too_long() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "patient", 50, 3000)
    await manager.acquire("hero", "stale", 50, 0)
    await manager.acquire("hero", "a", 50, 3000)

    reclaimed = await manager.sweep_expired(LEASE_MS)

    assert [x.locked_by_agent_id for x in reclaimed] == ["stale"]
    assert kinds(events) == [(LockEventKind.TIMED_OUT, "hero", "stale")]
    assert {x.locked_by_agent_id for x in manager.active_locks()} == {"a", "patient"}


async def test_lapsed_waiters_are_never_promoted() -> None:
    manager, events, clock = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "b", 90, 0)
    await manager.acquire("hero", "a", 50, 4000)
    clock.now = LEASE_MS
    assert await manager.release("hero", "a") is None
    assert kinds(events) == [(LockEventKind.TIMED_OUT, "hero", "b")]
    assert manager.active_locks() == ()


async def test_lapsed_holder_cannot_release_after_reclaim() -> None:
    manager, _, clock = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "b", 50, 4000)
    await manager.sweep_expired(LEASE_MS)
    clock.now = LEASE_MS
    assert await manager.release("hero", "a") is None
    assert manager.active_locks()[0].locked_by_agent_id == "b"


async def test_newcomer_takes_an_unswept_lapsed_lease_when_nobody_queues() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    lock = await manager.acquire("hero", "b", 50, LEASE_MS)
    assert lock.state is LockState.ACQUIRED
    assert lock.acquired_at_ms == LEASE_MS
    assert kinds(events) == [
        (LockEventKind.TIMED_OUT, "hero", "a"),
        (LockEventKind.PROMOTED, "hero", "b"),
    ]


async def test_newcomer_cannot_jump_the_queue_on_an_unswept_lapsed_lease() -> None:
    manager, _, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "queued", 90, 4000)
    lock = await manager.acquire("hero", "newcomer", 10, LEASE_MS)
    assert lock.state is LockState.WAITING
    holder = next(x for x in manager.active_locks() if x.state is LockState.ACQUIRED)
    assert holder.locked_by_agent_id == "queued"


async def test_lapsed_holder_reacquiring_is_regranted_when_nobody_queues() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    lock = await manager.acquire("hero", "a", 50, LEASE_MS)
    assert lock.state is LockState.ACQUIRED
    assert lock.acquired_at_ms == LEASE_MS
    assert kinds(events) == [(LockEventKind.TIMED_OUT, "hero", "a")]


async def test_lapsed_holder_reacquiring_goes_behind_the_queue() -> None:
    manager, _, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    await manager.acquire("hero", "b", 50, 4000)
    lock = await manager.acquire("hero", "a", 50, LEASE_MS)
    assert lock.state is LockState.WAITING
    holder = next(x for x in manager.active_locks() if x.state is LockState.ACQUIRED)
    assert holder.locked_by_agent_id == "b"


async def test_lapsed_request_starts_a_new_wait_when_polled() -> None:
    manager, events, _ = make_manager()
    await manager.acquire("hero", "a", 50, 0)
    first = await manager.acquire("hero", "b", 50, 0)
    await manager.acquire("hero", "a", 50, 4000)
    second = await manager.acquire("hero", "b", 50, LEASE_MS)
    assert second.state is LockState.WAITING
    assert second.acquired_at_ms == LEASE_MS
    assert second.lamport_ts > first.lamport_ts
    assert kinds(events) == [(LockEventKind.TIMED_OUT, "hero", "b")]


async def test_invalid_request_changes_nothing() -> None:
    manager, _, _ = make_manager()
    held = await manager.acquire("hero", "a", 50, 0)
    with pytest.raises(ValidationError):
        await manager.acquire("hero", "b", 0, 0)
    with pytest.raises(ValidationError):
        await manager.acquire("bad id!", "b", 50, 0)
    assert manager.active_locks() == (held,)


async def test_concurrent_acquires_grant_exactly_one_holder() -> None:
    manager, _, _ = make_manager()
    locks = await asyncio.gather(
        *(manager.acquire("hero", f"agent-{index}", 50, 0) for index in range(50))
    )
    assert sum(lock.state is LockState.ACQUIRED for lock in locks) == 1
    assert sum(lock.state is LockState.WAITING for lock in locks) == 49


@pytest.mark.parametrize("seed", range(5))
async def test_random_workload_preserves_lock_invariants(seed: int) -> None:
    rng = random.Random(seed)  # nosec B311
    manager, events, clock = make_manager()
    agents = [f"agent-{index}" for index in range(6)]
    targets = [f"target-{index}" for index in range(5)]
    priorities = {agent: rng.randint(1, 100) for agent in agents}

    for _ in range(600):
        clock.now += rng.randint(0, 400)
        agent, target = rng.choice(agents), rng.choice(targets)
        action = rng.random()
        if action < 0.6:
            lock = await manager.acquire(target, agent, priorities[agent], clock.now)
            assert lock.target_id == target
            assert lock.locked_by_agent_id == agent
            assert lock.state in {
                LockState.ACQUIRED,
                LockState.WAITING,
                LockState.CONFLICT_RESOLVED,
            }
        elif action < 0.9:
            await manager.release(target, agent)
        else:
            await manager.sweep_expired(clock.now)

        active = manager.active_locks()
        # find_wait_cycle raises if two agents hold one element, so this checks both invariants.
        assert find_wait_cycle(active) == ()
        holders = {x.target_id for x in active if x.state is LockState.ACQUIRED}
        assert {x.target_id for x in active} == holders, "a queue exists with no holder"
    assert {e.kind for e in events} == set(LockEventKind)


# --- hub: session lifecycle -----------------------------------------------------------------


async def make_hub(
    *,
    clock: FakeClock | None = None,
    bus: InMemoryEventBus | RedisEventBus | None = None,
    listener_capacity: int = 256,
    **settings: Any,
) -> SwarmSessionHub:
    hub = SwarmSessionHub(
        make_settings(**settings),
        bus=bus or InMemoryEventBus(),
        clock=clock or FakeClock(),
        listener_capacity=listener_capacity,
    )
    await hub.create_session(SESSION, make_bundle())
    return hub


async def commit(hub: SwarmSessionHub, delta: StreamDelta, *also_locking: str) -> SwarmSnapshot:
    """Take the locks a well-behaved agent would hold, then apply the delta."""
    manager = hub.lock_manager(delta.session_id)
    for target_id in (delta.target_id, *also_locking):
        await manager.acquire(target_id, delta.agent_id, 50, hub._clock())
    return await hub.apply_delta(delta.session_id, delta)


async def test_create_session_records_the_base_frame() -> None:
    hub = SwarmSessionHub(make_settings(), bus=InMemoryEventBus(), clock=FakeClock(5))
    bundle = make_bundle()
    snapshot = await hub.create_session(SESSION, bundle)
    assert snapshot.frame_seq == 0
    assert snapshot.step_name == "SESSION_CREATED"
    assert snapshot.timestamp_ms == 5
    assert snapshot.bundle_hash == bundle.state_hash()
    assert hub.timeline(SESSION).snapshots == (snapshot,)
    await hub.shutdown()


async def test_duplicate_session_is_rejected_and_the_original_survives() -> None:
    hub = await make_hub()
    with pytest.raises(SessionAlreadyExistsError) as caught:
        await hub.create_session(SESSION, make_bundle())
    assert caught.value.status == 409
    await commit(hub, set_code("<Hero v2 />"))
    await hub.shutdown()


async def test_invalid_session_id_is_rejected_up_front() -> None:
    hub = SwarmSessionHub(make_settings(), bus=InMemoryEventBus())
    with pytest.raises(ValidationError):
        await hub.create_session("bad id!", make_bundle())
    await hub.shutdown()


async def test_close_returns_the_timeline_and_forgets_the_session() -> None:
    hub = await make_hub()
    await commit(hub, set_code("<Hero v2 />"))
    timeline = await hub.close_session(SESSION)
    assert [frame.frame_seq for frame in timeline.snapshots] == [0, 1]
    assert len(timeline.deltas) == 1

    frame = make_frame()
    with pytest.raises(SessionNotFoundError) as caught:
        await hub.close_session(SESSION)
    assert caught.value.status == 404
    with pytest.raises(SessionNotFoundError):
        await hub.apply_delta(SESSION, set_code("x", seq=1))
    with pytest.raises(SessionNotFoundError):
        await hub.broadcast_frame(SESSION, frame)
    with pytest.raises(SessionNotFoundError):
        hub.subscribe(SESSION)
    with pytest.raises(SessionNotFoundError):
        hub.timeline(SESSION)
    with pytest.raises(SessionNotFoundError):
        hub.lock_manager(SESSION)


async def test_session_id_can_be_reused_after_close() -> None:
    hub = await make_hub()
    await hub.close_session(SESSION)
    snapshot = await hub.create_session(SESSION, make_bundle())
    assert snapshot.frame_seq == 0
    await hub.shutdown()


async def test_shutdown_closes_sessions_and_ends_listeners() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    await hub.shutdown()
    with pytest.raises(StopAsyncIteration):
        await next_event(listener)
    with pytest.raises(SessionNotFoundError):
        hub.timeline(SESSION)


async def test_hub_builds_its_own_bus_from_default_settings() -> None:
    get_settings.cache_clear()
    hub = SwarmSessionHub()
    await hub.create_session(SESSION, make_bundle())
    await hub.broadcast_frame(SESSION, make_frame())
    await hub.shutdown()
    get_settings.cache_clear()


async def test_shutdown_leaves_an_injected_bus_open() -> None:
    bus = InMemoryEventBus()
    hub = await make_hub(bus=bus)
    other = await make_hub(bus=bus)
    await hub.shutdown()
    listener = other.subscribe(SESSION)
    await bus.publish(
        session_channel(SESSION),
        CursorEvent(origin="another-worker", session_id=SESSION, frame=make_frame()),
    )
    assert isinstance(await next_event(listener), CursorEvent)
    await other.shutdown()


# --- hub: cursor broadcast ------------------------------------------------------------------


async def test_broadcast_reaches_every_listener() -> None:
    hub = await make_hub()
    first, second = hub.subscribe(SESSION), hub.subscribe(SESSION)
    await hub.broadcast_frame(SESSION, make_frame(x=7))
    for listener in (first, second):
        event = await next_event(listener)
        assert isinstance(event, CursorEvent)
        assert event.frame.x == 7
        assert event.session_id == SESSION
    await hub.shutdown()


async def test_sessions_are_isolated() -> None:
    hub = await make_hub()
    await hub.create_session("session-2", make_bundle())
    other = hub.subscribe("session-2")
    await hub.broadcast_frame(SESSION, make_frame())
    await assert_silent(other)
    await hub.shutdown()


async def test_pump_caps_the_frame_rate_and_conflates_to_the_latest_frame() -> None:
    release = asyncio.Event()
    pauses: list[float] = []

    async def gated_sleep(seconds: float) -> None:
        pauses.append(seconds)
        await release.wait()
        release.clear()

    hub = SwarmSessionHub(
        make_settings(broadcast_fps=30), bus=InMemoryEventBus(), sleep=gated_sleep
    )
    await hub.create_session(SESSION, make_bundle())
    listener = hub.subscribe(SESSION)

    await hub.broadcast_frame(SESSION, make_frame(x=1))
    leading = await next_event(listener)
    assert isinstance(leading, CursorEvent)
    assert leading.frame.x == 1
    assert pauses == [pytest.approx(1 / 30)]

    for x in (2, 3, 4):
        await hub.broadcast_frame(SESSION, make_frame(x=x))
    await assert_silent(listener)

    release.set()
    trailing = await next_event(listener)
    assert isinstance(trailing, CursorEvent)
    assert trailing.frame.x == 4, "the agent's final position must survive conflation"
    await assert_silent(listener)
    await hub.shutdown()


async def test_slow_consumer_stalls_nobody_and_stays_bounded() -> None:
    hub = await make_hub()
    slow, fast = hub.subscribe(SESSION), hub.subscribe(SESSION)
    for x in range(2000):
        await hub.broadcast_frame(SESSION, make_frame(f"agent-{x % 4}", x=x))
    await asyncio.sleep(0)
    await hub.broadcast_frame(SESSION, make_frame("agent-0", x=3000))

    seen: dict[str, int] = {}
    while seen.get("agent-0") != 3000:
        event = await next_event(fast)
        assert isinstance(event, CursorEvent)
        seen[event.frame.agent_id] = event.frame.x

    drained = []
    while slow._cursors:
        drained.append(await next_event(slow))
    assert len(drained) <= 4
    await hub.shutdown()


def test_listener_conflates_cursors_and_counts_what_it_dropped() -> None:
    listener = SessionListener(4, lambda _listener: None)
    for x in (1, 2, 3):
        listener.offer(CursorEvent(origin="w", session_id=SESSION, frame=make_frame("a", x=x)))
    listener.offer(CursorEvent(origin="w", session_id=SESSION, frame=make_frame("b", x=9)))
    assert listener.dropped == 2


async def test_listener_delivers_state_ahead_of_cursors() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    cursor = CursorEvent(origin="w", session_id=SESSION, frame=make_frame())
    listener.offer(cursor)
    await commit(hub, set_code("<Hero v2 />"))
    assert isinstance(await next_event(listener), StateEvent)
    assert await next_event(listener) == cursor
    await hub.shutdown()


async def test_state_overflow_drops_oldest_and_keeps_the_latest_full_state() -> None:
    hub = await make_hub(listener_capacity=2)
    listener = hub.subscribe(SESSION)
    for seq in range(5):
        await commit(hub, set_code(f"<Hero v{seq} />", seq=seq))
    assert listener.dropped == 3
    frames = []
    for _ in range(2):
        event = await next_event(listener)
        assert isinstance(event, StateEvent)
        frames.append(event)
    assert [event.delta.delta_seq for event in frames] == [3, 4]
    assert frames[-1].snapshot.bundle_state.frontend.components[0].code == "<Hero v4 />"
    await hub.shutdown()


async def test_closed_listener_drains_then_ends_and_gets_nothing_new() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    await commit(hub, set_code("<Hero v2 />"))
    listener.close()
    await commit(hub, set_code("<Hero v3 />", seq=1))
    received = [event async for event in listener]
    assert len(received) == 1
    await hub.shutdown()


async def test_closing_a_session_wakes_a_blocked_listener() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    reader = asyncio.create_task(anext(listener, None))
    await asyncio.sleep(0)
    await hub.close_session(SESSION)
    assert await asyncio.wait_for(reader, WAIT_S) is None


# --- hub: deltas and timeline ---------------------------------------------------------------


async def test_apply_delta_returns_records_and_broadcasts_the_frame() -> None:
    clock = FakeClock(50)
    hub = await make_hub(clock=clock)
    listener = hub.subscribe(SESSION)
    clock.now = 60
    delta = set_code("<Hero v2 />", seq=7)

    snapshot = await commit(hub, delta)

    assert snapshot.frame_seq == 1
    assert snapshot.step_name == "DELTA_APPLIED"
    assert snapshot.timestamp_ms == 60
    assert snapshot.bundle_state.frontend.components[0].code == "<Hero v2 />"
    timeline = hub.timeline(SESSION)
    assert timeline.deltas == (delta,)
    assert timeline.snapshots[-1] == snapshot
    event = await next_event(listener)
    assert isinstance(event, StateEvent)
    assert event.delta == delta
    assert event.snapshot == snapshot
    await hub.shutdown()


async def test_failed_delta_leaves_the_session_untouched() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    before = hub.timeline(SESSION)
    with pytest.raises(ASTSyntaxCompilationError):
        await commit(
            hub,
            make_delta(
                {"op": "replace", "path": "/frontend/title", "value": "Partial"},
                {"op": "test", "path": "/frontend/title", "value": "Acme"},
            ),
            FRONTEND_DOCUMENT_LOCK,
        )
    assert hub.timeline(SESSION) == before
    await assert_silent(listener)
    # The failed delta's sequence number was never consumed.
    assert (await commit(hub, set_code("<Hero v2 />"))).frame_seq == 1
    await hub.shutdown()


async def test_delta_for_another_session_is_rejected() -> None:
    hub = await make_hub()
    with pytest.raises(DeltaRejectedError, match="addressed to session other") as caught:
        await hub.apply_delta(SESSION, set_code("x", session_id="other"))
    assert caught.value.status == 409
    assert caught.value.extensions["delta_seq"] == 0
    await hub.shutdown()


@pytest.mark.parametrize("repeat", [5, 4])
async def test_stale_or_duplicate_delta_is_rejected(repeat: int) -> None:
    hub = await make_hub()
    await commit(hub, set_code("<Hero v2 />", seq=5))
    before = hub.timeline(SESSION)
    with pytest.raises(DeltaRejectedError, match="does not follow 5"):
        await hub.apply_delta(SESSION, set_code("<Hero v3 />", seq=repeat))
    assert hub.timeline(SESSION) == before
    await hub.shutdown()


async def test_sequence_gaps_are_allowed() -> None:
    hub = await make_hub()
    await commit(hub, set_code("<Hero v2 />", seq=3))
    await commit(hub, set_code("<Hero v3 />", seq=9))
    assert [delta.delta_seq for delta in hub.timeline(SESSION).deltas] == [3, 9]
    await hub.shutdown()


class YieldingBus(InMemoryEventBus):
    """Suspends inside publish, which is where concurrent writers could interleave."""

    async def publish(self, channel: str, event: SwarmEvent) -> None:
        await asyncio.sleep(0)
        await super().publish(channel, event)


async def test_concurrent_deltas_are_serialized_without_lost_updates() -> None:
    hub = await make_hub(bus=YieldingBus())
    listener = hub.subscribe(SESSION)
    count = 25
    snapshots = await asyncio.gather(
        *(commit(hub, set_code(f"<Hero v{seq} />", seq=seq)) for seq in range(count))
    )
    assert [snapshot.frame_seq for snapshot in snapshots] == list(range(1, count + 1))
    assert snapshots[-1].bundle_state.frontend.components[0].ast_version == count + 1
    published = [await next_event(listener) for _ in range(count)]
    assert [e.delta.delta_seq for e in published if isinstance(e, StateEvent)] == list(range(count))
    await hub.shutdown()


async def test_workers_see_commits_in_order_even_when_a_publish_stalls() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    published: list[int] = []

    class StallsFirstPublish(InMemoryEventBus):
        async def publish(self, channel: str, event: SwarmEvent) -> None:
            assert isinstance(event, StateEvent)
            if event.delta.delta_seq == 0:
                entered.set()
                await gate.wait()
            published.append(event.delta.delta_seq)
            await super().publish(channel, event)

    hub = await make_hub(bus=StallsFirstPublish())
    first = asyncio.create_task(commit(hub, set_code("<Hero v1 />", seq=0)))
    await asyncio.wait_for(entered.wait(), WAIT_S)
    second = asyncio.create_task(commit(hub, set_code("<Hero v2 />", seq=1)))
    for _ in range(20):
        await asyncio.sleep(0)
    assert published == [], "a later commit overtook one still being published"

    gate.set()
    await asyncio.gather(first, second)
    assert published == [0, 1]
    await hub.shutdown()


async def test_timeline_replays_to_the_recorded_state() -> None:
    clock = FakeClock()
    hub = await make_hub(clock=clock)
    for seq in range(10):
        clock.now += 10
        final = await commit(hub, set_code(f"<Hero v{seq} />", seq=seq))
    timeline = await hub.close_session(SESSION)
    assert replay(timeline) == final.bundle_state
    assert timeline == ReplayTimeline.model_validate_json(timeline.model_dump_json())


async def test_timeline_survives_a_backwards_clock() -> None:
    clock = FakeClock(1000)
    hub = await make_hub(clock=clock)
    clock.now = 500
    snapshot = await commit(hub, set_code("<Hero v2 />"))
    assert snapshot.timestamp_ms == 1000
    hub.timeline(SESSION)
    await hub.shutdown()


async def test_close_waits_for_writers_and_rejects_those_queued_behind_it() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()

    class BlockingBus(InMemoryEventBus):
        async def publish(self, channel: str, event: SwarmEvent) -> None:
            entered.set()
            await gate.wait()
            await super().publish(channel, event)

    hub = await make_hub(bus=BlockingBus())
    in_flight = asyncio.create_task(commit(hub, set_code("<Hero v2 />")))
    await asyncio.wait_for(entered.wait(), WAIT_S)
    closing = asyncio.create_task(hub.close_session(SESSION))
    late = asyncio.create_task(commit(hub, set_code("<Hero v3 />", seq=1)))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not closing.done()
    assert not late.done()

    gate.set()

    assert (await in_flight).frame_seq == 1
    assert len((await closing).deltas) == 1
    with pytest.raises(SessionNotFoundError):
        await late


# --- hub: lock enforcement ------------------------------------------------------------------


async def assert_locked_out(
    hub: SwarmSessionHub, delta: StreamDelta, target_id: str, *competing: str
) -> None:
    """Assert a 409 naming the target, and that the rejected delta changed nothing at all."""
    before = hub.timeline(SESSION)
    with pytest.raises(ElementLockConflictError) as caught:
        await hub.apply_delta(SESSION, delta)
    problem = caught.value.to_problem().to_json_dict()
    assert problem["status"] == 409
    assert problem["resource_uri"] == element_resource_uri(target_id)
    assert problem["competing_agent_ids"] == sorted({delta.agent_id, *competing})
    assert hub.timeline(SESSION) == before


async def hold(hub: SwarmSessionHub, *target_ids: str, agent: str = AGENT) -> None:
    manager = hub.lock_manager(SESSION)
    for target_id in target_ids:
        await manager.acquire(target_id, agent, 50, hub._clock())


async def test_unlocked_write_is_rejected_and_changes_nothing() -> None:
    hub = await make_hub()
    listener = hub.subscribe(SESSION)
    delta = set_code("<Hero v2 />", seq=4)

    await assert_locked_out(hub, delta, "hero")
    await assert_silent(listener)

    # The rejected delta consumed nothing, so it succeeds unchanged once the lock is held.
    assert (await commit(hub, delta)).frame_seq == 1
    await hub.shutdown()


async def test_write_under_another_agents_lock_names_both_agents() -> None:
    hub = await make_hub()
    await hold(hub, "hero", agent="agent-b")
    await assert_locked_out(hub, set_code("<Hero v2 />"), "hero", "agent-b")
    await hub.shutdown()


async def test_queued_request_does_not_authorize_a_write() -> None:
    clock = FakeClock()
    hub = await make_hub(clock=clock)
    manager = hub.lock_manager(SESSION)
    await manager.acquire("hero", "agent-b", 50, clock.now)
    request = await manager.acquire("hero", AGENT, 90, clock.now)
    assert request.state is LockState.WAITING
    await assert_locked_out(hub, set_code("<Hero v2 />"), "hero", "agent-b")
    await hub.shutdown()


async def test_released_lock_does_not_authorize_a_write() -> None:
    hub = await make_hub()
    await commit(hub, set_code("<Hero v2 />"))
    await hub.lock_manager(SESSION).release("hero", AGENT)
    await assert_locked_out(hub, set_code("<Hero v3 />", seq=1), "hero")
    await hub.shutdown()


async def test_lease_stops_authorizing_at_expiry_even_before_a_sweep() -> None:
    clock = FakeClock(0)
    hub = await make_hub(clock=clock)
    await hub.lock_manager(SESSION).acquire("hero", AGENT, 50, 0)

    clock.now = LEASE_MS - 1
    await hub.apply_delta(SESSION, set_code("<Hero v2 />"))

    clock.now = LEASE_MS
    await assert_locked_out(hub, set_code("<Hero v3 />", seq=1), "hero")
    await hub.shutdown()


async def test_deadlock_victim_loses_its_write_access() -> None:
    clock = FakeClock()
    hub = await make_hub(clock=clock)
    manager = hub.lock_manager(SESSION)
    await manager.acquire("hero", "low", 10, clock.now)
    await manager.acquire("t2", "high", 90, clock.now)
    await manager.acquire("t2", "low", 10, clock.now)
    await manager.acquire("hero", "high", 90, clock.now)
    await assert_locked_out(hub, set_code("<Hero v2 />", agent="low"), "hero", "high")
    await hub.apply_delta(SESSION, set_code("<Hero v3 />", agent="high"))
    await hub.shutdown()


async def test_target_lock_is_required_even_if_the_patched_elements_are_held() -> None:
    hub = await make_hub()
    await hold(hub, "features")
    delta = make_delta({"op": "replace", "path": "/frontend/components/1/code", "value": "<F />"})
    await assert_locked_out(hub, delta, "hero")
    await hub.shutdown()


@pytest.mark.parametrize(
    "_label, patches, locked_out",
    [
        (
            "write to another component",
            [{"op": "replace", "path": "/frontend/components/1/code", "value": "<F />"}],
            "features",
        ),
        (
            "write to another component's nested field",
            [{"op": "replace", "path": "/frontend/components/1/box/width", "value": 7}],
            "features",
        ),
        (
            "remove another component",
            [{"op": "remove", "path": "/frontend/components/1"}],
            "features",
        ),
        (
            "index shifted by an earlier remove",
            [
                {"op": "remove", "path": "/frontend/components/0"},
                {"op": "replace", "path": HERO_CODE, "value": "<Smuggled />"},
            ],
            "features",
        ),
        (
            "move into another component",
            [
                {
                    "op": "move",
                    "from": "/frontend/components/0/title",
                    "path": "/frontend/components/1/title",
                }
            ],
            "features",
        ),
        (
            "copy out of another component",
            [
                {
                    "op": "copy",
                    "from": "/frontend/components/1/title",
                    "path": "/frontend/components/0/title",
                }
            ],
            "features",
        ),
        (
            "precondition on another component",
            [{"op": "test", "path": "/frontend/components/1/code", "value": "<Hero />"}],
            "features",
        ),
        (
            "rewriting the whole collection",
            [{"op": "replace", "path": "/frontend/components", "value": []}],
            "features",
        ),
        (
            "rewriting the whole domain",
            [{"op": "remove", "path": "/frontend"}],
            "features",
        ),
        (
            "site field",
            [{"op": "replace", "path": "/frontend/title", "value": "Renamed"}],
            FRONTEND_DOCUMENT_LOCK,
        ),
    ],
)
async def test_cross_element_patch_outside_the_held_lease_is_rejected(
    _label: str, patches: list[dict[str, Any]], locked_out: str
) -> None:
    hub = await make_hub()
    await hold(hub, "hero")
    await assert_locked_out(hub, make_delta(*patches), locked_out)
    await hub.shutdown()


async def test_cross_element_patch_passes_once_both_leases_are_held() -> None:
    hub = await make_hub()
    delta = make_delta({"op": "replace", "path": "/frontend/components/1/code", "value": "<F />"})
    snapshot = await commit(hub, delta, "features")
    assert snapshot.bundle_state.frontend.components[1].code == "<F />"
    await hub.shutdown()


async def test_rejection_comes_before_the_operation_is_evaluated() -> None:
    """An agent cannot probe an element it has no lock on through ``test`` outcomes."""
    hub = await make_hub()
    await hold(hub, "hero")
    wrong_guess = make_delta(
        {"op": "test", "path": "/frontend/components/1/code", "value": "<Wrong />"}
    )
    await assert_locked_out(hub, wrong_guess, "features")
    await hub.shutdown()


async def test_earlier_operations_of_a_rejected_delta_leave_no_trace() -> None:
    hub = await make_hub()
    await hold(hub, "hero")
    before = hub.timeline(SESSION).snapshots[-1].bundle_hash
    delta = make_delta(
        {"op": "replace", "path": HERO_CODE, "value": "<Allowed />"},
        {"op": "replace", "path": "/frontend/components/1/code", "value": "<Denied />"},
    )
    await assert_locked_out(hub, delta, "features")
    assert hub.timeline(SESSION).snapshots[-1].bundle_hash == before
    await hub.shutdown()


async def test_adding_or_renaming_needs_a_lock_on_the_new_id() -> None:
    hub = await make_hub()
    await hold(hub, "hero")
    component = make_component("cta", y=200).model_dump(mode="json")
    add = make_delta({"op": "add", "path": "/frontend/components/-", "value": component})
    await assert_locked_out(hub, add, "cta")

    rename = make_delta({"op": "replace", "path": "/frontend/components/0/id", "value": "renamed"})
    await assert_locked_out(hub, rename, "renamed")

    snapshot = await commit(hub, rename, "renamed")
    assert snapshot.bundle_state.frontend.components[0].id == "renamed"
    await hub.shutdown()


async def test_inserting_before_a_locked_element_does_not_need_its_lock() -> None:
    hub = await make_hub()
    await hold(hub, "features", agent="agent-b")
    original = hub.timeline(SESSION).snapshots[-1].bundle_state.frontend.components[1]
    banner = make_component("banner", y=50).model_dump(mode="json")
    insert = make_delta(
        {"op": "add", "path": "/frontend/components/1", "value": banner},
        target="banner",
    )

    snapshot = await commit(hub, insert)

    assert [node.id for node in snapshot.bundle_state.frontend.components] == [
        "hero",
        "banner",
        "features",
    ]
    assert snapshot.bundle_state.frontend.components[2] == original
    await hub.shutdown()


async def test_element_smuggled_through_a_scratch_member_is_caught_when_it_lands() -> None:
    hub = await make_hub()
    await hold(hub, "hero", FRONTEND_DOCUMENT_LOCK)
    component = make_component("cta", y=200).model_dump(mode="json")
    delta = make_delta(
        {"op": "add", "path": "/frontend/draft", "value": component},
        {"op": "move", "from": "/frontend/draft", "path": "/frontend/components/-"},
    )
    await assert_locked_out(hub, delta, "cta")
    await hub.shutdown()


async def test_reordering_needs_the_document_lock_though_no_pointer_names_it() -> None:
    hub = await make_hub()
    order = DataModelSchema.model_validate(
        {"name": "Order", "fields": {"total": {"python_type": "int"}}, "table_name": "orders"}
    ).model_dump(mode="json")
    grow = make_delta(
        {"op": "add", "path": "/backend/models/-", "value": order},
        domain="backend",
        target="Order",
    )
    await commit(hub, grow)
    await hold(hub, "Lead")
    swap = make_delta(
        {"op": "move", "from": "/backend/models/0", "path": "/backend/models/1"},
        seq=1,
        domain="backend",
        target="Lead",
    )
    await assert_locked_out(hub, swap, BACKEND_DOCUMENT_LOCK)
    await commit(hub, swap, BACKEND_DOCUMENT_LOCK)
    await hub.shutdown()


@pytest.mark.parametrize(
    "patch, missing",
    [
        (
            {"op": "replace", "path": "/backend/service_name", "value": "renamed"},
            BACKEND_DOCUMENT_LOCK,
        ),
        (
            {"op": "replace", "path": "/backend/endpoints/0/summary", "value": "Renamed"},
            "api_endpoint_post_leads",
        ),
    ],
)
async def test_backend_writes_need_their_own_locks(patch: dict[str, Any], missing: str) -> None:
    hub = await make_hub()
    await hold(hub, "Lead")
    delta = make_delta(patch, domain="backend", target="Lead")
    await assert_locked_out(hub, delta, missing)
    await commit(hub, delta, missing)
    await hub.shutdown()


async def test_model_lock_authorizes_a_model_edit() -> None:
    hub = await make_hub()
    delta = make_delta(
        {"op": "replace", "path": "/backend/models/0/fields/contact/indexed", "value": False},
        domain="backend",
        target="Lead",
    )
    snapshot = await commit(hub, delta)
    assert not snapshot.bundle_state.backend.models[0].fields["contact"].indexed
    await hub.shutdown()


async def test_read_only_delta_needs_locks_on_what_it_reads() -> None:
    hub = await make_hub()
    delta = make_delta({"op": "test", "path": "/frontend/title", "value": "Acme"})
    await assert_locked_out(hub, delta, "hero")
    await hold(hub, "hero")
    await assert_locked_out(hub, delta, FRONTEND_DOCUMENT_LOCK)
    assert (await commit(hub, delta, FRONTEND_DOCUMENT_LOCK)).frame_seq == 1
    await hub.shutdown()


async def test_timeline_records_who_wrote_each_delta() -> None:
    hub = await make_hub()
    await commit(hub, set_code("<Hero v2 />", agent="agent-b"))
    timeline = hub.timeline(SESSION)
    assert [delta.agent_id for delta in timeline.deltas] == ["agent-b"]
    restored = ReplayTimeline.model_validate_json(timeline.model_dump_json())
    assert restored.deltas[0].agent_id == "agent-b"
    await hub.shutdown()


async def test_live_holder_ignores_lapsed_and_absent_locks() -> None:
    manager = AsyncLockManager(lease_ms=LEASE_MS, clock=FakeClock(0))
    await manager.acquire("hero", AGENT, 50, 0)
    assert manager.live_holder("hero", LEASE_MS - 1) == AGENT
    assert manager.live_holder("hero", LEASE_MS) is None
    assert manager.live_holder("features", 0) is None


# --- patch engine: pointer resolution -------------------------------------------------------


def touched(*patches: Mapping[str, Any], domain: str = "frontend") -> frozenset[str]:
    document = make_bundle().model_dump(mode="json")
    return elements_touched(document, make_delta(*patches, domain=domain).patches[0])


COMPONENT = make_component("cta", y=200).model_dump(mode="json")


@pytest.mark.parametrize(
    "patch, domain, expected",
    [
        ({"op": "replace", "path": HERO_CODE, "value": "x"}, "frontend", {"hero"}),
        (
            {"op": "test", "path": "/frontend/components/1/box/x", "value": 0},
            "frontend",
            {"features"},
        ),
        ({"op": "remove", "path": "/frontend/components/1"}, "frontend", {"features"}),
        (
            {"op": "replace", "path": "/frontend/components/1", "value": COMPONENT},
            "frontend",
            {"features", "cta"},
        ),
        ({"op": "add", "path": "/frontend/components/0", "value": COMPONENT}, "frontend", {"cta"}),
        ({"op": "add", "path": "/frontend/components/-", "value": COMPONENT}, "frontend", {"cta"}),
        ({"op": "add", "path": "/frontend/components/0/title", "value": "x"}, "frontend", {"hero"}),
        (
            {"op": "replace", "path": "/frontend/components/0/id", "value": "new"},
            "frontend",
            {"hero", "new"},
        ),
        ({"op": "replace", "path": "/frontend/components/0/id", "value": 7}, "frontend", {"hero"}),
        (
            {"op": "replace", "path": "/frontend/components/0/box/x", "value": 7},
            "frontend",
            {"hero"},
        ),
        (
            {"op": "replace", "path": "/frontend/title", "value": "x"},
            "frontend",
            {FRONTEND_DOCUMENT_LOCK},
        ),
        (
            {"op": "replace", "path": "/frontend/components", "value": [COMPONENT]},
            "frontend",
            {FRONTEND_DOCUMENT_LOCK, "hero", "features", "cta"},
        ),
        (
            {"op": "replace", "path": "/frontend/components", "value": "x"},
            "frontend",
            {FRONTEND_DOCUMENT_LOCK, "hero", "features"},
        ),
        (
            {"op": "replace", "path": "/frontend", "value": {"components": [COMPONENT]}},
            "frontend",
            {FRONTEND_DOCUMENT_LOCK, "hero", "features", "cta"},
        ),
        (
            {"op": "replace", "path": "/frontend", "value": "x"},
            "frontend",
            {FRONTEND_DOCUMENT_LOCK, "hero", "features"},
        ),
        (
            {
                "op": "move",
                "from": "/frontend/components/0/title",
                "path": "/frontend/components/1/title",
            },
            "frontend",
            {"hero", "features"},
        ),
        (
            {"op": "copy", "from": "/frontend/components/0", "path": "/frontend/components/-"},
            "frontend",
            {"hero"},
        ),
        (
            {"op": "move", "from": "/frontend/components/0", "path": "/frontend/draft"},
            "frontend",
            {"hero", FRONTEND_DOCUMENT_LOCK},
        ),
        (
            {"op": "replace", "path": "/backend/models/0/table_name", "value": "x"},
            "backend",
            {"Lead"},
        ),
        (
            {"op": "replace", "path": "/backend/models/0/name", "value": "Client"},
            "backend",
            {"Lead", "Client"},
        ),
        ({"op": "remove", "path": "/backend/endpoints/0"}, "backend", {"api_endpoint_post_leads"}),
        (
            {"op": "replace", "path": "/backend/endpoints/0/id", "value": "new"},
            "backend",
            {"api_endpoint_post_leads", "new"},
        ),
        (
            {"op": "remove", "path": "/backend"},
            "backend",
            {BACKEND_DOCUMENT_LOCK, "Lead", "api_endpoint_post_leads"},
        ),
        ({"op": "remove", "path": "/backend/models"}, "backend", {BACKEND_DOCUMENT_LOCK, "Lead"}),
        (
            {"op": "replace", "path": "/backend/service_name", "value": "x"},
            "backend",
            {BACKEND_DOCUMENT_LOCK},
        ),
        # Pointers that resolve to nothing touch nothing, and then fail when applied.
        ({"op": "remove", "path": "/frontend/components/9"}, "frontend", set()),
        ({"op": "remove", "path": "/frontend/components/first"}, "frontend", set()),
        ({"op": "remove", "path": "/frontend/components/-"}, "frontend", set()),
    ],
)
def test_pointer_resolves_to_its_lock_targets(
    patch: dict[str, Any], domain: str, expected: set[str]
) -> None:
    assert touched(patch, domain=domain) == expected


@pytest.mark.parametrize(
    "document, path, expected",
    [
        ({"frontend": "x"}, "/frontend/components/0/code", set()),
        ({"frontend": "x"}, "/frontend/components", {FRONTEND_DOCUMENT_LOCK}),
        ({"frontend": "x"}, "/frontend", {FRONTEND_DOCUMENT_LOCK}),
        ({"frontend": {"components": "x"}}, "/frontend/components/0/code", set()),
        ({"frontend": {"components": ["x"]}}, "/frontend/components/0/code", set()),
        ({"frontend": {"components": [{"id": 7}]}}, "/frontend/components/0/code", set()),
        (
            {"frontend": {"components": ["x", {"id": "y"}]}},
            "/frontend/components",
            {FRONTEND_DOCUMENT_LOCK, "y"},
        ),
    ],
)
def test_resolution_tolerates_shapes_an_earlier_operation_broke(
    document: dict[str, Any], path: str, expected: set[str]
) -> None:
    operation = PatchOperation.model_validate({"op": "test", "path": path, "value": 1})
    assert elements_touched(document, operation) == expected


def test_moved_value_brings_its_element_ids_with_it() -> None:
    document = make_bundle().model_dump(mode="json")
    document["frontend"]["draft"] = COMPONENT
    operation = PatchOperation.model_validate(
        {"op": "move", "from": "/frontend/draft", "path": "/frontend/components/-"}
    )
    assert elements_touched(document, operation) == {FRONTEND_DOCUMENT_LOCK, "cta"}


def test_authorizer_sees_each_operation_against_the_working_document() -> None:
    seen: list[frozenset[str]] = []
    PatchEngine().apply_patch(
        make_bundle(),
        make_delta(
            {"op": "remove", "path": "/frontend/components/0"},
            {"op": "replace", "path": HERO_CODE, "value": "<Shifted />"},
        ),
        authorize=seen.append,
    )
    assert seen == [frozenset({"hero"}), frozenset({"features"})]


def test_authorizer_rejection_passes_through_with_the_bundle_untouched() -> None:
    bundle = make_bundle()
    before = bundle.state_hash()

    def deny(elements: frozenset[str]) -> None:
        if "features" in elements:
            raise ElementLockConflictError(competing_agent_ids=("a",), resource_uri="urn:x")

    with pytest.raises(ElementLockConflictError):
        PatchEngine().apply_patch(
            bundle,
            make_delta(
                {"op": "replace", "path": HERO_CODE, "value": "<Allowed />"},
                {"op": "remove", "path": "/frontend/components/1"},
            ),
            authorize=deny,
        )
    assert bundle.state_hash() == before


def test_missing_copy_source_is_a_patch_failure_not_a_lock_failure() -> None:
    with pytest.raises(ASTSyntaxCompilationError, match="does not exist"):
        PatchEngine().apply_patch(
            make_bundle(),
            make_delta({"op": "copy", "from": "/frontend/nope", "path": "/frontend/title"}),
            authorize=lambda _elements: None,
        )


def test_changed_elements_is_empty_without_changes() -> None:
    assert changed_elements(make_bundle(), make_bundle()) == frozenset()


def test_changed_elements_sees_document_fields() -> None:
    bundle = make_bundle()
    retitled = bundle.model_copy(
        update={"frontend": bundle.frontend.model_copy(update={"title": "Renamed"})}
    )
    assert changed_elements(bundle, retitled) == {FRONTEND_DOCUMENT_LOCK}
    renamed = bundle.model_copy(
        update={"backend": bundle.backend.model_copy(update={"service_name": "renamed"})}
    )
    assert changed_elements(bundle, renamed) == {BACKEND_DOCUMENT_LOCK}


def test_changed_elements_sees_reordered_components() -> None:
    bundle = make_bundle()
    swapped = bundle.model_copy(
        update={
            "frontend": SiteBlueprint(
                site_id="site-1",
                title="Acme",
                theme="dark",
                components=(make_component("features", y=0), make_component("hero", y=100)),
            )
        }
    )
    assert changed_elements(bundle, swapped) == {"hero", "features", FRONTEND_DOCUMENT_LOCK}


def test_changed_elements_sees_reordered_backend_collections() -> None:
    bundle = make_bundle()
    second_model = DataModelSchema.model_validate(
        {"name": "Order", "fields": {"total": {"python_type": "int"}}, "table_name": "orders"}
    )
    second_endpoint = ApiEndpointNode.create(
        path="/api/v1/orders", method=HttpMethod.GET, summary="List", handler_code=HANDLER
    )
    grown = bundle.model_copy(
        update={
            "backend": bundle.backend.model_copy(
                update={
                    "models": (*bundle.backend.models, second_model),
                    "endpoints": (*bundle.backend.endpoints, second_endpoint),
                }
            )
        }
    )
    assert changed_elements(bundle, grown) == {"Order", second_endpoint.id}

    models_swapped = grown.model_copy(
        update={
            "backend": grown.backend.model_copy(
                update={"models": tuple(reversed(grown.backend.models))}
            )
        }
    )
    assert changed_elements(grown, models_swapped) == {BACKEND_DOCUMENT_LOCK}

    endpoints_swapped = grown.model_copy(
        update={
            "backend": grown.backend.model_copy(
                update={"endpoints": tuple(reversed(grown.backend.endpoints))}
            )
        }
    )
    assert changed_elements(grown, endpoints_swapped) == {BACKEND_DOCUMENT_LOCK}


# --- hub: locks and maintenance -------------------------------------------------------------


async def test_snapshot_reports_active_agents_and_locks_in_stable_order() -> None:
    clock = FakeClock()
    hub = await make_hub(clock=clock)
    manager = hub.lock_manager(SESSION)
    lock = await manager.acquire("hero", "agent-b", 50, clock.now)
    await hub.broadcast_frame(SESSION, make_frame("agent-b", active_lock=lock))
    await hub.broadcast_frame(SESSION, make_frame("agent-a"))
    snapshot = await commit(hub, set_code("<Hero v2 />", agent="agent-b"))
    assert [agent.agent_id for agent in snapshot.active_agents] == ["agent-a", "agent-b"]
    assert snapshot.active_agents[1].active_lock == lock
    assert snapshot.active_locks == (lock,)
    await hub.shutdown()


async def test_stale_lock_claim_in_a_frame_cannot_block_a_delta() -> None:
    clock = FakeClock()
    hub = await make_hub(clock=clock)
    manager = hub.lock_manager(SESSION)
    lock = await manager.acquire("other", "agent-a", 50, clock.now)
    await hub.broadcast_frame(SESSION, make_frame("agent-a", active_lock=lock))
    await manager.release("other", "agent-a")

    snapshot = await commit(hub, set_code("<Hero v2 />"))

    assert snapshot.active_agents[0].active_lock is None
    assert [held.target_id for held in snapshot.active_locks] == ["hero"]
    await hub.shutdown()


async def test_lock_events_reach_listeners() -> None:
    clock = FakeClock()
    hub = await make_hub(clock=clock)
    listener = hub.subscribe(SESSION)
    manager = hub.lock_manager(SESSION)
    await manager.acquire("t1", "low", 10, clock.now)
    await manager.acquire("t2", "high", 90, clock.now)
    await manager.acquire("t2", "low", 10, clock.now)
    await manager.acquire("t1", "high", 90, clock.now)
    notice = await next_event(listener)
    assert isinstance(notice, LockNotice)
    assert notice.event.kind is LockEventKind.PREEMPTED
    assert notice.event.lock.locked_by_agent_id == "low"
    await hub.shutdown()


async def test_sweep_reclaims_leases_and_notifies() -> None:
    clock = FakeClock(0)
    hub = await make_hub(clock=clock)
    listener = hub.subscribe(SESSION)
    await hub.lock_manager(SESSION).acquire("hero", "agent-a", 50, 0)
    clock.now = LEASE_MS
    assert await hub.sweep() == ()
    assert hub.lock_manager(SESSION).active_locks() == ()
    notice = await next_event(listener)
    assert isinstance(notice, LockNotice)
    assert notice.event.kind is LockEventKind.TIMED_OUT
    await hub.shutdown()


async def test_sweep_closes_only_idle_sessions() -> None:
    clock = FakeClock(0)
    hub = await make_hub(clock=clock, session_ttl_seconds=10)
    await hub.create_session("session-2", make_bundle())
    clock.now = 9_000
    await commit(hub, set_code("<Hero v2 />"))
    await hub.broadcast_frame("session-2", make_frame())
    assert await hub.sweep(now_ms=10_000) == ()
    clock.now = 18_999
    assert await hub.sweep() == ()
    assert await hub.sweep(now_ms=19_000) == (SESSION, "session-2")
    with pytest.raises(SessionNotFoundError):
        hub.timeline(SESSION)
    await hub.shutdown()


# --- multi-worker replication ---------------------------------------------------------------


async def test_in_memory_bus_supports_several_handlers_per_channel() -> None:
    bus = InMemoryEventBus()
    first: list[SwarmEvent] = []
    second: list[SwarmEvent] = []
    await bus.subscribe("c", first.append)
    await bus.subscribe("c", second.append)
    event = CursorEvent(origin="w", session_id=SESSION, frame=make_frame())
    await bus.publish("c", event)
    await bus.publish("nobody", event)
    await bus.unsubscribe("c", first.append)
    await bus.publish("c", event)
    await bus.unsubscribe("c", second.append)
    await bus.publish("c", event)
    assert first == [event]
    assert second == [event, event]
    await bus.close()


async def test_commit_replicates_to_other_workers_exactly_once() -> None:
    bus = InMemoryEventBus()
    writer, replica = await make_hub(bus=bus), await make_hub(bus=bus)
    local, remote = writer.subscribe(SESSION), replica.subscribe(SESSION)

    snapshot = await commit(writer, set_code("<Hero v2 />"))

    for listener in (local, remote):
        event = await next_event(listener)
        assert isinstance(event, StateEvent)
        assert event.snapshot == snapshot
        await assert_silent(listener)
    timeline = replica.timeline(SESSION)
    assert timeline.snapshots[-1].bundle_hash == snapshot.bundle_hash
    assert replay(timeline).state_hash() == snapshot.bundle_hash
    await writer.shutdown()
    await replica.shutdown()


async def test_worker_does_not_receive_its_own_events_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = InMemoryEventBus()
    hub = await make_hub(bus=bus)
    listener = hub.subscribe(SESSION)

    with caplog.at_level(logging.WARNING):
        await commit(hub, set_code("<Hero v2 />"))
        assert isinstance(await next_event(listener), StateEvent)

        await hub.broadcast_frame(SESSION, make_frame(x=1))
        assert isinstance(await next_event(listener), CursorEvent)
        await assert_silent(listener)

        await hub.lock_manager(SESSION).acquire("hero", "agent-a", 50, 0)
        await hub.sweep(now_ms=LEASE_MS)
        assert isinstance(await next_event(listener), LockNotice)
        await assert_silent(listener)

    assert caplog.records == [], "an echoed commit would be logged as an out-of-order delta"
    await hub.shutdown()


async def test_replica_continues_the_log_when_it_takes_over_writing() -> None:
    bus = InMemoryEventBus()
    first, second = await make_hub(bus=bus), await make_hub(bus=bus)
    await commit(first, set_code("<Hero v2 />", seq=0))
    snapshot = await commit(second, set_code("<Hero v3 />", seq=1))
    assert snapshot.frame_seq == 2
    assert snapshot.bundle_state.frontend.components[0].ast_version == 3
    assert first.timeline(SESSION).snapshots[-1].bundle_hash == snapshot.bundle_hash
    await first.shutdown()
    await second.shutdown()


async def test_cursor_frames_and_lock_notices_replicate() -> None:
    bus = InMemoryEventBus()
    writer, replica = await make_hub(bus=bus), await make_hub(bus=bus)
    remote = replica.subscribe(SESSION)

    await writer.broadcast_frame(SESSION, make_frame(x=11))
    cursor = await next_event(remote)
    assert isinstance(cursor, CursorEvent)
    assert cursor.frame.x == 11

    await writer.lock_manager(SESSION).acquire("hero", "agent-a", 50, 0)
    await writer.sweep(now_ms=LEASE_MS)
    notice = await next_event(remote)
    assert isinstance(notice, LockNotice)
    assert notice.event.kind is LockEventKind.TIMED_OUT

    # A remote frame counts towards the replica's next snapshot.
    snapshot = await commit(replica, set_code("<Hero v2 />"))
    assert snapshot.active_agents[0].agent_id == "agent-a"
    await writer.shutdown()
    await replica.shutdown()


async def test_replica_ignores_history_rewrites() -> None:
    bus = InMemoryEventBus()
    hub = await make_hub(bus=bus)
    listener = hub.subscribe(SESSION)
    latest = await commit(hub, set_code("<Hero v2 />", seq=5))
    await next_event(listener)
    before = hub.timeline(SESSION)

    await bus.publish(
        session_channel(SESSION),
        StateEvent(
            origin="another-worker",
            session_id=SESSION,
            delta=set_code("<Hero v0 />", seq=5),
            snapshot=latest,
        ),
    )

    assert hub.timeline(SESSION) == before
    await assert_silent(listener)
    await hub.shutdown()


async def test_replica_ignores_events_for_another_session() -> None:
    bus = InMemoryEventBus()
    hub = await make_hub(bus=bus)
    listener = hub.subscribe(SESSION)
    await bus.publish(
        session_channel(SESSION),
        CursorEvent(origin="another-worker", session_id="other", frame=make_frame()),
    )
    await assert_silent(listener)
    await hub.shutdown()


async def test_replica_timeline_stays_ordered_under_clock_skew() -> None:
    bus = InMemoryEventBus()
    behind, ahead = FakeClock(1_000), FakeClock(9_000)
    writer, replica = await make_hub(bus=bus, clock=behind), await make_hub(bus=bus, clock=ahead)
    snapshot = await commit(writer, set_code("<Hero v2 />"))
    assert snapshot.timestamp_ms == 1_000
    assert replica.timeline(SESSION).snapshots[-1].timestamp_ms == 9_000
    await writer.shutdown()
    await replica.shutdown()


def test_state_event_rejects_a_delta_from_another_session() -> None:
    snapshot = PatchEngine().generate_snapshot(make_bundle(), 1, "STEP", (), ())
    with pytest.raises(ValidationError, match="belongs to another session"):
        StateEvent(
            origin="w",
            session_id=SESSION,
            delta=set_code("x", session_id="other"),
            snapshot=snapshot,
        )


# --- redis bus ------------------------------------------------------------------------------


class FakeBroker:
    def __init__(self) -> None:
        self.subscribers: dict[str, list[FakePubSub]] = {}
        self.fail_publish = False
        self.fail_subscribe = False
        self.fail_unsubscribe = False
        self.fail_reads = 0
        self.published: list[tuple[str, str]] = []

    def deliver(self, channel: str, data: bytes) -> None:
        for subscriber in self.subscribers.get(channel, ()):
            subscriber.inbox.put_nowait(
                {"type": "message", "channel": channel.encode(), "data": data}
            )


class FakePubSub:
    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker
        self.inbox: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self.closed = False
        self.subscribe_calls = 0
        self.polled_empty = asyncio.Event()

    async def subscribe(self, *channels: str) -> None:
        if self.broker.fail_subscribe:
            raise RedisConnectionError("down")
        self.subscribe_calls += 1
        for channel in channels:
            self.broker.subscribers.setdefault(channel, []).append(self)

    async def unsubscribe(self, *channels: str) -> None:
        if self.broker.fail_unsubscribe:
            raise RedisConnectionError("down")
        for channel in channels:
            self.broker.subscribers[channel].remove(self)

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0
    ) -> Mapping[str, Any] | None:
        assert ignore_subscribe_messages
        if self.broker.fail_reads:
            self.broker.fail_reads -= 1
            raise RedisConnectionError("down")
        try:
            return await asyncio.wait_for(self.inbox.get(), 0.005 if timeout else 0)
        except TimeoutError:
            self.polled_empty.set()
            return None

    async def aclose(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self, broker: FakeBroker) -> None:
        self.broker = broker
        self.closed = False
        self.pubsub_instance = FakePubSub(broker)

    def publish(self, channel: str, message: str) -> Awaitable[int]:
        return self._publish(channel, message)

    async def _publish(self, channel: str, message: str) -> int:
        if self.broker.fail_publish:
            raise RedisConnectionError("down")
        self.broker.published.append((channel, message))
        self.broker.deliver(channel, message.encode())
        return 1

    def pubsub(self) -> FakePubSub:
        return self.pubsub_instance

    async def aclose(self) -> None:
        self.closed = True


def make_redis_bus(broker: FakeBroker) -> tuple[RedisEventBus, FakeRedis]:
    client = FakeRedis(broker)
    return RedisEventBus(client, retry_delay_s=0), client


async def test_create_event_bus_selects_transport_from_settings() -> None:
    assert isinstance(create_event_bus(make_settings()), InMemoryEventBus)
    bus = create_event_bus(make_settings(redis_url=RedisDsn("redis://cache:6379/0")))
    assert isinstance(bus, RedisEventBus)
    # The client connects lazily, so neither building nor closing it touches the network.
    await bus.close()


async def test_redis_bus_replicates_commits_through_json_between_workers() -> None:
    broker = FakeBroker()
    bus_a, _ = make_redis_bus(broker)
    bus_b, _ = make_redis_bus(broker)
    writer, replica = await make_hub(bus=bus_a), await make_hub(bus=bus_b)
    remote = replica.subscribe(SESSION)
    delta = make_delta(
        {"op": "add", "path": "/frontend/draft", "value": "Moved"},
        {"op": "move", "from": "/frontend/draft", "path": "/frontend/title"},
        {"op": "copy", "from": "/frontend/title", "path": "/frontend/components/0/title"},
        {"op": "test", "path": "/frontend/components/0/connected_endpoint_id", "value": None},
    )

    snapshot = await commit(writer, delta, FRONTEND_DOCUMENT_LOCK)

    event = await next_event(remote)
    assert isinstance(event, StateEvent)
    assert event.delta == delta
    assert event.snapshot == snapshot
    assert replica.timeline(SESSION).deltas == (delta,)
    await writer.shutdown()
    await replica.shutdown()
    await bus_a.close()
    await bus_b.close()


async def test_redis_bus_carries_cursor_and_lock_events() -> None:
    broker = FakeBroker()
    bus_a, _ = make_redis_bus(broker)
    bus_b, _ = make_redis_bus(broker)
    writer, replica = await make_hub(bus=bus_a), await make_hub(bus=bus_b)
    remote = replica.subscribe(SESSION)

    await writer.broadcast_frame(SESSION, make_frame(x=3))
    cursor = await next_event(remote)
    assert isinstance(cursor, CursorEvent)
    assert cursor.frame.x == 3

    await writer.lock_manager(SESSION).acquire("hero", "agent-a", 50, 0)
    await writer.sweep(now_ms=LEASE_MS)
    notice = await next_event(remote)
    assert isinstance(notice, LockNotice)
    assert notice.event.problem is not None
    assert notice.event.problem.status == 408
    await writer.shutdown()
    await replica.shutdown()
    await bus_a.close()
    await bus_b.close()


async def test_redis_bus_discards_invalid_messages_and_keeps_reading(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = FakeBroker()
    bus, _ = make_redis_bus(broker)
    hub = await make_hub(bus=bus)
    listener = hub.subscribe(SESSION)
    channel = session_channel(SESSION)
    snapshot = PatchEngine().generate_snapshot(make_bundle(), 1, "STEP", (), ())
    forged = StateEvent(
        origin="another-worker", session_id=SESSION, delta=set_code("x", seq=1), snapshot=snapshot
    ).model_dump_json()
    tampered = forged.replace(snapshot.bundle_hash, "0" * 64)
    assert tampered != forged

    with caplog.at_level(logging.WARNING):
        for garbage in (b"not json", b"\xff\xfe", b'{"kind":"unknown"}', tampered.encode()):
            broker.deliver(channel, garbage)
        valid = CursorEvent(origin="another-worker", session_id=SESSION, frame=make_frame(x=5))
        broker.deliver(channel, valid.model_dump_json().encode())
        assert await next_event(listener) == valid

    assert sum("discarding invalid event" in record.message for record in caplog.records) == 4
    assert hub.timeline(SESSION).deltas == ()
    await hub.shutdown()
    await bus.close()


async def test_commit_survives_a_redis_publish_outage(caplog: pytest.LogCaptureFixture) -> None:
    broker = FakeBroker()
    bus, _ = make_redis_bus(broker)
    hub = await make_hub(bus=bus)
    listener = hub.subscribe(SESSION)
    broker.fail_publish = True

    with caplog.at_level(logging.WARNING):
        snapshot = await commit(hub, set_code("<Hero v2 />"))

    assert hub.timeline(SESSION).snapshots[-1] == snapshot
    assert isinstance(await next_event(listener), StateEvent)
    assert "dropping state event" in caplog.text
    await hub.shutdown()
    await bus.close()


async def test_redis_reader_recovers_from_read_failures(caplog: pytest.LogCaptureFixture) -> None:
    broker = FakeBroker()
    bus, _ = make_redis_bus(broker)
    hub = await make_hub(bus=bus)
    listener = hub.subscribe(SESSION)
    broker.fail_reads = 3
    valid = CursorEvent(origin="another-worker", session_id=SESSION, frame=make_frame())

    with caplog.at_level(logging.WARNING):
        broker.deliver(session_channel(SESSION), valid.model_dump_json().encode())
        assert await next_event(listener) == valid

    assert "event bus read failed" in caplog.text
    await hub.shutdown()
    await bus.close()


async def test_redis_reader_keeps_listening_after_an_idle_poll() -> None:
    broker = FakeBroker()
    bus, client = make_redis_bus(broker)
    received: asyncio.Queue[SwarmEvent] = asyncio.Queue()
    await bus.subscribe("c", received.put_nowait)

    await asyncio.wait_for(client.pubsub_instance.polled_empty.wait(), WAIT_S)
    event = CursorEvent(origin="w", session_id=SESSION, frame=make_frame())
    broker.deliver("c", event.model_dump_json().encode())

    assert await asyncio.wait_for(received.get(), WAIT_S) == event
    await bus.close()


async def test_failed_subscription_aborts_session_creation_cleanly() -> None:
    broker = FakeBroker()
    bus, _ = make_redis_bus(broker)
    hub = SwarmSessionHub(make_settings(), bus=bus)
    broker.fail_subscribe = True
    with pytest.raises(RedisConnectionError):
        await hub.create_session(SESSION, make_bundle())
    with pytest.raises(SessionNotFoundError):
        hub.timeline(SESSION)

    broker.fail_subscribe = False
    await hub.create_session(SESSION, make_bundle())
    await hub.shutdown()
    await bus.close()


async def test_redis_bus_shares_one_subscription_per_channel() -> None:
    broker = FakeBroker()
    bus, client = make_redis_bus(broker)
    first: list[SwarmEvent] = []
    second: list[SwarmEvent] = []
    await bus.subscribe("c", first.append)
    await bus.subscribe("c", second.append)
    assert client.pubsub_instance.subscribe_calls == 1

    await bus.unsubscribe("c", first.append)
    assert broker.subscribers["c"] == [client.pubsub_instance]
    await bus.unsubscribe("c", second.append)
    assert broker.subscribers["c"] == []
    await bus.close()


async def test_unsubscribe_outage_does_not_fail_session_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = FakeBroker()
    bus, _ = make_redis_bus(broker)
    hub = await make_hub(bus=bus)
    broker.fail_unsubscribe = True
    with caplog.at_level(logging.WARNING):
        await hub.close_session(SESSION)
    assert "could not unsubscribe" in caplog.text
    await bus.close()


async def test_redis_bus_close_releases_every_resource() -> None:
    broker = FakeBroker()
    bus, client = make_redis_bus(broker)
    await bus.subscribe("c", lambda _event: None)
    await bus.close()
    assert client.closed
    assert client.pubsub_instance.closed
    # A second close finds no reader left to stop.
    await bus.close()


# --- errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, status, slug",
    [
        (SessionNotFoundError(session_id="s"), 404, "session-not-found"),
        (SessionAlreadyExistsError(session_id="s"), 409, "session-already-exists"),
        (DeltaRejectedError("stale", session_id="s", delta_seq=1), 409, "delta-rejected"),
    ],
)
def test_session_errors_render_as_problems(error: SiteBuilderError, status: int, slug: str) -> None:
    document = error.to_problem().to_json_dict()
    assert document["status"] == status
    assert str(document["type"]).endswith(slug)
    assert document["session_id"] == "s"
