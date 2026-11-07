import ast
import builtins
import gzip
import hashlib
import io
import json
import re
import tarfile
import tomllib
import zipfile
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from realtime_agentic_site_builder.agents.provider import MockLLMProvider
from realtime_agentic_site_builder.agents.runtime import wireframe_code
from realtime_agentic_site_builder.agents.tsx import find_tsx_issue
from realtime_agentic_site_builder.config import Settings
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
    MAX_REPORTED_ISSUES,
    ProjectSynthesisError,
)
from realtime_agentic_site_builder.gateway.service import GatewayService, create_app
from realtime_agentic_site_builder.synthesis import engine
from realtime_agentic_site_builder.synthesis.engine import (
    SynthesizedProject,
    project_slug,
    synthesize_project,
)
from realtime_agentic_site_builder.synthesis.exporter import (
    CHUNK_BYTES,
    EPOCH_DATE_TIME,
    EPOCH_SECONDS,
    FILE_MODE,
    ArchiveFormat,
    ProjectArchive,
    build_tar_gz,
    build_zip,
    export_bundle,
)
from realtime_agentic_site_builder.synthesis.templates import render

ROUTE_FILE = "backend/app/routers/routes_leads.py"
SESSIONS = "/api/v1/sessions"
SECTION = (
    'export default function Section() {\n  return <section className="p-8">Hi</section>;\n}\n'
)


def lead_model() -> DataModelSchema:
    return DataModelSchema.model_validate(
        {
            "name": "Lead",
            "table_name": "leads",
            "fields": {
                "email": {"python_type": "EmailStr", "indexed": True},
                "id": {"python_type": "UUID", "default_factory": "uuid.uuid4"},
                "created_at": {"python_type": "datetime", "default_factory": "datetime.utcnow"},
                "note": {"python_type": "str", "nullable": True},
                "score": {"python_type": "float", "nullable": True},
                "active": {"python_type": "bool", "nullable": True},
                # Plausible names that would shadow the modules the generated types come from.
                "uuid": {"python_type": "str", "nullable": True},
                "datetime": {"python_type": "datetime", "nullable": True},
            },
        }
    )


def counter_model() -> DataModelSchema:
    return DataModelSchema.model_validate(
        {
            "name": "Counter",
            "table_name": "user",
            "fields": {"count": {"python_type": "int"}},
        }
    )


def endpoint(
    path: str = "/api/v1/leads",
    method: HttpMethod = HttpMethod.POST,
    code: str = "async def handler() -> dict[str, str]:\n    return {'status': 'ok'}",
    *,
    summary: str = "Create a lead",
    request: str | None = None,
    response: str | None = None,
) -> ApiEndpointNode:
    return ApiEndpointNode.create(
        path=path,
        method=method,
        summary=summary,
        handler_code=code,
        request_model_ref=request,
        response_model_ref=response,
    )


def component(
    component_id: str = "hero",
    code: str = SECTION,
    *,
    y: int = 0,
    x: int = 0,
    width: int = 100,
    height: int = 50,
) -> ComponentNode:
    return ComponentNode.create(
        id=component_id,
        component_type=ComponentType.FEATURES,
        title="Section",
        code=code,
        box=BoundingBox(x=x, y=y, width=width, height=height),
    )


def make_bundle(
    *,
    components: Iterable[ComponentNode] = (),
    models: Iterable[DataModelSchema] = (),
    endpoints: Iterable[ApiEndpointNode] = (),
    dialect: DatabaseDialect = DatabaseDialect.SQLITE,
    title: str = "Acme",
    theme: str = "dark",
) -> FullstackProjectBundle:
    return FullstackProjectBundle(
        project_id="Acme:Site_1",
        frontend=SiteBlueprint.model_validate(
            {"site_id": "s", "title": title, "theme": theme, "components": tuple(components)}
        ),
        backend=BackendBlueprint(
            service_name="acme-api",
            database_dialect=dialect,
            models=tuple(models),
            endpoints=tuple(endpoints),
        ),
    )


def rich_bundle(dialect: DatabaseDialect = DatabaseDialect.SQLITE) -> FullstackProjectBundle:
    create = endpoint(
        code=(
            "async def create(payload: Lead, session: DbSession):\n"
            "    row = models.Lead(**payload.model_dump())\n"
            "    session.add(row)\n"
            "    await session.commit()\n"
            "    return row\n"
        ),
        request="Lead",
        response="Lead",
    )
    listing = endpoint(
        method=HttpMethod.GET,
        code=(
            "async def handler(session: DbSession, limit: int = Query(10)):\n"
            "    return (await session.scalars(select(models.Lead).limit(limit))).all()\n"
        ),
        summary="List leads",
    )
    removal = endpoint(
        "/api/v1/leads/{lead_id}",
        HttpMethod.DELETE,
        code=(
            "import logging\n\n"
            "async def helper() -> None:\n"
            "    logging.getLogger(__name__).info('deleting')\n\n"
            "async def handler(lead_id: uuid.UUID, session: DbSession) -> Response:\n"
            "    await helper()\n"
            "    raise HTTPException(status.HTTP_404_NOT_FOUND, 'No such lead')\n"
        ),
        summary='Delete a "lead"',
    )
    ping = endpoint(
        "/api/v1/ping/{key}",
        HttpMethod.PUT,
        code="async def handler(key) -> dict[str, str]:\n    return {'key': key}",
        request="Counter",
    )
    return make_bundle(
        components=(
            component("hero", wireframe_code("hero")),
            component("left", '<p className="p-4">Left</p><p>Note</p>', y=50, width=50),
            component("right", SECTION, y=50, x=50, width=150),
            component("contact-form", SECTION, y=100),
        ),
        models=(lead_model(), counter_model()),
        endpoints=(create, listing, removal, ping),
        dialect=dialect,
    )


def files_of(bundle: FullstackProjectBundle) -> Mapping[str, str]:
    return synthesize_project(bundle).files


def issues_of(bundle: FullstackProjectBundle) -> list[str]:
    with pytest.raises(ProjectSynthesisError) as caught:
        synthesize_project(bundle)
    return cast("list[str]", caught.value.extensions["issues"])


# --- generated source is well formed --------------------------------------------------------


def python_files(project: Mapping[str, str]) -> dict[str, str]:
    return {path: text for path, text in project.items() if path.endswith(".py")}


@pytest.mark.parametrize(
    "bundle",
    [
        rich_bundle(DatabaseDialect.SQLITE),
        rich_bundle(DatabaseDialect.POSTGRESQL),
        make_bundle(),
    ],
    ids=["sqlite", "postgresql", "empty"],
)
class TestEveryProject:
    def test_all_python_parses_and_compiles(self, bundle: FullstackProjectBundle) -> None:
        sources = python_files(files_of(bundle))
        assert "backend/app/main.py" in sources
        for path, source in sources.items():
            ast.parse(source, filename=path)
            compile(source, path, "exec", dont_inherit=True)

    def test_all_tsx_is_balanced(self, bundle: FullstackProjectBundle) -> None:
        project = files_of(bundle)
        sources = [text for path, text in project.items() if path.endswith((".tsx", ".ts"))]
        assert len(sources) >= 4
        for source in sources:
            assert find_tsx_issue(source) is None

    def test_configuration_parses(self, bundle: FullstackProjectBundle) -> None:
        project = files_of(bundle)
        package = json.loads(project["frontend/package.json"])
        assert package["dependencies"]["react"].startswith("^19")
        assert "tailwindcss" in package["devDependencies"]
        assert json.loads(project["frontend/tsconfig.json"])["compilerOptions"]["strict"] is True
        pyproject = tomllib.loads(project["backend/pyproject.toml"])
        assert pyproject["tool"]["uv"]["package"] is False
        assert package["name"] == pyproject["project"]["name"] == "acme-site-1"

    def test_paths_are_relative_and_sorted(self, bundle: FullstackProjectBundle) -> None:
        paths = list(files_of(bundle))
        assert paths == sorted(paths)
        for path in paths:
            assert not path.startswith("/")
            assert ".." not in path.split("/")


def test_project_has_every_promised_file() -> None:
    assert {
        ".dockerignore",
        ".gitignore",
        "Dockerfile",
        "README.md",
        "backend/.env.example",
        "backend/alembic.ini",
        "backend/app/config.py",
        "backend/app/database.py",
        "backend/app/errors.py",
        "backend/app/main.py",
        "backend/app/models.py",
        "backend/app/schemas.py",
        "backend/app/handlers/post_leads.py",
        "backend/app/routers/routes_leads.py",
        "backend/app/routers/routes_ping.py",
        "backend/migrations/env.py",
        "backend/migrations/script.py.mako",
        "backend/migrations/versions/0001_initial.py",
        "backend/pyproject.toml",
        "frontend/index.html",
        "frontend/package.json",
        "frontend/postcss.config.js",
        "frontend/src/App.tsx",
        "frontend/src/api/client.ts",
        "frontend/src/api/types.ts",
        "frontend/src/components/ContactForm.tsx",
        "frontend/src/main.tsx",
        "frontend/tailwind.config.js",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
    } <= set(files_of(rich_bundle()))


def test_generated_files_hold_no_machine_paths_or_clock_reads() -> None:
    for text in files_of(rich_bundle()).values():
        assert not re.search(r"/(?:home|Users|Volumes)/", text)
        assert "1980" not in text


def test_dialect_selects_driver_and_database_configuration() -> None:
    sqlite = files_of(rich_bundle(DatabaseDialect.SQLITE))
    postgres = files_of(rich_bundle(DatabaseDialect.POSTGRESQL))
    assert "aiosqlite" in sqlite["backend/pyproject.toml"]
    assert "asyncpg" in postgres["backend/pyproject.toml"]
    assert "sqlite+aiosqlite:///./app.db" in sqlite["backend/app/config.py"]
    assert "sqlite" in sqlite["Dockerfile"]
    assert "database_url: str\n" in postgres["backend/app/config.py"], "no default to fall back on"
    assert "DATABASE_URL" not in postgres["Dockerfile"]
    assert "postgresql+asyncpg" in postgres["backend/.env.example"]
    assert "-e DATABASE_URL" in postgres["README.md"]


def test_text_cannot_break_out_of_the_generated_source() -> None:
    title = 'Quote " back\\slash @@SLUG@@ & co'
    summary = 'A "quoted" \\ summary'
    project = files_of(make_bundle(title=title, endpoints=[endpoint(summary=summary)]))
    registration = ast.parse(project[ROUTE_FILE]).body[-1]
    assert isinstance(registration, ast.Expr)
    assert isinstance(registration.value, ast.Call)
    keywords = {item.arg: item.value for item in registration.value.keywords}
    assert isinstance(keywords["summary"], ast.Constant)
    assert keywords["summary"].value == summary
    assert tomllib.loads(project["backend/pyproject.toml"])["project"]["description"] == title
    assert (
        "<title>Quote &quot; back\\slash @@SLUG@@ &amp; co</title>"
        in project["frontend/index.html"]
    )
    assert "@@SLUG@@" in project["README.md"], "values are never expanded a second time"
    assert find_tsx_issue(project["frontend/src/App.tsx"]) is None
    assert json.dumps(title) in project["frontend/src/App.tsx"]


def test_render_fails_loudly_on_a_placeholder_it_was_not_given() -> None:
    assert render("@@A@@-@@B@@", A="@@B@@", B="x") == "@@B@@-x"
    with pytest.raises(KeyError):
        render("@@MISSING@@")


@pytest.mark.parametrize(
    ("project_id", "slug"),
    [("Acme:Site_1", "acme-site-1"), ("s-abc", "s-abc"), ("A..b--C", "a-b-c")],
)
def test_project_slug(project_id: str, slug: str) -> None:
    assert project_slug(project_id) == slug


# --- backend synthesis ----------------------------------------------------------------------


def classes_in(source: str) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}


def attributes_of(definition: ast.ClassDef) -> list[str]:
    return [
        node.target.id
        for node in definition.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


def test_models_and_schemas_mirror_the_blueprint() -> None:
    project = files_of(rich_bundle())
    orm = classes_in(project["backend/app/models.py"])
    schemas = classes_in(project["backend/app/schemas.py"])
    declared = {"id", "active", "created_at", "datetime", "email", "note", "score", "uuid"}
    assert set(attributes_of(orm["Lead"])) == declared
    assert set(attributes_of(schemas["Lead"])) == declared
    assert attributes_of(orm["Counter"]) == ["id", "count"], "a table needs a primary key"
    assert attributes_of(schemas["Counter"]) == ["count"], "but the schema is as declared"
    assert attributes_of(orm["Lead"])[0] == "id"
    source = project["backend/app/models.py"]
    assert "primary_key=True, default=_uuid.uuid4" in source
    assert '_sa.Index("ix_leads_email", "email")' in source
    assert "note: _orm.Mapped[str | None]" in source
    schema_source = project["backend/app/schemas.py"]
    assert "email: _pydantic.EmailStr\n" in schema_source
    assert "note: str | None = None\n" in schema_source
    assert "id: _uuid.UUID = _pydantic.Field(default_factory=_uuid.uuid4)" in schema_source


def test_no_field_can_shadow_what_generated_types_refer_to() -> None:
    """A field called ``uuid`` with a default must not change what later annotations mean."""
    project = files_of(rich_bundle())
    for module in ("models", "schemas"):
        source = project[f"backend/app/{module}.py"]
        assert "import uuid as _uuid" in source
        assert "import datetime as _datetime" in source
        assert "uuid: " in source
        assert ": uuid." not in source
        assert ": datetime." not in source


def test_modules_import_only_what_they_use() -> None:
    plain = DataModelSchema.model_validate(
        {"name": "Plain", "table_name": "plain", "fields": {"count": {"python_type": "int"}}}
    )
    project = files_of(make_bundle(models=[plain]))
    for module in ("models", "schemas"):
        assert "_uuid" not in project[f"backend/app/{module}.py"]
        assert "_datetime" not in project[f"backend/app/{module}.py"]
    empty = files_of(make_bundle())
    assert "pydantic" not in empty["backend/app/schemas.py"]
    assert "_sa" not in empty["backend/app/models.py"]
    assert "import sqlalchemy" not in empty["backend/migrations/versions/0001_initial.py"]
    assert "alembic" not in empty["backend/migrations/versions/0001_initial.py"]


def function_body(source: str, name: str) -> list[str]:
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return [ast.unparse(statement) for statement in function.body]


def test_migration_creates_what_the_models_declare_and_undoes_it_in_reverse() -> None:
    source = files_of(rich_bundle())["backend/migrations/versions/0001_initial.py"]
    upgrade = "\n".join(function_body(source, "upgrade"))
    assert upgrade.index("'user'") < upgrade.index("'leads'"), "models are ordered by class name"
    assert "sa.Column('id', sa.Integer(), primary_key=True)" in upgrade
    assert "sa.Column('email', sa.String(320), nullable=False)" in upgrade
    assert "sa.Column('note', sa.Text(), nullable=True)" in upgrade
    assert "op.create_index('ix_leads_email', 'leads', ['email'])" in upgrade
    assert function_body(source, "downgrade") == [
        "op.drop_table('leads')",
        "op.drop_table('user')",
    ]
    empty = files_of(make_bundle())["backend/migrations/versions/0001_initial.py"]
    assert function_body(empty, "upgrade") == ["pass"]
    assert function_body(empty, "downgrade") == ["pass"]


def test_long_identifiers_get_short_stable_index_names_and_formatted_layout() -> None:
    long_table = "t" * 60
    long_field = "f" * 30
    model = DataModelSchema.model_validate(
        {
            "name": "Long",
            "table_name": long_table,
            "fields": {
                long_field: {
                    "python_type": "datetime",
                    "indexed": True,
                    "nullable": True,
                    "default_factory": "datetime.datetime.now",
                },
                "another": {"python_type": "str", "indexed": True},
                "id": {"python_type": "int", "indexed": True},
            },
        }
    )
    project = files_of(make_bundle(models=[model]))
    models, schemas = project["backend/app/models.py"], project["backend/app/schemas.py"]
    names = re.findall(r'_sa\.Index\(\s*"(ix_[^"]+)"', models)
    assert len(names) == 2, "the primary key is indexed already"
    assert all(len(name) <= 63 for name in names)
    assert set(names) <= set(
        re.findall(r'"(ix_[^"]+)"', project["backend/migrations/versions/0001_initial.py"])
    )
    assert "__table_args__ = (\n" in models
    assert re.findall(r"^\s+(\S*)Index\(", models, re.MULTILINE) == ["_sa.", "_sa."]
    for source in (models, schemas):
        ast.parse(source)
        assert max(len(line) for line in source.splitlines()) <= 100
    assert f"    {long_field}: _datetime.datetime | None = _pydantic.Field(\n" in schemas


def test_import_lines_wrap_like_the_formatter_wraps_them() -> None:
    endpoints = [endpoint(f"/api/v1/resource-with-long-name-{index}") for index in range(8)]
    main = files_of(make_bundle(endpoints=endpoints))["backend/app/main.py"]
    assert "from app.routers import (\n" in main
    ast.parse(main)
    assert max(len(line) for line in main.splitlines()) <= 100


def test_a_long_validator_signature_is_wrapped() -> None:
    name = "Model" + "x" * 60
    model = DataModelSchema.model_validate(
        {"name": name, "table_name": "wide", "fields": {"count": {"python_type": "int"}}}
    )
    source = files_of(
        make_bundle(
            models=[model],
            endpoints=[endpoint("/api/v1/resource-" + "y" * 30, request=name)],
        )
    )[f"backend/app/routers/routes_resource_{'y' * 30}.py"]
    ast.parse(source)
    assert f"\n    payload: schemas.{name}\n) -> None:\n" in source


def test_routes_are_grouped_by_resource_and_registered_in_full() -> None:
    project = files_of(rich_bundle())
    leads = project[ROUTE_FILE]
    assert "from app.handlers import delete_leads_lead_id, get_leads, post_leads" in leads
    assert "post_leads.create," in leads, "the only async function need not be named handler"
    assert 'methods=["POST"]' in leads
    assert 'operation_id="post_leads"' in leads
    assert "response_model=schemas.Lead" in leads
    assert "response_model=None" in leads, "return annotations are never treated as a schema"
    main = project["backend/app/main.py"]
    assert "from app.routers import routes_leads, routes_ping" in main
    assert main.index("include_router(routes_leads") < main.index('app.mount("/"')


def test_a_resource_with_no_name_still_routes() -> None:
    project = files_of(make_bundle(endpoints=[endpoint("/api/v1/-", HttpMethod.DELETE)]))
    source = project["backend/app/routers/routes_.py"]
    assert source.startswith('"""Routes of the root resource."""')
    assert "from fastapi import APIRouter\n" in source
    assert "from app import schemas" not in source
    assert "backend/app/handlers/delete_.py" in project


def test_entry_point_prefers_the_function_named_handler() -> None:
    code = "async def helper():\n    pass\n\nasync def handler():\n    await helper()\n"
    assert (
        "post_leads.handler," in files_of(make_bundle(endpoints=[endpoint(code=code)]))[ROUTE_FILE]
    )
    code = "async def first():\n    pass\n\nasync def second():\n    pass\n"
    assert "post_leads.first," in files_of(make_bundle(endpoints=[endpoint(code=code)]))[ROUTE_FILE]


def test_handler_module_holds_the_code_verbatim_after_a_preamble() -> None:
    code = "async def handler():\n    return models.Lead  # verbatim\n"
    source = files_of(make_bundle(models=[lead_model()], endpoints=[endpoint(code=code)]))[
        "backend/app/handlers/post_leads.py"
    ]
    assert source.endswith(code.rstrip("\n") + "\n")
    assert "from app.schemas import Lead\n" in source
    assert source.startswith('"""Handler for POST /api/v1/leads.')


def test_everything_the_agent_is_told_is_in_scope_resolves() -> None:
    code = (
        "async def handler(\n"
        "    request: Request,\n"
        "    response: Response,\n"
        "    session: DbSession,\n"
        "    limit: Annotated[int, Query()] = 1,\n"
        "    other: Any = Depends(get_session),\n"
        ") -> None:\n"
        "    print(datetime.datetime, uuid.UUID, HTTPException, status, select, models, Lead)\n"
    )
    assert files_of(make_bundle(models=[lead_model()], endpoints=[endpoint(code=code)]))


BODY_REQUIRING = "_require_post_leads_body"


@pytest.mark.parametrize(
    ("signature", "validated"),
    [
        ("", True),
        ("session: DbSession", True),
        ("limit: int = 1", True),
        ("limit: Annotated[int, Query()] = 1", True),
        ("request: Request", True),
        ("when: datetime.datetime | None = None", True),
        ("*, flag: bool = False", True),
        ("first: int, /, second: str = 'Lead'", True),
        ("session: Any = Depends(get_session)", True),
        ("payload: Lead", False),
        ("payload: 'Lead'", False),
        ("payload: Lead | None = None", False),
        ("*, payload: Lead", False),
        ("payload: dict[str, Any]", False),
        ("payload: Annotated[dict[str, Any], Body()]", False),
    ],
)
def test_declared_request_model_is_enforced_exactly_when_the_handler_does_not_read_a_body(
    signature: str, validated: bool
) -> None:
    """FastAPI would otherwise nest two body parameters under their names, breaking the client."""
    preamble = "from fastapi import Body\n\n" if "Body" in signature else ""
    code = f"{preamble}async def handler({signature}):\n    return None\n"
    source = files_of(
        make_bundle(models=[lead_model()], endpoints=[endpoint(code=code, request="Lead")])
    )[ROUTE_FILE]
    assert (BODY_REQUIRING in source) is validated
    if validated:
        assert f"async def {BODY_REQUIRING}(payload: schemas.Lead) -> None:" in source
        assert f"dependencies=[Depends({BODY_REQUIRING})]" in source


def test_path_parameter_is_not_mistaken_for_a_body() -> None:
    source = files_of(
        make_bundle(
            models=[lead_model()],
            endpoints=[
                endpoint(
                    "/api/v1/leads/{payload}",
                    code="async def handler(payload: dict[str, str]):\n    return payload\n",
                    request="Lead",
                )
            ],
        )
    )[ROUTE_FILE]
    assert "_require_post_leads_payload_body" in source


def test_a_model_named_like_a_framework_type_counts_as_the_body() -> None:
    request_model = DataModelSchema.model_validate(
        {"name": "Request", "table_name": "requests", "fields": {"url": {"python_type": "str"}}}
    )
    source = files_of(
        make_bundle(
            models=[request_model],
            endpoints=[
                endpoint(code="async def handler(payload: Request):\n    pass\n", request="Request")
            ],
        )
    )
    assert "_require_post_leads_body" not in source[ROUTE_FILE]
    assert "from app.schemas import Request\n" in source["backend/app/handlers/post_leads.py"]


def test_no_request_model_means_no_validator() -> None:
    source = files_of(make_bundle(endpoints=[endpoint()]))[ROUTE_FILE]
    assert "Depends" not in source
    assert "schemas" not in source


# --- what would keep a project from starting is refused -------------------------------------


@pytest.mark.parametrize(
    "name", ["metadata", "registry", "_private", "__tablename__", "model_config", "json", "str"]
)
def test_field_names_the_stack_reserves_are_refused(name: str) -> None:
    model = DataModelSchema.model_validate(
        {"name": "Bad", "table_name": "bad", "fields": {name: {"python_type": "str"}}}
    )
    assert issues_of(make_bundle(models=[model])) == [
        f"model Bad: field name {name} is reserved by the generated stack"
    ]


def test_identifiers_longer_than_postgres_allows_are_refused() -> None:
    model = DataModelSchema.model_validate(
        {
            "name": "Bad",
            "table_name": "t" * 64,
            "fields": {"f" * 64: {"python_type": "str"}},
        }
    )
    assert issues_of(make_bundle(models=[model])) == [
        f"model Bad: table name {'t' * 64} cannot be used",
        f"model Bad: field name {'f' * 64} is too long for a column",
    ]


def test_the_migration_history_table_cannot_be_taken() -> None:
    model = DataModelSchema.model_validate(
        {"name": "Bad", "table_name": "alembic_version", "fields": {"a": {"python_type": "str"}}}
    )
    assert "alembic_version" in issues_of(make_bundle(models=[model]))[0]


@pytest.mark.parametrize(
    ("factory", "python_type"),
    [("os.abort", "str"), ("uuid.uuid4", "int"), ("datetime.utcnow", "str")],
)
def test_default_factories_outside_the_allowlist_never_become_code(
    factory: str, python_type: str
) -> None:
    model = DataModelSchema.model_validate(
        {
            "name": "Bad",
            "table_name": "bad",
            "fields": {"value": {"python_type": python_type, "default_factory": factory}},
        }
    )
    assert issues_of(make_bundle(models=[model])) == [
        f"model Bad: field value cannot use default factory {factory} for type {python_type}"
    ]
    assert factory not in engine._schemas_module([model])
    assert factory not in engine._models_module([model])


def test_string_uuid_factory_is_converted() -> None:
    model = DataModelSchema.model_validate(
        {
            "name": "Fine",
            "table_name": "fine",
            "fields": {"key": {"python_type": "str", "default_factory": "uuid4"}},
        }
    )
    assert (
        "default=lambda: str(_uuid.uuid4())"
        in files_of(make_bundle(models=[model]))["backend/app/models.py"]
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("async def handler(*args):\n    pass\n", "takes variadic arguments"),
        ("async def handler(**kwargs):\n    pass\n", "takes variadic arguments"),
        ("async def handler(payload):\n    pass\n", "parameter payload needs a type annotation"),
        ("async def handler(x: Missing):\n    pass\n", "name Missing is not defined"),
        ("async def handler():\n    return db\n", "name db is not defined"),
        ("async def handler():\n    def f():\n        return db\n", "name db is not defined"),
        ("import requests\n\nasync def handler():\n    pass\n", "importing requests would fail"),
        ("async def handler():\n    import boto3.s3\n", "importing boto3.s3 would fail"),
        ("from ..models import X\n\nasync def handler():\n    pass\n", "importing ..models would"),
        ("from . import x\n\nasync def handler():\n    pass\n", "importing . would fail"),
        ("from app import nope\n\nasync def handler():\n    pass\n", "importing app.nope would"),
        ("import app.secrets\n\nasync def handler():\n    pass\n", "importing app.secrets would"),
        ("async def handler():\n    pass\nreturn 1\n", "does not compile: 'return' outside"),
        (
            "from __future__ import annotations\n\nasync def handler():\n    pass\n",
            "does not compile: from __future__ imports must occur",
        ),
    ],
)
def test_handlers_that_would_break_the_service_are_refused(code: str, expected: str) -> None:
    issues = issues_of(make_bundle(endpoints=[endpoint(code=code)]))
    assert len(issues) >= 1
    assert all(issue.startswith("endpoint POST /api/v1/leads: ") for issue in issues)
    assert any(expected in issue for issue in issues), issues


@pytest.mark.parametrize(
    "code",
    [
        "import json\nimport fastapi.testclient\n\nasync def handler():\n    return json.dumps\n",
        "from app.models import Lead\n\nasync def handler():\n    return Lead\n",
        "from app import models, schemas\n\nasync def handler():\n    return models, schemas\n",
        "from pydantic import BaseModel\n\nasync def handler():\n    return BaseModel\n",
        "VALUE = 1\n\nclass Local:\n    pass\n\nasync def handler():\n    return VALUE, Local\n",
        "async def handler():\n    return __file__, __name__, print\n",
    ],
)
def test_handlers_that_resolve_everything_are_accepted(code: str) -> None:
    assert files_of(make_bundle(models=[lead_model()], endpoints=[endpoint(code=code)]))


def test_path_parameters_do_not_need_an_annotation() -> None:
    bundle = make_bundle(
        endpoints=[endpoint("/api/v1/leads/{key}", code="async def handler(key):\n    pass\n")]
    )
    assert files_of(bundle)


def test_a_repeated_path_parameter_is_refused() -> None:
    bundle = make_bundle(endpoints=[endpoint("/api/v1/{a}/x/{a}")])
    assert issues_of(bundle) == ["endpoint POST /api/v1/{a}/x/{a}: a path parameter appears twice"]


def test_a_get_cannot_carry_a_request_model() -> None:
    bundle = make_bundle(
        models=[lead_model()], endpoints=[endpoint(method=HttpMethod.GET, request="Lead")]
    )
    assert issues_of(bundle) == [
        "endpoint GET /api/v1/leads: a GET request cannot carry a request model"
    ]


@pytest.mark.parametrize("failure", [RecursionError, MemoryError])
def test_source_too_deep_to_compile_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(engine, "compile", explode, raising=False)
    assert issues_of(make_bundle(endpoints=[endpoint()])) == [
        "endpoint POST /api/v1/leads: is too deeply nested to compile"
    ]


def test_syntax_warnings_in_handlers_stay_out_of_the_logs(
    recwarn: pytest.WarningsRecorder,
) -> None:
    code = "async def handler():\n    return 1 is 1, '\\d'\n"
    files_of(make_bundle(endpoints=[endpoint(code=code)]))
    assert [w for w in recwarn if issubclass(w.category, SyntaxWarning)] == []


def test_generating_never_runs_a_handler() -> None:
    code = (
        "import builtins\n\nbuiltins.SYNTHESIS_RAN_THIS = True\n\nasync def handler():\n    pass\n"
    )
    files_of(make_bundle(endpoints=[endpoint(code=code)]))
    assert not hasattr(builtins, "SYNTHESIS_RAN_THIS")


def test_every_problem_is_reported_at_once_and_the_report_is_bounded() -> None:
    models = [
        DataModelSchema.model_validate(
            {"name": f"Bad{i}", "table_name": f"bad{i}", "fields": {"str": {"python_type": "str"}}}
        )
        for i in range(MAX_REPORTED_ISSUES + 5)
    ]
    with pytest.raises(ProjectSynthesisError) as caught:
        synthesize_project(
            make_bundle(
                models=models,
                components=[component("hero", "not markup")],
                endpoints=[endpoint(code="async def handler(*a):\n    pass\n")],
            )
        )
    error = caught.value
    assert f"until {MAX_REPORTED_ISSUES + 7} problem(s)" in error.detail
    assert len(cast("list[str]", error.extensions["issues"])) == MAX_REPORTED_ISSUES
    problem = error.to_problem().to_json_dict()
    assert problem["status"] == 422
    assert str(problem["type"]).endswith("/project-synthesis-failure")


# --- frontend synthesis ---------------------------------------------------------------------


def test_app_lays_out_rows_in_reading_order() -> None:
    app = files_of(rich_bundle())["frontend/src/App.tsx"]
    assert app.index("<Hero />") < app.index("<Left />") < app.index("<Right />")
    assert app.index("<Right />") < app.index("<ContactForm />")
    assert 'import ContactForm from "./components/ContactForm";' in app
    assert '<div className="flex w-full">' in app
    assert 'style={{ flex: "50 1 0%" }}' in app
    assert 'style={{ flex: "150 1 0%" }}' in app
    assert app.count("flex w-full") == 1, "components alone in a row need no wrapper"
    assert 'data-theme="dark"' in app


def test_a_row_extends_to_its_tallest_member() -> None:
    app = files_of(
        make_bundle(
            components=[
                component("tall", y=0, width=50, height=100),
                component("short", y=0, x=50, width=50, height=10),
                component("beside", y=50, x=50, width=50, height=10),
                component("below", y=100),
            ]
        )
    )["frontend/src/App.tsx"]
    assert app.count("flex w-full") == 1
    assert app.index("<Beside />") < app.index("</div>\n      <Below />")


@pytest.mark.parametrize("theme", ["dark", "light", "cyberpunk", "minimal", "enterprise"])
def test_every_theme_has_classes(theme: str) -> None:
    app = files_of(make_bundle(theme=theme))["frontend/src/App.tsx"]
    assert f'data-theme="{theme}"' in app
    assert "min-h-screen" in app


def test_an_empty_site_still_renders_its_title() -> None:
    app = files_of(make_bundle(title="Empty"))["frontend/src/App.tsx"]
    assert '{"Empty"}' in app
    assert "import " not in app


def test_component_names_are_valid_and_distinct_on_any_file_system() -> None:
    components = [
        component("hero-section", y=0),
        component("herosection", y=50),  # Differs only in case from the name above.
        component("herosection2", y=100),  # Already the name the collision above resolves to.
        component("1-side", y=150),
        component("018f4d6e-7f6a", y=200),
        component("---", y=250),
        component("app", y=300),  # Must not collide with the root component.
    ]
    project = files_of(make_bundle(components=components))
    names = [
        path.removeprefix("frontend/src/components/").removesuffix(".tsx")
        for path in project
        if path.startswith("frontend/src/components/")
    ]
    assert len(names) == len(components)
    assert len({name.casefold() for name in names}) == len(names)
    assert "app" not in {name.casefold() for name in names}
    assert all(re.fullmatch(r"[A-Z][A-Za-z0-9]*", name) for name in names)
    assert "HeroSection" in names
    assert "Component1Side" in names
    assert "Component" in names
    for name in names:
        assert f"import {name} from" in project["frontend/src/App.tsx"]


def test_complete_modules_are_kept_verbatim() -> None:
    project = files_of(make_bundle(components=[component("hero", SECTION)]))
    assert project["frontend/src/components/Hero.tsx"] == SECTION


@pytest.mark.parametrize(
    "code",
    [
        "function A() { return <p />; }\nexport { A as default };",
        "export default function () { return <p />; }",
    ],
)
def test_every_form_of_default_export_counts(code: str) -> None:
    source = files_of(make_bundle(components=[component("hero", code)]))[
        "frontend/src/components/Hero.tsx"
    ]
    assert source == f"{code}\n"


def test_snippets_are_wrapped_into_a_named_default_export() -> None:
    source = files_of(make_bundle(components=[component("side-a", "<p>A</p><p>B</p>")]))[
        "frontend/src/components/SideA.tsx"
    ]
    assert "export default function SideA() {" in source
    assert "<>\n      <p>A</p><p>B</p>\n    </>" in source
    assert find_tsx_issue(source) is None


def test_wireframes_export_without_their_marker_turning_into_visible_text() -> None:
    source = files_of(make_bundle(components=[component("hero", wireframe_code("hero"))]))[
        "frontend/src/components/Hero.tsx"
    ]
    assert "wireframe" not in source
    assert '<section data-component="hero" />' in source


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("const a = 1;", "must default-export a component or be a JSX snippet"),
        ("export function Named() { return null; }", "must default-export"),
        ("<Hero />", "snippet uses undefined components: Hero"),
        ("<A.B><C /></A.B>", "snippet uses undefined components: A.B, C"),
        ("export default function A() { return <div>; }", "<div>"),
        ("<div>", "<div>"),
    ],
)
def test_components_that_would_not_build_are_refused(code: str, expected: str) -> None:
    issues = issues_of(make_bundle(components=[component("hero", code)]))
    assert any(issue.startswith("component hero: ") and expected in issue for issue in issues), (
        issues
    )


@pytest.mark.parametrize(
    "statement",
    [
        'import { useState } from "react";',
        'import { createRoot } from "react-dom/client";',
        'import "react";',
        'import { ApiError } from "../api/client";',
        'import Other from "./Other";',
        'import { type Lead } from "../api/types";',
        'export { useState } from "react";',
        'const lazy = () => import("./Other");',
        'import {\n  useState,\n  useEffect,\n} from "react";',
    ],
)
def test_imports_that_resolve_are_accepted(statement: str) -> None:
    code = f"{statement}\nexport default function A() {{ return <p />; }}"
    components = [component("hero", code), component("other", y=50)]
    assert files_of(make_bundle(components=components))


@pytest.mark.parametrize(
    ("statement", "specifier"),
    [
        ('import motion from "framer-motion";', "framer-motion"),
        ('import x from "@scope/pkg/sub";', "@scope/pkg/sub"),
        ('import "left-pad";', "left-pad"),
        ('import "./missing.css";', "./missing.css"),
        ('import A from "./Missing";', "./Missing"),
        ('import A from "../../../../etc/passwd";', "../../../../etc/passwd"),
        ('const lazy = () => import("hallucinated");', "hallucinated"),
        ('const x = require("fs");', "fs"),
    ],
)
def test_invented_dependencies_are_refused_rather_than_installed(
    statement: str, specifier: str
) -> None:
    code = f"{statement}\nexport default function A() {{ return <p />; }}"
    assert issues_of(make_bundle(components=[component("hero", code)])) == [
        f"component hero: import of {specifier} cannot be resolved"
    ]


def test_scanning_for_imports_stays_linear_on_hostile_input() -> None:
    code = "import\n" * 100_000 + "export default function A() { return <p />; }"
    assert files_of(make_bundle(components=[component("hero", code)]))


# --- API client synthesis -------------------------------------------------------------------


def test_types_mirror_the_models() -> None:
    types = files_of(rich_bundle())["frontend/src/api/types.ts"]
    assert "export interface Lead {" in types
    assert "  email: string;\n" in types
    assert "  id?: string;\n" in types, "the server supplies it"
    assert "  note?: string | null;\n" in types
    assert "  score?: number | null;\n" in types
    assert "  active?: boolean | null;\n" in types
    assert "export interface Counter {\n  count: number;\n}" in types
    assert files_of(make_bundle())["frontend/src/api/types.ts"] == "export {};\n"


def test_client_has_a_typed_function_and_hook_per_endpoint() -> None:
    client = files_of(rich_bundle())["frontend/src/api/client.ts"]
    assert 'import type * as Models from "./types";' in client
    assert (
        "export function postLeads(body: Models.Lead, options?: RequestOptions): "
        "Promise<Models.Lead> {" in client
    )
    assert 'return request<Models.Lead>("POST", `/api/v1/leads`, options, body);' in client
    assert "export function getLeads(options?: RequestOptions): Promise<unknown> {" in client
    assert "export function usePostLeads() {\n  return useApiCall(postLeads);\n}" in client
    assert "export function useGetLeads() {" in client
    assert (
        "export function deleteLeadsLeadId(params: { lead_id: string | number }, "
        "options?: RequestOptions)" in client
    )
    assert "`/api/v1/leads/${encodeURIComponent(String(params.lead_id))}`" in client
    assert (
        "export function putPingKey(params: { key: string | number }, body: Models.Counter, "
        "options?: RequestOptions)" in client
    )


def test_client_without_endpoints_or_models_has_just_the_wrapper() -> None:
    client = files_of(make_bundle())["frontend/src/api/client.ts"]
    assert "Models" not in client
    assert "export class ApiError extends Error" in client
    assert "application/json" in client
    assert "useApiCall" in client


def test_client_without_models_does_not_import_them() -> None:
    assert (
        "Models" not in files_of(make_bundle(endpoints=[endpoint()]))["frontend/src/api/client.ts"]
    )


def test_a_response_model_alone_imports_the_types() -> None:
    bundle = make_bundle(models=[lead_model()], endpoints=[endpoint(response="Lead")])
    client = files_of(bundle)["frontend/src/api/client.ts"]
    assert "Promise<Models.Lead>" in client
    assert "import type * as Models" in client


def test_a_model_named_like_a_global_type_shadows_nothing_in_the_client() -> None:
    model = DataModelSchema.model_validate(
        {"name": "Promise", "table_name": "p", "fields": {"id": {"python_type": "int"}}}
    )
    client = files_of(make_bundle(models=[model], endpoints=[endpoint(response="Promise")]))[
        "frontend/src/api/client.ts"
    ]
    assert "Promise<Models.Promise>" in client
    assert "interface Promise" not in client


def test_client_names_never_collide_or_land_on_a_reserved_word() -> None:
    bundle = make_bundle(
        endpoints=[
            endpoint("/api/v1/a-1"),
            endpoint("/api/v1/a1"),
            endpoint("/api/v1/-", HttpMethod.DELETE),
        ]
    )
    client = files_of(bundle)["frontend/src/api/client.ts"]
    functions = re.findall(r"export function (\w+)\(", client)
    assert "delete" not in functions
    assert "delete_" in functions
    assert "useDelete" in functions
    assert {"postA1", "postA12", "usePostA1", "usePostA12"} <= set(functions)
    assert len(set(functions)) == len(functions)


# --- deterministic packaging ----------------------------------------------------------------


@pytest.mark.parametrize("archive_format", list(ArchiveFormat))
def test_identical_bundles_produce_identical_archives(archive_format: ArchiveFormat) -> None:
    bundle = rich_bundle()
    first = export_bundle(bundle, archive_format)
    second = export_bundle(rich_bundle(), archive_format)
    assert first.content == second.content
    assert first.sha256 == second.sha256 == hashlib.sha256(first.content).hexdigest()


@pytest.mark.parametrize("archive_format", list(ArchiveFormat))
def test_a_serialization_roundtrip_changes_no_byte(archive_format: ArchiveFormat) -> None:
    bundle = rich_bundle()
    roundtripped = FullstackProjectBundle.model_validate_json(bundle.model_dump_json())
    assert (
        export_bundle(bundle, archive_format).sha256
        == export_bundle(roundtripped, archive_format).sha256
    )


def test_field_order_cannot_change_the_archive_of_bundles_that_compare_equal() -> None:
    forward = {"a": {"python_type": "str"}, "b": {"python_type": "int"}}
    backward = dict(reversed(forward.items()))
    for_model = {"name": "Thing", "table_name": "things"}
    one = make_bundle(models=[DataModelSchema.model_validate({**for_model, "fields": forward})])
    two = make_bundle(models=[DataModelSchema.model_validate({**for_model, "fields": backward})])
    assert one == two
    assert one.state_hash() == two.state_hash()
    assert (
        export_bundle(one, ArchiveFormat.ZIP).sha256 == export_bundle(two, ArchiveFormat.ZIP).sha256
    )


def test_model_order_cannot_change_the_schema_either() -> None:
    one = files_of(make_bundle(models=[lead_model(), counter_model()]))
    two = files_of(make_bundle(models=[counter_model(), lead_model()]))
    assert one["backend/app/models.py"] == two["backend/app/models.py"]
    assert (
        one["backend/migrations/versions/0001_initial.py"]
        == two["backend/migrations/versions/0001_initial.py"]
    )


def test_a_change_to_the_bundle_changes_the_archive() -> None:
    assert (
        export_bundle(make_bundle(title="One"), ArchiveFormat.ZIP).sha256
        != export_bundle(make_bundle(title="Two"), ArchiveFormat.ZIP).sha256
    )


def test_zip_entries_are_sorted_with_normalized_metadata() -> None:
    archive = export_bundle(rich_bundle(), ArchiveFormat.ZIP)
    assert archive.filename == "acme-site-1.zip"
    assert archive.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive.content)) as opened:
        assert opened.testzip() is None
        infos = opened.infolist()
        names = [info.filename for info in infos]
        assert names == sorted(names)
        for info in infos:
            assert info.filename.startswith("acme-site-1/")
            assert ".." not in info.filename.split("/")
            assert info.date_time == EPOCH_DATE_TIME
            assert info.external_attr == FILE_MODE << 16
            assert info.create_system == 3
            assert info.compress_type == zipfile.ZIP_DEFLATED
        main = opened.read("acme-site-1/backend/app/main.py").decode()
    assert main == files_of(rich_bundle())["backend/app/main.py"]


def test_tar_gz_entries_are_sorted_with_normalized_metadata() -> None:
    archive = export_bundle(rich_bundle(), ArchiveFormat.TAR_GZ)
    assert archive.filename == "acme-site-1.tar.gz"
    assert archive.media_type == "application/gzip"
    header = archive.content[:10]
    assert header[3] == 0, "no file name is recorded"
    assert header[4:8] == b"\x00\x00\x00\x00", "no timestamp is recorded"
    with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as opened:
        members = opened.getmembers()
        names = [member.name for member in members]
        assert names == sorted(names)
        for member in members:
            assert member.mtime == EPOCH_SECONDS
            assert member.mode == FILE_MODE
            assert (member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "")
        extracted = opened.extractfile("acme-site-1/backend/app/main.py")
        assert extracted is not None
        assert extracted.read().decode() == files_of(rich_bundle())["backend/app/main.py"]
    assert gzip.decompress(archive.content).startswith(b"acme-site-1/")


def test_epoch_constants_name_the_same_moment() -> None:
    assert EPOCH_DATE_TIME == (1980, 1, 1, 0, 0, 0)
    assert EPOCH_SECONDS == 315532800


def test_entry_order_does_not_depend_on_the_order_given() -> None:
    forward = {"a.txt": b"a", "b.txt": b"b"}
    backward = dict(reversed(forward.items()))
    assert build_zip(forward) == build_zip(backward)
    assert build_tar_gz(forward) == build_tar_gz(backward)


def test_exporting_touches_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = rich_bundle()

    def refuse(*args: Any, **_kwargs: Any) -> None:
        raise AssertionError(f"opened {args[0]!r}")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    for archive_format in ArchiveFormat:
        assert export_bundle(bundle, archive_format).content


def test_chunks_reassemble_to_the_archive_within_the_chunk_size() -> None:
    content = b"x" * (CHUNK_BYTES * 2 + 5)
    chunks = list(ProjectArchive("a.zip", "application/zip", content).chunks())
    assert [len(chunk) for chunk in chunks] == [CHUNK_BYTES, CHUNK_BYTES, 5]
    assert b"".join(chunks) == content
    assert list(ProjectArchive("a.zip", "application/zip", b"").chunks()) == []


def test_project_type_is_what_the_engine_returns() -> None:
    project = synthesize_project(make_bundle())
    assert isinstance(project, SynthesizedProject)
    assert project.slug == "acme-site-1"


# --- the export route -----------------------------------------------------------------------


@asynccontextmanager
async def gateway(**options: Any) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    settings = Settings(gateway_secret=SecretStr("synthesis-signing-key-for-unit-tests"))
    app = create_app(settings, provider=MockLLMProvider(), **options)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://gateway") as http,
    ):
        yield app, http


def auth(app: FastAPI, user_id: str = "alice") -> dict[str, str]:
    service: GatewayService = app.state.gateway
    return {"Authorization": f"Bearer {service.authenticator.issue(user_id)}"}


async def session_with(
    app: FastAPI, http: httpx.AsyncClient, bundle: FullstackProjectBundle
) -> str:
    response = await http.post(
        SESSIONS, json={"bundle": bundle.model_dump(mode="json")}, headers=auth(app)
    )
    assert response.status_code == 201, response.text
    session_id: str = response.json()["session_id"]
    return session_id


async def test_export_streams_a_zip_of_the_session() -> None:
    async with gateway() as (app, http):
        session_id = await session_with(app, http, rich_bundle())
        response = await http.get(
            f"{SESSIONS}/{session_id}/export", params={"format": "zip"}, headers=auth(app)
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == 'attachment; filename="acme-site-1.zip"'
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.content == export_bundle(rich_bundle(), ArchiveFormat.ZIP).content


async def test_export_defaults_to_zip_and_can_produce_tar_gz() -> None:
    async with gateway() as (app, http):
        session_id = await session_with(app, http, rich_bundle())
        default = await http.get(f"{SESSIONS}/{session_id}/export", headers=auth(app))
        tarball = await http.get(
            f"{SESSIONS}/{session_id}/export", params={"format": "tar.gz"}, headers=auth(app)
        )
    assert default.headers["content-type"] == "application/zip"
    assert tarball.headers["content-type"] == "application/gzip"
    assert 'filename="acme-site-1.tar.gz"' in tarball.headers["content-disposition"]
    assert tarball.content == export_bundle(rich_bundle(), ArchiveFormat.TAR_GZ).content


async def test_repeated_exports_are_identical_and_open_to_viewers() -> None:
    async with gateway() as (app, http):
        session_id = await session_with(app, http, rich_bundle())
        owner = await http.get(f"{SESSIONS}/{session_id}/export", headers=auth(app))
        viewer = await http.get(f"{SESSIONS}/{session_id}/export", headers=auth(app, "bob"))
    assert viewer.status_code == 200
    assert owner.content == viewer.content


async def test_export_requires_a_token() -> None:
    async with gateway() as (_app, http):
        response = await http.get(f"{SESSIONS}/any/export")
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"


async def test_export_of_an_unknown_session_is_a_problem() -> None:
    async with gateway() as (app, http):
        response = await http.get(f"{SESSIONS}/nope/export", headers=auth(app))
    assert response.status_code == 404
    assert response.json()["type"].endswith("/session-not-found")


async def test_export_rejects_an_unknown_format() -> None:
    async with gateway() as (app, http):
        session_id = await session_with(app, http, make_bundle())
        response = await http.get(
            f"{SESSIONS}/{session_id}/export", params={"format": "rar"}, headers=auth(app)
        )
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"


async def test_a_bundle_that_cannot_be_exported_gets_a_problem_listing_why() -> None:
    bundle = make_bundle(components=[component("hero", "<Hero />")])
    async with gateway() as (app, http):
        session_id = await session_with(app, http, bundle)
        response = await http.get(f"{SESSIONS}/{session_id}/export", headers=auth(app))
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["type"].endswith("/project-synthesis-failure")
    assert body["instance"] == f"{SESSIONS}/{session_id}/export"
    assert body["issues"] == ["component hero: snippet uses undefined components: Hero"]


async def test_export_is_in_the_openapi_document() -> None:
    async with gateway() as (_app, http):
        document = (await http.get("/openapi.json")).json()
    route = document["paths"][f"{SESSIONS}/{{session_id}}/export"]["get"]
    assert {"application/zip", "application/gzip"} <= set(route["responses"]["200"]["content"])
    assert {"session_id", "format"} <= {parameter["name"] for parameter in route["parameters"]}


async def test_what_a_mock_swarm_run_produces_can_be_exported() -> None:
    async with gateway() as (app, http):
        response = await http.post(
            SESSIONS, json={"prompt": "Build a landing page"}, headers=auth(app)
        )
        session_id = response.json()["session_id"]
        service: GatewayService = app.state.gateway
        runtime = service._sessions[session_id]
        assert runtime.run is not None
        await runtime.run
        exported = await http.get(f"{SESSIONS}/{session_id}/export", headers=auth(app))
    assert exported.status_code == 200, exported.text
    with zipfile.ZipFile(io.BytesIO(exported.content)) as opened:
        names = opened.namelist()
        root = names[0].split("/")[0]
        router = opened.read(f"{root}/backend/app/routers/routes_leads.py").decode()
    assert f"{root}/frontend/src/components/Hero.tsx" in names
    assert f"{root}/frontend/src/components/Contact.tsx" in names
    assert "async def _require_post_leads_body(payload: schemas.Lead) -> None:" in router
