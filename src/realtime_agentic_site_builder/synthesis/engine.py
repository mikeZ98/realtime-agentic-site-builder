"""Compiles a bundle into the source tree of a standalone FastAPI and React project.

The bundle's code was written by language models, so nothing here ever runs it: handlers are only
parsed and compiled. What must hold is that the output starts and builds, because one bad handler
would otherwise keep the whole generated service from booting. Whatever would break it is collected
and reported together as a :class:`ProjectSynthesisError`, rather than shipped.

Generation is a pure function of the bundle. It reads no clock, and it orders everything by a
property of the input, so equal bundles yield byte-identical files.
"""

import ast
import builtins
import hashlib
import html
import json
import posixpath
import re
import symtable
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

from realtime_agentic_site_builder.agents.tsx import find_tsx_issue
from realtime_agentic_site_builder.domain.backend import (
    API_PREFIX,
    ApiEndpointNode,
    DatabaseDialect,
    DataModelSchema,
    FieldDefinition,
    FullstackProjectBundle,
    HttpMethod,
)
from realtime_agentic_site_builder.domain.blueprint import ComponentNode
from realtime_agentic_site_builder.domain.errors import ProjectSynthesisError
from realtime_agentic_site_builder.synthesis import templates
from realtime_agentic_site_builder.synthesis.templates import render

_LINE_LENGTH: Final = 100
_PRIMARY_KEY: Final = "id"
_MAX_SQL_IDENTIFIER: Final = 63  # PostgreSQL's limit, and it truncates silently beyond it.
_ALEMBIC_TABLE: Final = "alembic_version"
_RESERVED_FIELDS: Final = frozenset(dir(BaseModel)) | {
    # Declarative attributes of SQLAlchemy's base class.
    "metadata",
    "registry",
    # A class attribute with one of these names would change what later annotations mean.
    "bool",
    "float",
    "int",
    "str",
}

_PATH_PARAMETER: Final = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_THIRD_PARTY_IN_PROJECT: Final = frozenset({"fastapi", "pydantic", "sqlalchemy", "starlette"})
_PROJECT_MODULES: Final = frozenset(
    {"app.config", "app.database", "app.errors", "app.models", "app.schemas"}
)
_MODULE_GLOBALS: Final = frozenset(dir(builtins)) | {"__file__"}
# Parameters FastAPI never reads from the body, so they leave room for the declared request model.
_SCALAR_ANNOTATIONS: Final = frozenset(
    {"None", "Optional", "bool", "date", "datetime", "float", "int", "str", "uuid", "UUID"}
)
_NON_BODY_MARKERS: Final = frozenset(
    {
        "BackgroundTasks",
        "Cookie",
        "DbSession",
        "Depends",
        "Header",
        "Path",
        "Query",
        "Request",
        "Response",
    }
)

_FRONTEND_PACKAGES: Final = frozenset({"react", "react-dom"})
_RESERVED_COMPONENT_NAMES: Final = frozenset({"app"})
_CLIENT_TEMPLATE_NAMES: Final = frozenset({"request", "useapicall", "usecallback", "usestate"})
_LEADING_COMMENTS: Final = re.compile(r"\A\s*(?:/\*.*?\*/\s*)*", re.DOTALL)
_DEFAULT_EXPORT: Final = re.compile(r"\bexport\s+default\b|\bas\s+default\b")
_COMPONENT_TAG: Final = re.compile(r"<([A-Z][A-Za-z0-9_.]*)")
# Bounded, so that hostile input cannot make these scan to the end of the file from every line.
_MODULE_SPECIFIER: Final = re.compile(
    r"""^\s*(?:import|export)\b[^'";]{0,2000}?\bfrom\s*['"]([^'"]{1,512})['"]"""
    r"""|^\s*import\s*['"]([^'"]{1,512})['"]"""
    r"""|\b(?:import|require)\s*\(\s*['"]([^'"]{1,512})['"]""",
    re.MULTILINE,
)
_THEME_CLASSES: Final[Mapping[str, str]] = {
    "dark": "min-h-screen bg-slate-950 text-slate-100",
    "light": "min-h-screen bg-white text-slate-900",
    "cyberpunk": "min-h-screen bg-black text-fuchsia-300",
    "minimal": "min-h-screen bg-stone-50 text-stone-900",
    "enterprise": "min-h-screen bg-slate-100 text-slate-800",
}


@dataclass(frozen=True, slots=True)
class _ColumnType:
    schema: str
    orm: str
    column: str
    typescript: str


# Generated classes refer to these through underscore-prefixed imports, which no field can shadow.
_COLUMN_TYPES: Final[Mapping[str, _ColumnType]] = {
    "str": _ColumnType("str", "str", "sa.Text()", "string"),
    "int": _ColumnType("int", "int", "sa.Integer()", "number"),
    "float": _ColumnType("float", "float", "sa.Float()", "number"),
    "bool": _ColumnType("bool", "bool", "sa.Boolean()", "boolean"),
    "EmailStr": _ColumnType("_pydantic.EmailStr", "str", "sa.String(320)", "string"),
    "datetime": _ColumnType(
        "_datetime.datetime", "_datetime.datetime", "sa.DateTime(timezone=True)", "string"
    ),
    "UUID": _ColumnType("_uuid.UUID", "_uuid.UUID", "sa.Uuid()", "string"),
}
_UTC_NOW: Final = "lambda: _datetime.datetime.now(_datetime.UTC)"
# Only known factories are emitted: this text becomes code that runs on every insert.
_DEFAULT_FACTORIES: Final[Mapping[tuple[str, str], str]] = {
    ("uuid.uuid4", "UUID"): "_uuid.uuid4",
    ("uuid4", "UUID"): "_uuid.uuid4",
    ("uuid.uuid4", "str"): "lambda: str(_uuid.uuid4())",
    ("uuid4", "str"): "lambda: str(_uuid.uuid4())",
    ("datetime.datetime.now", "datetime"): _UTC_NOW,
    ("datetime.datetime.utcnow", "datetime"): _UTC_NOW,
    ("datetime.now", "datetime"): _UTC_NOW,
    ("datetime.utcnow", "datetime"): _UTC_NOW,
}


@dataclass(frozen=True, slots=True)
class SynthesizedProject:
    slug: str
    files: Mapping[str, str]


def project_slug(project_id: str) -> str:
    """Name the project in a form npm, PEP 508, Docker and a file name all accept."""
    return re.sub(r"[^a-z0-9]+", "-", project_id.lower()).strip("-")


def synthesize_project(bundle: FullstackProjectBundle) -> SynthesizedProject:
    """Compile the bundle, or report everything that keeps it from becoming a working project."""
    issues: list[str] = []
    slug = project_slug(bundle.project_id)
    files = {
        **synthesize_config(bundle, slug),
        **synthesize_backend(bundle, issues),
        **synthesize_frontend(bundle, issues),
    }
    if issues:
        raise ProjectSynthesisError(issues)
    return SynthesizedProject(slug=slug, files=dict(sorted(files.items())))


def _python_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _typescript_string(text: str) -> str:
    return json.dumps(text)


def _indent(lines: Sequence[str], width: int) -> str:
    return "\n".join(f"{' ' * width}{line}" for line in lines)


def _call(head: str, arguments: Sequence[str], *, indent: int) -> list[str]:
    """Lay out ``head(arguments)`` the way the formatter would, given the indent it will sit at."""
    joined = ", ".join(arguments)
    if indent + len(head) + len(joined) + 2 <= _LINE_LENGTH:
        return [f"{head}{joined})"]
    if indent + 4 + len(joined) <= _LINE_LENGTH:
        return [head, f"    {joined}", ")"]
    return [head, *(f"    {argument}," for argument in arguments), ")"]


def _from_import(module: str, names: Sequence[str]) -> str:
    if not names:
        return ""
    lines = _call(f"from {module} import (", names, indent=0)
    if len(lines) == 1:
        return f"from {module} import {', '.join(names)}\n"
    return "\n".join(lines) + "\n"


def _imports(source: str, candidates: Sequence[tuple[str, str]]) -> str:
    """Keep only the imports whose name the source uses, since a project's lint runs on them."""
    return "".join(line for name, line in candidates if f"{name}." in source)


# --- project configuration ------------------------------------------------------------------


def synthesize_config(bundle: FullstackProjectBundle, slug: str) -> dict[str, str]:
    title = bundle.frontend.title
    sqlite = bundle.backend.database_dialect is DatabaseDialect.SQLITE
    example_url = (
        "sqlite+aiosqlite:///./app.db"
        if sqlite
        else "postgresql+asyncpg://app:change-me@localhost:5432/app"
    )
    readme = render(
        templates.README,
        TITLE=title,
        STATE_HASH=bundle.state_hash(),
        SLUG=slug,
        DOCKER_RUN_ARGS="" if sqlite else " -e DATABASE_URL",
        DOCKER_NOTE="" if sqlite else "Set `DATABASE_URL` to your PostgreSQL database first.\n",
        DEV_ENV_STEP="" if sqlite else "cp .env.example .env  # then point it at your database\n",
    )
    docker_env = 'ENV DATABASE_URL="sqlite+aiosqlite:////app/data/app.db"\n' if sqlite else ""
    return {
        ".dockerignore": templates.DOCKERIGNORE,
        ".gitignore": templates.GITIGNORE,
        "Dockerfile": render(templates.DOCKERFILE, DOCKER_ENV=docker_env),
        "README.md": readme,
        "backend/.env.example": render(templates.ENV_EXAMPLE, DATABASE_URL=example_url),
        "backend/alembic.ini": templates.ALEMBIC_INI,
        "backend/pyproject.toml": render(
            templates.PYPROJECT,
            SLUG=slug,
            DESCRIPTION=_python_string(title),
            DATABASE_DRIVER="aiosqlite>=0.20,<1" if sqlite else "asyncpg>=0.29,<1",
        ),
        "frontend/package.json": _package_json(slug),
        "frontend/tsconfig.json": _json_file(templates.TSCONFIG),
        "frontend/vite.config.ts": templates.VITE_CONFIG,
        "frontend/tailwind.config.js": templates.TAILWIND_CONFIG,
        "frontend/postcss.config.js": templates.POSTCSS_CONFIG,
    }


def _json_file(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def _package_json(slug: str) -> str:
    return _json_file(
        {
            "name": slug,
            "private": True,
            "version": "0.1.0",
            "type": "module",
            # Type errors in generated components are reported by `typecheck`, not fatal to a build.
            "scripts": {"dev": "vite", "build": "vite build", "typecheck": "tsc"},
            "dependencies": templates.PACKAGE_JSON_DEPENDENCIES,
            "devDependencies": templates.PACKAGE_JSON_DEV_DEPENDENCIES,
        }
    )


# --- backend --------------------------------------------------------------------------------


def synthesize_backend(bundle: FullstackProjectBundle, issues: list[str]) -> dict[str, str]:
    backend = bundle.backend
    models = _ordered_models(backend.models)
    sqlite = backend.database_dialect is DatabaseDialect.SQLITE
    default_url = ' = "sqlite+aiosqlite:///./app.db"' if sqlite else ""
    for model in models:
        issues.extend(_model_issues(model))
    files = {
        "backend/app/__init__.py": templates.APP_INIT,
        "backend/app/config.py": render(templates.CONFIG, DATABASE_URL_DEFAULT=default_url),
        "backend/app/database.py": templates.DATABASE,
        "backend/app/errors.py": templates.ERRORS,
        "backend/app/models.py": _models_module(models),
        "backend/app/schemas.py": _schemas_module(models),
        "backend/migrations/env.py": templates.ALEMBIC_ENV,
        "backend/migrations/script.py.mako": templates.ALEMBIC_SCRIPT,
        "backend/migrations/versions/0001_initial.py": _migration(models),
        "backend/app/handlers/__init__.py": '"""One module per endpoint."""\n',
        "backend/app/routers/__init__.py": '"""One module per top-level API resource."""\n',
    }
    names = {model.name for model in models}
    routes: dict[str, list[_EndpointRoute]] = {}
    for endpoint in backend.endpoints:
        route = _synthesize_endpoint(endpoint, names, issues)
        files[f"backend/app/handlers/{route.module}.py"] = route.handler_source
        routes.setdefault(_router_module(endpoint), []).append(route)
    routers = sorted(routes)
    for router in routers:
        files[f"backend/app/routers/{router}.py"] = _router_source(router, routes[router])
    files["backend/app/main.py"] = render(
        templates.MAIN,
        TITLE=_python_string(backend.service_name),
        ROUTER_IMPORTS=_from_import("app.routers", routers),
        ROUTER_INCLUDES="".join(f"app.include_router({router}.router)\n" for router in routers),
    )
    return files


def _ordered_fields(model: DataModelSchema) -> list[tuple[str, FieldDefinition]]:
    """Order fields independently of the mapping's insertion order, which equality ignores."""
    return sorted(model.fields.items(), key=lambda item: (item[0] != _PRIMARY_KEY, item[0]))


def _ordered_models(models: Sequence[DataModelSchema]) -> list[DataModelSchema]:
    return sorted(models, key=lambda model: model.name)


def _model_issues(model: DataModelSchema) -> Iterator[str]:
    if len(model.table_name) > _MAX_SQL_IDENTIFIER or model.table_name == _ALEMBIC_TABLE:
        yield f"model {model.name}: table name {model.table_name} cannot be used"
    for name, field in _ordered_fields(model):
        if name.startswith("_") or name in _RESERVED_FIELDS:
            yield f"model {model.name}: field name {name} is reserved by the generated stack"
        elif len(name) > _MAX_SQL_IDENTIFIER:
            yield f"model {model.name}: field name {name} is too long for a column"
        if field.default_factory is not None and _default_factory(field) is None:
            yield (
                f"model {model.name}: field {name} cannot use default factory "
                f"{field.default_factory} for type {field.python_type}"
            )


def _default_factory(field: FieldDefinition) -> str | None:
    return _DEFAULT_FACTORIES.get((field.default_factory or "", field.python_type))


def _index_name(table: str, field: str) -> str:
    name = f"ix_{table}_{field}"
    if len(name) <= _MAX_SQL_IDENTIFIER:
        return name
    return f"ix_{hashlib.sha256(name.encode()).hexdigest()[:32]}"


def _indexed_fields(model: DataModelSchema) -> list[str]:
    return [
        name for name, field in _ordered_fields(model) if field.indexed and name != _PRIMARY_KEY
    ]


def _models_module(models: Sequence[DataModelSchema]) -> str:
    classes = "".join(_model_class(model) for model in models)
    standard = _imports(
        classes,
        [("_datetime", "import datetime as _datetime\n"), ("_uuid", "import uuid as _uuid\n")],
    )
    # Bound to names that start with an underscore, which no field can have: see _model_issues.
    sqlalchemy = "import sqlalchemy as _sa\n" if classes else ""
    return (
        '"""The database tables."""\n\n'
        f"{standard}{chr(10) if standard else ''}"
        f"{sqlalchemy}from sqlalchemy import orm as _orm\n\n\n"
        "class _Base(_orm.DeclarativeBase):\n"
        "    pass\n\n\n"
        f"{classes}"
        "metadata = _Base.metadata\n"
    )


def _model_class(model: DataModelSchema) -> str:
    lines = [f"__tablename__ = {_python_string(model.table_name)}"]
    indexed = _indexed_fields(model)
    if indexed:
        lines.extend(
            _table_args(
                [
                    (_python_string(_index_name(model.table_name, name)), _python_string(name))
                    for name in indexed
                ]
            )
        )
    lines.append("")
    if _PRIMARY_KEY not in model.fields:
        lines.append("id: _orm.Mapped[int] = _orm.mapped_column(_sa.Integer(), primary_key=True)")
    for name, field in _ordered_fields(model):
        column = _COLUMN_TYPES[field.python_type]
        is_key = name == _PRIMARY_KEY
        annotation = column.orm if is_key or not field.nullable else f"{column.orm} | None"
        arguments = [column.column.replace("sa.", "_sa.")]
        if is_key:
            arguments.append("primary_key=True")
        factory = _default_factory(field)
        if factory is not None:
            arguments.append(f"default={factory}")
        lines.extend(
            _call(f"{name}: _orm.Mapped[{annotation}] = _orm.mapped_column(", arguments, indent=4)
        )
    body = "\n".join(f"    {line}" if line else "" for line in lines)
    return f"class {model.name}(_Base):\n{body}\n\n\n"


def _table_args(indexes: Sequence[tuple[str, str]]) -> list[str]:
    calls = ", ".join(f"_sa.Index({name}, {column})" for name, column in indexes)
    single = f"__table_args__ = ({calls},)"
    if len(single) + 4 <= _LINE_LENGTH:
        return [single]
    lines = ["__table_args__ = ("]
    for index in indexes:
        call = _call("_sa.Index(", index, indent=8)
        call[-1] += ","
        lines.extend(f"    {line}" for line in call)
    lines.append(")")
    return lines


def _schemas_module(models: Sequence[DataModelSchema]) -> str:
    classes = "\n\n".join(_schema_class(model) for model in models)
    standard = _imports(
        classes,
        [("_datetime", "import datetime as _datetime\n"), ("_uuid", "import uuid as _uuid\n")],
    )
    if not classes:
        return '"""The shapes requests and responses take."""\n'
    return (
        '"""The shapes requests and responses take."""\n\n'
        f"{standard}{chr(10) if standard else ''}"
        "import pydantic as _pydantic\n\n\n"
        f"{classes}"
    )


def _schema_class(model: DataModelSchema) -> str:
    lines = ["model_config = _pydantic.ConfigDict(from_attributes=True)", ""]
    for name, field in _ordered_fields(model):
        kind = _COLUMN_TYPES[field.python_type].schema
        annotation = f"{kind} | None" if field.nullable else kind
        factory = _default_factory(field)
        if factory is not None:
            lines.extend(
                _call(
                    f"{name}: {annotation} = _pydantic.Field(",
                    [f"default_factory={factory}"],
                    indent=4,
                )
            )
        elif field.nullable:
            lines.append(f"{name}: {annotation} = None")
        else:
            lines.append(f"{name}: {annotation}")
    body = "\n".join(f"    {line}" if line else "" for line in lines)
    return f"class {model.name}(_pydantic.BaseModel):\n{body}\n"


def _migration(models: Sequence[DataModelSchema]) -> str:
    upgrade: list[str] = []
    for model in models:
        table = _python_string(model.table_name)
        upgrade.append("op.create_table(")
        upgrade.append(f"    {table},")
        if _PRIMARY_KEY not in model.fields:
            upgrade.append('    sa.Column("id", sa.Integer(), primary_key=True),')
        for name, field in _ordered_fields(model):
            column = _COLUMN_TYPES[field.python_type].column
            constraint = (
                "primary_key=True" if name == _PRIMARY_KEY else f"nullable={field.nullable}"
            )
            upgrade.append(f"    sa.Column({_python_string(name)}, {column}, {constraint}),")
        upgrade.append(")")
        for name in _indexed_fields(model):
            index = _python_string(_index_name(model.table_name, name))
            upgrade.append(f"op.create_index({index}, {table}, [{_python_string(name)}])")
    downgrade = [f"op.drop_table({_python_string(m.table_name)})" for m in reversed(models)]
    return render(
        templates.MIGRATION,
        IMPORTS="import sqlalchemy as sa\nfrom alembic import op\n\n" if models else "",
        UPGRADE=_indent(upgrade or ["pass"], 4),
        DOWNGRADE=_indent(downgrade or ["pass"], 4),
    )


def _endpoint_module(endpoint: ApiEndpointNode) -> str:
    """Endpoint ids are unique, lowercase and safe, so this cannot collide on any file system."""
    return endpoint.id.removeprefix("api_endpoint_")


def _router_module(endpoint: ApiEndpointNode) -> str:
    resource = endpoint.path.removeprefix(API_PREFIX).split("/", 1)[0]
    return f"routes_{re.sub(r'[^a-z0-9]+', '_', resource).strip('_')}"


@dataclass(frozen=True, slots=True)
class _EndpointRoute:
    module: str
    entry: str
    endpoint: ApiEndpointNode
    handler_source: str
    body_validator: str | None


def _synthesize_endpoint(
    endpoint: ApiEndpointNode, models: Set[str], issues: list[str]
) -> _EndpointRoute:
    module = _endpoint_module(endpoint)
    label = f"endpoint {endpoint.method.value} {endpoint.path}"
    path_names = _PATH_PARAMETER.findall(endpoint.path)
    path_parameters = set(path_names)
    if len(path_parameters) != len(path_names):
        issues.append(f"{label}: a path parameter appears twice")
    if endpoint.method is HttpMethod.GET and endpoint.request_model_ref is not None:
        issues.append(f"{label}: a GET request cannot carry a request model")

    tree = _parse(endpoint.handler_code)
    entry = _entry_point(tree)
    issues.extend(f"{label}: {issue}" for issue in _signature_issues(entry, path_parameters))
    for imported in sorted(set(_imported_modules(tree))):
        if not _is_importable(imported):
            issues.append(f"{label}: importing {imported} would fail in the generated project")

    schema_names = ", ".join(sorted(models))
    source = (
        render(
            templates.HANDLER_PREAMBLE,
            METHOD=endpoint.method.value,
            PATH=endpoint.path,
            SCHEMA_IMPORT=f"from app.schemas import {schema_names}\n" if schema_names else "",
        )
        + endpoint.handler_code
        + "\n"
    )
    issues.extend(f"{label}: {issue}" for issue in _source_issues(source, module))

    binds_body = any(
        _reads_body(argument, default, path_parameters, models)
        for argument, default in _parameters(entry)
    )
    needs_validator = endpoint.request_model_ref is not None and not binds_body
    return _EndpointRoute(
        module=module,
        entry=entry.name,
        endpoint=endpoint,
        handler_source=source,
        body_validator=f"_require_{module}_body" if needs_validator else None,
    )


def _parse(code: str) -> ast.Module:
    # The bundle's own validators have already parsed this successfully.
    with warnings.catch_warnings(action="ignore", category=SyntaxWarning):
        return ast.parse(code)


def _entry_point(tree: ast.Module) -> ast.AsyncFunctionDef:
    """Pick the function to route to. The bundle's validators guarantee that there is one."""
    functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
    named = [function for function in functions if function.name == "handler"]
    return (named or functions)[0]


def _parameters(function: ast.AsyncFunctionDef) -> list[tuple[ast.arg, ast.expr | None]]:
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    padding: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults))
    return [
        *zip(positional, [*padding, *arguments.defaults], strict=True),
        *zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True),
    ]


def _signature_issues(function: ast.AsyncFunctionDef, path_parameters: Set[str]) -> Iterator[str]:
    if function.args.vararg is not None or function.args.kwarg is not None:
        yield f"{function.name} takes variadic arguments, which cannot be routed"
    for argument, default in _parameters(function):
        unbound = argument.annotation is None and default is None
        if unbound and argument.arg not in path_parameters:
            yield f"{function.name}: parameter {argument.arg} needs a type annotation"


def _annotation_names(annotation: ast.expr | None) -> set[str]:
    """Collect the names in an annotation, including those in a quoted forward reference."""
    names: set[str] = set()
    for node in () if annotation is None else ast.walk(annotation):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
    return names


def _default_names(default: ast.expr | None) -> set[str]:
    nodes = () if default is None else ast.walk(default)
    return {node.id for node in nodes if isinstance(node, ast.Name)}


def _reads_body(
    argument: ast.arg, default: ast.expr | None, path_parameters: Set[str], models: Set[str]
) -> bool:
    """Tell whether FastAPI would read this parameter from the request body."""
    if argument.arg in path_parameters:
        return False
    names = _annotation_names(argument.annotation) | _default_names(default)
    if names & models or "Body" in names:
        return True
    if names & _NON_BODY_MARKERS:
        return False
    return not names <= _SCALAR_ANNOTATIONS


def _imported_modules(tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module == "app":
                yield from (f"app.{alias.name}" for alias in node.names)
            else:
                # A relative import keeps its leading dots, so it never counts as importable.
                yield f"{'.' * node.level}{node.module or ''}"


def _is_importable(module: str) -> bool:
    root = module.partition(".")[0]
    if root == "app":
        return module in _PROJECT_MODULES
    return root in _THIRD_PARTY_IN_PROJECT or root in sys.stdlib_module_names


def _source_issues(source: str, module: str) -> Iterator[str]:
    """Compile without running, which also catches what parsing alone lets through.

    That covers ``return`` outside a function and a ``__future__`` import that the preamble has
    pushed off the top of the file. Then, because names in a signature are evaluated on import, one
    unresolved name there would keep the whole service from starting.
    """
    filename = f"{module}.py"
    with warnings.catch_warnings(action="ignore", category=SyntaxWarning):
        try:
            compile(source, filename, "exec", dont_inherit=True)
            table = symtable.symtable(source, filename, "exec")
        except SyntaxError as error:
            yield f"does not compile: {error.msg}"
            return
        except (RecursionError, MemoryError):
            yield "is too deeply nested to compile"
            return
    defined = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    for name in sorted(_referenced_globals(table) - defined - _MODULE_GLOBALS):
        yield f"name {name} is not defined"


def _referenced_globals(table: symtable.SymbolTable) -> set[str]:
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_global() and symbol.is_referenced()
    }
    for child in table.get_children():
        names |= _referenced_globals(child)
    return names


def _registration(route: _EndpointRoute) -> str:
    endpoint = route.endpoint
    response_model = (
        "None" if endpoint.response_model_ref is None else f"schemas.{endpoint.response_model_ref}"
    )
    lines = [
        "router.add_api_route(",
        f"    {_python_string(endpoint.path)},",
        f"    {route.module}.{route.entry},",
        f'    methods=["{endpoint.method.value}"],',
        f"    summary={_python_string(endpoint.summary)},",
        f'    operation_id="{route.module}",',
        # Explicit, so that an arbitrary return annotation is never treated as a response schema.
        f"    response_model={response_model},",
    ]
    prefix = ""
    if route.body_validator is not None:
        lines.append(f"    dependencies=[Depends({route.body_validator})],")
        signature = f"payload: schemas.{endpoint.request_model_ref}"
        definition = _call(f"async def {route.body_validator}(", [signature], indent=0)
        definition[-1] += " -> None:"
        prefix = (
            "\n".join(definition)
            + '\n    """Hold requests to the declared schema: the handler binds no body."""\n\n\n'
        )
    lines.append(")")
    return prefix + "\n".join(lines) + "\n"


def _router_source(router: str, routes: Sequence[_EndpointRoute]) -> str:
    validated = any(route.body_validator is not None for route in routes)
    uses_schemas = validated or any(
        route.endpoint.response_model_ref is not None for route in routes
    )
    modules = sorted({route.module for route in routes})
    schema_import = "from app import schemas\n" if uses_schemas else ""
    project_imports = schema_import + _from_import("app.handlers", modules)
    return (
        f'"""Routes of the {router.removeprefix("routes_") or "root"} resource."""\n\n'
        f"from fastapi import APIRouter{', Depends' if validated else ''}\n\n"
        f"{project_imports}\n"
        "router = APIRouter()\n\n\n" + "\n\n".join(_registration(route) for route in routes)
    )


# --- frontend -------------------------------------------------------------------------------


def synthesize_frontend(bundle: FullstackProjectBundle, issues: list[str]) -> dict[str, str]:
    site = bundle.frontend
    names = _component_names(site.components)
    files = {
        "frontend/index.html": render(templates.INDEX_HTML, TITLE=html.escape(site.title)),
        "frontend/src/App.tsx": _app_module(bundle, names),
        "frontend/src/api/client.ts": _api_client(bundle),
        "frontend/src/api/types.ts": _api_types(bundle),
        "frontend/src/index.css": templates.INDEX_CSS,
        "frontend/src/main.tsx": templates.MAIN_TSX,
    }
    for component in site.components:
        path = f"frontend/src/components/{names[component.id]}.tsx"
        files[path] = _component_module(component, names[component.id], issues)
    for component in site.components:
        path = f"frontend/src/components/{names[component.id]}.tsx"
        issues.extend(_import_issues(component, path, files))
    return files


def _unique_name(name: str, used: set[str]) -> str:
    """Pick a name that differs in more than case: macOS and Windows would merge the files."""
    candidate = name
    suffix = 2
    while candidate.casefold() in used:
        candidate = f"{name}{suffix}"
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def _component_names(components: Sequence[ComponentNode]) -> dict[str, str]:
    used = set(_RESERVED_COMPONENT_NAMES)
    names: dict[str, str] = {}
    for component in components:
        pascal = "".join(part.capitalize() for part in component.id.split("-"))
        if not pascal or pascal[0].isdigit():
            pascal = f"Component{pascal}"
        names[component.id] = _unique_name(pascal, used)
    return names


def _component_module(component: ComponentNode, name: str, issues: list[str]) -> str:
    label = f"component {component.id}"
    code = component.code
    if not _DEFAULT_EXPORT.search(code):
        snippet = _LEADING_COMMENTS.sub("", code, count=1)
        if not snippet.startswith("<"):
            issues.append(f"{label}: code must default-export a component or be a JSX snippet")
        unresolved = sorted(set(_COMPONENT_TAG.findall(snippet)))
        if unresolved:
            # A snippet has no imports, so nothing in it can name another component.
            issues.append(f"{label}: snippet uses undefined components: {', '.join(unresolved)}")
        code = render(templates.BARE_COMPONENT, NAME=name, CODE=snippet)
    problem = find_tsx_issue(code)
    if problem is not None:
        issues.append(f"{label}: {problem}")
    return code if code.endswith("\n") else f"{code}\n"


def _module_specifiers(code: str) -> Iterator[str]:
    for match in _MODULE_SPECIFIER.finditer(code):
        yield match[1] or match[2] or match[3]


def _resolves(specifier: str, importer: str, files: Mapping[str, str]) -> bool:
    if specifier.startswith("."):
        target = posixpath.normpath(posixpath.join(posixpath.dirname(importer), specifier))
        return any(
            candidate in files
            for candidate in (target, f"{target}.tsx", f"{target}.ts", f"{target}/index.ts")
        )
    return specifier.split("/", 1)[0] in _FRONTEND_PACKAGES


def _import_issues(component: ComponentNode, path: str, files: Mapping[str, str]) -> Iterator[str]:
    """Refuse an import rather than add the dependency: models invent convincing package names."""
    for specifier in sorted(set(_module_specifiers(files[path]))):
        if not _resolves(specifier, path, files):
            yield f"component {component.id}: import of {specifier} cannot be resolved"


def _layout_rows(components: Sequence[ComponentNode]) -> list[list[ComponentNode]]:
    """Group side-by-side components. They are sorted by ``y`` and cannot overlap."""
    rows: list[list[ComponentNode]] = []
    bottom = 0
    for component in components:
        if rows and component.box.y < bottom:
            rows[-1].append(component)
            bottom = max(bottom, component.box.bottom)
        else:
            rows.append([component])
            bottom = component.box.bottom
    return rows


def _app_module(bundle: FullstackProjectBundle, names: Mapping[str, str]) -> str:
    site = bundle.frontend
    lines: list[str] = []
    for row in _layout_rows(site.components):
        if len(row) == 1:
            lines.append(f"<{names[row[0].id]} />")
            continue
        lines.append('<div className="flex w-full">')
        for component in row:
            flex = f'{{{{ flex: "{component.box.width} 1 0%" }}}}'
            lines.append(f"  <div style={flex}>")
            lines.append(f"    <{names[component.id]} />")
            lines.append("  </div>")
        lines.append("</div>")
    if not lines:
        lines = [f'<h1 className="p-8 text-2xl">{{{_typescript_string(site.title)}}}</h1>']
    imports = "".join(
        f'import {name} from "./components/{name}";\n' for name in sorted(names.values())
    )
    return render(
        templates.APP_TSX,
        IMPORTS=f"{imports}\n" if imports else "",
        THEME_CLASSES=_THEME_CLASSES[site.theme],
        THEME=site.theme,
        BODY=_indent(lines, 6),
    )


def _api_types(bundle: FullstackProjectBundle) -> str:
    interfaces = [_interface(model) for model in _ordered_models(bundle.backend.models)]
    return "\n".join(interfaces) if interfaces else "export {};\n"


def _interface(model: DataModelSchema) -> str:
    lines = [f"export interface {model.name} {{"]
    for name, field in _ordered_fields(model):
        kind = _COLUMN_TYPES[field.python_type].typescript
        optional = "?" if field.nullable or field.default_factory is not None else ""
        lines.append(f"  {name}{optional}: {kind}{' | null' if field.nullable else ''};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _api_client(bundle: FullstackProjectBundle) -> str:
    endpoints = bundle.backend.endpoints
    uses_models = any(
        endpoint.request_model_ref is not None or endpoint.response_model_ref is not None
        for endpoint in endpoints
    )
    used = set(_CLIENT_TEMPLATE_NAMES)
    functions = [_endpoint_client(endpoint, used) for endpoint in endpoints]
    # Types live in their own module, so that a model called Promise shadows nothing here.
    imports = 'import type * as Models from "./types";\n\n' if uses_models else ""
    source = render(templates.API_CLIENT, MODELS_IMPORT=imports)
    return "\n".join([source, *functions])


def _endpoint_client(endpoint: ApiEndpointNode, used: set[str]) -> str:
    words = _endpoint_module(endpoint).split("_")
    camel = words[0] + "".join(word.capitalize() for word in words[1:])
    # Only a DELETE on a path with no name of its own reaches a reserved word: `delete`.
    function = _unique_name("delete_" if camel == "delete" else camel, used)
    hook = _unique_name(f"use{function[0].upper()}{function[1:].rstrip('_')}", used)
    parameters: list[str] = []
    path_names = _PATH_PARAMETER.findall(endpoint.path)
    if path_names:
        fields = "; ".join(f"{name}: string | number" for name in path_names)
        parameters.append(f"params: {{ {fields} }}")
    if endpoint.request_model_ref is not None:
        parameters.append(f"body: Models.{endpoint.request_model_ref}")
    parameters.append("options?: RequestOptions")
    url = _PATH_PARAMETER.sub(
        lambda match: f"${{encodeURIComponent(String(params.{match[1]}))}}", endpoint.path
    )
    response_model = endpoint.response_model_ref
    response = "unknown" if response_model is None else f"Models.{response_model}"
    body = ", body" if endpoint.request_model_ref is not None else ""
    return (
        f"/** {endpoint.method.value} {endpoint.path} */\n"
        f"export function {function}({', '.join(parameters)}): Promise<{response}> {{\n"
        f'  return request<{response}>("{endpoint.method.value}", `{url}`, options{body});\n'
        "}\n\n"
        f"export function {hook}() {{\n"
        f"  return useApiCall({function});\n"
        "}\n"
    )
