"""The live DOM mirror: RFC 6902 alone keeps a viewer's layout equal to the committed one.

Three claims, each checked against the real patch engine rather than a model of it:

- Patches, then the server's corrections, turn the layout before a commit into exactly the layout
  it committed, including stamped versions and normalized text the page cannot derive itself.
- The page's own JavaScript applies RFC 6902 exactly as the server does, operation by operation,
  success and failure alike. That runs under Node when it is installed, and is skipped otherwise.
- A viewer whose copy is not the one a commit applied to is never sent corrections for it.
"""

import copy
import json
import random
import shutil
import subprocess
from typing import Any

import pytest
from e2e_support import OWNER, VIEWER, gateway
from pydantic import SecretStr

from realtime_agentic_site_builder.canvas.mirror import corrections, json_diff
from realtime_agentic_site_builder.canvas.viewer import _RENDERED_SCRIPT
from realtime_agentic_site_builder.config import Settings
from realtime_agentic_site_builder.core.hub import StateEvent, SwarmSessionHub
from realtime_agentic_site_builder.core.patch_engine import PatchEngine, replay_operations
from realtime_agentic_site_builder.domain.backend import FullstackProjectBundle
from realtime_agentic_site_builder.domain.blueprint import BoundingBox, ComponentNode, ComponentType
from realtime_agentic_site_builder.domain.errors import SiteBuilderError
from realtime_agentic_site_builder.domain.swarm import PatchOperation, StreamDelta
from realtime_agentic_site_builder.gateway.connection import CanvasConnection

DOCUMENT = "frontend:document"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is not installed")
# Titles the server normalizes: entities, compatibility forms, markup and stray whitespace.
TITLES = ("Plain", "Terms &amp; Conditions", "Next…", "Ｆｕｌｌ", " Hi ")
THEMES = ("dark", "light", "cyberpunk", "minimal", "enterprise")


def component(component_id: str, y: int, kind: ComponentType = ComponentType.HERO) -> Any:
    return ComponentNode.create(
        id=component_id,
        component_type=kind,
        title=component_id.title(),
        code="export default function S() { return <section />; }",
        box=BoundingBox(x=0, y=y, width=400, height=100),
    ).model_dump(mode="json")


def bundle() -> FullstackProjectBundle:
    kinds = (ComponentType.NAVBAR, ComponentType.HERO, ComponentType.PRICING, ComponentType.FOOTER)
    return FullstackProjectBundle.model_validate(
        {
            "project_id": "mirror",
            "frontend": {
                "site_id": "s",
                "title": "Start",
                "theme": "dark",
                "components": [
                    component(f"c{index}", 120 * index, kind) for index, kind in enumerate(kinds)
                ],
            },
            "backend": {"service_name": "api", "database_dialect": "SQLITE"},
        }
    )


def operation(**fields: Any) -> PatchOperation:
    return PatchOperation.model_validate(fields)


# --- the difference the server sends --------------------------------------------------------


def test_a_diff_descends_where_shapes_match_and_replaces_where_they_do_not() -> None:
    old = {"a": [1, {"b": True}], "c~/d": "x", "same": [1, 2]}
    new = {"a": [1, {"b": 1}], "c~/d": "y", "same": [1, 2]}
    assert [op.model_dump() for op in json_diff(old, new, "")] == [
        {"op": "replace", "path": "/a/1/b", "value": 1},
        {"op": "replace", "path": "/c~0~1d", "value": "y"},
    ], "JSON's true is not 1, and keys are escaped as RFC 6901 requires"
    assert [op.model_dump() for op in json_diff({"a": 1}, {"b": 1}, "/x")] == [
        {"op": "replace", "path": "/x", "value": {"b": 1}}
    ]
    assert [op.model_dump() for op in json_diff([1], [1, 2], "/x")] == [
        {"op": "replace", "path": "/x", "value": [1, 2]}
    ]
    assert list(json_diff({"a": [None]}, {"a": [None]}, "")) == []


def test_corrections_carry_exactly_what_patches_alone_miss() -> None:
    before = bundle().frontend.model_dump(mode="json")
    patches = (operation(op="replace", path="/frontend/components/1/title", value="A &amp; B"),)
    committed, _ = PatchEngine().apply_patch(bundle(), delta(patches))
    after = committed.frontend.model_dump(mode="json")
    fixes = corrections(before, patches, after)
    assert {fix.path: fix.value for fix in fixes} == {
        "/frontend/components/1/title": "A & B",
        "/frontend/components/1/ast_version": 2,
    }
    unchanged = (operation(op="test", path="/frontend/theme", value="dark"),)
    assert corrections(before, unchanged, before) == ()


def delta(patches: tuple[PatchOperation, ...], seq: int = 0) -> StreamDelta:
    return StreamDelta(
        session_id="s",
        agent_id="agent-coder",
        delta_seq=seq,
        target_domain="frontend",
        target_id=DOCUMENT,
        patches=patches,
        emitted_at_ms=0,
    )


def random_patches(rng: random.Random, frontend: dict[str, Any]) -> tuple[PatchOperation, ...]:
    """A commit an agent might send: retitles, recodes, resizes, copies, inserts and removals."""
    components = frontend["components"]
    count = len(components)
    pick = rng.randrange(max(count, 1))
    base = f"/frontend/components/{pick}"
    bottom = max((item["box"]["y"] + item["box"]["height"] for item in components), default=0)
    choices = [
        lambda: operation(op="replace", path="/frontend/title", value=rng.choice(TITLES)),
        lambda: operation(op="replace", path="/frontend/theme", value=rng.choice(THEMES)),
        lambda: operation(op="test", path="/frontend/theme", value=frontend["theme"]),
        lambda: operation(
            op="add",
            path="/frontend/components/-",
            value=component(f"n{rng.randrange(99)}", bottom + 20),
        ),
    ]
    if count:
        choices += [
            lambda: operation(op="replace", path=f"{base}/title", value=rng.choice(TITLES)),
            lambda: operation(op="replace", path=f"{base}/code", value=f"// v{rng.randrange(9)}"),
            lambda: operation(op="replace", path=f"{base}/box/height", value=rng.randint(20, 100)),
            lambda: operation(
                op="copy",
                from_=f"/frontend/components/{rng.randrange(count)}/title",
                path=f"{base}/title",
            ),
            lambda: operation(op="remove", path=base),
        ]
    return tuple(rng.choice(choices)() for _ in range(rng.randint(1, 3)))


def engine_commits(seed: int, steps: int) -> list[tuple[dict[str, Any], list[Any], dict[str, Any]]]:
    """Run random commits through the engine, each as (before, patches + corrections, after)."""
    rng, engine, state = random.Random(seed), PatchEngine(), bundle()
    commits = []
    for seq in range(steps):
        before = state.frontend.model_dump(mode="json")
        patches = random_patches(rng, before)
        try:
            state, _ = engine.apply_patch(state, delta(patches, seq))
        except SiteBuilderError:
            continue
        after = state.frontend.model_dump(mode="json")
        sent = [op.model_dump() for op in (*patches, *corrections(before, patches, after))]
        commits.append((before, sent, after))
    return commits


def test_patches_then_corrections_reproduce_every_commit_of_a_random_workload() -> None:
    commits = engine_commits(seed=7, steps=400)
    assert len(commits) > 200, "most random commits should be accepted"
    corrected = 0
    for before, sent, after in commits:
        document = {"frontend": copy.deepcopy(before)}
        replay_operations(document, [PatchOperation.model_validate(op) for op in sent])
        assert document["frontend"] == after
        corrected += len(sent) > 0
    assert corrected == len(commits)


# --- the page's own RFC 6902, against the engine --------------------------------------------


def page_patch_source() -> str:
    start = _RENDERED_SCRIPT.index("// <patch>")
    return _RENDERED_SCRIPT[start : _RENDERED_SCRIPT.index("// </patch>")]


RUNNER = """
const chunks = [];
process.stdin.on("data", (chunk) => chunks.push(chunk));
process.stdin.on("end", () => {
  const { cases, commits } = JSON.parse(Buffer.concat(chunks).toString());
  const single = cases.map(([doc, op]) => {
    try { applyOperation(doc, op); return ["ok", doc]; } catch (error) { return ["error"]; }
  });
  const whole = commits.map(([before, ops]) => applyPatches({ frontend: before }, ops).frontend);
  process.stdout.write(JSON.stringify({ single, whole }));
});
"""


def run_page_patcher(cases: list[Any], commits: list[Any]) -> dict[str, Any]:
    assert NODE is not None
    result = subprocess.run(  # noqa: S603 - a fixed interpreter and a script from this repository
        [NODE, "-e", page_patch_source() + RUNNER],
        input=json.dumps({"cases": cases, "commits": commits}),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


KEYS = ("a", "b", "a/b", "m~n", "", "0", "-")


def random_json(rng: random.Random, depth: int = 0) -> Any:
    roll = rng.random()
    if depth < 3 and roll < 0.3:
        return {rng.choice(KEYS): random_json(rng, depth + 1) for _ in range(rng.randint(0, 3))}
    if depth < 3 and roll < 0.55:
        return [random_json(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return rng.choice([None, True, False, 0, 1, 2.5, "x", "", "1"])


def pointers(value: Any, prefix: str = "") -> list[str]:
    found = [prefix] if prefix else []
    if isinstance(value, dict):
        for key, item in value.items():
            found += pointers(item, f"{prefix}/{key.replace('~', '~0').replace('/', '~1')}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += pointers(item, f"{prefix}/{index}")
    return found


def random_case(rng: random.Random) -> tuple[Any, dict[str, Any]] | None:
    document = {"root": random_json(rng)}
    known = pointers(document)
    junk = ["/root/zz", "/root/9", "/root/-", "/root/01", "/nope/x", "/root/a~1b", "/root/m~0n"]

    def where() -> str:
        if known and rng.random() < 0.7:
            chosen = rng.choice(known)
            return chosen + rng.choice(["", "", "/-", "/0", "/a"])
        return rng.choice(junk)

    fields: dict[str, Any] = {
        "op": rng.choice(["add", "remove", "replace", "move", "copy", "test"])
    }
    fields["path"] = where()
    if fields["op"] in {"add", "replace", "test"}:
        fields["value"] = (
            random_json(rng, 2)
            if rng.random() < 0.6
            else copy.deepcopy(rng.choice([document["root"], None, 1, True]))
        )
    if fields["op"] in {"move", "copy"}:
        fields["from"] = where()
    try:
        return document, PatchOperation.model_validate(fields).model_dump()
    except ValueError:
        return None


def engine_outcome(document: Any, raw: dict[str, Any]) -> list[Any]:
    patched = copy.deepcopy(document)
    try:
        replay_operations(patched, [PatchOperation.model_validate(raw)])
    except Exception:  # noqa: BLE001 - any failure is the outcome being compared
        return ["error"]
    return ["ok", patched]


@needs_node
def test_the_page_applies_each_operation_exactly_as_the_engine_does() -> None:
    rng = random.Random(2026)
    cases = [case for _ in range(4000) if (case := random_case(rng)) is not None]
    page = run_page_patcher([list(case) for case in cases], [])
    expected = [engine_outcome(document, raw) for document, raw in cases]
    outcomes = {outcome[0] for outcome in expected}
    assert outcomes == {"ok", "error"}, "the cases must exercise both"
    for (document, raw), mine, theirs in zip(cases, page["single"], expected, strict=True):
        assert mine == theirs, (document, raw)


@needs_node
def test_the_page_reaches_every_committed_layout_from_what_the_server_sends() -> None:
    commits = engine_commits(seed=11, steps=300)
    page = run_page_patcher([], [[before, sent] for before, sent, _ in commits])
    assert page["whole"] == [after for _, _, after in commits]


@needs_node
def test_the_page_refuses_keys_that_would_reach_an_objects_prototype() -> None:
    cases = [
        [{}, {"op": "add", "path": f"/{key}/polluted", "value": 1}]
        for key in ("__proto__", "constructor", "prototype")
    ]
    assert run_page_patcher(cases, [])["single"] == [["error"]] * 3


# --- what each viewer is sent ---------------------------------------------------------------


async def test_a_viewer_is_never_sent_corrections_against_a_layout_it_does_not_hold() -> None:
    hub = SwarmSessionHub(Settings(gateway_secret=SecretStr("x" * 40)))
    start = bundle()
    await hub.create_session("s", start)
    listener = hub.subscribe("s", lossless=True)
    await hub.lock_manager("s").acquire("c0", "agent-coder", 70, hub.now_ms())
    patches = (operation(op="replace", path="/frontend/components/0/title", value="A &amp; B"),)
    for seq in range(2):
        await hub.apply_delta("s", delta(patches, seq).model_copy(update={"target_id": "c0"}))
    events = []
    while (event := listener.poll()) is not None:
        if isinstance(event, StateEvent):
            events.append(event)
    viewer = object.__new__(CanvasConnection)
    viewer._mirror_seq, viewer._mirror = 0, start.frontend.model_dump(mode="json")

    first = json.loads(viewer._describe(events[0]))
    assert first["corrections"], "the first commit follows the layout the viewer holds"
    viewer._mirror_seq = -5
    assert json.loads(viewer._describe(events[1]))["corrections"] == []
    assert viewer._mirror_seq == 2
    await hub.shutdown()


async def test_a_canvas_viewer_mirrors_each_commit_exactly_over_the_socket() -> None:
    async with gateway() as harness:
        session_id = await harness.create_session(OWNER, bundle=bundle().model_dump(mode="json"))
        viewer = harness.socket(session_id, VIEWER, query="view=canvas")
        hello = await viewer.connect()
        mirror = {"frontend": hello["frontend"]}
        owner = harness.socket(session_id)
        await owner.connect()
        owner.send({"type": "lock.acquire", "target_id": "c2", "request_id": "l"})
        assert (await owner.next_of("ack", "error"))["kind"] == "ack"
        owner.send(
            {
                "type": "delta",
                "request_id": "d",
                "target_domain": "frontend",
                "target_id": "c2",
                "patches": [
                    {"op": "replace", "path": "/frontend/components/2/title", "value": "Q &amp; A"}
                ],
            }
        )
        sent = await viewer.next_of("canvas.delta")
        replay_operations(
            mirror,
            [PatchOperation.model_validate(op) for op in sent["patches"] + sent["corrections"]],
        )
        committed = harness.service.hub.timeline(session_id).snapshots[-1]
        assert mirror["frontend"] == committed.bundle_state.frontend.model_dump(mode="json")
        assert mirror["frontend"]["components"][2]["title"] == "Q & A"
        await owner.drop()
        await viewer.drop()
