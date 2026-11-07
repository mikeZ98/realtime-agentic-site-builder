"""Measure how many deliberate faults the test suite catches in the safety-critical modules.

Every applicable site is mutated, one at a time: comparison operators are flipped or relaxed,
arithmetic and boolean operators swapped, ``not`` dropped, numbers nudged and booleans inverted.
Each mutant runs the test files that cover its module, in a private copy of the repository under
``.cache/mutation/``. A mutant is *killed* when a test fails or the run times out, and *survives*
when every test still passes. Nothing outside ``.cache/`` is touched.

    uv run python scripts/mutation-test.py              # every target
    uv run python scripts/mutation-test.py --list       # count mutants without running them
    uv run python scripts/mutation-test.py core/lock_manager.py --workers 4

Module-level constants written in upper case are configuration, not logic, and are not mutated.
The values the documentation promises are pinned by contract tests instead.

Survivors are printed with their location, so each can be judged: a missing test, or an
equivalent mutant that no test could ever tell apart from the original. Those reviewed as
equivalent are listed in ``EQUIVALENT`` with the reason, reported separately, and excluded from the
adjusted score. The raw score counts them as survivors.
"""

import argparse
import ast
import os
import shutil
import subprocess  # nosec B404: runs this repository's own test suite, never external input
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parent.parent
PACKAGE: Final = Path("src/realtime_agentic_site_builder")
WORKSPACE: Final = ROOT / ".cache" / "mutation"

# Each module is mutated against the tests that are meant to pin it down.
TARGETS: Final[dict[str, tuple[str, ...]]] = {
    "core/lock_manager.py": ("tests/unit/test_core.py", "tests/unit/test_contracts.py"),
    "core/patch_engine.py": (
        "tests/unit/test_core.py",
        "tests/unit/test_hardening.py",
        "tests/unit/test_contracts.py",
    ),
    "core/hub.py": (
        "tests/unit/test_core.py",
        "tests/unit/test_hardening.py",
        "tests/unit/test_gateway.py",
        "tests/unit/test_contracts.py",
    ),
    "domain/swarm.py": (
        "tests/unit/test_schemas.py",
        "tests/unit/test_core.py",
        "tests/unit/test_contracts.py",
    ),
    "gateway/auth.py": ("tests/unit/test_gateway.py", "tests/unit/test_contracts.py"),
    "canvas/projection.py": ("tests/unit/test_canvas.py", "tests/unit/test_contracts.py"),
    "canvas/compositor.py": ("tests/unit/test_canvas.py", "tests/unit/test_contracts.py"),
    "synthesis/exporter.py": ("tests/unit/test_synthesis.py", "tests/unit/test_contracts.py"),
}
# Reviewed survivors no test could ever kill, keyed by module, source line and mutation.
_STUB = "self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0"
_REPLAY_LOOP = (
    "for position, (operation, raw) in enumerate(zip(delta.patches, patch, strict=True), 1):"
)
_DESCRIBE = "first = error.errors(include_url=False, include_context=False, include_input=False)[0]"
_SEGMENT = "earlier, later = next(pair for pair in pairwise(self._knots) if t_ms <= pair[1].t_ms)"
EQUIVALENT: Final[dict[tuple[str, str, str], str]] = {
    ("core/hub.py", _STUB, "False -> True"): "a default in a Protocol stub, which never runs",
    ("core/hub.py", _STUB, "0.0 -> 1.0"): "a default in a Protocol stub, which never runs",
    ("core/hub.py", "self.last_committed_seq = -1", "1 -> 2"): (
        "only compared against delta sequence numbers, which cannot be negative"
    ),
    ("core/hub.py", "self.reserved_delta_seq = -1", "1 -> 2"): (
        "max() with last_delta_seq, at least -1, makes -1 and -2 reserve the same number"
    ),
    ("core/patch_engine.py", _REPLAY_LOOP, "True -> False"): (
        "both sequences are built from the same operations, so their lengths always match"
    ),
    ("core/patch_engine.py", _DESCRIBE, "False -> True"): (
        "these flags add keys to each error, and only loc and msg are ever read"
    ),
    ("canvas/projection.py", "if speed <= MAX_SPEED_PX_PER_MS:", "LtE -> Lt"): (
        "at exactly the limit, scaling by limit / speed multiplies by 1.0"
    ),
    ("canvas/projection.py", "self._correction_at = 0.0", "0.0 -> 1.0"): (
        "the correction starts as a zero vector, so when it was set cannot matter"
    ),
    ("canvas/projection.py", "knot = _Knot(t_ms, position, Point(0.0, 0.0))", "0.0 -> 1.0"): (
        "the velocity is overwritten on the next line"
    ),
    ("canvas/projection.py", "if t_ms <= first.t_ms:", "LtE -> Lt"): (
        "at the first knot, interpolation from u = 0 returns its position and tangent too"
    ),
    ("canvas/projection.py", _SEGMENT, "LtE -> Lt"): (
        "at a knot, the end of one segment and the start of the next agree in value and slope"
    ),
}

_COMPARE_SWAPS: Final[dict[type[ast.cmpop], tuple[type[ast.cmpop], ...]]] = {
    ast.Lt: (ast.LtE, ast.GtE),
    ast.LtE: (ast.Lt, ast.Gt),
    ast.Gt: (ast.GtE, ast.LtE),
    ast.GtE: (ast.Gt, ast.Lt),
    ast.Eq: (ast.NotEq,),
    ast.NotEq: (ast.Eq,),
    ast.Is: (ast.IsNot,),
    ast.IsNot: (ast.Is,),
    ast.In: (ast.NotIn,),
    ast.NotIn: (ast.In,),
}
_BINARY_SWAPS: Final[dict[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
}


@dataclass(frozen=True, slots=True)
class Mutant:
    module: str
    index: int
    line: int
    description: str
    source: str

    @property
    def equivalence(self) -> str | None:
        return EQUIVALENT.get((self.module, self.source, self.description))


def _annotation_nodes(tree: ast.AST) -> set[int]:
    """Collect nodes inside annotations and docstrings, which no runtime behaviour depends on."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        annotations: list[ast.AST | None] = []
        if isinstance(node, ast.AnnAssign | ast.arg):
            annotations.append(node.annotation)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            annotations.append(node.returns)
        for annotation in annotations:
            if annotation is not None:
                skipped.update(id(child) for child in ast.walk(annotation))
    return skipped


class _Mutator(ast.NodeTransformer):
    """Walk the tree in a fixed order, applying the mutation numbered ``target`` only."""

    def __init__(self, target: int, skipped: set[int]) -> None:
        self.target = target
        self.skipped = skipped
        self.count = 0
        self.applied: tuple[int, str] | None = None

    def _site(self, node: ast.AST, description: str) -> bool:
        hit = self.count == self.target
        if hit:
            self.applied = (getattr(node, "lineno", 0), description)
        self.count += 1
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if id(node) in self.skipped:
            return node
        for position, operator in enumerate(node.ops):
            for replacement in _COMPARE_SWAPS.get(type(operator), ()):
                name = f"{type(operator).__name__} -> {replacement.__name__}"
                if self._site(node, name):
                    node.ops[position] = replacement()
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        replacement = _BINARY_SWAPS.get(type(node.op))
        if id(node) in self.skipped or replacement is None:
            return node
        if self._site(node, f"{type(node.op).__name__} -> {replacement.__name__}"):
            node.op = replacement()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if id(node) in self.skipped:
            return node
        swapped = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        if self._site(node, f"{type(node.op).__name__} -> {type(swapped).__name__}"):
            node.op = swapped
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        skip = id(node) in self.skipped or not isinstance(node.op, ast.Not)
        if not skip and self._site(node, "drop not"):
            return node.operand
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if id(node) in self.skipped:
            return node
        value = node.value
        if isinstance(value, bool):
            if self._site(node, f"{value} -> {not value}"):
                node = ast.copy_location(ast.Constant(not value), node)
        elif isinstance(value, int | float) and self._site(node, f"{value!r} -> {value + 1!r}"):
            node = ast.copy_location(ast.Constant(value + 1), node)
        return node


def _configuration(tree: ast.Module) -> set[int]:
    """Collect the values of module-level upper-case constants, which are configuration."""
    skipped: set[int] = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        value = getattr(node, "value", None)
        if value is not None and names and all(name.isupper() for name in names):
            skipped.update(id(child) for child in ast.walk(value))
    return skipped


def _parse(source: str) -> tuple[ast.Module, set[int]]:
    tree = ast.parse(source)
    skipped = _annotation_nodes(tree) | _configuration(tree)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        first = body[0] if isinstance(body, list) and body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            skipped.add(id(first.value))
    return tree, skipped


def mutants_of(module: str) -> Iterator[Mutant]:
    source = (ROOT / PACKAGE / module).read_text()
    tree, skipped = _parse(source)
    counter = _Mutator(-1, skipped)
    counter.visit(tree)
    lines = source.splitlines()
    for index in range(counter.count):
        _, (line, description) = mutate(source, index)
        yield Mutant(module, index, line, description, lines[line - 1].strip())


def mutate(source: str, index: int) -> tuple[str, tuple[int, str]]:
    tree, skipped = _parse(source)
    mutator = _Mutator(index, skipped)
    tree = mutator.visit(tree)
    assert mutator.applied is not None
    return ast.unparse(ast.fix_missing_locations(tree)), mutator.applied


def _prepare(worker: int) -> Path:
    copy = WORKSPACE / f"worker-{worker}"
    shutil.rmtree(copy, ignore_errors=True)
    copy.mkdir(parents=True)
    for name in ("src", "tests"):
        shutil.copytree(ROOT / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "pyproject.toml", copy / "pyproject.toml")
    return copy


def _run_tests(copy: Path, module: str, timeout_s: float) -> str:
    tests = TARGETS[module]
    command = [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:randomly"]
    command += [*tests]
    environment = {
        **os.environ,
        # The copy's source must win over the editable install of the original.
        "PYTHONPATH": str(copy / "src"),
        "PYTHONPYCACHEPREFIX": str(copy / ".pycache"),
    }
    try:
        result = subprocess.run(  # nosec B603: a fixed command over this repository's own tests
            command,
            cwd=copy,
            env=environment,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    return "survived" if result.returncode == 0 else "killed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("modules", nargs="*", default=list(TARGETS), help="paths below the package")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    parser.add_argument("--list", action="store_true", help="count mutants and stop")
    parser.add_argument("--min-score", type=float, default=0.0, help="fail below this kill rate")
    arguments = parser.parse_args()

    mutants = [mutant for module in arguments.modules for mutant in mutants_of(module)]
    for module in arguments.modules:
        print(f"{module:<28} {sum(m.module == module for m in mutants):>4} mutants")
    if arguments.list:
        return

    copies = [_prepare(worker) for worker in range(arguments.workers)]
    sources = {module: (ROOT / PACKAGE / module).read_text() for module in arguments.modules}
    baseline: dict[str, float] = {}
    for module in arguments.modules:
        started = time.perf_counter()
        if _run_tests(copies[0], module, 600) != "survived":
            raise SystemExit(f"the tests for {module} fail before any mutation")
        baseline[module] = time.perf_counter() - started

    outcomes: dict[Mutant, str] = {}
    free = list(copies)

    def run(mutant: Mutant) -> None:
        copy = free.pop()
        target = copy / PACKAGE / mutant.module
        try:
            target.write_text(mutate(sources[mutant.module], mutant.index)[0])
            timeout_s = max(baseline[mutant.module] * 5, 20.0)
            outcomes[mutant] = _run_tests(copy, mutant.module, timeout_s)
        finally:
            target.write_text(sources[mutant.module])
            free.append(copy)

    started = time.perf_counter()
    with ThreadPoolExecutor(arguments.workers) as pool:
        list(pool.map(run, mutants))
    elapsed = time.perf_counter() - started

    print()
    for module in arguments.modules:
        mine = [mutant for mutant in mutants if mutant.module == module]
        killed = sum(outcomes[m] != "survived" for m in mine)
        timeouts = sum(outcomes[m] == "timeout" for m in mine)
        print(f"{module:<28} {killed:>4}/{len(mine):<4} killed  ({timeouts} by timeout)")
    survivors = [mutant for mutant in mutants if outcomes[mutant] == "survived"]
    equivalent = [mutant for mutant in survivors if mutant.equivalence is not None]
    unexplained = [mutant for mutant in survivors if mutant.equivalence is None]
    killed_total = len(mutants) - len(survivors)
    raw = killed_total / len(mutants) if mutants else 1.0
    judged = len(mutants) - len(equivalent)
    adjusted = killed_total / judged if judged else 1.0
    print(f"{'raw':<28} {killed_total:>4}/{len(mutants):<4} killed  {raw:.2%} in {elapsed:.0f}s")
    print(
        f"{'excluding equivalent':<28} {killed_total:>4}/{judged:<4} killed  {adjusted:.2%}"
        f"  ({len(equivalent)} reviewed as equivalent)"
    )
    if equivalent:
        print("\nequivalent, by review:")
        for mutant in equivalent:
            print(f"  {PACKAGE / mutant.module}:{mutant.line}  {mutant.description}")
            print(f"      {mutant.equivalence}")
    if unexplained:
        print("\nsurvivors:")
        for mutant in unexplained:
            print(f"  {PACKAGE / mutant.module}:{mutant.line}  {mutant.description}")
            print(f"      {mutant.source}")
    if adjusted < arguments.min_score:
        raise SystemExit(
            f"kill rate {adjusted:.2%} is below the required {arguments.min_score:.2%}"
        )


if __name__ == "__main__":
    main()
