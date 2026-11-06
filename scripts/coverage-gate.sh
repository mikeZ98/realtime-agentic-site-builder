#!/usr/bin/env bash
# Runs the test suite under coverage and enforces two gates:
#   1. the absolute floor, [tool.coverage.report] fail_under in pyproject.toml;
#   2. zero regression against the committed baseline in scripts/coverage-baseline.txt.
# --update-baseline ratchets the baseline to the current value. It can only raise it, because a
# regression fails first; lowering it takes a deliberate, reviewable manual edit.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PYTHONPYCACHEPREFIX="./.cache/pycache"

readonly BASELINE_FILE="scripts/coverage-baseline.txt"
readonly REPORT_FILE=".cache/coverage/coverage.json"

update_baseline=false
case "${1:-}" in
    "") ;;
    --update-baseline) update_baseline=true ;;
    *)
        echo "usage: $0 [--update-baseline]" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "$REPORT_FILE")"
rm -f "$REPORT_FILE"
uv run --locked pytest --cov --cov-report=term-missing --cov-report=json

current="$(
    uv run --locked python -c \
        'import json, sys; print(format(json.load(open(sys.argv[1]))["totals"]["percent_covered"], ".2f"))' \
        "$REPORT_FILE"
)"

if [[ -f "$BASELINE_FILE" ]]; then
    baseline="$(tr -d '[:space:]' <"$BASELINE_FILE")"
    if ! awk -v c="$current" -v b="$baseline" 'BEGIN { exit !(c + 0 >= b + 0) }'; then
        echo "coverage-gate: FAIL, coverage regressed from ${baseline}% to ${current}%" >&2
        exit 1
    fi
elif [[ "$update_baseline" == false ]]; then
    echo "coverage-gate: FAIL, $BASELINE_FILE is missing; create it with --update-baseline" >&2
    exit 1
else
    baseline="none"
fi

if [[ "$update_baseline" == true ]]; then
    printf '%s\n' "$current" >"$BASELINE_FILE"
    echo "coverage-gate: baseline updated from ${baseline} to ${current}%"
    exit 0
fi

echo "coverage-gate: OK, ${current}% (baseline ${baseline}%)"
if [[ "$current" != "$baseline" ]]; then
    echo "coverage-gate: coverage improved; run with --update-baseline to lock it in"
fi
