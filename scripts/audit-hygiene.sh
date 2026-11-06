#!/usr/bin/env bash
# Audits the working tree (tracked and untracked, non-ignored files) for absolute machine paths,
# leaked credentials and dotenv files, and for tool caches living outside .cache/.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
# shellcheck source=../.githooks/_policy.sh
source .githooks/_policy.sh

status=0

policy_scan "absolute machine path" "$ABSOLUTE_PATH_RE" --untracked || status=1
policy_scan "credential token" "$TOKEN_RE" --untracked || status=1
policy_scan "hardcoded credential" "$CREDENTIAL_ASSIGNMENT_RE" -i --untracked || status=1

while IFS= read -r -d '' file; do
    if policy_is_dotenv "${file##*/}"; then
        policy_error "dotenv file is tracked or unignored: $file"
        status=1
    fi
done < <(git ls-files --cached --others --exclude-standard -z)

stray_caches="$(
    find . \( -path ./.git -o -path ./.cache -o -path ./.venv -o -name node_modules \) -prune -o \
        \( -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \
        -o -name .ruff_cache -o -name htmlcov \) -o -type f -name .coverage \) -print
)"
if [[ -n "$stray_caches" ]]; then
    policy_error "caches outside .cache/ (is direnv loaded?):"
    sed 's/^/    /' <<<"$stray_caches" >&2
    status=1
fi

if [[ "$status" -eq 0 ]]; then
    echo "audit-hygiene: OK"
fi
exit "$status"
