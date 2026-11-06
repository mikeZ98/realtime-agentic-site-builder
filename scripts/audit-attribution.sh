#!/usr/bin/env bash
# Audits commit history on all branches, tags and remotes: sole author and committer, linear
# history, Conventional Commits, no co-author trailers. Catches anything that bypassed the hooks.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
# shellcheck source=../.githooks/_policy.sh
source .githooks/_policy.sh

if ! git rev-parse -q --verify HEAD >/dev/null; then
    echo "audit-attribution: no commits yet"
    exit 0
fi

status=0
count=0
while IFS= read -r sha; do
    count=$((count + 1))
    IFS=$'\x1f' read -r author_name author_email committer_name committer_email parents \
        <<<"$(git log -1 --format='%an%x1f%ae%x1f%cn%x1f%ce%x1f%P' "$sha")"
    problems=""
    [[ "$author_name" == "$EXPECTED_NAME" ]] || problems+=" author-name='$author_name'"
    [[ "$author_email" == "$EXPECTED_EMAIL" ]] || problems+=" author-email='$author_email'"
    [[ "$committer_name" == "$EXPECTED_NAME" ]] || problems+=" committer-name='$committer_name'"
    [[ "$committer_email" == "$EXPECTED_EMAIL" ]] || problems+=" committer-email='$committer_email'"
    [[ "$parents" != *" "* ]] || problems+=" merge-commit"
    if [[ -n "$problems" ]]; then
        policy_error "${sha:0:12}:$problems"
        status=1
    fi
    if ! policy_check_message "$(git log -1 --format=%B "$sha")"; then
        policy_error "${sha:0:12}: invalid commit message"
        status=1
    fi
done < <(git rev-list --branches --tags --remotes)

if [[ "$status" -eq 0 ]]; then
    echo "audit-attribution: OK ($count commits)"
fi
exit "$status"
