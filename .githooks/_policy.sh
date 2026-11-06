#!/usr/bin/env bash
# Repository policy: sole-author identity and staged-content hygiene.
# Executed by pre-commit. Sourced by commit-msg and scripts/audit-*.sh for its rules.
set -euo pipefail

readonly EXPECTED_NAME="mikeZ98"
readonly EXPECTED_EMAIL="145975972+mikeZ98@users.noreply.github.com"
readonly CONVENTIONAL_COMMIT_RE='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9_-]+\))?: .+$'
readonly FORBIDDEN_TRAILER_RE='^[[:space:]]*co-authored-by:|generated (with|by) \[?(claude|copilot|cursor|codex|aider|chatgpt)|noreply@anthropic\.com'

# Written so that none of these patterns match their own source text.
readonly ABSOLUTE_PATH_RE='(^|[^A-Za-z0-9._-])/(home|Users|Volumes)/[A-Za-z0-9._-]'
readonly TOKEN_RE='AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}|(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|-----BEGIN ([A-Z]+ )?PRIVATE KEY-----'
readonly CREDENTIAL_ASSIGNMENT_RE="(password|passwd|secret|api[_-]?key|auth[_-]?token|access[_-]?token)[A-Za-z0-9_-]*[\"']?[[:space:]]*[:=][[:space:]]*[\"'][^\"'[:space:]]{8,}[\"']"

policy_error() {
    printf 'policy: %s\n' "$*" >&2
}

# Succeeds when the given basename is a dotenv file that must never be committed.
policy_is_dotenv() {
    case "$1" in
        .env.example) return 1 ;;
        .env | .env.*) return 0 ;;
        *) return 1 ;;
    esac
}

# policy_scan <label> <regex> <git-grep args...>
# Reports file:line only, so that a matched secret is never echoed to the terminal or CI logs.
policy_scan() {
    local label="$1" regex="$2" hits rc=0
    shift 2
    hits="$(git --literal-pathspecs grep -nIE -e "$regex" "$@")" || rc=$?
    case "$rc" in
        0)
            policy_error "$label found:"
            cut -d: -f1,2 <<<"$hits" | sed 's/^/    /' >&2
            return 1
            ;;
        1) return 0 ;;
        *)
            policy_error "git grep failed while scanning for $label (exit $rc)"
            return 1
            ;;
    esac
}

# policy_check_message <message>: Conventional Commits subject, no tool-generated trailers.
policy_check_message() {
    local message="$1" subject status=0
    subject="$(grep -m1 '[^[:space:]]' <<<"$message" || true)"
    if ! grep -Eq "$CONVENTIONAL_COMMIT_RE" <<<"$subject"; then
        policy_error "subject is not a Conventional Commit: '$subject'"
        status=1
    fi
    if grep -iEq "$FORBIDDEN_TRAILER_RE" <<<"$message"; then
        policy_error "message contains a co-author or tool-generated attribution trailer"
        status=1
    fi
    return "$status"
}

policy_check_identity() {
    local role ident name email status=0
    for role in AUTHOR COMMITTER; do
        # git exports --author overrides to hooks, so `git var` sees the identity actually used.
        ident="$(git var "GIT_${role}_IDENT")"
        name="${ident%% <*}"
        email="${ident#*<}"
        email="${email%%>*}"
        if [[ "$name" != "$EXPECTED_NAME" ]]; then
            policy_error "$role name is '$name', expected '$EXPECTED_NAME'"
            status=1
        fi
        if [[ "$email" != "$EXPECTED_EMAIL" ]]; then
            policy_error "$role email is '$email', expected '$EXPECTED_EMAIL'"
            status=1
        fi
    done
    return "$status"
}

# Scans index content rather than the worktree: what gets committed is what is checked.
policy_check_staged() {
    local file status=0
    local -a files=()
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(git diff --cached --name-only --diff-filter=ACMR -z)
    if [[ "${#files[@]}" -eq 0 ]]; then
        return 0
    fi

    for file in "${files[@]}"; do
        if policy_is_dotenv "${file##*/}"; then
            policy_error "dotenv file staged: $file"
            status=1
        fi
    done
    policy_scan "absolute machine path" "$ABSOLUTE_PATH_RE" --cached -- "${files[@]}" || status=1
    policy_scan "credential token" "$TOKEN_RE" --cached -- "${files[@]}" || status=1
    policy_scan "hardcoded credential" "$CREDENTIAL_ASSIGNMENT_RE" -i --cached -- "${files[@]}" ||
        status=1
    return "$status"
}

policy_main() {
    local status=0
    policy_check_identity || status=1
    policy_check_staged || status=1
    return "$status"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    policy_main
fi
