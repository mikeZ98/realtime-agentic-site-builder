# realtime-agentic-site-builder

Asynchronous, real-time multi-agent site builder and swarm simulation gateway.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, optionally, [direnv](https://direnv.net/).

```sh
uv sync
git config core.hooksPath .githooks
direnv allow
```

Git does not carry `core.hooksPath` across clones, so run the second command in every new clone.

## Quality gates

```sh
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests
uv run pytest
scripts/coverage-gate.sh       # add --update-baseline to ratchet after coverage improves
scripts/audit-attribution.sh
scripts/audit-hygiene.sh
```
