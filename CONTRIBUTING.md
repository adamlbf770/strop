# Contributing

Thanks for helping improve Strop. This page covers how to set up the project and what a change
should look like.

## Setup

Strop uses [uv](https://docs.astral.sh/uv/) and needs Python 3.11 or newer.

```bash
git clone https://github.com/baranylcn/strop.git
cd strop
uv sync
```

Run the checks before you open a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The same checks run in CI on Linux, Windows, and macOS. The tests use scripted models and make
no network calls.

## Releasing

1. Set the new version in `src/strop/__init__.py` and add a section for it to `CHANGELOG.md`.
   A test fails if the changelog has no entry for the current version.
2. Commit and push to `main`, and wait until CI is green.
3. Tag that exact commit and push the tag: `git tag vX.Y.Z` and `git push origin vX.Y.Z`. The
   release workflow refuses a tag that does not match `strop.__version__`.

## Project layout

```text
src/strop/
  cli/          Typer commands. Parse options, call a service, print the result.
  services/     Workflows used by the commands (evaluation, the optimization loop, export).
  config/       Typed configuration for strop.yaml.
  data/         Dataset loading, validation, and splitting.
  prompts/      Prompt files, rendering, revision rules, and diffs.
  metrics/      Metrics, the registry that resolves their names, and the metric suite.
  evaluation/   Running a prompt over rows, output parsing, budgets.
  analysis/     Failure classification and LLM-proposed patterns.
  optimization/ The optimizer meta-prompt, candidate validation, and acceptance rules.
  llm/          The client interface, LiteLLM client, retries, and caching.
  storage/      SQLite access and numbered SQL migrations.
  reporting/    Terminal and file output.
tests/
  unit/         Mirrors src/strop.
  integration/  Whole commands, run with scripted models.
examples/       Runnable example projects.
docs/           User documentation.
```

## Conventions

- Keep modules small and focused, and keep the `cli` layer free of business logic.
- The `optimization` and `analysis` packages must not import partitions or storage. A test
  enforces this, because it is what keeps the test set out of the optimizer.
- Add a migration file in `src/strop/storage/migrations/` for every schema change. Never edit a
  migration that has been released.
- Every change comes with tests. Use the scripted models in `tests/integration/conftest.py`
  instead of calling a real provider.
- Do not invent numbers. Costs and token counts come from providers; when they are missing, show
  them as unavailable.
- Commit messages are one line in the form `type: summary`, for example `fix: keep test rows out
  of the optimizer view`. Types: feat, fix, perf, refactor, docs, test, build, ci, chore.
- Documentation avoids marketing language and describes only what exists.
