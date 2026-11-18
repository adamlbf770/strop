# Strop

Strop is a command-line tool, in early development, that will improve LLM prompts by measuring them against your own dataset and keeping a revision only when the measured results get better.

## Status

Pre-alpha. The package currently contains project scaffolding only: no commands, no evaluation, no optimization. It is not published on PyPI yet. Sections marked "Planned" describe the intended behavior and are not implemented.

## Planned behavior

Strop cannot know what a correct output is. You define it with labeled data, rule checks, a Python function, or an LLM-judge rubric, and Strop measures your definition.

Planned workflow:

1. Run your prompt on a validation set and score the outputs with your metric. This is the baseline.
2. Collect the failures from a separate training set and group them by error type.
3. Ask an optimizer model for revised prompts that address those failures.
4. Test each revision on the same validation rows. Keep a revision only if the primary metric improves and no constraint you set is broken.
5. Repeat for a fixed number of turns (`--max-turns`).
6. Evaluate the final prompt once on a held-out test set and show a diff against the original prompt.

Design goals:

- Runs locally from the command line. No server, no account, no telemetry.
- Works with any model that [LiteLLM](https://docs.litellm.ai/docs/providers) can call. You choose the models and provide the API keys through environment variables.
- Dataset rows are sent only to the model providers you configure.
- Every prompt version, dataset version, and result is stored locally, so a run can be reproduced and inspected.

## Install

Strop is not on PyPI yet. To work with the current source you need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/baranylcn/strop.git
cd strop
uv sync
uv run python -c "import strop; print(strop.__version__)"
```

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## License

MIT. See [LICENSE](https://github.com/baranylcn/strop/blob/main/LICENSE).
