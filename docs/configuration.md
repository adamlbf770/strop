# Configuration reference

`strop init` writes `strop.yaml` in your project directory. Edit it by hand any time. Strop
rejects unknown keys, so a typo is reported instead of ignored. Command-line flags override the
values in this file for a single run.

Paths in `strop.yaml` are relative to the project directory.

## Complete example

```yaml
project:
  name: ticket-routing
  task_description: Route a customer message to a support category

prompt:
  file: prompt.yaml

dataset:
  path: data.csv
  id_column: ticket_id
  columns:
    customer_message: input
    category: expected
    channel: metadata

output:
  type: label
  labels: [billing, refund, exchange, technical, other]

models:
  target:
    model: <provider/model>
    temperature: 0.0
  optimizer:
    model: <provider/model>

metrics:
  primary: macro_f1
  secondary: [accuracy]
  constraints:
    valid_output_rate: {min: 0.98}
    recall:refund: {min: 0.80}

split:
  train: 0.6
  validation: 0.2
  test: 0.2
  seed: 42

optimization:
  max_turns: 5
  candidates_per_turn: 3
  patience: 2
  approval: auto

budgets:
  max_total_model_calls: 500
  max_wall_time_minutes: 60

execution:
  batch_size: 10
  max_concurrency: 3
  retry_attempts: 3
```

Only `project`, `dataset`, `output`, `models.target`, and `metrics.primary` are required. Every
other value has the default shown in the tables below.

## project

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | Project name. |
| `task_description` | required | One sentence describing what the prompt must do. The optimizer and the judge see it. |

## prompt

| Key | Default | Meaning |
| --- | --- | --- |
| `file` | `prompt.yaml` | The starting prompt. See [Prompt files](#prompt-files). |

## dataset

| Key | Default | Meaning |
| --- | --- | --- |
| `path` | required | CSV or JSONL file. |
| `format` | from the extension | `csv` or `jsonl`. Extensions `.csv`, `.jsonl`, and `.ndjson` are recognized. |
| `id_column` | none | Column with a unique id per row. Without it, rows are numbered by position. |
| `columns` | required | Maps column names to roles: `input`, `expected`, `metadata`, or `ignore`. |

Rules for `columns`:

- At least one column must be `input`. Input column names must be usable as prompt variables:
  letters, digits, and underscores, not starting with a digit.
- At most one column may be `expected`. It is required by metrics that compare with a reference
  (accuracy, F1, exact match, field metrics, similarity).
- `metadata` columns are kept with the row but never sent to a model.
- Columns that are not listed are ignored.

Files must be UTF-8. JSONL values that are not strings are stored as JSON text, so an expected
JSON object in a JSONL file works for `json` outputs.

Invalid rows (empty inputs, empty expected values, labels outside `output.labels`, expected
values that are not valid JSON for `json` outputs, duplicate ids) are reported with their row
numbers. Commands stop on invalid rows unless you pass `--skip-invalid` to `strop data split`.

## output

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | required | `label`, `json`, or `text`. |
| `labels` | none | Allowed labels. Required for `label` (at least two, no duplicates). |
| `schema_file` | none | JSON Schema file the output must satisfy. Only for `json`. |
| `match` | `strict` | For `label` only. `strict` accepts the label alone, ignoring case, quotes, and a final period. `contains` also accepts an output that contains exactly one label as a whole word. |

An output that does not fit the contract is recorded as an invalid output. It is never dropped.
For `json`, a JSON object inside a Markdown code fence is accepted.

## models

Each role takes a model name that LiteLLM understands, such as `<provider>/<model>`. Strop has no
default model.

| Role | Required | Purpose |
| --- | --- | --- |
| `target` | yes | The model the prompt is optimized for. |
| `optimizer` | for `optimize` | Proposes revised prompts. |
| `analyzer` | no | Proposes failure patterns. Falls back to `optimizer`. |
| `judge` | for judge metrics | Scores judge rubrics. |

Settings for each role:

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | required | Model name. |
| `temperature` | `0.0` | Between 0 and 2. Responses are only cached at temperature 0. |
| `max_tokens` | none | Output token limit. |
| `timeout_s` | `60` | Seconds to wait for one call. |
| `params` | `{}` | Extra keyword arguments passed to LiteLLM, for example `api_base`. |

Parameters a model does not support are dropped instead of causing an error.

## metrics

| Key | Default | Meaning |
| --- | --- | --- |
| `primary` | required | The metric the loop tries to improve. |
| `secondary` | `[]` | Metrics that are reported but do not decide anything. |
| `constraints` | `{}` | Limits a candidate must respect, as `name: {min: x}` or `{max: x}`. |
| `rules` | `{}` | Deterministic output checks. Each becomes the metric `rule:<name>`. |
| `python` | none | A custom metric file. It becomes the metric `custom`. |
| `judges` | `{}` | Judge rubrics. Each becomes the metric `judge:<name>`. |

The metrics `valid_output_rate`, every rule, `custom`, and every judge are always reported.
See [Metrics](metrics.md) for all metric names.

### rules

```yaml
metrics:
  rules:
    short: {type: max_words, value: 120}
    polite: {type: not_matches, value: "(?i)stupid|idiot"}
    complete: {type: contains_all, value: [refund, days]}
```

| Type | Value | Passes when |
| --- | --- | --- |
| `max_chars`, `min_chars` | whole number | the output has at most or at least that many characters |
| `max_words`, `min_words` | whole number | the output has at most or at least that many words |
| `matches` | regular expression | the expression is found in the output |
| `not_matches` | regular expression | the expression is not found in the output |
| `contains_all` | list of strings | every string is present, ignoring case |
| `contains_none` | list of strings | no string is present, ignoring case |

Rules check the raw model output. An empty output fails every rule. Outputs lost to provider
errors are not scored.

### python

```yaml
metrics:
  python:
    file: metric.py
    timeout_s: 120
    direction: maximize
```

`metric.py` defines `score(output, row)`. `output` is the raw model text. `row` is a dictionary
with `inputs` (the input columns), `expected` (or `None`), and `value` (the parsed output, or
`None` if it was invalid). Return a number from 0 to 1, a bool, or
`{"score": 0.5, "feedback": "short explanation"}`. The feedback is shown to the optimizer.

The file runs in a separate Python process with a timeout. It is your own code, so Strop runs it
as written. Dataset content is passed to it as data and is never executed. Anything the file
prints is ignored.

### judges

```yaml
metrics:
  judges:
    helpful:
      criteria: The answer gives the customer a clear next step.
      scale_min: 1
      scale_max: 5
      instructions: ""
```

A judge model scores each output as an integer from `scale_min` to `scale_max` and gives a short
reason. Scores are normalized to 0 to 1. Judge scores are estimates and are labeled as subjective
in every report. Judge metrics need `models.judge`.

## split

| Key | Default | Meaning |
| --- | --- | --- |
| `train` | `0.6` | Share of rows used for failure analysis and optimizer evidence. |
| `validation` | `0.2` | Share of rows used to compare prompts. |
| `test` | `0.2` | Share of rows kept for the final check. May be `0`. |
| `seed` | `42` | Seed for the shuffle. |
| `stratify` | `true` | Keep class proportions the same in every partition. |

The ratios must sum to 1. Stratifying only applies when the expected values are class-like. If
most expected values are unique (free text, JSON objects, numbers), the split is random.

## optimization

| Key | Default | Meaning |
| --- | --- | --- |
| `max_turns` | `5` | Turns per run. |
| `candidates_per_turn` | `3` | Revised prompts proposed per turn. |
| `target` | none | Stop when the primary metric reaches this value. |
| `patience` | `2` | Stop after this many turns in a row without an accepted prompt. |
| `approval` | `auto` | `auto` promotes the winner. `manual` pauses for `strop approve` or `strop reject`. |
| `min_improvement` | `0.01` | Smallest primary metric gain a candidate needs. |
| `train_sample` | none | Analyze a seeded sample of this many train rows per turn. |
| `instructions` | `""` | Extra guidance for the optimizer. |
| `evidence_examples` | `12` | Failed examples shown to the analyzer and optimizer. |
| `evidence_chars` | `400` | Each text field of an example is cut to this length. |
| `max_class_regression` | `0.10` | Largest allowed drop in any class F1. `null` turns the check off. |
| `max_validity_regression` | `0.02` | Largest allowed drop in the valid output rate. |
| `min_validation_samples` | `10` | Fewest validation rows needed to compare prompts. |
| `require_paired_wins` | `true` | The candidate must win on more examples than it loses. |

## budgets

| Key | Default | Meaning |
| --- | --- | --- |
| `max_total_model_calls` | `500` | Fresh model calls per run. Cached answers are free and are not counted. `null` turns the limit off. |
| `max_total_cost_usd` | none | Cost limit. Only calls that report a cost count towards it. |
| `max_wall_time_minutes` | `60` | Time limit per run. `null` turns the limit off. |

## execution

| Key | Default | Meaning |
| --- | --- | --- |
| `batch_size` | `10` | Rows per batch. Budgets are checked between batches. |
| `max_concurrency` | `3` | Simultaneous model calls. |
| `retry_attempts` | `3` | Attempts for a call that fails with a temporary error. |

## Prompt files

A prompt file is YAML with chat messages:

```yaml
messages:
  - role: system
    content: |-
      You route customer messages to a support team.
      Reply with the category only.
  - role: user
    content: |-
      Message: {{customer_message}}
optimize:
  - system
```

- Roles are `system`, `user`, and `assistant`.
- `{{name}}` is replaced by the input column called `name`. Every variable must match an input
  column.
- `optimize` lists the roles the optimizer may rewrite. Other messages are never changed. It
  defaults to `system` if there is a system message, otherwise `user`.
- Wrap text that must never change in `<strop:fixed>...</strop:fixed>`. The optimizer must keep
  it exactly, and the markers are removed before the prompt is sent to a model.
- A file ending in `.md` or `.txt` is treated as a single user message that the optimizer may
  rewrite.

Your prompt file is never overwritten. Every revision is stored as a new immutable version in
`.strop/`.
