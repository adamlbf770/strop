# How Strop works

## The three partitions

`strop data split` shuffles your rows with a seed and stores the result, so the same data and
settings always give the same split.

| Partition | Used for |
| --- | --- |
| `train` | Finding failures. The analyzer and the optimizer only see failed train examples. |
| `validation` | Comparing prompts. Every candidate is scored on the same validation rows. |
| `test` | One final check with `strop test --confirm`. |

The test partition is locked in code, not only in the command line:

- The evaluation service refuses the test partition unless it was created for `strop test`.
- The optimizer and analysis packages have no access to partitions or storage. They receive a
  fixed view of train failures and validation scores. A test checks that they cannot import
  either.
- `strop test` asks for `--confirm` and refuses a second run without `--allow-rerun`, because
  tuning the prompt after looking at the test result makes the result optimistic.

The split is saved with the dataset's hash. If the data file changes later, Strop asks you to
split again instead of mixing old and new rows.

## One turn of the loop

`strop optimize --max-turns N` repeats the following up to N times:

1. Take the best prompt so far and score it on validation. A stored result is reused when the
   prompt, model, output contract, and metrics have not changed.
2. Run it on the train rows and classify every failure.
3. Ask the analyzer for failure patterns. These are hypotheses, and each must cite example ids.
4. Ask the optimizer for `--candidates` revised prompts. Each reply must be valid JSON and must
   cite example ids. A reply that edits a frozen message, drops a `{{variable}}`, changes a
   `<strop:fixed>` span, or does not change the prompt is discarded and recorded with its reason.
5. Score each valid candidate on the validation rows.
6. Apply the acceptance rules. Of the candidates that pass, the one with the largest gain wins.
7. The winner becomes the best prompt (or waits for approval), and the next turn starts from it.

Every candidate is stored as a prompt version, including rejected and invalid ones, together
with its summary, evidence, and the exact reason for the decision. `strop history` lists them.

## Acceptance rules

A candidate becomes the best prompt only if all of these hold:

- The validation partition has at least `min_validation_samples` rows.
- The primary metric improves by at least `min_improvement`.
- Every constraint holds. If the best prompt already violates a constraint, the candidate may
  not make it worse.
- The valid output rate does not fall by more than `max_validity_regression`.
- No class loses more than `max_class_regression` of its F1.
- With `require_paired_wins`, the candidate wins on more validation examples than it loses.

If a candidate fails, every failed rule is listed in its reason. The best prompt stays as it was.

## Approval and resuming

With `--approval manual`, the winning candidate is stored as pending and the command exits with
code 10 after printing the next steps. Review it with `strop diff v4`, then run
`strop approve v4` or `strop reject v4`, and continue with `strop optimize --resume`.

A run that is stopped by a budget (exit code 11) or by Ctrl-C (exit code 12) is saved after every
candidate. `strop optimize --resume` continues the unfinished turn without repeating work that
is stored or cached.

## When a run stops

The final report always says why:

| Reason | Meaning |
| --- | --- |
| `max_turns` | The turn limit was reached. |
| `target_reached` | The primary metric reached `--target` on validation. |
| `patience` | `--patience` turns in a row produced no accepted prompt. |
| `no_failures` | The best prompt has no failures left on the train rows. |
| `budget` | The call, cost, or time budget was reached. |
| `awaiting_approval` | A candidate is waiting for `strop approve` or `strop reject`. |

When a run ends, Strop prints the metric before and after and a colored diff between the
initial prompt (v1) and the final best prompt.

## Prompt versions

The prompt file is registered as `v1` on the first run. Versions are immutable: a database
trigger refuses updates and deletes. `strop rollback v2` only moves the pointer to the best
version. Your own edits are added with `strop prompt add --file edited.yaml --use`. If you edit
the prompt file without registering it, Strop warns that it differs from every stored version.

## Caching and repeatability

- Responses at temperature 0 are cached by model, messages, and settings. Responses at a higher
  temperature are never cached, so repeated sampling stays varied.
- `--no-cache` skips the cache for one command. `strop clean --cache --yes` empties it.
- `strop baseline --repeats N` runs the baseline N times; runs after the first skip the cache and
  report how much the primary metric moved between runs.
- Each stored evaluation records the model and a fingerprint of the settings that affect it.

## Local data

Everything lives in your project directory.

| Path | Content |
| --- | --- |
| `strop.yaml` | Your configuration. |
| `.strop/strop.db` | SQLite database: splits, prompt versions, evaluations, per-example results, runs, candidates, cached responses. |
| `exports/` | Written by `strop export`. |

`.strop/` contains model outputs and cached responses for your dataset rows. Keep it, and your
`.env` file, out of version control. API keys are read from the environment or from `.env` and
are never written to the database, logs, or reports.

## Failure handling

- A failed model call is recorded as `provider_error` for that example and does not stop the
  run. Temporary errors (rate limits, timeouts, outages) are retried with exponential backoff.
- Rate-limit pressure is bounded by `max_concurrency`.
- Budgets are checked between batches, so a run can finish the current batch before it stops.
