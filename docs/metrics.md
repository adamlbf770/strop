# Metrics

A metric turns the outputs of a prompt into a number. Strop cannot know what a correct output is.
You define success with one of these, and Strop measures exactly that.

| Way to define success | Use it when | Trust |
| --- | --- | --- |
| Labeled data | You have an expected output per row (classification, extraction). | Deterministic. |
| Rules | You can state checks on the output (length, required words, format). | Deterministic. |
| A Python function | You can compute a score in code. | As good as your code. |
| A judge rubric | Quality needs judgment and you have no reference answers. | Subjective estimate. |

Every metric is computed on one partition at a time, and reports show the partition and the
number of examples next to each value.

## Choosing the primary metric

The primary metric is the one the loop tries to improve. Everything else is secondary or a
constraint:

- `secondary` metrics are reported only.
- `constraints` are limits a candidate must respect, written as `name: {min: x}` or
  `{max: x}`. A candidate that breaks one is rejected, even if the primary metric improves.

For classification, `macro_f1` weighs every class equally, which suits unbalanced data.
`accuracy` suits balanced data. Use `recall:<label>` as a constraint for classes you cannot
afford to miss.

## Built-in metrics

### Reference-based

These compare the output with the `expected` column. Invalid outputs count as wrong.

| Name | Meaning |
| --- | --- |
| `accuracy` | Share of outputs equal to the expected value. |
| `exact_match` | Same as accuracy, for any output type. |
| `macro_f1`, `micro_f1`, `weighted_f1` | F1 averaged over classes, pooled, or weighted by class size. |
| `macro_precision`, `micro_precision`, `weighted_precision` | Precision, averaged the same way. |
| `macro_recall`, `micro_recall`, `weighted_recall` | Recall, averaged the same way. |
| `precision:<label>`, `recall:<label>`, `f1:<label>` | Scores for one class. |
| `field_accuracy` | Average share of expected JSON fields whose value matches exactly. |
| `field_completeness` | Average share of expected JSON fields present in the output. |
| `field_f1` | Average F1 between the output's and the expected object's (field, value) pairs. |
| `similarity` | Average character-level similarity between output and expected text, from 0 to 1. |
| `numeric_close` | Share of outputs whose number is within `1e-6` of the expected number. |
| `numeric_close:<tolerance>` | The same with an absolute tolerance, for example `numeric_close:0.5`. |

Macro averages skip classes that have no examples and no predictions in the partition. Reports
for classification metrics include a per-class table and a confusion matrix in the stored data.

### Reference-free

| Name | Meaning |
| --- | --- |
| `valid_output_rate` | Share of outputs that fit the output contract. Always reported. |
| `json_valid` | Share of outputs that parse as JSON. |
| `schema_valid` | Share of outputs that parse as JSON and satisfy `output.schema_file`. |
| `rule:<name>` | Pass rate of one rule from `metrics.rules`. |
| `custom` | Your Python metric from `metrics.python`. |
| `judge:<name>` | A judge rubric from `metrics.judges`. Subjective. |

Metrics that need an expected value cannot be used when no dataset column has the role
`expected`. Strop tells you which flag to set.

## Failure categories

Every failed example gets one deterministic label. These are facts derived from the stored
outputs, not guesses.

| Label | Meaning |
| --- | --- |
| `provider_error` | The model call failed after retries. |
| `empty_output` | The model returned nothing. |
| `truncated` | The output was cut off by the token limit and could not be used. |
| `invalid_format` | The output does not fit the output contract (unknown label, invalid JSON). |
| `schema_violation` | The output is JSON but breaks the schema. |
| `wrong_label` | A valid label that is not the expected one. |
| `missing_field` | A JSON output lacks fields that the expected object has. |
| `wrong_value` | A valid output that differs from the expected value. |
| `rule_violation` | A correct output that fails one of your rules. |
| `low_score` | A valid output with a primary score below 1, for graded metrics. |

## Judge scores

A judge is another language model. It can be inconsistent, it can prefer its own style, and it
can be wrong. Strop treats its scores as estimates:

- Reports mark judge metrics as subjective.
- Judge scores are never mixed into deterministic metrics.
- The judge sees the task, your criteria, the input, the reference answer if there is one, and
  the model's answer. Everything from the dataset is passed as data, and the judge is told not to
  follow instructions found in it.
- Use a different model as judge than the target model when you can.
- Use `--sample` and read some outputs yourself before trusting a judged metric for a decision
  that matters.
