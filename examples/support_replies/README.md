# Example: answering customer questions

A task with no expected answers. The dataset has 60 synthetic customer questions. Quality is
measured without reference answers:

- Two rubrics in `rubric.yaml` are scored by a judge model: `judge:helpful` and `judge:polite`.
  Judge scores are subjective estimates, and Strop labels them that way.
- Two rules check each answer deterministically: `short` (at most 120 words) and `no_promises`
  (must not contain "guarantee").
- `metric.py` is an optional custom metric written in Python. It runs in a separate process.

Files:

- `data.csv`: `question_id`, `question`, `topic`.
- `prompt.yaml`: the starting prompt. Only the system message may be rewritten.
- `rubric.yaml`: the judge rubrics.
- `metric.py`: the custom metric, named `custom` in reports.

## Run it

Export the API keys your providers need. Use a judge model that is different from the target
model if you can, because a model tends to rate its own style highly.

```bash
cd examples/support_replies

strop init \
  --name support-replies \
  --task "Answer a customer question for an online shop" \
  --prompt prompt.yaml \
  --data data.csv --id-column question_id \
  --input question --metadata topic \
  --output-type text \
  --rubric-file rubric.yaml --metric-file metric.py \
  --rule short=max_words:120 \
  --rule 'no_promises=not_matches:(?i)guarantee' \
  --metric judge:helpful --secondary-metric judge:polite \
  --constraint "rule:short>=0.95" \
  --target-model <provider/model> \
  --optimizer-model <provider/model> \
  --judge-model <provider/model>

strop data split
strop baseline
strop optimize --max-turns 5
strop test --confirm
```

Each judged example also gets a short reason from the judge. Those reasons are passed to the
optimizer as evidence, together with the failed rules and the custom metric's feedback.
