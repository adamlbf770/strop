# Example: routing support tickets

A labeled classification task. The dataset has 150 synthetic customer messages, 30 for each
category: `billing`, `refund`, `exchange`, `technical`, and `other`. The messages are written
to overlap on purpose. For example, "The jacket was the wrong size and I do not want another
one, just give me my money back" is a refund, while "The jacket arrived damaged, can you send
me a new one?" is an exchange. The starting prompt in `prompt.yaml` does not explain that
difference, which gives the optimizer something to find.

Files:

- `data.csv`: `ticket_id`, `customer_message`, `category` (the expected answer), `channel`.
- `prompt.yaml`: the starting prompt. Only the system message may be rewritten.

## Run it

You need an API key for the provider you choose, exported as the environment variable that
provider expects (for example `OPENAI_API_KEY`). Replace `<provider/model>` with any model
name LiteLLM supports.

```bash
cd examples/ticket_routing

strop init \
  --name ticket-routing \
  --task "Route a customer message to a support category" \
  --prompt prompt.yaml \
  --data data.csv --id-column ticket_id \
  --input customer_message --expected category --metadata channel \
  --output-type label --labels billing,refund,exchange,technical,other \
  --metric macro_f1 --constraint "valid_output_rate>=0.98" \
  --target-model <provider/model> --optimizer-model <provider/model>

strop data check
strop data split
strop baseline
strop optimize --max-turns 5
strop test --confirm
strop export
```

`strop optimize` prints a colored diff between the starting prompt and the final prompt when
it finishes. `strop export` writes the final prompt and a report to `exports/`.

The results depend on the models you choose, so this example does not list expected scores.
