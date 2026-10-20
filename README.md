# hermes-provider-requesty

[Requesty](https://requesty.ai) as a native model provider for
[Hermes Agent](https://github.com/NousResearch/hermes-agent).

```bash
hermes plugins install requestyai/hermes-provider-requesty#requesty
hermes plugins enable requesty
echo 'REQUESTY_API_KEY=<your key>' >> ~/.hermes/.env   # https://app.requesty.ai/api-keys
hermes --provider requesty -m claude-sonnet-5-5
```

Or run `hermes model` and pick **Requesty**.

## What you get

| | |
|---|---|
| **Model picker** | Requesty's [managed policies](https://docs.requesty.ai/api-reference/endpoint/models-managed-list) (`claude-sonnet-5-5`, `gpt-6.1-sol`, `gemini-3.8-flash`, …): curated routes with automatic failover. Add `@eu` to stay on EU-hosted providers. Any model id (`anthropic/claude-sonnet-5-5`) or org policy (`policy/my-policy`) also works with `-m`. |
| **Reasoning** | `/reasoning` and `agent.reasoning_effort` map onto Requesty's `reasoning_effort`, sent only to models the catalog marks as reasoning-capable. Defaults to `medium`. |
| **Prompt caching** | Done by Requesty server-side. Nothing to configure. |
| **Tracing** | Each Hermes session is sent as `requesty.trace_id`, so a conversation groups together in the Requesty dashboard. |
| **Context and cost** | Context windows and per-token prices (tiered and cache rates included) come from the Requesty catalog, so compression and `/usage` are accurate. |

## Configuration

| Variable | |
|---|---|
| `REQUESTY_API_KEY` | Required. |
| `REQUESTY_BASE_URL` | Optional. `https://router.eu.requesty.ai/v1` for the EU endpoint, or a proxy. |

Put them in `~/.hermes/.env`. Alias: `--provider rq`.

## Data disclosure

Prompts, tool calls and your `REQUESTY_API_KEY` are sent to `router.requesty.ai`
(or `REQUESTY_BASE_URL`), along with the Hermes session id as a trace id. The
plugin also reads Requesty's public model catalogs without credentials. It sends
no attribution headers and no telemetry.

## Development

```bash
python -m unittest discover -s tests   # offline; stubs the Hermes runtime
ruff check requesty tests && ruff format --check requesty tests
hermes plugins validate ./requesty
```

## License

MIT
