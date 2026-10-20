Requesty provider installed.

1. Add your key to `~/.hermes/.env` (get one at https://app.requesty.ai/api-keys):

   ```
   REQUESTY_API_KEY=<your key>
   ```

2. Pick a model: run `hermes model` and choose **Requesty**, or launch with

   ```
   hermes --provider requesty -m claude-sonnet-5-5
   ```

   The picker lists Requesty's managed policies — curated routes with
   automatic failover. Append `@eu` (e.g. `claude-sonnet-5-5@eu`) to stay on
   EU-hosted providers. Any model id (`anthropic/claude-sonnet-5-5`) or your
   own policy (`policy/my-policy`) also works.

3. Optional: `REQUESTY_BASE_URL=https://router.eu.requesty.ai/v1` routes all
   traffic through Requesty's EU endpoint.
