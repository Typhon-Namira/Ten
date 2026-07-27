# AI Provider Routing

TEN uses one provider-neutral reasoning boundary:

1. Cerebras is attempted first with model `gpt-oss-120b`.
2. Groq is the ordered fallback with model `llama-3.1-8b-instant`.
3. A 400 request-validation failure is terminal because the same invalid TEN payload must not be sent elsewhere.
4. Authentication, quota, rate-limit, model-availability, network, and 5xx failures may fall back.
5. Each provider receives at most one retry, and only for transient network or 5xx failures.

Cerebras uses strict JSON Schema response formatting. Groq's
`llama-3.1-8b-instant` uses JSON Object Mode because Groq does not list that model
for strict constrained JSON Schema output. The same application-side TEN schema
validator remains mandatory: missing fields, malformed JSON, and unexpected fields
are rejected. Groq receives at most one correction request after malformed JSON.

The required Railway variables are:

```text
TEN_CEREBRAS_API_KEY
TEN_CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
TEN_CEREBRAS_MODEL=gpt-oss-120b
TEN_GROQ_API_KEY
TEN_GROQ_BASE_URL=https://api.groq.com/openai/v1
TEN_GROQ_MODEL=llama-3.1-8b-instant
```

Remove superseded provider variables from the Railway service after this revision is deployed.
The application never logs API keys, authorization headers, prompts, compact analysis payloads,
or raw provider response bodies.

## Cycle idempotency

Reasoning is eligible for every synchronized UMS cycle, including successive one-minute
boundaries. The durable key is derived from:

```text
normalized instrument
+ exact UMS market-data boundary
+ UMS schema/cycle version
+ prompt, context, response-schema, and reasoning-policy contract
```

Duplicate delivery of the same immutable cycle reuses the persisted forecast. A later UMS
boundary creates a fresh key and may issue a fresh provider request. Dashboard endpoints only
read persisted state and cannot invoke the provider router.

## Operational states

Each provider reports one of `HEALTHY`, `STANDBY`, `RATE_LIMITED`, `QUOTA_EXHAUSTED`,
`AUTH_FAILED`, `UNAVAILABLE`, `CIRCUIT_OPEN`, or `UNCONFIGURED`, including its model, last
success, last failure, sanitized failure code, and circuit deadline. Rate-limit reset headers
control short circuits when present; daily quota exhaustion opens a longer circuit. Groq request
and token limits are tracked separately, and token exhaustion is terminal for that fallback
attempt even while request quota remains. When both providers fail, TEN persists the typed
terminal reason and remains fail-closed for publication.

## Deployment verification

Run:

```text
python -m alembic upgrade head
python scripts/ai_reasoning_smoke_test.py
```

Confirm logs show the selected provider/model, cycle boundary, attempt, fallback state, sanitized
result, latency, rate-limit metadata, and circuit transitions. The smoke script is shadow-only and
uses in-memory persistence.
