# AI Provider Routing

TEN uses one four-account Groq reasoning pool. Each eligible five-minute job walks
the accounts in fixed order and stops after the first valid response:

```text
groq_1 -> groq_2 -> groq_3 -> groq_4
```

Each account has an independent state and cooldown. Authentication, configuration,
quota, rate-limit, and exhausted retryable transport failures advance to the next
account. Only transport errors and HTTP 408/5xx receive the configured bounded retry
on the same account. JSON or analysis-schema failures remain validation failures and
never masquerade as quota or transport errors.

Groq receives JSON Object Mode requests. TEN's strict application schema remains
authoritative: missing fields, malformed JSON, and unexpected properties are rejected.
One bounded correction request is allowed on the same account.

## Environment

```text
TEN_AI_PRIMARY_PROVIDER=groq
TEN_GROQ_POOL_ENABLED=true
TEN_GROQ_POOL_SIZE=4
TEN_GROQ_API_KEY_1
TEN_GROQ_API_KEY_2
TEN_GROQ_API_KEY_3
TEN_GROQ_API_KEY_4
TEN_GROQ_BASE_URL=https://api.groq.com/openai/v1
TEN_GROQ_MODEL=gpt-oss-120b
TEN_GROQ_REQUEST_TIMEOUT_SECONDS=60
TEN_GROQ_MAX_RETRIES_PER_ACCOUNT=1
TEN_GROQ_RATE_LIMIT_COOLDOWN_SECONDS=3600
TEN_GROQ_QUOTA_COOLDOWN_SECONDS=86400
TEN_GROQ_POOL_STRATEGY=ordered_failover
TEN_AI_OUTPUT_PROFILE=compact
TEN_AI_TARGET_OUTPUT_TOKENS=900
TEN_AI_MAX_OUTPUT_TOKENS=1400
TEN_AI_INPUT_TOKEN_BUDGET=3500
TEN_AI_TOKEN_SAFETY_MARGIN=256
```

Five-minute production analysis uses the compact output profile. A provider
`finish_reason=length` consumes at most one fresh `compact_retry` on the same
account. Output-budget failures never open an account circuit and never trigger
four-account failover.

Compact schema `compact-1.1` exposes deterministic supply and demand zones to
the model as request-scoped `SZ*` and `DZ*` catalogs. The provider returns only
catalog references; TEN resolves accepted references to the original zone
midpoints. Prices, objects, empty strings, sentinels, and unknown IDs fail
closed. Operational dashboard metrics default to the current deployment,
current prompt/schema/profile, and the last 24 hours; historical totals are
reported separately.

`TEN_GROQ_API_KEY` is temporarily accepted only when
`TEN_GROQ_API_KEY_1` is absent. It maps to `groq_1`, emits a deprecation warning,
and never creates a fifth account.

TEN never logs or returns API keys, authorization headers, prompts, or complete
provider payloads.

## Cycle idempotency

The durable five-minute cycle claim is acquired before pool selection. Failover stays
inside that single job. Duplicate snapshot delivery, application restarts, concurrent
workers, and dashboard reads cannot create additional provider calls for a completed
cycle or more than one persisted analysis.

## Operational states

Each account reports `AVAILABLE`, `RATE_LIMITED`, `QUOTA_EXHAUSTED`,
`CONFIGURATION_ERROR`, `CIRCUIT_OPEN`, `DISABLED`, or `UNKNOWN`, plus its safe
HTTP/error metadata, calls, successful analyses, failures, token usage, and cooldown.
The active account is assigned only after a validated analysis is durable.

## Deployment verification

Apply migrations, configure the four account variables, and run the analytical smoke
cycle. Confirm one eligible cycle produces one successful Groq call, one persisted
analysis, no calls after the successful account, and no provider calls from repeated
dashboard reads.
