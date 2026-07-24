# TEN OpenRouter Request Payload Audit

Measured: 2026-07-24  
Production request: `5b681b44-8c00-56dd-a36d-0f29d28ca6f1`  
Production request timestamp: `2026-07-24T13:27:03.311588Z`  
Sanitized API-key fingerprint: `sha256:1f6fe8fd2f69`

No API key, Authorization header, prompt content, provider payload, or private market value is included in this report.

## Executive finding

The production failure was caused by request size, not a generally unusable API key.

- The real TEN request was **211,104 bytes**.
- OpenRouter counted **48,161 prompt tokens**.
- The authoritative HTTP 402 message was:

  ```text
  Prompt tokens limit exceeded: 48161 > 23881.
  ```

- A 140-byte request from the same Railway runtime and current key fingerprint returned HTTP 200 and exactly `OK`, using 15 prompt tokens and 2 completion tokens.

The prior error mapping converted every HTTP 402 to `openrouter_insufficient_credits`. That hid the authoritative prompt-token allowance failure. Revision 2 classifies this response as `key_limit_exhausted`.

## Exact request-construction path

```text
IntegrationCoordinator.process
  backend/app/integration/service.py
  → UnifiedMarketState and QuantForecastResult passed in memory

AIReasoningService.process
  backend/app/ai_reasoning/service.py
  → reads active signal, latest forecast, latest proposal, recent memory

AIReasoningRequestBuilder.build
  backend/app/ai_reasoning/request_builder.py
  → converted every UMS EvidenceItem into AIReasoningRequest

ExistingOpenRouterReasoningProvider.reason
  backend/app/ai_reasoning/provider.py
  → loaded prompt
  → serialized request.model_dump()
  → appended response_contract

PromptLoader.load
  backend/app/ai/prompts/loader.py

HttpOpenRouterClient.complete_json
  backend/app/ai/openrouter_client/client.py
  → two messages
  → response_format={"type":"json_object"}
  → POST /chat/completions
```

### Previous database readers

| Reader | Repository method | Data added to old request |
|---|---|---|
| Active signal | `active_signals(instrument)` | Full persisted `ManagedSignal` model |
| Latest AI forecast | `latest_forecast(instrument)` | Full previous forecast payload |
| Latest proposal | `latest_proposal()` | Full previous proposal payload |
| Market memory | `recent_memory(instrument, 20)` | Up to 20 history entries summarized across multiple categories |

The request did not read dashboard state or previous prompt text. It was stateless at the HTTP message level, but it serialized large current analytical evidence plus full previous forecast/proposal objects.

## Measured production request

| Metric | Value |
|---|---:|
| Request body | 211,104 bytes |
| Messages | 2 |
| System prompt | 901 characters |
| User prompt | 191,562 characters |
| Assistant history | 0 characters |
| Tool definitions | 0 bytes |
| `response_format` schema | 22 bytes |
| Response-contract JSON, unescaped | 2,660 bytes |
| Candle records | 0 |
| SMC object references/groups | 17 |
| Liquidity object references/groups | 10 |
| Volume Profile entries/groups | 6 |
| Compact Quant scalar features | 14 |
| Actual OpenRouter prompt tokens | 48,161 |
| Conservative local wire estimate | 70,369 tokens |
| Configured maximum output | 3,200 tokens |
| Model | `meta-llama/llama-3.3-70b-instruct` |
| Routing | OpenRouter default provider routing |
| Published model context | 131K tokens |

The section estimator uses `ceil(serialized wire bytes / 3)`. It is intentionally conservative because exact Llama tokenization is not installed. The provider-reported 48,161 tokens is authoritative for the total.

### Reconciled wire breakdown

| Section | Records | Serialized bytes | Conservative tokens | Request share | Required | Summarize | Remove |
|---|---:|---:|---:|---:|---|---|---|
| HTTP envelope/message structure | 2 | 195 | 65 | 0.09% | Yes | No | No |
| System prompt | 1 | 911 | 304 | 0.43% | Yes | Yes | No |
| `analysis_request` | 52 fields | 206,964 | 68,988 | 98.04% | Some fields | Yes | Most fields |
| `response_contract` | 5 groups | 2,985 | 995 | 1.41% | Yes | Yes | No |
| User JSON structure | 2 groups | 49 | 17 | 0.02% | Yes | No | No |
| **Total** |  | **211,104** | **70,369** | **99.99% rounding** |  |  |  |

### Largest old analysis fields

| Field | Serialized bytes | Records/groups | Finding |
|---|---:|---:|---|
| `volume_profile_evidence` | 48,883 | 3 | Full nested profile dictionaries survived collection bounding |
| `liquidity_pools` | 41,325 | 3 | Full nested liquidity structures |
| `smc_evidence` | 22,225 | 3 | Large nested SMC summaries/objects |
| `market_regime` | 15,786 | 3 | Repeated raw regime payloads |
| `institutional_flow_evidence` | 13,420 | 3 | Full nested engine payloads |
| `economic_event_context` | 12,553 | 3 | Full calendar evidence |
| `session_context` | 3,487 | 12 | Repeated evidence references |
| `momentum_evidence` | 2,509 | 9 | Repeated references |
| `previous_ai_forecast` | 2,207 | 57 fields | Full previous forecast |
| `liquidity_sweeps_and_raids` | 1,927 | 7 | Repeated references |
| `volatility_evidence` | 1,792 | 6 | Repeated references |
| `trend_evidence` | 1,655 | 6 | Repeated references |
| `displacement_evidence` | 1,635 | 6 | Repeated references |
| `supported_timeframe_states` | 1,632 | 3 | Required but reducible |

The old `_bounded_value()` bounded list cardinality but recursively preserved every dictionary key. Three evidence objects could therefore still contain tens of thousands of bytes. Evidence was also repeated across semantic categories.

## Minimal versus real result

| Metric | Minimal request | Real TEN request |
|---|---:|---:|
| Key fingerprint | `1f6fe8fd2f69` | Same current Railway runtime |
| Model | Llama 3.3 70B Instruct | Llama 3.3 70B Instruct |
| Request bytes | 140 | 211,104 |
| Provider input tokens | 15 | 48,161 |
| Maximum output tokens | 5 | 3,200 |
| Routing | OpenRouter default | OpenRouter default |
| HTTP status | 200 | 402 |
| Error code | — | `402` |
| Result/message | Exact `OK` | Prompt-token limit `48,161 > 23,881` |

The real request was not replayed because that would violate the one-request-per-cycle invariant. Its result came from the persisted typed failure for the captured production request.

## New compact boundary

`backend/app/ai_reasoning/llm_context.py` defines `LLMAnalysisContext`, which is independent from ORM records and engine models. Its collections are structurally bounded:

| Field | Maximum | Omission rule |
|---|---:|---|
| M1/M5/M15 trends | 3 | Missing timeframe omitted; availability remains explicit |
| Supply zones | 3 | No qualifying zone → empty |
| Demand zones | 3 | No qualifying zone → empty |
| Relevant order blocks | 3 | No qualifying block → empty |
| Relevant fair-value gaps | 3 | No qualifying gap → empty |
| Nearest liquidity levels | 5 | No valid positive level → empty |
| HVNs/LVNs | 3 each | Unavailable profile → empty with typed status |
| Material changes | 5 | Only newest deterministic summaries |
| Previous decisions | 1 | No prior forecast → omitted |
| Active positions | 1 | No active signal → omitted |
| Risk flags/reason codes | 5 | Extras deterministically omitted |

It contains only cycle identity, symbol, boundary/cutoff, current price, regime/trend/SMC summaries, nearest levels/zones, compact Volume Profile, Institutional Flow, Quant and risk summaries, current position, one previous decision, and material changes.

The provider now serializes:

```text
analysis_context
response_contract
```

It cannot serialize `UnifiedMarketState`, engine objects, ORM objects, candle arrays, previous proposal history, dashboard data, or arbitrary dictionaries directly.

## New response contract

The model returns only:

```text
decision: LONG | SHORT | WAIT
confidence: 0..1
rationale: <=500 characters
risk_flags: <=5
proposal: actionable geometry or null
```

The deterministic validator expands this compact result into existing internal forecast/proposal models. WAIT uses `proposal=null`. Full chain-of-thought and raw provider output are not persisted.

## Token, size, and cost guards

The provider measures the exact canonical body before HTTP:

- target input: 4,000 tokens;
- warning: 8,000 tokens;
- hard rejection: 16,000 tokens;
- normal output: 1,000 tokens;
- absolute output: 2,000 tokens;
- maximum conservative request cost: USD 0.05.

Preflight rejection produces `request_too_large` or `maximum_cost_exceeded`, lists only sanitized section names/sizes, makes no HTTP request, and does not open payment backoff.

HTTP classification now distinguishes:

```text
payment_blocked
key_limit_exhausted
request_too_large
context_limit_exceeded
maximum_cost_exceeded
no_eligible_provider
rate_limited
provider_unavailable
invalid_request
authentication_failed
```

## Measured new representative request

The production request cannot be transformed in place without deploying unreviewed code. The new boundary was therefore measured locally with the production request path and a representative synchronized XAUUSD state fixture:

| Metric | Old production | New representative | Reduction |
|---|---:|---:|---:|
| Request body | 211,104 B | 4,921 B | 97.67% |
| Input tokens | 48,161 actual | 1,641 conservative | 96.59% |
| Maximum output | 3,200 | 1,000 | 68.75% |
| Conservative maximum cost | $0.057287 | $0.003957 | 93.09% |

The maximum-cost estimate uses the configured conservative OpenRouter price ceilings of $1.04/M input tokens and $2.25/M output tokens. Actual routed cost can be lower. The failed production request was rejected before completion, so this is a maximum-request estimate rather than a billed-cost claim.

## Validation

Tests prove:

1. OpenRouter receives only `LLMAnalysisContext`.
2. Candle arrays and raw engine fields are absent.
3. Every collection has a Pydantic maximum.
4. No message/history collection exists.
5. A normal request remains below 4,000 conservative tokens.
6. Oversized requests are rejected before client invocation.
7. Oversized requests are not classified as credit/payment failures.
8. Compact WAIT validates with a 1,000-token output ceiling.
9. The service makes at most one provider call per cycle.
10. Increasing old memory history does not increase prompt size beyond five material changes.
11. Prompt-token HTTP 402 is classified as `key_limit_exhausted`.
12. Raw provider output is not persisted in failure records.

## Code locations changed

- `backend/app/ai_reasoning/llm_context.py`
- `backend/app/ai_reasoning/provider.py`
- `backend/app/ai_reasoning/request_builder.py`
- `backend/app/ai_reasoning/models.py`
- `backend/app/ai_reasoning/config.py`
- `backend/app/ai_reasoning/service.py`
- `backend/app/ai_reasoning/repository.py`
- `backend/app/ai_reasoning/validation.py`
- `backend/app/ai/openrouter_client/client.py`
- `backend/app/ai_reasoning/prompts/new_market_analysis_v1.txt`
- `backend/app/ai_reasoning/prompts/existing_signal_monitoring_v1.txt`
- `configs/ai_reasoning.yaml`
- focused AI reasoning and OpenRouter tests
