# Configuration

`configs/smc.yaml` contains typed and hashed pivot, structure, displacement, imbalance, Order Block, dealing-range, MTF, quality, lifecycle, batching, checkpoint, and bounded-recalculation settings for SMC Production 1.0. Exact definitions and defaults are documented in [SMC_ENGINE.md](SMC_ENGINE.md).

`configs/liquidity.yaml` contains immutable validated groups for absolute/tick/ATR/percentage tolerances, equal-level clustering, pool limits, sweep and reclaim classification, DST-aware sessions, previous-period references, symbol-specific round levels, deterministic ranking, multi-timeframe bounds, processing limits, input quality and persistence policy. Its SHA-256-derived version is included in every snapshot, event, feature and persisted object. Exact semantics are documented in [LIQUIDITY_ENGINE.md](LIQUIDITY_ENGINE.md).

`configs/volume_profile.yaml` contains immutable validated volume-source semantics, Decimal-aligned grid methods, allocation assumptions, value-area percentage, node/shelf/gap thresholds, bounded profile windows, multi-timeframe depth, and durable-persistence policy. Unknown, missing, and synthetic semantics remain explicitly degraded. See [VOLUME_PROFILE_ENGINE.md](VOLUME_PROFILE_ENGINE.md).

`configs/market_regime.yaml` strictly validates version compatibility, dependency policy, source/family weights, classification thresholds, evidence bounds, correlation caps, replay-safe decay, persistence/checkpointing, processing/retention, repository mode, and multi-timeframe hierarchy/depth. Invalid settings fail startup rather than silently changing semantics. See [MARKET_REGIME_ENGINE.md](MARKET_REGIME_ENGINE.md).

`configs/economic_calendar.yaml` strictly validates provider modes and priority, timezone, timeout/retry/rate bounds, importance/relevance weights, risk-window ordering, freshness thresholds, API limits, retention, cache, batch size, repository mode, and version compatibility. The production default is explicitly disabled/degraded until a real authenticated live adapter is configured. See [ECONOMIC_CALENDAR_ENGINE.md](ECONOMIC_CALENDAR_ENGINE.md).

TEN treats YAML as the runtime composition source.

| File | Purpose |
|---|---|
| `engine_registry.yaml` | Engine version and enabled selection |
| `pipeline.yaml` | Ordered execution and confidence-factor mapping |
| `feature_flags.yaml` | Independent rollout controls |
| `confidence.yaml` | Deterministic weights and AI bonus cap |
| `market_data.yaml` | Provider-neutral market-data settings |
| `smc.yaml` | SMC engine settings |
| `liquidity.yaml` | Liquidity engine settings |
| `volume_profile.yaml` | Volume Profile grid, allocation, profile, quality, and persistence settings |
| `flow.yaml` | Flow-estimation settings |
| `volume_profile.yaml` | Profile settings |
| `economic.yaml` | Event risk windows |
| `economic_calendar.yaml` | Economic Calendar providers, normalization, risk context, retention, and API bounds |
| `ai.yaml` | Provider-neutral AI model and prompt policy |
| `ai_scoring.yaml` | Deterministic intelligence aggregation policy |
| `signal_decision.yaml` | Fail-closed analytical decision policy, rules, validity, and replay controls |
| `signal.yaml` | Scenario construction settings |
| `market_regime.yaml` | Disabled future regime contract |
| `replay.yaml` | Strict Replay Production 1.0 ordering, graph, limits, speed, checkpoint, lease, isolation, point-in-time, trace, retention, and source policy |

`YamlConfigRepository` restricts names to safe stems, uses `yaml.safe_load`, requires mapping roots, and validates documents into Pydantic models. Secrets remain environment variables.

Feature flags are applied before construction. A required pipeline step referencing a disabled engine fails configuration validation rather than silently producing partial analysis.
## Institutional Flow (`configs/flow.yaml`)

The versioned Institutional Flow configuration controls evidence limits and age, minimum quality, decay, correlation caps and discounts, participation/activity/inventory/conflict thresholds, per-source weights, production persistence requirements, request bounds, and multi-timeframe depth. Configuration is validated at startup and included in every snapshot, feature, checkpoint, and event.
# AI Scoring

`configs/ai_scoring.yaml` strictly defines the weighted policy, version/hash identity, approved source groups, component direction/confidence/risk weights, source freshness thresholds, minimum evidence/group requirements, confidence ceilings, conflict thresholds/penalties, non-overlapping directional labels, API limits, retention windows, persistence requirement, and replay-event policy. Configuration is data only; arbitrary expressions or module names are never evaluated.

# Signal Decision

`configs/signal_decision.yaml` defines the approved policy/version, exhaustive AI-label direction mapping, observation/eligibility thresholds, hard risk and quality gates, conflict penalty ceiling, timeframe freshness windows, Economic Calendar phases, exhaustive Market Regime outcomes, cooldown, reversal, hysteresis, validity, API limits, retention, persistence policy, and replay publication controls. Pydantic validation rejects unknown keys, missing mappings/timeframes, overlapping thresholds, negative durations, invalid versions, unknown outcomes, and zero validity. YAML never selects arbitrary imports or executable expressions.

# Replay

`configs/replay.yaml` is fail-closed and strictly validates compatibility, canonical ordering, the dependency graph, approved scope and sources, resource bounds, speed, checkpoints, worker leases, isolation, deterministic hashing, point-in-time enforcement, trace limits, retention, and event-cycle ceilings. It cannot import providers or execute arbitrary expressions. See [REPLAY_ENGINE.md](REPLAY_ENGINE.md).
