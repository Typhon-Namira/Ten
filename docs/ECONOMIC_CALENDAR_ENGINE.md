# TEN Economic Calendar Engine Production 1.0

The Economic Calendar Engine is TEN's provider-neutral, point-in-time macro-event context system. It owns provider observations, normalized event identity, schedules, publication lifecycle, revisions, reconciliation, instrument relevance, bounded risk-window context, quality, freshness, persistence, replay, and read-only APIs. It never creates signals, orders, position sizes, or trading instructions. Every analytical output carries `probabilistic_context=true` and `trading_instruction=false`.

## Architecture and ownership

Provider adapters emit immutable `ProviderEventObservation` objects. Normalization parses values, timezones, names, countries, currencies, status, importance, and category without importing another analytical engine. Reconciliation selects fields by configured provider priority while preserving observation IDs and explicit conflicts. The service appends revisions, persists canonical events, creates snapshots and instrument contexts, publishes namespaced features and typed idempotent events, and checkpoints synchronization identity state.

Dependency direction is strictly external provider → adapter → Economic Calendar normalization/persistence → downstream consumers. Economic Calendar does not import SMC, Liquidity, Volume Profile, Institutional Flow, Market Regime, AI Scoring, Signal, or execution implementations.

## Provider abstraction

The provider protocol exposes identity, version, timezone, capabilities, bounded full/incremental fetching, lookup, and health. Supported modes are `live_provider`, `file_import`, `static_fixture`, `in_memory_test_provider`, and `disabled`. This repository ships deterministic memory/fixture, safe JSON/CSV file-import, and disabled adapters. No production provider credential is committed or assumed. The deployed default is deliberately degraded/disabled until an authenticated live adapter is configured. Fixtures can never report themselves as live providers.

Imports are restricted to JSON/CSV files below the configured import root, limited to 10 MB, parsed without code execution, and stripped of credential-shaped fields. Provider payload storage is sanitized and bounded; authorization headers and secrets are excluded.

## Event lifecycle and temporal safety

Canonical IDs use normalized country, currency, release name, and scheduled release date; same-provider identity is retained across tentative-to-exact and reschedule updates. Core/headline distinctions remain in the canonical name. Weak matches are not merged. Provider observations remain independently addressable.

`actual`, `forecast`, `previous`, and `revised_previous` are separate values. First release, later revisions, corrections, schedule changes, cancellations, postponements, reschedules, metadata changes, and provider conflicts produce append-only typed revisions. Unavailable or unparseable values are `None`, never zero. Percent, basis/value suffixes, currencies, localized separators, negative parentheses, placeholders, and text values are handled explicitly. Surprise is calculated only for comparable numeric actual/forecast values; its direction is indicator-relative and has no asset/trade direction.

Scheduled time, provider publication time, response time, ingestion time, and `available_at` are distinct. When a provider lacks publication time, response/ingestion availability is used and is never backdated. All canonical timestamps are timezone-aware UTC; original timezone/representation remains traceable. Historical reads select only state with `available_at <= as_of`, excluding later discovery, forecasts, schedule changes, cancellations, conflicts, releases, and revisions. Replay uses an injected analysis clock and never reads wall time inside analytics.

## Context, quality, and degradation

Instrument mapping supports currency pairs and configured commodity, index, and crypto overrides. Context exposes previous/next/active relevant events, direct matches, bounded importance/relevance/freshness/quality/cluster scores, configured pre/imminent/post/cooldown phases, transparent explanation, and limitations. Nearby events form deterministic clusters while retaining component counts and applying bounded correlation-aware scoring.

Quality combines source quality, normalization confidence, completeness, conflicts, timing certainty, and freshness. Freshness thresholds classify fresh, aging, stale, critical, or unknown. Disabled, stale, unreachable, rate-limited, or failed providers are visible in health and snapshot degradation; partial fixture/file results never masquerade as live synchronization.

## Persistence, recovery, and multi-instance safety

The in-memory repository is bounded and lock-protected. The PostgreSQL adapter uses deterministic primary keys, unique observation/revision/context/checkpoint constraints, upserts, transactions, and indexed time queries. The additive idempotent migration is `migrations/20260718_economic_calendar_v1.sql`. Checkpoints contain engine/schema/configuration/normalization versions, cursors/tokens, last observation, identity state, and a SHA-256 payload hash. Missing checkpoints produce a clean start; corrupt or incompatible checkpoints fail explicitly.

Multi-instance safety comes from deterministic IDs and database uniqueness. Production live adapters should additionally use a database advisory lock or deployment leader lease before polling; the bundled scheduler starts only for a configured live provider and prevents overlapping synchronization within an instance.

## APIs and integrations

Under the configured API prefix, read-only routes are available at `/economic-calendar`: health, config, metrics, providers, events/detail/revisions/observations, upcoming/recent/active, snapshot/history, symbol context, clusters, conflicts, and explanations. Timestamps require timezone offsets, pages and windows are bounded, responses are deterministically ordered, and provider secrets/raw authorization data are never returned.

Feature Store records use the `economic_calendar` namespace and include versions, symbol, analysis/boundary times, source event IDs, next/previous/released values, proximity, risk phase/score, clusters, quality, freshness, degradation, conflicts, revisions, and safety flags. Event Bus messages use deterministic IDs and contain versions, source IDs, previous/current state, changes, quality, freshness, conflicts, and safety flags. Duplicate persisted state does not emit a second identical message.

## Configuration, deployment, and limitations

`configs/economic_calendar.yaml` is strict and versioned. It validates provider modes/priorities/timezones, retries/timeouts/rate bounds, windows, weights, freshness thresholds, pagination, retention, cache, and batch limits. PostgreSQL is selected only when database initialization succeeds; production does not silently claim durable persistence when it is using memory.

Railway runs the existing FastAPI lifecycle, registers metadata before table creation, restores the calendar checkpoint, exposes provider/degradation state, and shuts down the scheduler and database session cleanly. Live synchronization remains unavailable until a real authenticated adapter and credential are configured. This limitation does not affect fixture/file contract tests or deterministic replay semantics.
