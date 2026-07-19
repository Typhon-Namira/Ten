# Replay Engine Production 1.0

The Replay Engine deterministically reconstructs TEN's historical analytical behavior from immutable, version-pinned sources. It is an orchestration boundary, not a market-data owner: providers expose typed historical-event adapters and Replay never imports their implementations. It is not a backtester and contains no order, fill, position, brokerage, P&L, or performance simulation.

## Correctness model

Every request pins a dataset identity, manifest or query cutoff, time range, instruments, timeframes, selected engines, engine/config versions, ordering version, and execution mode. Historical events carry occurred, published, and available timestamps. Processing is gated by `available_at`; final candle OHLCV becomes available no earlier than candle close and ingestion, while Economic Calendar revisions appear only when their revision was available.

Canonical ordering is `(available_at, source_priority, source_sequence, source, source_event_id, payload_hash)`. Sources are merged with a bounded k-way merge and equal timestamps are processed as a deterministic group. IDs and payload hashes are content-derived. The virtual clock is monotonic, bounded by the request, and independent of wall-clock speed.

## Isolation and graph

Replay has a dedicated event bus and feature store. Live buses, live feature state, providers, and live-wired engine services are not invoked. Typed processors declare compatibility and dependencies; the registry builds a deterministic topological graph and fails closed for missing, incompatible, or cyclic dependencies. Generated analytical events are drained deterministically with a configured cycle ceiling.

## Lifecycle and durability

Sessions move through `created`, `validating`, `ready`, `running`, `pausing`, `paused`, `resuming`, `cancelling`, `cancelled`, `completed`, `failed`, and `recovering` using an explicit transition graph. Pause and cancel take effect at safe event-group boundaries. Step mode processes one timestamp group.

Sessions, transitions, checkpoints, trace rows, output references, and worker leases are durable. Checkpoints contain source cursors, virtual time, processor state, counts, hash-chain state, and progress. Writes use optimistic row versions; expiring leases coordinate multiple workers. Recovery resumes from the latest valid checkpoint without replaying committed work. Failures are sanitized and persisted.

## Determinism and observability

Semantic output hashes form a SHA-256 chain over canonical analytical output, excluding wall-clock and storage identities. Identical requests may produce separate sessions so completed runs can be compared. Health, metrics, summaries, transitions, checkpoints, trace records, AI Scoring references, Signal Decision references, and comparison results are exposed through bounded APIs.

## Operation

Configuration is in `configs/replay.yaml`. The API lifecycle initializes Replay after Signal Decision and stops it first. The API process does not run an embedded worker by default; production processing requires a separately deployed worker using the shared database. The default PostgreSQL registry supports historical candles and Economic Calendar revisions. Replay reports degraded health when durable storage or configured historical sources are unavailable.

Operators must apply `migrations/20260719_replay_engine_v1.sql`, configure shared PostgreSQL storage, deploy the worker independently, register immutable datasets, and monitor leases, failures, backlog, checkpoint age, and deterministic comparisons.
