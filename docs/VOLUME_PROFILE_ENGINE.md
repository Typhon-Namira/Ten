# Volume Profile Engine Production 1.0

## Purpose and boundaries

The Volume Profile Engine converts replay-visible candle volume into deterministic analytical distributions across price. It consumes normalized candles only through Market Data interfaces and optional typed SMC/Liquidity evidence. It does not fetch providers, detect market structure or liquidity pools, infer institutional intent or regime, or emit entries, orders, stops, targets, sizing, or recommendations.

Candle volume does not reveal exact volume-at-price. Every bucket is an analytical allocation approximation. Exchange volume, broker volume, tick volume, synthetic volume, missing volume, and unknown semantics remain distinct. Tick volume is not centralized exchange volume. Optional directional allocation is estimated candle geometry, not true bid/ask delta, and is disabled by default.

## Architecture and profiles

`MarketDataService -> VolumeProfileContext -> deterministic analyzer -> snapshot -> repository / Feature Store / Event Bus / read-only API`.

The domain has no FastAPI, SQLAlchemy, provider, broker, or deployment dependency. `VolumeProfileContext` accepts optional typed SMC references, Liquidity source IDs, instrument tick size, and traceable anchors. Cross-engine inputs are linked as evidence; their detection is never duplicated.

Snapshots support fixed-range, developing, session, daily, weekly, monthly, composite, anchored, and multi-timeframe contexts. Sessions use the Market Data session engine, including timezone, DST, weekend, and configured-holiday behavior. Completed prior periods are immutable; the current period is developing. Anchors become visible only at their availability timestamp. Composite profiles retain bounded constituent references. Unsupported `W1`/`MN1` inputs remain explicit empty entries rather than fabricated data.

Replay, historical, incremental, recovery, fixed-range, anchored, and composite processing use the same analyzer. IDs include the source boundary, processing mode, volume semantics, anchors, Liquidity references, and configuration version. A query at T consumes only candles and evidence available at T. Completed references are marked tested only by later visible candle intersections; future tests never modify an earlier stored snapshot.

## Price grid and volume conservation

Grid methods are tick-size, fixed increment, configured row count, percentage, ATR-normalized, and bounded automatic. Decimal arithmetic aligns base and row size to the instrument tick. Buckets are `[lower, upper)`; only the final bucket includes its upper bound. Stable IDs prevent floating-point identity drift. Bin limits bound memory.

Allocation methods are close, typical price `(high + low + close) / 3`, uniform intersected-range distribution, and normalized body/wick weighting. Directional estimates, when enabled, are bounded candle-location estimates. For every profile, `sum(bucket.volume) == included source volume` within documented numerical tolerance; estimated buy plus estimated sell equals total bucket volume.

## POC, Value Area, nodes, shelves, and gaps

POC is the largest-volume bucket. Ties resolve by distance to weighted mean price, then lower stable bucket index. Value Area defaults to 70% and uses deterministic POC expansion: compare adjacent upper/lower volume, choose the larger, and choose the lower side on ties. VAH, VAL, achieved percentage, included volume, and overshoot are retained.

HVNs are positive local maxima above the configured percentile. LVNs are internal local minima below the configured percentile; exterior zero buckets are excluded. Shelves require a configured multi-bin width above mean volume and remain distinct from single-bin HVNs. Volume gaps are bounded low-activity regions inside traded price with positive surrounding activity; they are not SMC FVGs, liquidity voids, or missing-data gaps.

## Shape, migration, confluence, and tested references

Shape evidence records skewness, excess kurtosis, POC location, concentration, mode count, and elongation. Results may be D, P, b, double-distribution, trend, thin, multimodal, or undefined, with an alternative and conflicting evidence. Shape is not accumulation/distribution or directional intent.

Migration records POC/VAH/VAL changes, normalized bucket movement, elapsed time, direction, confidence, and quality. Confluence links overlapping profile POCs with optional SMC/Liquidity IDs, preserves source diversity, and discounts correlated profile evidence. POC, VAH, VAL, HVN, and LVN tested states use price intersection after completion, including first test and count; they are analytical references, not targets or sweep classifications.

## Persistence, checkpoints, features, and events

PostgreSQL stores immutable snapshot JSONB, indexed profile objects, and SHA-256 protected checkpoints. Writes are conflict-safe and transactional. Startup rejects corrupt or engine-incompatible checkpoints and recovers the latest valid state without unsafe serialization. Explicit in-memory mode is degraded when production persistence is required.

The `volume_profile` Feature Store namespace publishes developing POC/value area, completed sessions, nearest HVN/LVN, shelves, gaps, shape, migration, confluence, confidence, quality, traceability, and version fields. Typed deterministic events cover profiles, POC migration, value-area changes, nodes, shelves, gaps, shapes, anchors, composites, degradation, replay, and checkpoint recovery. Reprocessing the same snapshot is idempotent in-process.

## API, configuration, health, and limitations

Read-only endpoints live under `/volume-profile`: health, metrics, config, state, snapshot, profiles, developing/completed and period families, bounded fixed range, POC, value area, nodes, shelves, gaps, shapes, migrations, confluences, MTF, and replay. Pagination is capped at 5,000 objects; candle inputs at 100,000; fixed ranges at the configured day limit. No mutation endpoint exists.

`configs/volume_profile.yaml` versions volume-source policy, grids, allocation, nodes, value area, profile windows, MTF depth, and persistence. Health is degraded for ephemeral persistence, unobserved input, or unknown/missing/synthetic semantics. Metrics include observations, source volume, profile/bucket/node counts, migrations, confluences, failures, recovery, replay, latency, and latest boundary.

The engine has no order-book or trade-tape knowledge and cannot reconstruct exact intrabar trading. Outputs are analytical observations, not trading instructions.

## Complexity

Allocation is `O(candles * intersected_bins)`, bounded by the maximum bins; node and value-area scans are `O(bins)`, and period grouping is `O(candles)`. Persistence reads one bounded latest snapshot/checkpoint per series. On the validation host, a repeated narrow-range M15 series produced 13 profiles/25 maximum buckets for 2,000 candles in 1.659s (1.22 MiB traced peak, 0.40 MiB snapshot), 15 profiles/25 buckets for 10,000 in 6.546s (1.68 MiB traced peak, 0.48 MiB snapshot), and 15 profiles/25 buckets for 100,000 in 3.811s without tracing overhead (0.43 MiB snapshot). These measurements demonstrate bounded behavior for this workload, not universal linear-time claims.
