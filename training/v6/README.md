# TEN V6 Research Pipeline

## Purpose

TEN V6 is the execution-aware XAUUSD research pipeline.

The V6 line investigates causal technical state, multi-horizon
direction, execution-aware outcomes, daily opportunity selection,
and economic trade quality.

## Main pipeline

1. Build technical features and market structure.
2. Build multi-horizon outcome targets.
3. Build execution-aligned targets.
4. Build daily opportunity targets.
5. Train technical/execution models.
6. Diagnose WHEN versus WHAT.
7. Diagnose directional learnability.
8. Train the end-to-end multi-scale execution model.

## Key versions

- V6.0 / V6.1: dual-brain and profit-aware experiments
- V6.2: technical feature/setup pipeline
- V6.3: advanced technical brain
- V6.4: technical experts
- V6.5: technical mixture-of-experts
- V6.6.1: multisurface technical brain
- V6.6.2: execution policy backtests
- V6.7.0: execution-aligned target correction
- V6.7.1: execution precision brain
- V6.7.2: daily opportunity brain
- V6.7.3: directional utility brain
- V6.8.0: end-to-end multi-scale execution brain

## V6.8 architecture

The model consumes three causal contexts:

- Recent: 24 M5 steps
- Intraday: 96 steps, stride 3
- Regime: 60 steps, stride 24

Outputs include:

- LONG/SHORT direction
- horizon-specific direction
- execution net-return estimates
- win probabilities
- daily opportunity scores
- within-day rank

## Validation protocol

2023 is used for champion selection.

The champion policy is frozen before opening the 2024 benchmark.

V6.8 does not evaluate 2025 or 2026.

## V6.8 frozen result

Champion epoch: 2

Frozen policy:

- quantile: 0.90
- threshold: 0.9149430990219116
- direction confidence: 0.30

2024 frozen benchmark:

- trades: 246
- coverage: 94.9807%
- win rate: 42.2764%
- mean net: -1.22523 bps
- profit factor: 0.90193

V6.8 therefore has sufficient daily coverage but does not have
acceptable economic performance. It is retained as a research
checkpoint and reproducibility reference, not as a production model.

## External datasets

Large datasets are intentionally excluded from Git.

Recovered dataset locations:

- training/v2/data_lake/xau/
- training/v6/data_lake/

The recovery copy contains:

- V2 XAU: 2 files, 50,880,491 bytes
- V6 data lake: 29 files, 4,131,767,040 bytes

See `data_sha256.txt` for the recovered data manifest.

## Checkpoints

Twelve V6 model checkpoints were recovered separately from Git.

See `checkpoint_sha256.txt` for the SHA256 manifest.

Model weights are intentionally excluded from normal Git history.

## Recovery status

The V6 source pipeline, experiment results, checkpoint manifests,
and dataset manifests are preserved.

Large data and model weights are stored outside the repository.

## Safety

Do not blindly run `git add .` in a training workspace.

Git ignores large training datasets, NumPy arrays, parquet files,
checkpoints, ONNX files, and local backup files.
