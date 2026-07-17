# Configuration

`configs/smc.yaml` contains typed and hashed pivot, structure, displacement, imbalance, Order Block, dealing-range, MTF, quality, lifecycle, batching, checkpoint, and bounded-recalculation settings for SMC Production 1.0. Exact definitions and defaults are documented in [SMC_ENGINE.md](SMC_ENGINE.md).

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
| `flow.yaml` | Flow-estimation settings |
| `volume_profile.yaml` | Profile settings |
| `economic.yaml` | Event risk windows |
| `ai.yaml` | OpenRouter model and prompt version |
| `signal.yaml` | Scenario construction settings |
| `market_regime.yaml` | Disabled future regime contract |
| `replay.yaml` | Disabled future replay contract |

`YamlConfigRepository` restricts names to safe stems, uses `yaml.safe_load`, requires mapping roots, and validates documents into Pydantic models. Secrets remain environment variables.

Feature flags are applied before construction. A required pipeline step referencing a disabled engine fails configuration validation rather than silently producing partial analysis.
