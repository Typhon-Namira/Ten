# Engine catalog

| Engine | Input | Output | Baseline responsibility | Upgrade boundary |
|---|---|---|---|---|
| Market data | Provider response or CSV | `Candle`, `Tick` | Normalize M1, M5, M15, H1, H4, D1 data | Implement `MarketDataProvider` or `RealtimeMarketDataProvider` |
| SMC / ICT | Ordered candles | `SMCResult` | Directional structure, close-based BOS, three-candle FVG, premium/discount position | Implement `SMCAnalyzer` |
| Liquidity | Ordered candles | `LiquidityResult` | Equal-level pools, sweep state, nearest buy/sell liquidity, UTC session | Implement `LiquidityAnalyzer` |
| Institutional flow | Ordered OHLCV | `FlowScore` | Estimate pressure, acceleration, delta proxy, and absorption probability | Implement `InstitutionalFlowEngine`; label licensed exchange data explicitly |
| Volume profile | Ordered OHLCV | `VolumeProfileResult` | POC, VAH, VAL, HVN/LVN nodes from configurable price bins | Implement `VolumeProfileAnalyzer` |
| Economic calendar | Timestamp and events | `NewsRiskResult` | Impact-aware risk windows and high-impact no-trade flag | Implement `EconomicCalendarEngine` and a provider adapter |
| AI scoring | `ScoringContext` only | `SignalScore` | Conservative confluence and quality assessment through OpenRouter | Implement `AIScoringEngine`; version every prompt |
| Signal | All validated engine results | `Signal` | Apply risk gating and create entry/stop/target scenarios | Implement `SignalEngine`; execution is out of scope |
| Market regime | Feature snapshot | `MarketRegimeResult` | Infrastructure only; no detection | Implement `MarketRegimeEngine` and enable its flag |
| Replay | Historical event source | `ReplayState` | Infrastructure only; no simulation | Implement `ReplayEngine` and enable its flag |

Baseline analyzers are intentionally transparent and deterministic. Runtime instances are selected by the registry, never directly constructed by the pipeline. Results are foundations for research, not claims that every future capability listed in the project vision is already implemented. All timestamps must be timezone-aware, all external observations must retain provider provenance, and all scoring changes must be replayable against frozen inputs.
