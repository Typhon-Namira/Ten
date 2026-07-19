# Engine catalog

| Engine | Input | Output | Baseline responsibility | Upgrade boundary |
|---|---|---|---|---|
| Market data | Provider response or CSV | `Candle`, `Tick` | Normalize M1, M5, M15, H1, H4, D1 data | Implement `MarketDataProvider` or `RealtimeMarketDataProvider` |
| SMC / ICT | Normalized candles from `MarketDataService` | `SMCAnalysisSnapshot` / compatible `SMCResult` | Confirmed swings, independent internal/external structure, BOS, CHoCH, and MSS | Extend `SMCAnalyzer`; see [SMC Engine](SMC_ENGINE.md) |
| Liquidity | Ordered candles | `LiquidityResult` | Equal-level pools, sweep state, nearest buy/sell liquidity, UTC session | Implement `LiquidityAnalyzer` |
| Institutional flow | Ordered OHLCV | `FlowScore` | Estimate pressure, acceleration, delta proxy, and absorption probability | Implement `InstitutionalFlowEngine`; label licensed exchange data explicitly |
| Volume profile | Ordered OHLCV | `VolumeProfileResult` | POC, VAH, VAL, HVN/LVN nodes from configurable price bins | Implement `VolumeProfileAnalyzer` |
| Economic calendar | Timestamp and events | `NewsRiskResult` | Impact-aware risk windows and high-impact no-trade flag | Implement `EconomicCalendarEngine` and a provider adapter |
| AI scoring | Versioned point-in-time engine evidence | `AIScoreSnapshot` | Deterministic direction, confidence, risk, alignment, quality, and composite intelligence | Extend approved deterministic policies; preserve replay and explanation contracts |
| Signal decision | Trusted persisted `AIScoreSnapshot` plus event/regime context | `SignalDecision` | Fail-closed eligibility, observation, blocking, sufficiency, validity, cooldown, and reversal policy | Extend approved rule/policy registries; execution remains prohibited |
| Signal | All validated engine results | `Signal` | Apply risk gating and create entry/stop/target scenarios | Implement `SignalEngine`; execution is out of scope |
| Market regime | Feature snapshot | `MarketRegimeResult` | Infrastructure only; no detection | Implement `MarketRegimeEngine` and enable its flag |
| Replay | Version-pinned point-in-time historical sources | `ReplaySession`, checkpoints, trace and analytical output references | Deterministic reconstruction of historical analytical behavior | Add typed source/processor adapters; backtesting, P&L and execution remain out of scope |

Baseline analyzers are intentionally transparent and deterministic. Runtime instances are selected by the registry, never directly constructed by the pipeline. Results are foundations for research, not claims that every future capability listed in the project vision is already implemented. All timestamps must be timezone-aware, all external observations must retain provider provenance, and all scoring changes must be replayable against frozen inputs.
