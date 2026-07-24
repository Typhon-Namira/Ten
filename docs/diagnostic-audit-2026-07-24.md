# TEN full-system diagnostic audit

Date: 2026-07-24
Scope: ground-up, evidence-based audit of database growth and pipeline non-function on the
live Railway deployment. **Diagnosis only — no code was changed in this pass.** Every claim
below is cited to a file:line in the current repo (HEAD `956731f`, 2026-07-23T23:59:15+04:00)
or explicitly marked as unverified/needs-live-access. Three independent research passes
(database write-paths, full pipeline trace, dashboard-panel mapping, env-var/silent-failure
audit) were run and cross-checked against each other and against direct manual reads of the
storage models, integration service, and outbox repository.

**No production/Railway access was available in this sandbox** — no `.env`, no `railway` CLI,
no `psql`, no Docker. Everything below is static-code evidence plus repo state. Section "What I
don't know" at the end lists exact queries/checks for you to run and paste back.

---

## Executive summary

Two things are true at once, and reconciling them is the crux of this audit:

1. **The dashboard almost certainly reads from a path that is off by default.** The page the
   user is most likely looking at (`/api/v1/dashboard/latest`, the AI-centric dashboard) is
   gated behind `ai_centric_shadow_mode`, which defaults to `false` in
   `configs/feature_flags.yaml:11`. With it off, that endpoint returns
   `"status": "pending"` / `reason: "ai_centric_shadow_mode_disabled"` forever — by design, not
   by bug (`backend/app/api/routes/dashboard.py:104-170`). Separately, the Market Intelligence
   panel's "Unavailable" values are what `safe_call()` (`backend/app/api/safe.py:15-18`) returns
   when the underlying engine-state calls time out or return empty — which happens whenever the
   two background workers (`market_data_worker`, `integration_worker`) are not running.
2. **Those two workers being off is a known, self-inflicted, silent trap already documented in
   the repo.** `.env.example:11-23` ships `TEN_INTEGRATION_WORKER_ENABLED=false` and
   `TEN_MARKET_DATA_WORKER_ENABLED=false` with an explicit warning that copying this file
   verbatim into Railway **permanently disables both workers**, because an explicit `false`
   overrides the production auto-enable safety net in `settings.py:111-118` (the net only fires
   "if the variable was never set at all"). If that's what happened on Railway, no candles are
   ever polled, no `NewCandle` event is ever published, and the entire analytical chain
   (SMC → Liquidity → Volume Profile → Institutional Flow → Market Regime → AI Scoring →
   Signal Decision → `operational_signals`) never runs a single cycle — this alone explains
   "most panels Unavailable, no scenarios/signals ever published, dashboard shows the app is
   barely processing anything."
3. **That same "barely processing anything" state is hard to reconcile with 5GB/5h of DB
   growth from the trading pipeline itself** — if the workers are truly off, the tables that
   pipeline writes to are not growing. The one background process confirmed to run
   **unconditionally, independent of both worker-enabled flags**, is the economic-calendar
   sync scheduler (`backend/app/engines/economic_calendar_engine/service.py:167-168`, started
   from `main.py:276` whenever any public-source provider is configured — true by default). It
   is also the one worker with **zero liveness visibility** anywhere in
   `/api/v1/system/diagnostics` (confirmed in the silent-failure audit below) — meaning it could
   be over-writing data with nobody able to see it from the dashboard.
4. Independent of whichever of the above is the dominant driver, this audit found a **structural
   gap that guarantees unbounded growth regardless of any bug**: roughly two dozen tables that
   the live pipeline writes to (every analytical engine's objects/snapshots, every
   integration/outbox table, market-evidence/unified-market-state tables, quant/AI-reasoning
   tables) have **no retention/pruning mechanism implemented at all** — not disabled, not
   misconfigured, simply never written. On top of that, three tables (`ai_score_snapshots`,
   `signal_decisions`, `replay_sessions`) *do* have a fully-implemented retention `cleanup()`
   method with a configured `retention.live_days`/`retention.replay_days` — but that method is
   **never called from anywhere in the codebase**, confirmed by grep. This is dead code that
   gives a false impression retention is handled.
5. The specific bug class you already fixed once (a dedup/identity key computed from a field
   that changes on every poll, causing infinite reprocessing) was searched for exhaustively
   across every `hashlib`/`fingerprint`/`*_hash` call site in the codebase. The original fix
   (`backend/app/integration/models.py:190-202`, `event_id` derived from `source_event_id`
   only, explicitly excluding the wall-clock `ingestion_time`) is confirmed live and correct.
   Every other dedup key traced (`ScoringInput.fingerprint()`, `SignalDecisionInput.fingerprint()`,
   `semantic_hash`, `snapshot_hash`, each engine's `analysis_timestamp`) was confirmed to
   deliberately exclude wall-clock fields. **No second instance of the exact bug was found** —
   two lower-priority areas (non-SMC/non-MarketRegime engine `snapshot_id` construction;
   `economic_calendar_engine` observation-record identity) were spot-checked but not traced with
   full rigor; see "What I don't know."

**Bottom line: this is not one bug. It's a stack of independently-true things — a silent
config trap that (if triggered) turns off the entire visible pipeline, a second silent config
gate that turns off the dashboard's primary read path even if the pipeline is healthy, an
unconditional background loop with zero observability, and a retention system that was
designed (per the schema and even partially coded) but never finished or wired up.** Which of
these is contributing how much to the specific 5GB/5h number cannot be determined without a
live database query and the actual Railway environment variable values — both listed explicitly
below for you to run.

---

## Database bloat: root causes ranked by likely contribution

I do not have production DB access, so I cannot give you the actual `pg_stat_user_tables`
output the task asked for. What follows is ranked by (a) structural certainty — confirmed by
reading the actual repository/model code, not inferred — and (b) how few unverified
preconditions (i.e. "is flag X actually on in Railway") each explanation requires.

### 1. No retention/pruning exists for ~25 tables the live pipeline writes to (highest confidence, largest blast radius)

**Evidence.** Grepping `backend/app` for `prune|cleanup|retention|purge|trim` and cross-checking
against every repository under `backend/app/engines/*/repository.py` and
`backend/app/integration/repository.py` turns up retention logic in exactly five places:

| Engine | Retention method | Wired up? |
|---|---|---|
| `market_regime_engine` | `repository.prune_history()`, called inline after every snapshot save — `market_regime_engine/service.py:237` | **Yes**, live |
| `economic_calendar_engine` | `repository.prune_history()`, called from the sync scheduler — `economic_calendar_engine/service.py:344` | **Yes**, live |
| `ai_scoring_engine` | `service.cleanup()` (defined `ai_scoring_engine/service.py:272`, calls `repository.prune()` twice) | **No** — zero call sites anywhere in `backend/` |
| `signal_decision_engine` | `service.cleanup()` (defined `signal_decision_engine/service.py:316`, calls `repository.prune()` twice) | **No** — zero call sites |
| `replay_engine` | `service.cleanup()` (defined `replay_engine/service.py:240`) | **No** — zero call sites |

Confirmed by direct grep of `\.cleanup\(\)` across `backend/`: **zero matches** outside the
three definitions themselves.

Tables with **no retention mechanism of any kind**, not even dead code — confirmed by absence
of any `prune`/`delete`/`cleanup` method in their repository file:
`smc_objects`, `smc_analysis_snapshots`, `liquidity_objects`, `liquidity_snapshots`,
`volume_profile_objects`, `volume_profile_snapshots`, `institutional_flow_evidence`,
`institutional_flow_snapshots`, `integration_events`, `integration_outbox`,
`integration_processed_events`, `integration_snapshots`, `operational_signals`,
`integration_event_trace`, `integration_data_quality_issues`, `market_evidence_frames`,
`unified_market_states`, `unified_market_state_timeframes`, `evidence_items`,
`unified_market_state_evidence_links`, `historical_candles`, `realtime_candles`,
`provider_metrics`, `market_quality_history`, `market_gap_history`, `market_latency_history`,
`market_synchronization_history`, `quant_forecast_*` (all 7 tables), `ai_reasoning_requests`,
`ai_market_forecasts`, `ai_forecast_scenarios`, `final_system_actions`,
`guardrail_evaluations`, `published_analytical_signals`, `managed_signals`,
`llm_usage_metrics` (confirmed via `backend/app/storage/models.py:1-1264`, full read, and
per-engine repository files).

**Why this matters regardless of which scenario below is true:** several of these tables store
large, duplicated JSONB payloads by design —
`market_evidence_frames` is documented in its own docstring as "**Full** pre-normalization
engine outputs for one closed timeframe candle" (`storage/models.py:768`), and `evidence_items`
as "State-specific evidence preserving the **complete raw analytical output**"
(`storage/models.py:828`) — i.e. the same engine output is intentionally stored at least twice
(once per evidence item, once folded into the frame), on top of the primary snapshot tables
(`smc_analysis_snapshots.payload`, `liquidity_snapshots.payload`, etc.) which also store full
JSONB engine output. **If the pipeline is producing even a modest, healthy cadence of cycles**
(one per closed M15 candle per symbol = every 15 minutes per instrument), this design — full
JSON payload written to 3-5 different tables per cycle, forever, with zero pruning on any of
them — is sufficient on its own to produce multi-GB growth over hours, with no bug required.
This is a **retention-was-never-implemented** cause, not a bug — distinguishing it from the
causes below per the task's request.

### 2. Unconditional economic-calendar background loop has zero liveness visibility (high confidence it runs; unconfirmed whether it's the dominant byte count)

**Evidence.** `EconomicCalendarService.start()` (`economic_calendar_engine/service.py:164-168`)
creates its `_poll()` scheduler task whenever `any(provider.mode.value in {"live_provider",
"public_web_source"} for provider in self.providers)` — true by default (the six keyless
public sources: BLS/BEA/Fed/Census/DOL/ECB). It is started from `main.py:276`, **independent of
`TEN_INTEGRATION_WORKER_ENABLED` / `TEN_MARKET_DATA_WORKER_ENABLED`** — i.e. this loop runs
even in the exact silent-trap scenario described in the executive summary where the main
trading pipeline is fully dark. Its retention (`prune_history`, called at
`economic_calendar_engine/service.py:344`) is wired, unlike the three dead `cleanup()`s above —
but I could not verify from static code alone whether the configured `keep_events` /
`keep_observations` / `keep_snapshots` limits are generous enough, or whether the prune call's
cadence keeps pace with the sync cadence, to actually bound growth in practice.

Separately, the silent-failure audit (below) found this is the **one background loop with no
`add_done_callback`, no `status()`/"crashed" field, and no presence in
`/api/v1/system/diagnostics`'s `"workers"` block** — so even if it were misbehaving (e.g. a
sync bug causing repeated full-history re-fetches instead of the intended incremental sync), an
operator watching the dashboard would have no way to see it. This combination — runs
unconditionally, invisible in diagnostics — makes it the most parsimonious single explanation
for "DB fills up while the dashboard shows nothing happening," because it requires the fewest
unverified assumptions (it doesn't depend on which way any of the disputed worker/shadow-mode
flags are actually set on Railway).

**Not fully verified:** whether `economic_calendar_provider_observations` (which stores a raw
`payload` and `payload_hash` per observation, `storage/models.py:411-419`) has a genuinely
stable identity per observation, or whether repeated polls of unchanged data could be inserting
duplicate rows. The `id` field is a plain `PGUUID` primary key with no additional unique
constraint in the model (`storage/models.py:413`) — I did not trace far enough into
`economic_calendar_engine/repository.py`'s insert path to confirm whether `id` is derived
deterministically from content (safe) or freshly generated per insert with a separate,
unenforced content-equality check (would allow duplicate accumulation). **This is the single
most valuable loose end to close before starting any fix.**

### 3. `realtime_candles` grows on every poll where the forming candle's price changes, not just on new closed candles (confirmed, moderate impact at current default config)

**Evidence.** `MarketDataService.latest()` (`engines/market_data_engine/service.py:129-183`)
only skips the write (`repository.append_realtime`, line 170) when the polled candle is
`same_identity` **and** has identical OHLC values to what's already stored
(line 147: `if previous is not None and same_identity and self._same_market_values(...)`). For
a currently-forming (not-yet-closed) bar, price legitimately changes on nearly every poll, so
this table receives a new row — storing the **full candle JSONB payload**
(`RealtimeCandleRecord.payload`, `storage/models.py:79`) — on close to every poll cycle. There
is no retention on this table (see #1). At the documented single-symbol/single-timeframe
default (`TEN_MARKET_DATA_SYMBOLS=["XAUUSD"]`, `TEN_MARKET_DATA_TIMEFRAMES=["M15"]`) and a
60-second poll interval, this is on the order of tens of KB/hour — not enough alone to explain
5GB/5h, but it compounds with everything else, and **scales linearly with any additional
symbols/timeframes configured on Railway that aren't reflected in `.env.example`** (unverified
— you'd need to check the live `TEN_MARKET_DATA_SYMBOLS`/`TEN_MARKET_DATA_TIMEFRAMES` values).

**Related discrepancy worth flagging:** the actual code default for the poll interval is **10
seconds** (`backend/app/core/config/settings.py:45`,
`market_data_poll_seconds: float = Field(default=10, ...)`), not the 60 seconds documented and
set explicitly in `.env.example:32`. If Railway's env doesn't set
`TEN_MARKET_DATA_POLL_SECONDS` at all (plausible if only some lines of `.env.example` were
copied), the live system polls 6x more often than the docs describe. Minor on its own, but
worth confirming — it's a one-line env var check.

### 4. If `ai_centric_shadow_mode` is actually enabled on Railway (contrary to the checked-in default), the shadow pipeline adds another full set of large, unretained JSONB tables (unconfirmed — depends entirely on live config)

If `TEN_AI_CENTRIC_SHADOW_MODE=true` is set on Railway (overriding the `configs/feature_flags.yaml:11`
default of `false`), then every cycle also writes `market_evidence_frames`, `unified_market_states`,
`unified_market_state_timeframes`, `evidence_items`, plus the quant-forecasting and AI-reasoning
tables — all in the no-retention list in #1, several storing "complete raw analytical output" by
design. This would directly contradict the dashboard showing "pending"/disabled for that same
flag (see Pipeline section, Gate B) — **so this scenario and the "dashboard is empty because the
flag is off" explanation cannot both be fully true at once**. Resolving this one flag's actual
live value resolves a large part of the ambiguity in this whole audit.

### Ruled out / checked and found not to be a factor

- **Outbox retry tight-looping.** `IntegrationRepository.fail()` (`integration/repository.py:326-350`)
  applies bounded exponential backoff (`min(2**attempts, 300)` seconds) before a failed item is
  reclaimable, and `IntegrationWorker.run()` only skips its poll-interval sleep when items were
  actually processed that cycle (`integration/worker.py:62-66`) — a persistently-failing candle
  cannot spin the worker in a tight loop. Confirmed by direct read.
- **The exact previously-fixed dedup bug recurring elsewhere.** See executive summary point 5 —
  searched exhaustively, not found recurring, with two areas flagged as not-100%-verified (see
  "What I don't know").
- **Legacy dead tables (`signals`, `analysis_results`, `engine_logs`).** Confirmed via grep that
  nothing outside their model definitions and `__init__.py` exports ever inserts into them — not
  a bloat source.

---

## Pipeline non-function: root causes ranked by how much each blocks signal generation

### 1. `TEN_INTEGRATION_WORKER_ENABLED=false` / `TEN_MARKET_DATA_WORKER_ENABLED=false` explicitly set on Railway (unconfirmed live value, but this is a known, documented, self-inflicted trap) — blocks 100% of the pipeline if true

If either is explicitly `false` on Railway (as opposed to unset), the production auto-enable
safety net does not apply (`settings.py:111-118`, only fires "if the variable was never set at
all" — `"market_data_worker_enabled" not in self.model_fields_set`). With
`market_data_worker_enabled=false`: `MarketDataWorker` never polls
(`engines/market_data_engine/worker.py:68-72`), so no candle is ever fetched, no
`NewCandle` event is ever published. With `integration_worker_enabled=false` (and the embedded
API worker fallback hardcoded off, `main.py:326`): even if candles somehow arrived, the outbox
is never drained. Either one alone is sufficient to produce every symptom described: "most
panels Unavailable," "no scenarios/signals ever published," "dashboard shows the app is barely
processing anything." This exact trap, and the fact that `/api/v1/system/diagnostics`'s
`enabled` field (not a "crashed" field) is the only place it's visible, is already documented
in the repo's own `.env.example:11-23`.

### 2. `TEN_INTEGRATION_ENABLED=false` or `TEN_LIVE_PIPELINE_ENABLED=false` (unconfirmed live value) — blocks 100% of the pipeline if true, and fails **completely silently**

Distinct from #1: `FullSystemIntegrationService.start()` — which is the only place
`NewCandle` gets a subscriber (`integration/service.py:66-69`) — is only called from
`main.py:425-426` if **both** `integration_config.enabled` and `integration_config.live_pipeline_enabled`
are true. If either is false, `MarketDataService.latest()` still runs and still calls
`event_bus.publish(NewCandle(...))` every closed candle — into a bus with **zero registered
handlers for that event type**. `InMemoryEventBus.publish()` (`events/bus.py:60-79`) builds an
empty task set and returns cleanly — no error, no log, no metric distinguishing this from "no
new candles happened." This is a more dangerous variant of #1 because there is **no diagnostics
field anywhere that would tell you this happened** — candles would still show as fetched, but
nothing downstream would ever fire.

### 3. `ai_centric_shadow_mode` defaults to `false` — blocks the specific dashboard endpoint the user is most likely watching, independent of #1/#2

`configs/feature_flags.yaml:11`. `FullSystemIntegrationService._run()` skips
`unified_market_state.capture_cycle(...)` entirely when this is off
(`integration/service.py:247-256`), so `unified_market_states` is never populated, and
`GET /api/v1/dashboard/latest` (`api/routes/dashboard.py:104-170`) returns
`"status": "pending"`, every stage `"not_available"`, reason
`"ai_centric_shadow_mode_disabled"` — **permanently, correctly, and by design**, even if the
primary trading pipeline (#1/#2 above resolved) is fully healthy and actively writing
`operational_signals`. The four related flags (`ai_signal_proposals/monitoring/publication/adjustments`,
`configs/feature_flags.yaml:12-15`) further gate the AI-reasoning/final-decision layer even if
shadow mode itself were turned on. **This does not affect whether real signals get produced
(`operational_signals`, via `signal_decision_engine`) — only whether this specific dashboard
page shows them.** If the user's mental model is "the dashboard is the pipeline," this flag
alone could produce the entire reported symptom even with a perfectly healthy backend.

### 4. Config files that look load-bearing but have zero runtime effect — not blocking, but a trap for anyone trying to fix #1-#3 by editing config

- **`configs/integration.yaml` is never loaded anywhere** — confirmed via grep, zero references
  to `"integration.yaml"` or a matching `load_model(...)` call in `backend/`. All
  `IntegrationConfig` values in production come from Pydantic field defaults plus explicit
  kwargs in `main.py:321-327`. The YAML's values (`worker.poll_seconds: 1`,
  `outbox_batch_size: 100`) currently happen to match the code defaults, so behavior is
  unaffected today — but editing this file to "fix" the pipeline would silently do nothing.
- **`configs/feature_flags.yaml`'s `EnableSMC`/`EnableLiquidity`/`EnableFlow`/`EnableVolumeProfile`/
  `EnableEconomicFilter`/`EnableAI`/`EnableMarketRegime`/`EnableReplay`/`EnableDashboardModules`
  flags (lines 2-10) are vestigial relative to the live pipeline.** They're only read by
  `EngineRegistry`/`PipelineManager.run()` (`services/registry.py:52`,
  `services/pipeline.py:97-152`) — a second, entirely separate orchestrator that is constructed
  at startup (`main.py:147-152`) but whose `.run()`/`.analyze()` method is **never called from
  anywhere in the running application** (confirmed via grep — the only live references touch
  `.event_bus`/`.feature_store`, not `.run()`). `FullSystemIntegrationService._run()` calls each
  real engine service (`self.smc.analyze_candles(...)`, etc.) **unconditionally** — there is no
  `if enabled:` check anywhere in that call chain. Toggling these flags does nothing to the live
  pipeline.
- **`GET /api/v1/signals` and `/api/v1/signals/latest` are permanently, correctly empty** —
  they read `InMemorySignalRepository` (`main.py:145`), which only the dead `PipelineManager.run()`
  path ever writes to (`.add()` has zero call sites). The frontend already knows this and avoids
  the endpoint (`frontend/src/services/api.ts:71-75` has a comment documenting exactly this), so
  it is not itself a user-visible symptom — but it's a landmine for any future external
  integration or monitoring hitting that URL expecting live data.

### 5. Miscellaneous silent-failure gaps found (lower severity, worth fixing but not primary suspects)

- `signal_decision_engine/service.py:219-220,235-236` (`_economic_context`/`_regime_context`):
  bare `except Exception: return None` — a real bug in the economic-calendar or market-regime
  context call is indistinguishable from "not configured."
- `replay_engine/worker.py:27-31`: `except Exception: continue` with no log — a failing replay
  session silently stops being processed, invisibly (low impact: `replay_worker_enabled`
  defaults `false`).
- 8+ occurrences of the pattern `except Exception: self.metrics.<counter>_failures_total += 1`
  with **no log line**, one per analytical engine's event-bus/feature-store publish call (e.g.
  `ai_scoring_engine/service.py:239-240,269-270`, `liquidity_engine/service.py:290-292`,
  `volume_profile_engine/service.py:238-239,317-319`, `institutional_flow_engine/service.py:330-332,350-352`,
  `market_regime_engine/service.py:473-475,514-516`, `signal_decision_engine/service.py:278-280,313-315`).
  Does not block the snapshot/decision itself being persisted, but a broken downstream
  publish would be invisible in logs, only visible via a metrics endpoint.
- `replay_engine/worker.py` and `economic_calendar_engine/service.py`'s scheduler tasks both
  lack the `add_done_callback` crash-detection pattern that `MarketDataWorker` and
  `IntegrationWorker` already have (`worker.py:71-72`/`33-34` vs. no equivalent in the other
  two) — if either task's own loop machinery dies outside its per-iteration try/except, it dies
  silently with no "crashed" signal anywhere in `/api/v1/system/diagnostics`.
- `MarketDataWorker`/`IntegrationWorker` themselves are confirmed **not** to have this gap — both
  have per-iteration try/except, a done-callback backstop, and a `status()["crashed"]` field
  surfaced in diagnostics. This is explicitly the fix already made in a prior session
  (`worker.py:60-66,74-86,128-141` carries the explanatory comments) and it is confirmed present
  in current code.

### Confirmed healthy / not a cause

- The full analytical call chain — `market_data.history()` → `smc.analyze_candles()` →
  `liquidity.analyze()` → `volume_profile.analyze()` → `institutional_flow.analyze()` →
  `market_regime.analyze_snapshot()` → `economic_calendar.context()` →
  (shadow-gated AI layer) → `ai_scoring.calculate()` → `signal_decision.evaluate()` →
  `operational_signals` — is fully wired, in the correct order, with every engine's
  service method confirmed to exist and be called with matching signatures
  (`integration/service.py:166-428`, cross-checked against each engine's `service.py`).
  **If the two worker-enabled flags and `integration_enabled`/`live_pipeline_enabled` are all
  true on Railway, this chain will run correctly.** This was verified by direct code trace, not
  assumed from a prior audit.
- Outbox claiming uses genuine `SELECT ... FOR UPDATE SKIP LOCKED` with 15-minute lease
  expiry and bounded exponential retry backoff — safe under multiple Railway replicas
  (`integration/repository.py:231-282,326-350`).
- Migrations are current through `20260723_0006_outbox_worker_leases.py`, matching what the
  prior persistence audit describes, and `railway.toml`'s `preDeployCommand = "python -m alembic
  upgrade head"` should apply them automatically on each deploy — **not independently verified
  against the live database's actual `alembic_version`,** see "What I don't know."

---

## Proposed fixes (description only — not implemented this pass)

| # | Root cause | Proposed fix (description only) | Risk/complexity | Session sizing |
|---|---|---|---|---|
| DB-1 | No retention on ~25 tables | Add scheduled pruning (age- or count-based) for every analytical/evidence/integration table, mirroring the pattern already proven in `market_regime_engine`/`economic_calendar_engine`. Needs a decision on retention windows per table (regulatory/audit tables like `signal_decisions`/`operational_signals` may need longer retention than raw evidence). | Medium — mechanical per-table, but many tables means many small changes plus a decision on what "safe to delete" means for audit-relevant records. | Dedicated session — real design decisions needed on retention windows, not just wiring. |
| DB-1b | `cleanup()` dead code for ai_scoring/signal_decision/replay | Wire the three existing `cleanup()` methods into a scheduled task (e.g. alongside the existing `market_regime`/`economic_calendar` pattern, or a shared periodic-maintenance loop in `main.py`). | Low — the retention logic already exists and is presumably tested; this is "call the function that's already there." | Quick fix. |
| DB-2 | Economic-calendar loop invisible + unverified dedup | (a) Add `status()`/liveness reporting to `EconomicCalendarService` and surface it in `/api/v1/system/diagnostics` alongside the other two workers. (b) Verify (via the query below) whether `economic_calendar_provider_observations` row count/growth is actually anomalous, and if so trace the exact insert path for a missing content-based dedup key. | Low for (a); unknown for (b) until verified. | (a) Quick fix. (b) Needs the live-DB check first — don't guess. |
| DB-3 | `realtime_candles` grows every price tick | Either stop persisting every tick (keep only the latest realtime value, already cached separately — `service.py:173`) or add retention/downsampling. Decide whether `realtime_candles`'s audit value (every observed tick) is actually needed, or whether `historical_candles` (already deterministically upserted) is sufficient. | Low-medium — depends on whether anything currently reads `realtime_candles` history for a real purpose (needs a quick usage check before removing). | Quick fix once the "is this data used" question is answered. |
| P-1 | Worker-enabled flags explicitly `false` on Railway | Delete `TEN_INTEGRATION_WORKER_ENABLED`/`TEN_MARKET_DATA_WORKER_ENABLED` from Railway's env entirely (letting the auto-enable safety net work), or set them explicitly `true`. This is a **Railway dashboard change, not a code change.** | Very low — one-line env var edit, but verify current value first (see below). | Quick fix, pending verification. |
| P-2 | `integration_enabled`/`live_pipeline_enabled` possibly false, fails silently | Same as P-1 (env var check/fix). Additionally, consider adding a startup-time warning/metric when `event_bus.publish()` is called for an event type with zero registered handlers — would have caught this class of misconfiguration immediately instead of needing a full audit. | Low for the env fix; low-medium for the "0 handlers" warning (touches the shared event bus). | Env check: quick fix. Handler-count warning: quick fix, good candidate for the same session. |
| P-3 | `ai_centric_shadow_mode` off by default | Confirm with the user whether the AI-centric dashboard is actually meant to be the primary view. If yes, enable the flag (and its four siblings as appropriate) once the primary pipeline (P-1/P-2) is confirmed healthy — enabling it before that just adds more unbounded writes (see DB-4) without producing visible value. | Low technically, but is a product decision, not just a bug fix. | Discuss with user before touching. |
| P-4 | Dead `configs/integration.yaml`/`pipeline.yaml`/`EnableX` flags | Either wire these into the live path for real, or delete/clearly-deprecate them to stop them from misleading whoever edits them next expecting an effect. | Low-medium — mostly cleanup, but touches config loading. | Follow-up session, not urgent. |
| P-5 | Silent `except Exception: return None` / metrics-only failure swallowing | Add `logger.exception(...)` calls at each of the ~10 sites listed above; low-risk, additive-only change. | Very low. | Quick fix, good first PR. |
| P-6 | Missing done-callback/crashed-status on replay/economic-calendar workers | Apply the same `add_done_callback` + `status()["crashed"]` pattern already used by `MarketDataWorker`/`IntegrationWorker` to the other two background loops. | Low — proven pattern, just needs replicating. | Quick fix. |

---

## What I don't know / need from you

I have no live database or Railway access in this sandbox. Please run the following and paste
results back — they resolve nearly every open ambiguity in this report:

1. **Actual Railway environment variable values** (the single highest-value thing to check —
   resolves P-1, P-2, P-3, and DB-4 simultaneously):
   ```
   TEN_ENVIRONMENT
   TEN_INTEGRATION_WORKER_ENABLED
   TEN_MARKET_DATA_WORKER_ENABLED
   TEN_INTEGRATION_ENABLED
   TEN_LIVE_PIPELINE_ENABLED
   TEN_AI_CENTRIC_SHADOW_MODE
   TEN_MARKET_DATA_SYMBOLS
   TEN_MARKET_DATA_TIMEFRAMES
   TEN_MARKET_DATA_POLL_SECONDS
   ```
   (Railway dashboard → your service → Variables tab; or `railway variables` if you have the CLI
   linked locally.)

2. **Database size and top tables** — run against the production `TEN_DATABASE_URL`:
   ```sql
   SELECT pg_size_pretty(pg_database_size(current_database()));

   SELECT relname AS table_name,
          pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
          n_live_tup AS row_count
   FROM pg_stat_user_tables
   ORDER BY pg_total_relation_size(relid) DESC
   LIMIT 15;
   ```

3. **Growth rate per table** — same query again in ~1 hour, diffed, or directly:
   ```sql
   SELECT 'economic_calendar_provider_observations' AS t, count(*), min(ingested_at), max(ingested_at) FROM economic_calendar_provider_observations
   UNION ALL
   SELECT 'realtime_candles', count(*), min(received_at), max(received_at) FROM realtime_candles
   UNION ALL
   SELECT 'smc_analysis_snapshots', count(*), min(created_at), max(created_at) FROM smc_analysis_snapshots
   UNION ALL
   SELECT 'operational_signals', count(*), min(effective_at), max(effective_at) FROM operational_signals
   UNION ALL
   SELECT 'unified_market_states', count(*), min(created_at), max(created_at) FROM unified_market_states;
   ```
   This directly tells us: is the trading pipeline producing cycles at all (row count/timespan
   in `smc_analysis_snapshots`/`operational_signals`), is shadow mode actually active
   (`unified_market_states` row count), and is the economic-calendar loop the outlier
   (`economic_calendar_provider_observations` row count vs. timespan — should be at most a few
   hundred real-world events, not thousands).

4. **Deployed commit / migration state**, to confirm the fixes already made this week are
   actually live (not just merged):
   ```sql
   SELECT * FROM alembic_version;
   ```
   Compare against `migrations/versions/20260723_0006_outbox_worker_leases.py` being the head
   revision. Also check Railway's deploy log for the currently-running deploy's commit SHA
   against `git log -1` (`956731f` at the time of this audit) — the app also logs its own
   `git_sha` per cycle (`integration/service.py:187`, reads `RAILWAY_GIT_COMMIT_SHA`), so
   `grep git_sha` in Railway's live logs would confirm this without a redeploy.

5. **Two hash/identity constructions I spot-checked but did not trace to full rigor** (lowest
   priority, but the task explicitly asked for exhaustive coverage of this pattern):
   - `economic_calendar_engine`'s provider-observation `id`/dedup path
     (`economic_calendar_engine/repository.py`, `storage/models.py:411-419`) — needs one more
     read to confirm whether repeated polls of unchanged events could insert duplicate rows.
   - Liquidity/Volume Profile/Institutional Flow/Economic Calendar engines' `snapshot_id`/
     `context_id` construction (confirmed deterministic for SMC and Market Regime via
     `stable_id(...)`; not confirmed to the same depth for the other three analytical engines).

6. **Railway resource metrics** (memory/CPU/restart count, OOM events) — not accessible from
   this sandbox at all. Railway's own metrics dashboard (Observability tab) for the service
   would show restart count and memory usage directly; if the deploy has been OOM-killed
   repeatedly, that's independently discoverable there and would also explain intermittent
   pipeline stalls without needing to trace it through code.

---

## Explicitly not done in this pass

No code was modified. No migrations were run. No config files were changed. This document is
diagnosis only, per the task's instructions — fixes should be selected and scoped in a follow-up
session once the live-environment questions above are answered, starting with whichever finding
above is both confirmed and ranked lowest-risk/quick-fix (candidates: P-1/P-2 env var checks,
and DB-1b wiring the three existing dead `cleanup()` calls).
