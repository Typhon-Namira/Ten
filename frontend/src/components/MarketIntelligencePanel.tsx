import type { ChartLiquidityPool, MarketIntelligence, SourceDiagnostic } from '../types'
import { normalizeEngineState, stateFromAvailability } from '../lib/engineState'
import { tradingContextBadge } from '../lib/economicState'
import { StateBadge } from './StateBadge'
import { EconomicStateBadge } from './EconomicStateBadge'
import { BarChart, BiasArrow, Gauge, HeatmapStrip, RadarChart, RadialProgress } from './widgets/Widgets'

const number = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const na = (value: unknown) => (value === null || value === undefined || value === '' ? '—' : String(value))
const time = (value: string | null) => (value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—')

/** Pulls whatever numeric fields exist in an untyped engine state/quality record into a small bar
 * chart — these records vary per engine (liquidity's `state` is flat, institutional_flow's is a
 * tree of sub-objects like `participation.score`) so this walks one level of nesting rather than
 * hardcoding field names the backend doesn't guarantee, and rather than only ever finding
 * top-level scalars and silently rendering an empty chart for a genuinely nested payload. */
function numericEntries(record: Record<string, unknown> | null | undefined): { label: string; value: number }[] {
  if (!record) return []
  const entries: { label: string; value: number }[] = []
  for (const [key, value] of Object.entries(record)) {
    if (typeof value === 'number') {
      entries.push({ label: key.replaceAll('_', ' ').slice(0, 10), value })
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [nestedKey, nestedValue] of Object.entries(value as Record<string, unknown>)) {
        if (typeof nestedValue === 'number') entries.push({ label: nestedKey.replaceAll('_', ' ').slice(0, 10), value: nestedValue })
      }
    }
    if (entries.length >= 6) break
  }
  return entries.slice(0, 6)
}

function DiagnosticRow({ item }: { item: SourceDiagnostic }) {
  const state = normalizeEngineState(item.status)
  return (
    <tr className={`source-diagnostics__row source-diagnostics__row--${item.status}`}>
      <td>{item.source.replaceAll('_', ' ')}</td>
      <td><StateBadge state={state} /></td>
      <td>{item.snapshot_found ? time(item.snapshot_timestamp) : '—'}</td>
      <td>{item.age_seconds === null ? '—' : `${Math.round(item.age_seconds)}s`}</td>
      <td>{item.freshness}</td>
      <td>{item.error ?? '—'}</td>
    </tr>
  )
}

export function MarketIntelligencePanel({ data, liquidityPools = [] }: { data: MarketIntelligence | null; liquidityPools?: ChartLiquidityPool[] }) {
  if (!data) {
    return <div className="empty-state"><h3>Loading market intelligence…</h3></div>
  }
  const candle = data.current_candle
  // A bare "—" for confidence/decision gives no way to tell "the pipeline hasn't reached this
  // stage yet" apart from "it ran and genuinely produced nothing" — surface the diagnostics
  // table's own per-source reason (its latest attempt's error, or its status) inline instead.
  const reasonFor = (source: string): string => {
    const diagnostic = data.diagnostics.find((item) => item.source === source)
    if (!diagnostic) return 'not yet attempted'
    return diagnostic.error ?? diagnostic.status.replaceAll('_', ' ')
  }

  const engineRows: { label: string; available: boolean }[] = [
    { label: 'SMC', available: data.current_bos !== null || data.current_choch !== null || data.current_bias !== null },
    { label: 'Liquidity', available: data.liquidity.available },
    { label: 'Volume profile', available: data.volume_profile.available },
    { label: 'Institutional flow', available: data.institutional_flow.available },
    { label: 'Market regime', available: data.market_regime.available },
    { label: 'Economic calendar', available: data.economic_status.available && !data.economic_status.degraded },
    { label: 'AI scoring', available: data.confidence_percent !== null },
    { label: 'Signal decision', available: data.decision_status !== null },
  ]

  return (
    <div className="intel">
      <div className="intel__gauges">
        <Gauge value={data.confidence_percent} label="Confidence" />
        <Gauge value={data.risk_percent} label="Market risk" invert />
        <RadialProgress value={data.scenario_readiness_percent} label="Scenario readiness" />
        <div className="intel__bias-block">
          <span className="intel__bias-block-label">Current bias</span>
          <BiasArrow direction={data.current_bias} />
          <span className="intel__bias-block-sub">{na(data.premium_discount)}</span>
        </div>
      </div>
      {(data.confidence_percent === null || data.scenario_readiness_percent === null) && (
        <p className="intel__reason">
          {data.confidence_percent === null && <>Confidence unavailable — {reasonFor('ai_scoring')}. </>}
          {data.scenario_readiness_percent === null && <>Scenario readiness unavailable — {reasonFor('signal_decision')}.</>}
        </p>
      )}

      <div className="intel-grid">
        <div className="intel-field"><span>Current symbol</span><b>{data.instrument}</b></div>
        <div className="intel-field"><span>Current session</span><b>{na(data.current_session.replaceAll('_', ' '))}</b></div>
        <div className="intel-field"><span>Current candle</span><b>{candle ? `${number.format(candle.close)} @ ${new Date(candle.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : '—'}</b></div>
        <div className="intel-field"><span>Spread</span><b>{data.spread === null ? 'unavailable' : number.format(data.spread)}</b></div>
        <div className="intel-field"><span>HTF bias</span><b>{data.htf_bias ? Object.entries(data.htf_bias).map(([tf, dir]) => `${tf}:${dir}`).join(' ') : '—'}</b></div>
        <div className="intel-field"><span>Current FVG</span><b>{data.current_fvg ? data.current_fvg.type.replaceAll('_', ' ') : 'none active'}</b></div>
        <div className="intel-field"><span>Current order block</span><b>{data.current_order_block ? data.current_order_block.type.replaceAll('_', ' ') : 'none active'}</b></div>
        <div className="intel-field"><span>Current BOS</span><b>{data.current_bos ? `${data.current_bos.direction} @ ${number.format(data.current_bos.price)}` : 'none recent'}</b></div>
        <div className="intel-field"><span>Current CHOCH</span><b>{data.current_choch ? `${data.current_choch.direction} @ ${number.format(data.current_choch.price)}` : 'none recent'}</b></div>
        <div className="intel-field"><span>Decision status</span><b>{data.decision_status ? `${data.decision_status.replaceAll('_', ' ')}${data.decision_active ? '' : ' (expired)'}` : `unavailable — ${reasonFor('signal_decision')}`}</b></div>
        <div className="intel-field"><span>Latest candle</span><b>{time(data.latest_candle_timestamp)}</b></div>
        <div className="intel-field"><span>Response generated</span><b>{time(data.generated_at)}</b></div>
      </div>

      <div className="intel__widgets">
        <div className="intel__widget-card">
          <p className="intel__widget-title">Market regime</p>
          <div className="regime-readout">
            <StateBadge state={stateFromAvailability(data.market_regime.available)} />
            <BiasArrow direction={data.market_regime.directional_bias} />
            <span>{na(data.market_regime.dominant_regime)} / {na(data.market_regime.trend_regime)}</span>
          </div>
          <BarChart
            // trend_strength/volatility_score/confidence are all 0-1 fractions on the backend —
            // scaled to 0-100 here so this bar chart shares the same percent axis as every other
            // widget on this panel instead of rendering as a near-invisible sliver.
            data={[
              { label: 'trend', value: (data.market_regime.trend_strength ?? 0) * 100, color: '#c4a359' },
              { label: 'vol', value: (data.market_regime.volatility_score ?? 0) * 100, color: '#8f7bd6' },
              { label: 'conf', value: (data.market_regime.confidence ?? 0) * 100, color: '#59b993' },
            ]}
            width={200}
            height={84}
          />
        </div>
        <div className="intel__widget-card">
          <p className="intel__widget-title">Liquidity distribution</p>
          <StateBadge state={stateFromAvailability(data.liquidity.available)} />
          <HeatmapStrip
            cells={[...liquidityPools]
              .sort((a, b) => a.lower - b.lower)
              .map((pool) => ({ value: pool.side === 'sell_side' ? -pool.strength : pool.strength, label: `${pool.side.replace('_side', '')} ${pool.lower.toFixed(0)}-${pool.upper.toFixed(0)} (${pool.lifecycle_state})` }))}
          />
        </div>
        <div className="intel__widget-card">
          <p className="intel__widget-title">Institutional flow</p>
          <StateBadge state={stateFromAvailability(data.institutional_flow.available)} />
          <BarChart data={numericEntries(data.institutional_flow.state ?? data.institutional_flow.quality).map((entry) => ({ ...entry, color: '#59b993' }))} width={200} height={84} />
        </div>
        <div className="intel__widget-card">
          <p className="intel__widget-title">Economic risk</p>
          <div className="economic-stage-list">
            <div><StateBadge state={data.economic_status.stages ? normalizeEngineState(data.economic_status.stages.provider_health.status) : 'waiting'} /><span>Provider</span></div>
            <div><StateBadge state={data.economic_status.stages ? normalizeEngineState(data.economic_status.stages.downloaded_events.status) : 'waiting'} /><span>Downloaded</span></div>
            <div><StateBadge state={data.economic_status.stages ? normalizeEngineState(data.economic_status.stages.mapped_events.status) : 'waiting'} /><span>Mapped</span></div>
            <div><StateBadge state={data.economic_status.stages ? normalizeEngineState(data.economic_status.stages.relevant_events.status) : 'waiting'} /><span>Relevant</span></div>
            <div>
              {data.economic_status.stages
                ? <EconomicStateBadge visual={tradingContextBadge(data.economic_status.stages.trading_context.context_state, data.economic_status.stages.trading_context.risk_window_phase)} detail={data.economic_status.stages.trading_context.reason ?? undefined} />
                : <StateBadge state="waiting" />}
              <span>Trading context</span>
            </div>
          </div>
          {data.economic_status.next_relevant_event && <p className="intel__widget-sub">Next: {data.economic_status.next_relevant_event}</p>}
        </div>
        <div className="intel__widget-card">
          <p className="intel__widget-title">Signal quality</p>
          <RadarChart
            axes={[
              { label: 'confidence', value: data.confidence_percent ?? 0 },
              { label: 'safety', value: data.risk_percent === null ? 0 : 100 - data.risk_percent },
              { label: 'readiness', value: data.scenario_readiness_percent ?? 0 },
              { label: 'regime', value: (data.market_regime.confidence ?? 0) * 100 },
              { label: 'econ', value: data.economic_status.stages ? 100 - (data.economic_status.stages.trading_context.risk_score ?? 0) * 100 : 0 },
            ]}
          />
        </div>
        <div className="intel__widget-card intel__widget-card--wide">
          <p className="intel__widget-title">Engine health matrix</p>
          <div className="status-matrix">
            {engineRows.map((row) => (
              <div className="status-matrix__cell" key={row.label}>
                <StateBadge state={stateFromAvailability(row.available)} />
                <span>{row.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <table className="source-diagnostics">
        <thead><tr><th>Source</th><th>Status</th><th>Snapshot at</th><th>Age</th><th>Freshness</th><th>Error</th></tr></thead>
        <tbody>{data.diagnostics.map((item) => <DiagnosticRow item={item} key={item.source} />)}</tbody>
      </table>
    </div>
  )
}
