import type { MarketIntelligence, SourceDiagnostic } from '../types'

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="intel-field">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  )
}

const number = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const na = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value)
const time = (value: string | null) => value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'

function DiagnosticRow({ item }: { item: SourceDiagnostic }) {
  return (
    <tr className={`source-diagnostics__row source-diagnostics__row--${item.status}`}>
      <td>{item.source.replaceAll('_', ' ')}</td>
      <td>{item.status}</td>
      <td>{item.snapshot_found ? time(item.snapshot_timestamp) : '—'}</td>
      <td>{item.age_seconds === null ? '—' : `${Math.round(item.age_seconds)}s`}</td>
      <td>{item.freshness}</td>
      <td>{item.error ?? '—'}</td>
    </tr>
  )
}

export function MarketIntelligencePanel({ data }: { data: MarketIntelligence | null }) {
  if (!data) {
    return <div className="empty-state"><h3>Loading market intelligence…</h3></div>
  }
  const candle = data.current_candle
  return (
    <div>
      <div className="intel-grid">
        <Field label="Current symbol" value={data.instrument} />
        <Field label="Current session" value={na(data.current_session.replaceAll('_', ' '))} />
        <Field label="Current candle" value={candle ? `${number.format(candle.close)} @ ${new Date(candle.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : '—'} />
        <Field label="Spread" value={data.spread === null ? 'unavailable' : number.format(data.spread)} />
        <Field label="Trend" value={na(data.market_regime.trend_regime)} />
        <Field label="Market regime" value={na(data.market_regime.dominant_regime)} />
        <Field label="Current bias" value={na(data.current_bias)} />
        <Field label="HTF bias" value={data.htf_bias ? Object.entries(data.htf_bias).map(([tf, dir]) => `${tf}:${dir}`).join(' ') : '—'} />
        <Field label="Liquidity direction" value={data.liquidity.available ? 'available' : 'unavailable'} />
        <Field label="Premium / discount" value={na(data.premium_discount)} />
        <Field label="Current FVG" value={data.current_fvg ? `${data.current_fvg.type.replaceAll('_', ' ')}` : 'none active'} />
        <Field label="Current order block" value={data.current_order_block ? `${data.current_order_block.type.replaceAll('_', ' ')}` : 'none active'} />
        <Field label="Current BOS" value={data.current_bos ? `${data.current_bos.direction} @ ${number.format(data.current_bos.price)}` : 'none recent'} />
        <Field label="Current CHOCH" value={data.current_choch ? `${data.current_choch.direction} @ ${number.format(data.current_choch.price)}` : 'none recent'} />
        <Field label="Institutional flow" value={data.institutional_flow.available ? 'available' : 'unavailable'} />
        <Field label="Volume profile" value={data.volume_profile.available ? 'available' : 'unavailable'} />
        <Field label="Economic status" value={data.economic_status.degraded ? 'degraded' : na(data.economic_status.risk_window_phase)} />
        <Field label="Confidence %" value={data.confidence_percent === null ? '—' : `${data.confidence_percent.toFixed(1)}%`} />
        <Field label="Scenario readiness %" value={data.scenario_readiness_percent === null ? '—' : `${data.scenario_readiness_percent.toFixed(1)}%`} />
        <Field label="Decision status" value={data.decision_status ? `${data.decision_status.replaceAll('_', ' ')}${data.decision_active ? '' : ' (expired)'}` : '—'} />
        <Field label="Latest candle" value={time(data.latest_candle_timestamp)} />
        <Field label="Response generated" value={time(data.generated_at)} />
      </div>
      <table className="source-diagnostics">
        <thead><tr><th>Source</th><th>Status</th><th>Snapshot at</th><th>Age</th><th>Freshness</th><th>Error</th></tr></thead>
        <tbody>{data.diagnostics.map((item) => <DiagnosticRow item={item} key={item.source} />)}</tbody>
      </table>
    </div>
  )
}
