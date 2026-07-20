import { Minus, TrendingDown, TrendingUp } from 'lucide-react'
import type { OperationalSignal } from '../types'

const DIRECTION_ICON: Record<string, typeof TrendingUp> = { bullish: TrendingUp, bearish: TrendingDown }

function DirectionIcon({ direction }: { direction: string }) {
  const Icon = DIRECTION_ICON[direction] ?? Minus
  return <Icon size={15} />
}

export function SignalTable({ signals }: { signals: OperationalSignal[] }) {
  if (!signals.length) {
    return (
      <div className="empty-state">
        <span className="empty-state__mark">10</span>
        <h3>Waiting for qualified confluence</h3>
        <p>No market scenario is active. TEN publishes only after its analysis pipeline completes.</p>
      </div>
    )
  }

  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Market</th><th>Bias</th><th>State</th><th>Confidence</th><th>Data quality</th><th>Effective</th><th>Expires</th></tr></thead>
        <tbody>{signals.map((signal) => (
          <tr key={signal.operational_signal_id}>
            <td><strong>{signal.instrument}</strong><small>{signal.timeframe}</small></td>
            <td><span className={`direction direction--${signal.direction}`}><DirectionIcon direction={signal.direction} />{signal.direction}</span></td>
            <td>{signal.state.replaceAll('_', ' ')}</td>
            <td><div className="confidence"><span style={{ width: `${signal.confidence}%` }} /><b>{Math.round(signal.confidence)}%</b></div></td>
            <td>{signal.data_quality_status}</td>
            <td>{new Date(signal.effective_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
            <td>{new Date(signal.expires_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}
