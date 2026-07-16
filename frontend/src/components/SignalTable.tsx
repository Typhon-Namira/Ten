import { Minus, TrendingDown, TrendingUp } from 'lucide-react'
import type { Signal } from '../types'

const number = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

function DirectionIcon({ direction }: Pick<Signal, 'direction'>) {
  return direction === 'long' ? <TrendingUp size={15} /> : direction === 'short' ? <TrendingDown size={15} /> : <Minus size={15} />
}

export function SignalTable({ signals }: { signals: Signal[] }) {
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
        <thead><tr><th>Market</th><th>Bias</th><th>Entry zone</th><th>Stop</th><th>Target</th><th>Confidence</th><th>Generated</th></tr></thead>
        <tbody>{signals.map((signal) => (
          <tr key={`${signal.symbol}-${signal.timestamp}`}>
            <td><strong>{signal.symbol}</strong><small>{signal.timeframe}</small></td>
            <td><span className={`direction direction--${signal.direction}`}><DirectionIcon direction={signal.direction} />{signal.direction}</span></td>
            <td>{number.format(signal.entry_zone[0])}–{number.format(signal.entry_zone[1])}</td>
            <td>{number.format(signal.stop_loss)}</td>
            <td>{number.format(signal.take_profit)}</td>
            <td><div className="confidence"><span style={{ width: `${signal.confidence * 100}%` }} /><b>{Math.round(signal.confidence * 100)}%</b></div></td>
            <td>{new Date(signal.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

