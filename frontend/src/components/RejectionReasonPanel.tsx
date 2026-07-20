import { AlertTriangle, CheckCircle2, HelpCircle, XCircle } from 'lucide-react'
import type { DiagnosticStatus, RejectionsResponse } from '../types'

const STATUS_ICON: Record<DiagnosticStatus, typeof CheckCircle2> = {
  passed: CheckCircle2,
  failed: XCircle,
  not_evaluated: HelpCircle,
  informational: AlertTriangle,
}

export function RejectionReasonPanel({ data }: { data: RejectionsResponse | null }) {
  if (!data || data.count === 0) {
    return (
      <div className="empty-state">
        <span className="empty-state__mark">10</span>
        <h3>No rejected setups recently</h3>
        <p>{data ? 'Every recent decision was eligible, or none has run yet.' : 'Waiting for decision history…'}</p>
      </div>
    )
  }
  return (
    <div className="rejection-list">
      {data.rejections.map((item) => (
        <details key={item.decision_id} className="rejection">
          <summary>
            <span className={`direction direction--${item.direction === 'bullish' ? 'long' : item.direction === 'bearish' ? 'short' : 'neutral'}`}>{item.direction}</span>
            <b>{item.state.replaceAll('_', ' ')}</b>
            <span>{new Date(item.as_of).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
            <span>{item.confidence_score.toFixed(1)}% confidence</span>
          </summary>
          <div className="rejection__body">
            {item.ai_score_unavailable && <p className="rejection__note">AI score snapshot unavailable ({item.ai_score_unavailable}) — diagnostics below are partial.</p>}
            <ul className="diagnostics">
              {item.diagnostics.map((entry) => {
                const Icon = STATUS_ICON[entry.status]
                return (
                  <li key={entry.key} className={`diagnostics__item diagnostics__item--${entry.status}`}>
                    <Icon size={14} />
                    <span>{entry.label}</span>
                    <small>{entry.detail}</small>
                  </li>
                )
              })}
            </ul>
          </div>
        </details>
      ))}
    </div>
  )
}
