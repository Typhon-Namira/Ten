import { AlertTriangle, CheckCircle2, HelpCircle, XCircle } from 'lucide-react'
import { AIExplanationPanel } from './AIExplanationPanel'
import { useExplain } from '../hooks/useExplain'
import { tenApi } from '../services/api'
import type { DiagnosticStatus, RejectedDecision, RejectionsResponse } from '../types'

const STATUS_ICON: Record<DiagnosticStatus, typeof CheckCircle2> = {
  passed: CheckCircle2,
  failed: XCircle,
  not_evaluated: HelpCircle,
  informational: AlertTriangle,
}

function primaryBlockers(item: RejectedDecision, limit = 3) {
  return item.diagnostics.filter((entry) => entry.status === 'failed').slice(0, limit)
}

function RejectionCard({ item }: { item: RejectedDecision }) {
  const blockers = primaryBlockers(item)
  const remaining = item.diagnostics.length - blockers.length
  const explain = useExplain(() => tenApi.explainRejection(item.decision_id))
  return (
    <details className="rejection">
      <summary>
        <div className="rejection__headline">
          <span className={`direction direction--${item.direction === 'bullish' ? 'long' : item.direction === 'bearish' ? 'short' : 'neutral'}`}>{item.direction}</span>
          <b>{item.state.replaceAll('_', ' ')}</b>
          <span>{new Date(item.as_of).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
          <span>{item.confidence_score.toFixed(1)}% confidence</span>
        </div>
        <ul className="rejection__primary-blockers">
          {blockers.length === 0 && <li className="rejection__primary-blockers--none">No individual check failed outright — see full detail</li>}
          {blockers.map((entry) => (
            <li key={entry.key}><XCircle size={13} />{entry.label}{entry.detail && <small> — {entry.detail}</small>}</li>
          ))}
        </ul>
        <span className="rejection__expand-hint">{remaining > 0 ? `+${remaining} more checks — click to expand` : 'click for full detail'}</span>
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
        <div className="rejection__explain">
          <AIExplanationPanel data={explain.data} loading={explain.loading} error={explain.error} onExplain={() => void explain.run()} actionLabel="Explain this rejection" />
        </div>
      </div>
    </details>
  )
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
      {data.rejections.map((item) => <RejectionCard item={item} key={item.decision_id} />)}
    </div>
  )
}
