import { AlertTriangle, BadgeCheck, Sparkles } from 'lucide-react'
import type { ExplainResponse } from '../types'

function ScoreMeter({ score }: { score: ExplainResponse['explainability_score'] }) {
  return (
    <div className="explain-score">
      <div className="explain-score__bar"><span style={{ width: `${score.percent}%` }} /></div>
      <span>{score.percent}% explanation confidence · {score.engines_available}/{score.engines_total} engines available · {score.evidence_citations} evidence citations</span>
    </div>
  )
}

/** Renders one grounded `/api/v1/explain/*` response. Every field here traces back to TEN's own
 * engine outputs — `evidence` cites exactly which snapshot backed the answer, and `error` (rather
 * than a fabricated explanation) is what renders if the provider failed or returned garbage. */
export function AIExplanationPanel({ data, loading, error, onExplain, actionLabel = 'Explain this' }: {
  data: ExplainResponse | null
  loading: boolean
  error: string | null
  onExplain: () => void
  actionLabel?: string
}) {
  if (loading) {
    return (
      <div className="explain-panel explain-panel--loading">
        <div className="skeleton skeleton-line" style={{ width: '70%' }} />
        <div className="skeleton skeleton-line" style={{ width: '92%' }} />
        <div className="skeleton skeleton-line" style={{ width: '55%' }} />
      </div>
    )
  }
  if (!data) {
    return (
      <div className="explain-panel explain-panel--empty">
        <Sparkles size={20} />
        <p>TEN's Explainability Layer can walk through this in plain language — grounded only in TEN's own engine outputs, nothing invented, nothing external.</p>
        <button onClick={onExplain}><Sparkles size={13} />{actionLabel}</button>
        {error && <p className="explain-panel__error">{error}</p>}
      </div>
    )
  }
  const { explanation, error: explainError, explainability_score: score, evidence } = data
  return (
    <div className="explain-panel">
      <ScoreMeter score={score} />
      {explainError && <div className="alert"><span>AI unavailable</span>{explainError}</div>}
      {explanation && (
        <>
          <p className="explain-panel__summary">{explanation.summary}</p>
          <div className="explain-columns">
            <div>
              <p className="explain-columns__label"><BadgeCheck size={12} /> Supporting</p>
              <ul>{explanation.primary_reasons.length ? explanation.primary_reasons.map((reason, index) => <li key={index}>{reason}</li>) : <li className="explain-empty-item">None cited</li>}</ul>
            </div>
            <div>
              <p className="explain-columns__label"><AlertTriangle size={12} /> Opposing</p>
              <ul>{explanation.opposing_factors.length ? explanation.opposing_factors.map((reason, index) => <li key={index}>{reason}</li>) : <li className="explain-empty-item">None cited</li>}</ul>
            </div>
          </div>
          {explanation.engine_breakdown.length > 0 && (
            <div className="explain-engines">
              {explanation.engine_breakdown.map((item) => (
                <div className="explain-engines__item" key={item.engine}><b>{item.engine.replaceAll('_', ' ')}</b><span>{item.influence}</span><small>{item.note}</small></div>
              ))}
            </div>
          )}
          {explanation.required_for_change.length > 0 && (
            <div className="explain-block"><p className="explain-columns__label">What would need to change</p><ul>{explanation.required_for_change.map((item, index) => <li key={index}>{item}</li>)}</ul></div>
          )}
          {explanation.caveats.length > 0 && (
            <div className="explain-block explain-block--muted"><p className="explain-columns__label">Caveats</p><ul>{explanation.caveats.map((item, index) => <li key={index}>{item}</li>)}</ul></div>
          )}
        </>
      )}
      {evidence.length > 0 && (
        <div className="explain-evidence">
          <span>Evidence</span>
          {evidence.map((item) => <code key={`${item.source}-${item.reference_id}`} title={item.timestamp ?? ''}>{item.source} #{item.reference_id.slice(0, 8)}</code>)}
        </div>
      )}
      <button className="explain-panel__refresh" onClick={onExplain} disabled={loading}>Re-explain</button>
    </div>
  )
}
