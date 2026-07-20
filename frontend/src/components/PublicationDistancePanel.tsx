import { TrendingUp } from 'lucide-react'
import type { MarketIntelligence, RejectionsResponse } from '../types'
import { Gauge } from './widgets/Widgets'

/** Replaces a bare "0 signals" with the reason why — how close the current analysis is to a
 * publishable scenario, not just that nothing has published yet. Sourced from the same
 * marketIntelligence/rejections data already on screen elsewhere, so this can never disagree
 * with the confidence gauge or the rejected-setups panel above it. */
export function PublicationDistancePanel({ intelligence, rejections }: { intelligence: MarketIntelligence | null; rejections: RejectionsResponse | null }) {
  const latestRejection = rejections?.rejections[0] ?? null
  const confidenceRule = latestRejection?.diagnostics.find((item) => item.key === 'confidence_too_low')
  const threshold = typeof confidenceRule?.threshold === 'number' ? confidenceRule.threshold : null
  const confidence = intelligence?.confidence_percent ?? null
  const distance = confidence !== null && threshold !== null ? Math.max(0, threshold - confidence) : null
  const blockers = latestRejection?.diagnostics.filter((item) => item.status === 'failed') ?? []

  return (
    <div className="empty-state empty-state--publication">
      <span className="empty-state__mark">10</span>
      <h3>No scenario published yet</h3>
      <p>TEN publishes only after its analysis pipeline clears every gate below.</p>
      <div className="publication-distance">
        <Gauge value={confidence} label="Current confidence" />
        <div className="publication-distance__stat">
          <span>Publication threshold</span>
          <b>{threshold === null ? 'not yet evaluated' : `${threshold.toFixed(0)}%`}</b>
        </div>
        <div className="publication-distance__stat">
          <span>Distance to publication</span>
          <b>{distance === null ? '—' : <><TrendingUp size={13} /> {distance.toFixed(1)} pts</>}</b>
        </div>
        <div className="publication-distance__stat">
          <span>Scenario readiness</span>
          <b>{intelligence?.scenario_readiness_percent === null || intelligence?.scenario_readiness_percent === undefined ? '—' : `${intelligence.scenario_readiness_percent.toFixed(0)}%`}</b>
        </div>
      </div>
      {blockers.length > 0 && (
        <div className="publication-distance__blockers">
          <p>Remaining blockers</p>
          <ul>
            {blockers.map((item) => <li key={item.key}>{item.label}{item.detail && <small> — {item.detail}</small>}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}
