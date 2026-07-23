import type { AIReasoningDashboard } from '../types'

const percentage = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const price = (value: number | null | undefined) => value == null ? '—' : value.toFixed(2)

export function AIReasoningPanel({ data }: { data: AIReasoningDashboard | null }) {
  const forecast = data?.forecast
  const proposal = data?.proposal
  const signal = data?.managed_signals[0]
  const history = signal ? data?.signal_histories[signal.signal_id] : null

  return <div className="ai-reasoning">
    <div className="ai-reasoning__banner">SHADOW-ONLY · AWAITING GUARDRAIL VALIDATION · NOT APPROVED FOR PUBLICATION</div>
    <div className="ai-reasoning__sections">
      <section>
        <h3>AI Market Forecast</h3>
        <dl>
          <div><dt>Direction</dt><dd>{forecast?.dominant_direction ?? 'unavailable'}</dd></div>
          <div><dt>BUY / SELL / NEUTRAL</dt><dd>{percentage(forecast?.buy_probability)} / {percentage(forecast?.sell_probability)} / {percentage(forecast?.neutral_probability)}</dd></div>
          <div><dt>Scenario</dt><dd>{forecast?.dominant_scenario ?? 'No validated forecast'}</dd></div>
          <div><dt>Alternative</dt><dd>{forecast?.alternative_scenarios[0]?.name ?? '—'}</dd></div>
          <div><dt>Setup family</dt><dd>{forecast?.selected_setup_family?.replaceAll('_', ' ') ?? '—'}</dd></div>
          <div><dt>Horizon / move</dt><dd>{forecast?.expected_horizon ?? '—'} · {percentage(forecast?.expected_base_move)}</dd></div>
          <div><dt>Confidence / uncertainty</dt><dd>{percentage(forecast?.forecast_confidence)} / {percentage(forecast?.uncertainty)}</dd></div>
          <div><dt>Evidence completeness</dt><dd>{percentage(forecast?.evidence_completeness)}</dd></div>
        </dl>
      </section>
      <section>
        <h3>AI Signal Proposal</h3>
        <dl>
          <div><dt>Recommendation</dt><dd>{proposal?.recommended_action ?? 'No validated proposal'}</dd></div>
          <div><dt>Direction / readiness</dt><dd>{proposal?.direction ?? '—'} · {proposal?.setup_readiness ?? '—'}</dd></div>
          <div><dt>Entry zone</dt><dd>{proposal?.entry_zone ? `${price(proposal.entry_zone.low)}–${price(proposal.entry_zone.high)}` : '—'}</dd></div>
          <div><dt>Stop Loss</dt><dd>{price(proposal?.stop_loss)}</dd></div>
          <div><dt>Take Profits</dt><dd>{proposal?.take_profit_levels.map(price).join(', ') || '—'}</dd></div>
          <div><dt>Risk-to-Reward</dt><dd>{proposal?.expected_risk_to_reward?.toFixed(2) ?? '—'}</dd></div>
          <div><dt>Invalidation / expiry</dt><dd>{price(proposal?.invalidation_price)} · {proposal?.expires_at ? new Date(proposal.expires_at).toLocaleString() : '—'}</dd></div>
          <div><dt>Evidence</dt><dd>{proposal ? `${proposal.supporting_evidence_ids.length} supporting · ${proposal.contradicting_evidence_ids.length} contradicting` : '—'}</dd></div>
        </dl>
      </section>
      <section>
        <h3>Signal Lifecycle</h3>
        <dl>
          <div><dt>State</dt><dd>{signal?.state ?? 'No managed signal'}</dd></div>
          <div><dt>Opportunity</dt><dd className="mono">{signal?.structural_opportunity_key.slice(0, 16) ?? '—'}</dd></div>
          <div><dt>Setup / direction</dt><dd>{signal ? `${signal.setup_family.replaceAll('_', ' ')} · ${signal.direction}` : '—'}</dd></div>
          <div><dt>Transitions</dt><dd>{history?.transitions.length ?? 0}</dd></div>
          <div><dt>Level revisions</dt><dd>{history?.revisions.length ?? 0}</dd></div>
          <div><dt>Monitoring evaluations</dt><dd>{history?.monitoring.length ?? 0}</dd></div>
        </dl>
      </section>
      <section>
        <h3>AI Health</h3>
        <dl>
          <div><dt>Provider</dt><dd>{data?.health.provider ?? 'OpenRouter'}</dd></div>
          <div><dt>Model</dt><dd>{data?.health.model_identifier ?? '—'}</dd></div>
          <div><dt>Prompt</dt><dd>{data?.health.prompt_version ?? '—'}</dd></div>
          <div><dt>Availability</dt><dd>{data?.health.provider_available == null ? 'not called' : data.health.provider_available ? 'available' : 'unavailable'}</dd></div>
          <div><dt>Latency</dt><dd>{data?.health.latest_latency_ms == null ? '—' : `${data.health.latest_latency_ms.toFixed(0)} ms`}</dd></div>
          <div><dt>Validation / retries</dt><dd>{data?.health.latest_validation_passed == null ? '—' : String(data.health.latest_validation_passed)} / {data?.health.latest_retry_count ?? 0}</dd></div>
          <div><dt>Failures / fallback</dt><dd>{data?.health.failed_requests ?? 0} · {data?.health.fallback_state ?? 'none'}</dd></div>
        </dl>
      </section>
    </div>
  </div>
}
