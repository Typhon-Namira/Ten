import type { AIReasoningDashboard } from '../types'

const percentage = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`
const price = (value: number | null | undefined) => value == null ? '—' : value.toFixed(2)

export function AIReasoningPanel({ data }: { data: AIReasoningDashboard | null }) {
  const forecast = data?.forecast
  const proposal = data?.proposal
  const signal = data?.managed_signals[0]
  const history = signal ? data?.signal_histories[signal.signal_id] : null
  const finalAction = signal ? data?.final_actions[signal.signal_id]?.at(-1) : null
  const publication = signal ? data?.publications[signal.signal_id] : null
  const failedGates = finalAction?.gate_evaluations.filter(item => item.status !== 'passed' && item.status !== 'not_applicable') ?? []
  const live = Boolean(data?.health.publication_enabled && publication)

  return <div className="ai-reasoning">
    <div className={`ai-reasoning__banner ${live ? 'ai-reasoning__banner--live' : ''}`}>
      {live ? 'ANALYTICAL LIVE · GUARDRAIL APPROVED · NO BROKER EXECUTION' : 'SHADOW / SAFE TEST · DETERMINISTIC GUARDRAILS · NO BROKER EXECUTION'}
    </div>
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
        <h3>Final Decision</h3>
        <dl>
          <div><dt>Action / approval</dt><dd>{finalAction ? `${finalAction.action} · ${finalAction.approval_state}` : 'Not evaluated'}</dd></div>
          <div><dt>Publication</dt><dd>{finalAction?.publication_state ?? 'disabled'}</dd></div>
          <div><dt>Final risk</dt><dd>{finalAction?.final_risk_classification ?? '—'}</dd></div>
          <div><dt>Risk-to-Reward</dt><dd>{finalAction?.final_risk_to_reward?.toFixed(2) ?? '—'}</dd></div>
          <div><dt>Gate results</dt><dd>{finalAction ? `${finalAction.gate_evaluations.length - failedGates.length} passed · ${failedGates.length} blocked/unavailable` : '—'}</dd></div>
          <div><dt>Modifications</dt><dd>{finalAction?.modifications.length ?? 0}</dd></div>
          <div><dt>Analytical publication</dt><dd>{publication ? `${publication.direction} · ${new Date(publication.published_at).toLocaleString()}` : 'None'}</dd></div>
          <div><dt>Broker execution</dt><dd>Not available</dd></div>
        </dl>
      </section>
      <section>
        <h3>Signal Monitoring</h3>
        <dl>
          <div><dt>State</dt><dd>{signal?.state ?? 'No managed signal'}</dd></div>
          <div><dt>Opportunity</dt><dd className="mono">{signal?.structural_opportunity_key.slice(0, 16) ?? '—'}</dd></div>
          <div><dt>Setup / direction</dt><dd>{signal ? `${signal.setup_family.replaceAll('_', ' ')} · ${signal.direction}` : '—'}</dd></div>
          <div><dt>Transitions</dt><dd>{history?.transitions.length ?? 0}</dd></div>
          <div><dt>Level revisions</dt><dd>{history?.revisions.length ?? 0}</dd></div>
          <div><dt>Monitoring evaluations</dt><dd>{history?.monitoring.length ?? 0}</dd></div>
          <div><dt>Outcomes</dt><dd>{history?.outcomes.length ?? 0}</dd></div>
        </dl>
      </section>
      <section>
        <h3>System Health and Usage</h3>
        <dl>
          <div><dt>Primary / active</dt><dd>{data?.health.primary_provider ?? 'cerebras'} / {data?.health.active_provider ?? 'Not selected'}</dd></div>
          <div><dt>Fallback</dt><dd>groq · {data?.health.fallback_status ?? 'STANDBY'}</dd></div>
          <div><dt>Model</dt><dd>{data?.health.model_identifier ?? '—'}</dd></div>
          <div><dt>Availability</dt><dd>{data?.health.provider_available == null ? 'not called' : data.health.provider_available ? 'available' : 'unavailable'}</dd></div>
          <div><dt>Provider states</dt><dd>{Object.entries(data?.health.providers ?? {}).map(([name, item]) => `${name}: ${item.status}`).join(' · ') || 'not called'}</dd></div>
          <div><dt>Latency</dt><dd>{data?.health.latest_latency_ms == null ? '—' : `${data.health.latest_latency_ms.toFixed(0)} ms`}</dd></div>
          <div><dt>LLM requests / tokens today</dt><dd>{data?.llm_usage.request_count ?? 0} / {data?.llm_usage.total_tokens ?? 'unavailable'}</dd></div>
          <div><dt>Guardrails / publications</dt><dd>{data?.health.guardrails.status ?? '—'} / {data?.health.guardrails.publications_succeeded ?? 0}</dd></div>
          <div><dt>Policy</dt><dd>{data?.health.guardrails.policy_versions.guardrails ?? '—'}</dd></div>
          <div><dt>Production readiness</dt><dd>{data?.production_readiness?.status ?? 'not measured'}</dd></div>
        </dl>
      </section>
    </div>
  </div>
}
