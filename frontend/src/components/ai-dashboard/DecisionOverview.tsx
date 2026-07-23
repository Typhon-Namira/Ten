import { ArrowDownRight, ArrowUpRight, RefreshCw, ShieldCheck } from 'lucide-react'
import type { AIReasoningDashboard, MarketIntelligence, QuantForecastResult } from '../../types'
import {
  actionTone,
  activePublication,
  activeSignal,
  decisionExplanation,
  decisionHeadline,
  humanize,
  latestFinalAction,
  pipelineSteps,
} from '../../lib/aiDashboard'
import { FreshnessIndicator, PipelineIcon, StatusBadge } from './Primitives'

function formatPrice(value: number | null | undefined): string {
  return value == null ? 'Unavailable' : value.toFixed(2)
}

function formatTime(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : 'Unavailable'
}

export function DashboardHeader({
  instrument,
  intelligence,
  reasoning,
  stale,
  lastUpdated,
  loading,
  onRefresh,
}: {
  instrument: string
  intelligence: MarketIntelligence | null
  reasoning: AIReasoningDashboard | null
  stale: boolean
  lastUpdated: Date | null
  loading: boolean
  onRefresh: () => Promise<void>
}) {
  const profile = reasoning?.runtime.operating_profile ?? 'safe_test'
  const aiOnline = reasoning?.health.provider_available
  const healthy = reasoning?.health.guardrails.status === 'healthy' && !stale
  return <header className="ai-statusbar">
    <div className="ai-statusbar__identity">
      <span className="ai-wordmark">TEN</span>
      <div><strong>{instrument}</strong><small>{intelligence?.latest_candle_timestamp ? formatTime(intelligence.latest_candle_timestamp) : 'Market timestamp unavailable'}</small></div>
    </div>
    <div className="ai-statusbar__badges" aria-label="System status">
      <StatusBadge tone="neutral">{humanize(profile)}</StatusBadge>
      <StatusBadge tone={aiOnline === true ? 'positive' : aiOnline === false ? 'negative' : 'neutral'} pulse={aiOnline === true}>
        AI {aiOnline === true ? 'online' : aiOnline === false ? 'offline' : 'not called'}
      </StatusBadge>
      <FreshnessIndicator stale={stale} timestamp={intelligence?.latest_candle_timestamp ?? null} />
      <StatusBadge tone={healthy ? 'positive' : 'warning'}><ShieldCheck size={12} />System {healthy ? 'healthy' : 'limited'}</StatusBadge>
      <span className="ai-updated">Updated {lastUpdated ? lastUpdated.toLocaleTimeString() : '—'}</span>
      <button className="ai-icon-button" type="button" aria-label="Refresh dashboard" onClick={() => void onRefresh()} disabled={loading}>
        <RefreshCw size={16} className={loading ? 'spin' : ''} />
      </button>
    </div>
    <div className="ai-statusbar__boundary"><strong>Analytical Intelligence Only</strong><span>No Broker Execution</span></div>
  </header>
}

export function FinalDecisionHero({
  intelligence,
  reasoning,
}: {
  intelligence: MarketIntelligence | null
  reasoning: AIReasoningDashboard | null
}) {
  const action = latestFinalAction(reasoning)
  const signal = activeSignal(reasoning)
  const publication = activePublication(reasoning)
  const tone = actionTone(action)
  const direction = action?.final_direction
  const Icon = direction === 'SELL' ? ArrowDownRight : ArrowUpRight
  return <section className={`decision-hero decision-hero--${tone}`} aria-labelledby="current-decision-title">
    <div className="decision-hero__status">
      <p>Final system action</p>
      <div className="decision-hero__headline">
        <span className="decision-hero__icon"><Icon size={28} /></span>
        <h1 id="current-decision-title" key={action?.final_action_id ?? 'none'}>{decisionHeadline(action)}</h1>
      </div>
      <p className="decision-hero__explanation">{decisionExplanation(action)}</p>
      <div className="decision-hero__badges">
        <StatusBadge tone={tone}>{humanize(action?.approval_state ?? 'unavailable')}</StatusBadge>
        <StatusBadge tone={publication ? 'positive' : 'neutral'}>{publication ? 'Published analytical signal' : 'Not published'}</StatusBadge>
        <StatusBadge tone="neutral">{signal ? humanize(signal.state) : 'No active signal'}</StatusBadge>
      </div>
    </div>
    <div className="decision-hero__price">
      <span>Current market</span>
      <strong>{formatPrice(intelligence?.current_candle?.close)}</strong>
      <small>{intelligence?.current_session ? `${humanize(intelligence.current_session)} session` : 'Session unavailable'}</small>
    </div>
    <div className="decision-hero__levels">
      <div><span>Entry zone</span><strong>{action?.final_entry ? `${formatPrice(action.final_entry.low)} – ${formatPrice(action.final_entry.high)}` : 'Unavailable'}</strong></div>
      <div><span>Stop loss</span><strong>{formatPrice(action?.final_stop_loss)}</strong></div>
      <div><span>Take profit</span><strong>{action?.final_take_profits.length ? action.final_take_profits.map(formatPrice).join(' · ') : 'Unavailable'}</strong></div>
      <div><span>Risk to reward</span><strong>{action?.final_risk_to_reward?.toFixed(2) ?? 'Unavailable'}</strong></div>
      <div><span>Decision time</span><strong>{formatTime(action?.created_at)}</strong></div>
      <div><span>AI confidence</span><strong>{reasoning?.forecast?.forecast_confidence == null ? 'Unavailable' : `${(reasoning.forecast.forecast_confidence * 100).toFixed(0)}%`}</strong></div>
    </div>
  </section>
}

export function DecisionPipeline({
  intelligence,
  quant,
  reasoning,
}: {
  intelligence: MarketIntelligence | null
  quant: QuantForecastResult | null
  reasoning: AIReasoningDashboard | null
}) {
  const steps = pipelineSteps(reasoning, intelligence != null, quant?.status ?? null)
  return <section className="decision-pipeline" aria-label="AI decision pipeline">
    {steps.map((step, index) => <div className={`decision-pipeline__step decision-pipeline__step--${step.status}`} key={step.id}>
      <span className="decision-pipeline__index">{index + 1}</span>
      <PipelineIcon status={step.status} />
      <div><strong>{step.label}</strong><small>{step.detail}</small></div>
      {index < steps.length - 1 && <i aria-hidden="true" />}
    </div>)}
  </section>
}
