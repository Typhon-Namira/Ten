import { BarChart3, CheckCircle2, Scale } from 'lucide-react'
import { useActiveSelection } from '../../hooks/useActiveSelection'
import { useAIDashboardData } from '../../hooks/useAIDashboardData'
import { humanize } from '../../lib/aiDashboard'
import { ActiveSignalMonitor, GuardrailSummary, OperationalHealth, ValidationSummary } from './DecisionCards'
import { DashboardHeader, DecisionPipeline, FinalDecisionHero } from './DecisionOverview'
import { AIReasoningSummary, MarketStateSummary, QuantForecastSummary } from './IntelligenceCards'
import { EmptyState, ErrorState, LoadingSkeleton, Metric, SectionHeader, StatusBadge } from './Primitives'
import { SystemStatusPanel } from './SystemStatusPanel'

export type DashboardView = 'overview' | 'signals' | 'performance' | 'calibration' | 'system'

function CalibrationPanel({ data }: { data: ReturnType<typeof useAIDashboardData> }) {
  const calibration = data.calibration
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Probability reliability" title="Calibration" action={<Scale size={19} />} />
    {!calibration ? <EmptyState title="Calibration unavailable" detail={humanize(data.aggregate?.calibration.reason ?? 'insufficient_validated_sample')} /> : <>
      <div className="health-grid">
        <Metric label="Calibration state" value={humanize(calibration.status)} />
        <Metric label="Validated sample" value={calibration.sample_count || 'Insufficient validated sample'} />
        <Metric label="Brier score" value={calibration.brier_score?.toFixed(4) ?? 'Unavailable'} />
        <Metric label="Log loss" value={calibration.log_loss?.toFixed(4) ?? 'Unavailable'} />
        <Metric label="Expected calibration error" value={calibration.expected_calibration_error?.toFixed(4) ?? 'Unavailable'} />
        <Metric label="Model" value={`${calibration.model_name} · ${calibration.model_version}`} />
      </div>
      <div className="calibration-note">
        <BarChart3 size={20} />
        <div><strong>{calibration.sample_count > 0 ? 'Measured probability reliability' : 'Insufficient validated sample'}</strong><p>Raw AI confidence is not presented as calibrated probability. Reliability becomes meaningful only after complete outcome observations exist.</p></div>
      </div>
    </>}
  </section>
}

function ReadinessDetail({ data }: { data: ReturnType<typeof useAIDashboardData> }) {
  const readiness = data.reasoning?.production_readiness
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Backend-authoritative assessment" title="Readiness checks" action={<CheckCircle2 size={19} />} />
    {!readiness ? <EmptyState title="Readiness not evaluated" detail={humanize(data.aggregate?.readiness.reason ?? 'insufficient_validated_sample')} /> : <div className="readiness-checks">
      {Object.entries(readiness.measured_checks).map(([name, check]) => <div key={name}>
        <StatusBadge tone={check.passed ? 'positive' : 'negative'}>{check.passed ? 'Passed' : 'Blocked'}</StatusBadge>
        <span><strong>{humanize(name)}</strong><small>Required: {typeof check.threshold === 'object' ? 'backend policy' : String(check.threshold)}</small></span>
      </div>)}
    </div>}
  </section>
}

export function AIDashboard({ view = 'overview' }: { view?: DashboardView }) {
  const { selection } = useActiveSelection()
  const data = useAIDashboardData(selection.instrument, selection.timeframe)
  const errorMessage = Object.values(data.errors)[0]
  const title = {
    overview: 'Current AI decision',
    signals: 'Analytical signals',
    performance: 'Measured performance',
    calibration: 'Probability calibration',
    system: 'System readiness',
  }[view]
  return <div className="ai-page">
    <DashboardHeader
      instrument={selection.instrument}
      intelligence={data.intelligence}
      reasoning={data.reasoning}
      stale={data.stale}
      lastUpdated={data.lastUpdated}
      loading={data.loading}
      onRefresh={data.refresh}
    />
    <div className="ai-page__intro"><div><p>TEN AI ANALYTICAL PLATFORM</p><h1>{title}</h1></div><span>Closed market data → validated analytical intelligence</span></div>
    {errorMessage && <ErrorState message={`${errorMessage}. Last known backend-authoritative values remain visible.`} />}
    {data.loading && !data.reasoning && !data.intelligence ? <LoadingSkeleton rows={6} /> : <>
      {view === 'overview' && <>
        <FinalDecisionHero intelligence={data.intelligence} reasoning={data.reasoning} unavailableReason={data.aggregate?.stages.final_action.reason} publicationReason={data.aggregate?.stages.publication.reason} />
        <DecisionPipeline intelligence={data.intelligence} quant={data.quant} reasoning={data.reasoning} stages={data.aggregate?.stages} />
        <div className="ai-card-grid">
          <MarketStateSummary data={data.intelligence} />
          <QuantForecastSummary forecast={data.quant} calibration={data.calibration} unavailableReason={data.aggregate?.stages.quant_forecast.reason} />
          <AIReasoningSummary data={data.reasoning} unavailableReason={data.aggregate?.stages.ai_reasoning.reason} />
          <GuardrailSummary data={data.reasoning} unavailableReason={data.aggregate?.stages.guardrails.reason} />
        </div>
        <ActiveSignalMonitor data={data.reasoning} market={data.intelligence} unavailableReason={data.aggregate?.stages.monitoring.reason} />
        <ValidationSummary data={data.reasoning} unavailableReason={data.aggregate?.readiness.reason} />
        <OperationalHealth data={data.reasoning} stale={data.stale} />
        <SystemStatusPanel data={data.systemStatus} />
      </>}
      {view === 'signals' && <>
        <FinalDecisionHero intelligence={data.intelligence} reasoning={data.reasoning} unavailableReason={data.aggregate?.stages.final_action.reason} publicationReason={data.aggregate?.stages.publication.reason} />
        <ActiveSignalMonitor data={data.reasoning} market={data.intelligence} unavailableReason={data.aggregate?.stages.monitoring.reason} />
        <div className="ai-card-grid"><GuardrailSummary data={data.reasoning} unavailableReason={data.aggregate?.stages.guardrails.reason} /><AIReasoningSummary data={data.reasoning} unavailableReason={data.aggregate?.stages.ai_reasoning.reason} /></div>
      </>}
      {view === 'performance' && <>
        <ValidationSummary data={data.reasoning} unavailableReason={data.aggregate?.readiness.reason} />
        <CalibrationPanel data={data} />
      </>}
      {view === 'calibration' && <>
        <CalibrationPanel data={data} />
        <QuantForecastSummary forecast={data.quant} calibration={data.calibration} unavailableReason={data.aggregate?.stages.quant_forecast.reason} />
      </>}
      {view === 'system' && <>
        <SystemStatusPanel data={data.systemStatus} />
        <ReadinessDetail data={data} />
        <OperationalHealth data={data.reasoning} stale={data.stale} />
        <DecisionPipeline intelligence={data.intelligence} quant={data.quant} reasoning={data.reasoning} stages={data.aggregate?.stages} />
      </>}
    </>}
  </div>
}
