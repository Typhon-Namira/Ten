import { BarChart3, Scale } from 'lucide-react'
import { useActiveSelection } from '../../hooks/useActiveSelection'
import { useAIDashboardData } from '../../hooks/useAIDashboardData'
import { ErrorBoundary } from '../ErrorBoundary'
import { AnalysisHistory, CurrentAnalyticalCycle, SignalHistory } from './AuthoritativeCycle'
import { DashboardHeader } from './DecisionOverview'
import { MarketStateSummary, QuantForecastSummary } from './IntelligenceCards'
import { EmptyState, ErrorState, LoadingSkeleton, Metric, SectionHeader } from './Primitives'
import { SystemStatusPanel } from './SystemStatusPanel'

export type DashboardView = 'overview' | 'signals' | 'performance' | 'calibration' | 'system'

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
      reasoning={null}
      latestCycle={data.latestCycle}
      stale={data.stale}
      lastUpdated={data.lastUpdated}
      loading={data.loading}
      onRefresh={data.refresh}
    />
    <div className="ai-page__intro">
      <div><p>TEN AI ANALYTICAL PLATFORM</p><h1>{title}</h1></div>
      <span>Closed market data → validated analytical intelligence</span>
    </div>
    {errorMessage && <ErrorState message={`${errorMessage}. Last known backend-authoritative values remain visible.`} />}
    {data.loading && !data.latestCycle && !data.intelligence ? <LoadingSkeleton rows={6} /> : <>
      {view === 'overview' && <>
        <ErrorBoundary label="Latest completed cycle"><CurrentAnalyticalCycle cycle={data.latestCycle} /></ErrorBoundary>
        <div className="ai-card-grid">
          <ErrorBoundary label="Market state"><MarketStateSummary data={data.intelligence} /></ErrorBoundary>
          <ErrorBoundary label="Quant forecast"><QuantForecastSummary forecast={data.quant} calibration={null} unavailableReason={data.latestCycle?.stages.quant_forecast?.reason} /></ErrorBoundary>
        </div>
        <ErrorBoundary label="System status"><SystemStatusPanel data={data.systemStatus} /></ErrorBoundary>
      </>}
      {view === 'signals' && <>
        <ErrorBoundary label="Latest completed cycle"><CurrentAnalyticalCycle cycle={data.latestCycle} /></ErrorBoundary>
        <ErrorBoundary label="Signal history"><SignalHistory page={data.signalHistory} /></ErrorBoundary>
        <ErrorBoundary label="Analysis history"><AnalysisHistory page={data.analysisHistory} /></ErrorBoundary>
      </>}
      {view === 'performance' && <section className="ai-card ai-card--wide">
        <SectionHeader eyebrow="Persisted sample semantics" title="Analytical performance" action={<BarChart3 size={19} />} />
        <div className="health-grid">
          <Metric label="Signals generated" value={data.latestCycle?.performance.signals_generated ?? 0} />
          <Metric label="Signals evaluated" value={data.latestCycle?.performance.signals_evaluated ?? 0} detail="Outcome evaluation is never inferred from generation" />
          <Metric label="Pending outcomes" value={data.latestCycle?.performance.signals_awaiting_outcome ?? 0} />
          <Metric label="Minimum required sample" value={data.latestCycle?.performance.minimum_required_sample ?? 30} />
        </div>
      </section>}
      {view === 'calibration' && <>
        <section className="ai-card ai-card--wide">
          <SectionHeader eyebrow="Observed outcomes only" title="Calibration" action={<Scale size={19} />} />
          <EmptyState title="Insufficient validated sample" detail="TEN does not present signal confidence as calibrated probability until persisted outcomes satisfy the minimum sample policy." />
          <div className="health-grid">
            <Metric label="Signals generated" value={data.latestCycle?.performance.signals_generated ?? 0} />
            <Metric label="Signals evaluated" value={data.latestCycle?.performance.signals_evaluated ?? 0} />
            <Metric label="Calibration sample" value={data.latestCycle?.performance.calibration_sample_size ?? 0} />
            <Metric label="Minimum required" value={data.latestCycle?.performance.minimum_required_sample ?? 30} />
          </div>
        </section>
        <ErrorBoundary label="Quant forecast"><QuantForecastSummary forecast={data.quant} calibration={null} unavailableReason={data.latestCycle?.stages.quant_forecast?.reason} /></ErrorBoundary>
      </>}
      {view === 'system' && <>
        <ErrorBoundary label="System status"><SystemStatusPanel data={data.systemStatus} /></ErrorBoundary>
        <ErrorBoundary label="Latest completed cycle"><CurrentAnalyticalCycle cycle={data.latestCycle} /></ErrorBoundary>
      </>}
    </>}
  </div>
}
