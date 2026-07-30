import { Route } from 'lucide-react'
import type { ForwardMarketScenario, LatestCompletedCycle } from '../../types'
import { humanize } from '../../lib/aiDashboard'
import { EmptyState, Metric, SectionHeader, StatusBadge } from './Primitives'

const price = (value: number | null | undefined) =>
  value == null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: 3 })

function ScenarioCard({ scenario }: { scenario: ForwardMarketScenario | null }) {
  if (!scenario) {
    return <section className="ai-card">
      <EmptyState title="Scenario not yet available" detail="Awaiting a completed point-in-time candle analysis for this horizon." />
    </section>
  }
  const tone = scenario.primary_direction === 'BULLISH'
    ? 'positive'
    : scenario.primary_direction === 'BEARISH'
      ? 'negative'
      : 'neutral'
  return <section className="ai-card">
    <SectionHeader eyebrow={`${scenario.timeframe} forward horizon`} title={`${scenario.timeframe} primary scenario`} action={<Route size={18} />} />
    <StatusBadge tone={tone}>{scenario.primary_direction}</StatusBadge>
    <p className="reasoning-copy">{scenario.expected_price_path}</p>
    <div className="health-grid">
      <Metric label="Type" value={humanize(scenario.scenario_type)} />
      <Metric label="Reference price" value={price(scenario.reference_market_price)} />
      <Metric label="Expected range" value={`${price(scenario.expected_range.low)} – ${price(scenario.expected_range.high)}`} />
      <Metric label="Likely close" value={`${price(scenario.expected_closing_zone.low)} – ${price(scenario.expected_closing_zone.high)}`} />
      <Metric label="Raw confidence" value={`${scenario.confidence.toFixed(0)}%`} />
      <Metric label="Calibration" value={humanize(scenario.calibration_status)} />
      <Metric label="Invalidation" value={price(scenario.invalidation_level)} />
      <Metric label="Expires" value={new Date(scenario.expiry).toLocaleTimeString()} />
      <Metric label="Scenario validity" value={scenario.scenario_validity} />
      <Metric label="Geometry" value={scenario.execution_geometry_validity} />
    </div>
    <p className="reasoning-copy">{scenario.narrative}</p>
  </section>
}

export function ForwardMarketScenarios({ cycle }: { cycle: LatestCompletedCycle | null }) {
  const scenarios = cycle?.forward_market_scenarios
  const combined = scenarios?.combined
  return <section className="scenario-forecast-section">
    <SectionHeader eyebrow="Evidence-grounded probability paths" title="Forward Market Scenarios" action={<Route size={19} />} />
    <div className="ai-card-grid">
      <ScenarioCard scenario={scenarios?.m5 ?? null} />
      <ScenarioCard scenario={scenarios?.m15 ?? null} />
    </div>
    <section className="ai-card ai-card--wide">
      <SectionHeader eyebrow="M15 authoritative comparison" title="Combined forward scenario" action={<Route size={18} />} />
      {!combined ? <EmptyState title="Combined scenario pending" detail="The combined assessment is produced only after an M15 close with a still-valid M5 scenario." /> : <>
        <StatusBadge tone={combined.agreement === 'ALIGNED' ? 'positive' : combined.agreement === 'CONFLICT' ? 'negative' : 'neutral'}>{combined.agreement}</StatusBadge>
        <p className="reasoning-copy">{combined.expected_price_path}</p>
        <div className="health-grid">
          <Metric label="Direction" value={combined.combined_direction} />
          <Metric label="Confidence" value={`${combined.confidence.toFixed(0)}%`} />
          <Metric label="Scenario validity" value={combined.scenario_validity} />
          <Metric label="Execution geometry" value={combined.execution_geometry_validity} />
          <Metric label="Entry" value={price(combined.geometry?.entry)} />
          <Metric label="Stop loss" value={price(combined.geometry?.stop_loss)} />
          <Metric label="Take profit" value={price(combined.geometry?.take_profit)} />
          <Metric label="Risk / reward" value={combined.geometry ? combined.geometry.risk_reward_ratio.toFixed(2) : 'Unavailable'} />
          <Metric label="Expiry" value={new Date(combined.expiry).toLocaleTimeString()} />
          <Metric label="Publication" value={humanize(combined.publication_status)} />
        </div>
        {combined.geometry == null
          ? <p className="reasoning-copy"><strong>Executable setup unavailable.</strong> {humanize(combined.geometry_rejection_reason ?? 'deterministic geometry unavailable')}</p>
          : <p className="reasoning-copy">Validated facts: {combined.geometry.basis_fact_identifiers.join(', ')}</p>}
      </>}
      <small>Analytical Intelligence Only · No Broker Execution</small>
    </section>
  </section>
}
