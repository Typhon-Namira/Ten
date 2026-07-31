import { Route } from 'lucide-react'
import type { CandidateMarketScenario, ForwardMarketScenario, LatestCompletedCycle } from '../../types'
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

function CandidateSummary({ candidate, title }: { candidate: CandidateMarketScenario; title: string }) {
  const tone = candidate.direction === 'BULLISH' ? 'positive' : candidate.direction === 'BEARISH' ? 'negative' : 'neutral'
  return <section className="ai-card">
    <SectionHeader eyebrow={title} title={humanize(candidate.scenario_type)} action={<Route size={18} />} />
    <StatusBadge tone={tone}>{candidate.direction} · {candidate.final_scenario_score.toFixed(1)}%</StatusBadge>
    <ol className="reasoning-copy">
      {candidate.path_sequence.map(stage => <li key={stage.stage_id}>
        {stage.label} · {price(stage.expected_price_area.low)}–{price(stage.expected_price_area.high)}
      </li>)}
    </ol>
    <div className="health-grid">
      <Metric label="Reference price" value={price(candidate.reference_price)} />
      <Metric label="Entry type" value={humanize(candidate.entry_type)} />
      <Metric label="Entry zone" value={candidate.entry_zone ? `${price(candidate.entry_zone.low)}–${price(candidate.entry_zone.high)}` : 'Unavailable'} />
      <Metric label="Entry" value={price(candidate.geometry?.entry)} />
      <Metric label="Stop loss" value={price(candidate.geometry?.stop_loss)} />
      <Metric label="Take profit" value={price(candidate.geometry?.take_profit)} />
      <Metric label="Risk / reward" value={candidate.geometry ? candidate.geometry.risk_reward_ratio.toFixed(2) : 'Unavailable'} />
      <Metric label="Geometry" value={candidate.geometry_validity} />
      <Metric label="Calibration" value={candidate.calibrated_probability == null ? `Pending (${candidate.calibration_sample_size} samples)` : `${(candidate.calibrated_probability * 100).toFixed(1)}%`} />
      <Metric label="Expires" value={new Date(candidate.expiry).toLocaleTimeString()} />
    </div>
    {candidate.rejection_reason && <p className="reasoning-copy"><strong>Execution unavailable:</strong> {humanize(candidate.rejection_reason)}</p>}
  </section>
}

export function PrimaryMarketScenario({ cycle }: { cycle: LatestCompletedCycle | null }) {
  const selection = cycle?.primary_market_scenario
  const attempt = cycle?.authoritative_simulation
  return <section className="scenario-forecast-section">
    <SectionHeader eyebrow="Authoritative M15 simulation" title="Primary Market Scenario" action={<Route size={19} />} />
    {!selection || !selection.primary
      ? <section className="ai-card ai-card--wide"><EmptyState
          title={attempt?.status ? humanize(attempt.status) : 'Primary Scenario pending'}
          detail={attempt?.failure_message || attempt?.skip_reason || selection?.rejection_reason
            ? humanize(attempt?.failure_message ?? attempt?.skip_reason ?? selection?.rejection_reason ?? '')
            : 'Awaiting a synchronized completed M15 simulation.'}
        /></section>
      : <>
        <CandidateSummary candidate={selection.primary} title="Selected Primary Scenario" />
        <section className="ai-card ai-card--wide">
          <SectionHeader eyebrow="Structured forecast" title="Expected Price Path" action={<Route size={18} />} />
          <ol className="scenario-path">
            {selection.primary.path_sequence.map(stage => <li key={stage.stage_id}>
              <strong>Stage {stage.sequence} · {stage.label}</strong>
              <span>{price(stage.expected_price_area.low)}–{price(stage.expected_price_area.high)}</span>
              <small>{stage.timing_seconds ? `Expected within ${Math.round(stage.timing_seconds / 60)} minutes · ` : ''}{humanize(stage.invalidation_condition)}</small>
            </li>)}
          </ol>
        </section>
        <div className="health-grid">
          <Metric label="Authoritative signal" value={selection.authoritative_action} />
          <Metric label="Selection status" value={selection.status} />
          <Metric label="Publication" value={selection.signal_eligible ? 'ELIGIBLE' : 'BLOCKED'} />
          <Metric label="Minimum score" value={`${selection.minimum_score.toFixed(0)}%`} />
        </div>
        <section className="ai-card ai-card--wide">
          <SectionHeader eyebrow="Deterministic ranking" title="Why This Scenario Was Selected" action={<Route size={18} />} />
          <p className="reasoning-copy">{selection.ranking_explanation}</p>
          <div className="health-grid">
            {selection.primary.score_components.map(component => <Metric
              key={component.name}
              label={humanize(component.name)}
              value={`${component.contribution >= 0 ? '+' : ''}${component.contribution.toFixed(1)}`}
              detail={component.reason}
            />)}
          </div>
        </section>
        {selection.alternative && <CandidateSummary candidate={selection.alternative} title="Alternative Market Scenario · Not authoritative" />}
        <section className="ai-card ai-card--wide">
          <SectionHeader eyebrow="Deterministic candidate comparison" title="Candidate Scenario Ranking" action={<Route size={18} />} />
          <div className="health-grid">
            {selection.ranked_candidates.map(candidate => <Metric
              key={candidate.candidate_id}
              label={`#${candidate.rank} ${humanize(candidate.scenario_type)}`}
              value={`${candidate.direction} · ${candidate.final_scenario_score.toFixed(1)}%`}
              detail={`${candidate.geometry_validity}${candidate.rejection_reason ? ` · ${humanize(candidate.rejection_reason)}` : ''}`}
            />)}
          </div>
        </section>
      </>}
  </section>
}

export function ForwardMarketScenarios({ cycle }: { cycle: LatestCompletedCycle | null }) {
  const scenarios = cycle?.forward_market_scenarios
  const combined = scenarios?.combined
  return <section className="scenario-forecast-section">
    <SectionHeader eyebrow="Underlying directional evidence" title="Supporting Directional Synthesis" action={<Route size={19} />} />
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
