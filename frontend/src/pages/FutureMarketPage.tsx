import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, Clock3, RefreshCw, Route, Target, Waves } from 'lucide-react'
import { useActiveSelection } from '../hooks/useActiveSelection'

interface PriceZone { low: number; high: number }
interface PathStage {
  sequence: number
  minute_from: number
  minute_to: number
  event: string
  expected_price_area: PriceZone | null
  invalidation_condition: string | null
}
interface ScenarioBranch {
  scenario_id: string
  scenario_type: string
  direction: 'BULLISH' | 'BEARISH' | 'RANGE' | 'INCONCLUSIVE'
  probability: number
  expected_range: PriceZone
  likely_close: PriceZone
  path: PathStage[]
  invalidation: string
  rank: number
}
interface OpportunityWindow {
  opportunity_id: string
  scenario_id: string
  direction: 'BULLISH' | 'BEARISH' | 'RANGE' | 'INCONCLUSIVE'
  state: 'WATCHING' | 'FORMING' | 'ARMED' | 'TRIGGERED' | 'INVALIDATED' | 'EXPIRED'
  expected_from_minute: number
  expected_to_minute: number
  entry_zone: PriceZone
  trigger_conditions: string[]
  invalidation_level: number
  targets: number[]
  probability: number
  quality: number
}
interface FutureMarketForecast {
  forecast_id: string
  instrument: string
  generated_at: string
  market_cutoff: string
  expires_at: string
  forecast_horizon_seconds: number
  forecast_cadence_seconds: number
  provider: string
  model_name: string
  model_version: string
  market_state: {
    regime: string
    uncertainty: number
    reference_price: number
    context_timeframes: string[]
  }
  dominant_scenario_id: string | null
  scenarios: ScenarioBranch[]
  opportunities: OpportunityWindow[]
}

const API_BASE_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, '') ?? ''
const price = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 3 })
const humanize = (value: string) => value.replaceAll('_', ' ').toLowerCase().replace(/(^|\s)\S/g, letter => letter.toUpperCase())

function tone(direction: ScenarioBranch['direction']) {
  return direction === 'BULLISH' ? 'positive' : direction === 'BEARISH' ? 'negative' : 'neutral'
}

function ScenarioCard({ scenario, dominant }: { scenario: ScenarioBranch; dominant: boolean }) {
  return <section className="ai-card">
    <div className="section-header">
      <div>
        <p>{dominant ? 'DOMINANT FUTURE' : `ALTERNATIVE #${scenario.rank}`}</p>
        <h2>{humanize(scenario.scenario_type)}</h2>
      </div>
      <Route size={19} />
    </div>
    <span className={`status-badge status-badge--${tone(scenario.direction)}`}>
      {scenario.direction} · {(scenario.probability * 100).toFixed(1)}%
    </span>
    <ol className="scenario-path">
      {scenario.path.map(stage => <li key={`${scenario.scenario_id}-${stage.sequence}`}>
        <strong>+{stage.minute_from}–{stage.minute_to}m · {humanize(stage.event)}</strong>
        <span>{stage.expected_price_area ? `${price(stage.expected_price_area.low)}–${price(stage.expected_price_area.high)}` : 'Price area evolving'}</span>
        {stage.invalidation_condition && <small>{humanize(stage.invalidation_condition)}</small>}
      </li>)}
    </ol>
    <div className="health-grid">
      <div><small>Expected range</small><strong>{price(scenario.expected_range.low)} – {price(scenario.expected_range.high)}</strong></div>
      <div><small>Likely close</small><strong>{price(scenario.likely_close.low)} – {price(scenario.likely_close.high)}</strong></div>
    </div>
  </section>
}

function OpportunityCard({ opportunity }: { opportunity: OpportunityWindow }) {
  return <section className="ai-card">
    <div className="section-header">
      <div><p>FUTURE OPPORTUNITY</p><h2>{opportunity.direction} window</h2></div>
      <Target size={19} />
    </div>
    <span className={`status-badge status-badge--${opportunity.state === 'INVALIDATED' ? 'negative' : opportunity.state === 'ARMED' || opportunity.state === 'TRIGGERED' ? 'positive' : 'neutral'}`}>
      {opportunity.state}
    </span>
    <div className="health-grid">
      <div><small>Expected window</small><strong>+{opportunity.expected_from_minute}m → +{opportunity.expected_to_minute}m</strong></div>
      <div><small>Entry area</small><strong>{price(opportunity.entry_zone.low)} – {price(opportunity.entry_zone.high)}</strong></div>
      <div><small>Probability</small><strong>{(opportunity.probability * 100).toFixed(1)}%</strong></div>
      <div><small>Quality</small><strong>{opportunity.quality.toFixed(0)} / 100</strong></div>
      <div><small>Invalidation</small><strong>{price(opportunity.invalidation_level)}</strong></div>
      <div><small>Targets</small><strong>{opportunity.targets.map(price).join(' → ')}</strong></div>
    </div>
    <div className="reasoning-copy">
      <strong>Waiting for</strong>
      <ul>{opportunity.trigger_conditions.map(item => <li key={item}>{humanize(item)}</li>)}</ul>
    </div>
  </section>
}

export function FutureMarketPage({ view = 'forecast' }: { view?: 'forecast' | 'opportunities' }) {
  const { selection } = useActiveSelection()
  const [forecast, setForecast] = useState<FutureMarketForecast | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const refresh = useCallback(async () => {
    setRefreshing(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/v2/future-market/latest?instrument=${encodeURIComponent(selection.instrument)}`, { cache: 'no-store' })
      if (!response.ok) throw new Error(response.status === 404 ? 'TEN 2.0 is waiting for its first completed scenario cycle.' : `Future-market API returned ${response.status}`)
      setForecast(await response.json() as FutureMarketForecast)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Future-market forecast unavailable')
    } finally {
      setRefreshing(false)
    }
  }, [selection.instrument])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 5_000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const dominant = useMemo(
    () => forecast?.scenarios.find(item => item.scenario_id === forecast.dominant_scenario_id) ?? forecast?.scenarios[0] ?? null,
    [forecast],
  )

  return <div className="ai-page">
    <header className="ai-page__intro">
      <div>
        <p>TEN 2.0 · FUTURE MARKET SIMULATOR</p>
        <h1>{view === 'forecast' ? `${selection.instrument} · Next 30 Minutes` : 'Upcoming Opportunities'}</h1>
      </div>
      <button type="button" onClick={() => void refresh()} disabled={refreshing}>
        <RefreshCw size={16} /> {refreshing ? 'Updating…' : 'Refresh future'}
      </button>
    </header>

    {error && <section className="ai-card ai-card--wide"><strong>Forecast not ready</strong><p className="reasoning-copy">{error}</p></section>}

    {forecast && <>
      <section className="ai-card ai-card--wide">
        <div className="section-header">
          <div><p>WORLD STATE</p><h2>30-minute scenario map</h2></div>
          <Waves size={20} />
        </div>
        <div className="health-grid">
          <div><small>Reference</small><strong>{price(forecast.market_state.reference_price)}</strong></div>
          <div><small>Regime</small><strong>{humanize(forecast.market_state.regime)}</strong></div>
          <div><small>Uncertainty</small><strong>{(forecast.market_state.uncertainty * 100).toFixed(1)}%</strong></div>
          <div><small>Horizon</small><strong>{forecast.forecast_horizon_seconds / 60} minutes</strong></div>
          <div><small>Refresh cadence</small><strong>{forecast.forecast_cadence_seconds / 60} minutes</strong></div>
          <div><small>Context</small><strong>{forecast.market_state.context_timeframes.join(' · ')}</strong></div>
        </div>
        <p className="reasoning-copy">TEN maps plausible futures and waits for scenario conditions. It does not treat a directional forecast as an immediate trade command.</p>
      </section>

      {view === 'forecast' && <>
        <div className="ai-card-grid">
          {forecast.scenarios.map(scenario => <ScenarioCard key={scenario.scenario_id} scenario={scenario} dominant={scenario.scenario_id === dominant?.scenario_id} />)}
        </div>
        <section className="ai-card ai-card--wide">
          <div className="section-header"><div><p>TIME AXIS</p><h2>What TEN expects to unfold</h2></div><Clock3 size={19} /></div>
          {dominant ? <ol className="scenario-path">
            {dominant.path.map(stage => <li key={`timeline-${stage.sequence}`}><strong>+{stage.minute_from}–{stage.minute_to} minutes</strong><span>{humanize(stage.event)}</span></li>)}
          </ol> : <p className="reasoning-copy">No dominant scenario is available.</p>}
        </section>
      </>}

      {(view === 'opportunities' || forecast.opportunities.length > 0) && <section>
        <div className="section-header"><div><p>OPPORTUNITY DISCOVERY</p><h2>Conditional entry windows</h2></div><Activity size={19} /></div>
        {forecast.opportunities.length ? <div className="ai-card-grid">{forecast.opportunities.map(item => <OpportunityCard key={item.opportunity_id} opportunity={item} />)}</div> : <section className="ai-card"><p className="reasoning-copy">No qualified opportunity is forming inside the current 30-minute scenario map.</p></section>}
      </section>}

      <section className="ai-card ai-card--wide">
        <small>Provider: {forecast.provider} · {forecast.model_name} · {forecast.model_version} · Generated {new Date(forecast.generated_at).toLocaleTimeString()}</small>
      </section>
    </>}
  </div>
}
