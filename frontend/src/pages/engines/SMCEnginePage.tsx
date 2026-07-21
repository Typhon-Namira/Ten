import { ChartNoAxesCombined } from 'lucide-react'
import { useActiveSelection } from '../../hooks/useActiveSelection'
import { useEngineDetail } from '../../hooks/useEngineDetail'
import { EngineDetailPage } from './EngineDetailPage'

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })

function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value)
}

function count(items: unknown, predicate: (item: Record<string, unknown>) => boolean): number {
  if (!isArray(items)) return 0
  return items.filter((item) => typeof item === 'object' && item !== null && predicate(item as Record<string, unknown>)).length
}

/** A genuine detection funnel built from real counted objects in the latest SMC snapshot — swings
 * feed structure events, structure events + displacement feed zones, zones settle into a
 * lifecycle. This is real data (every count below is `array.length` on an actual snapshot field),
 * not a fabricated sub-step timeline the backend doesn't measure. */
export function SMCEnginePage() {
  const { selection } = useActiveSelection()
  const { state } = useEngineDetail('/smc', '/snapshot', selection.instrument, selection.timeframe)

  const swings = state?.swings
  const events = state?.structure_events
  const displacements = state?.displacements
  const zones = state?.zones
  const dealingRanges = state?.dealing_ranges
  const structureState = state?.structure_state as Record<string, unknown> | undefined

  const funnel = [
    { label: 'Swings detected', value: isArray(swings) ? swings.length : 0 },
    { label: 'BOS', value: count(events, (e) => e.event_type === 'bos') },
    { label: 'CHOCH', value: count(events, (e) => e.event_type === 'choch') },
    { label: 'Displacements', value: isArray(displacements) ? displacements.length : 0 },
    { label: 'Order blocks', value: count(zones, (z) => typeof z.zone_type === 'string' && z.zone_type.includes('order_block')) },
    { label: 'Breakers', value: count(zones, (z) => typeof z.zone_type === 'string' && z.zone_type.includes('breaker')) },
    { label: 'FVGs', value: count(zones, (z) => typeof z.zone_type === 'string' && z.zone_type.includes('fvg')) },
    { label: 'Mitigation blocks', value: count(zones, (z) => typeof z.zone_type === 'string' && z.zone_type.includes('mitigation')) },
  ]
  const maxFunnel = Math.max(...funnel.map((f) => f.value), 1)
  // Computed as a plain top-level value rather than inside an IIFE in the JSX below — keeps this
  // page immune to the same "narrowing doesn't survive a nested closure" class of TS build error
  // fixed earlier in PipelineStageTracker.tsx.
  const latestDealingRange = isArray(dealingRanges) && dealingRanges.length > 0 ? (dealingRanges[dealingRanges.length - 1] as Record<string, unknown>) : null

  return (
    <EngineDetailPage
      eyebrow="STRUCTURE WORKSPACE"
      title="SMC / ICT Engine"
      description="Versioned smart-money and market-structure detection: swings → structure events → displacement → zones → lifecycle."
      icon={<ChartNoAxesCombined size={25} />}
      basePath="/smc"
      statePath="/snapshot"
      extra={
        <section className="panel">
          <div className="panel__head"><div><p className="eyebrow">DETECTION FUNNEL</p><h2>Swings → structure → zones</h2></div></div>
          <div className="panel-body">
            <div className="funnel">
              {funnel.map((stage) => (
                <div className="funnel__row" key={stage.label}>
                  <span className="funnel__label">{stage.label}</span>
                  <div className="funnel__bar-track"><div className="funnel__bar" style={{ width: `${(stage.value / maxFunnel) * 100}%` }} /></div>
                  <b className="funnel__value">{stage.value}</b>
                </div>
              ))}
            </div>
            {structureState && (
              <div className="intel-grid" style={{ marginTop: 16 }}>
                <div className="intel-field"><span>Current direction</span><b>{String(structureState.current_direction ?? '—')}</b></div>
                <div className="intel-field"><span>Internal direction</span><b>{String(structureState.internal_direction ?? '—')}</b></div>
                <div className="intel-field"><span>External direction</span><b>{String(structureState.external_direction ?? '—')}</b></div>
                <div className="intel-field"><span>Structure version</span><b>{String(structureState.state_version ?? '—')}</b></div>
              </div>
            )}
            {latestDealingRange && (
              <div className="intel-grid" style={{ marginTop: 1 }}>
                <div className="intel-field"><span>Range high</span><b>{number.format(Number(latestDealingRange.range_high))}</b></div>
                <div className="intel-field"><span>Equilibrium</span><b>{number.format(Number(latestDealingRange.equilibrium))}</b></div>
                <div className="intel-field"><span>Range low</span><b>{number.format(Number(latestDealingRange.range_low))}</b></div>
                <div className="intel-field"><span>Golden zone</span><b>{number.format(Number(latestDealingRange.golden_zone_low))} – {number.format(Number(latestDealingRange.golden_zone_high))}</b></div>
              </div>
            )}
          </div>
        </section>
      }
    />
  )
}
