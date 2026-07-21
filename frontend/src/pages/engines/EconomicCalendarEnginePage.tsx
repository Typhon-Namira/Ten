import { CalendarClock, CheckCircle2, XCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useActiveSelection } from '../../hooks/useActiveSelection'
import { fetchSafe } from '../../services/api'
import { normalizeEngineState } from '../../lib/engineState'
import { tradingContextBadge } from '../../lib/economicState'
import { EconomicStateBadge } from '../../components/EconomicStateBadge'
import { ProviderHealthPanel } from '../../components/ProviderHealthPanel'
import type { ProviderStatus } from '../../types'
import { EngineDetailPage } from './EngineDetailPage'

const POLL_MS = 5_000

interface StagedDiagnostics {
  provider_health: { status: string; reachable_providers: string[] }
  downloaded_events: { status: string; count: number }
  mapped_events: { status: string; mapped_count: number; unmapped_count: number }
  relevant_events: { status: string; active_count: number }
  trading_context: { status: string; context_state: string; risk_window_phase: string; reason: string | null }
}

/** Provider -> Download -> Mapping -> Relevance -> Trading context, each independently
 * observable, sourced from `/economic-calendar/diagnostics` (built to answer exactly "why did
 * this become unavailable" without collapsing every stage into one boolean) and
 * `/economic-calendar/providers` for the retry/latency/failure detail the brief asked for. */
export function EconomicCalendarEnginePage() {
  const { selection } = useActiveSelection()
  const [stages, setStages] = useState<StagedDiagnostics | null>(null)
  const [providers, setProviders] = useState<ProviderStatus[] | null>(null)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      const [nextStages, nextProviders] = await Promise.all([
        fetchSafe<StagedDiagnostics>(`/economic-calendar/diagnostics?symbol=${encodeURIComponent(selection.instrument)}`),
        fetchSafe<ProviderStatus[]>('/economic-calendar/providers'),
      ])
      if (cancelled) return
      setStages(nextStages)
      setProviders(nextProviders)
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [selection.instrument])

  const pipelineStages = stages ? [
    { label: 'Provider', status: stages.provider_health.status, detail: stages.provider_health.reachable_providers.join(', ') || 'no reachable provider' },
    { label: 'Download', status: stages.downloaded_events.status, detail: `${stages.downloaded_events.count} events` },
    { label: 'Mapping', status: stages.mapped_events.status, detail: `${stages.mapped_events.mapped_count} mapped / ${stages.mapped_events.unmapped_count} unmapped` },
    { label: 'Relevance filter', status: stages.relevant_events.status, detail: `${stages.relevant_events.active_count} active` },
    { label: 'Trading context', status: stages.trading_context.status, detail: stages.trading_context.reason ?? 'ready' },
  ] : []

  return (
    <EngineDetailPage
      eyebrow="RISK WORKSPACE"
      title="Economic Calendar Engine"
      description="Macro-event windows and configurable risk filtering — every stage of provider-to-decision reported independently."
      icon={<CalendarClock size={25} />}
      basePath="/economic-calendar"
      statePath="/diagnostics"
      extra={
        <>
          <section className="panel">
            <div className="panel__head">
              <div><p className="eyebrow">STAGED PIPELINE</p><h2>Provider → download → mapping → relevance → trading context</h2></div>
              {stages && <EconomicStateBadge visual={tradingContextBadge(stages.trading_context.context_state, stages.trading_context.risk_window_phase)} detail={stages.trading_context.reason ?? undefined} />}
            </div>
            <div className="panel-body">
              <div className="stage-flow stage-flow--horizontal">
                {pipelineStages.map((stage, index) => {
                  const canonical = normalizeEngineState(stage.status)
                  const ok = canonical === 'healthy' || canonical === 'running'
                  return (
                    <div className="stage-flow__node-wrap" key={stage.label}>
                      <div className={`stage-flow__node stage-flow__node--${canonical === 'healthy' ? 'success' : canonical === 'failed' || canonical === 'blocked' ? 'failed' : 'degraded'}`}>
                        {ok ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                      </div>
                      <span className="stage-flow__label">{stage.label}<br /><small>{stage.detail}</small></span>
                      {index < pipelineStages.length - 1 && <div className="stage-flow__connector stage-flow__connector--filled" />}
                    </div>
                  )
                })}
              </div>
              {!stages && <p className="widget-empty">Loading staged diagnostics…</p>}
            </div>
          </section>
          <section className="panel">
            <div className="panel__head"><div><p className="eyebrow">PROVIDER HEALTH</p><h2>Configured providers</h2></div><span>updates every 5s</span></div>
            <ProviderHealthPanel providers={providers} />
          </section>
        </>
      }
    />
  )
}
