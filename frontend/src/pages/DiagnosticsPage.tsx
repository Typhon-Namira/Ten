import { Stethoscope } from 'lucide-react'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useDashboard } from '../hooks/useDashboard'
import { normalizeEngineState } from '../lib/engineState'
import { StateBadge } from '../components/StateBadge'

const time = (value: string | null) => (value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'never')

export function DiagnosticsPage() {
  const { selection } = useActiveSelection()
  const { diagnostics } = useDashboard(selection.instrument, selection.timeframe)

  if (!diagnostics) {
    return <div className="page"><header><div><p className="eyebrow">SYSTEM</p><h1>Diagnostics</h1></div></header><div className="empty-state"><h3>Connecting…</h3></div></div>
  }
  const operationalHealthy = diagnostics.operational_state.startsWith('HEALTHY')
  const marketWorker = diagnostics.workers.market_data_worker
  const integrationWorker = diagnostics.workers.integration_worker

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SYSTEM</p><h1>Full diagnostics <em>report.</em></h1></div>
        <div className="page-icon"><Stethoscope size={25} /></div>
      </header>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">OPERATIONAL STATE</p><h2>{diagnostics.operational_state.replaceAll('_', ' ')}</h2></div><StateBadge state={operationalHealthy ? 'healthy' : 'limited'} /></div>
        <div className="intel-grid">
          <div className="intel-field"><span>Database</span><b><StateBadge state={normalizeEngineState(diagnostics.database.status)} /></b></div>
          <div className="intel-field"><span>Provider</span><b><StateBadge state={normalizeEngineState(diagnostics.provider.status)} /></b></div>
          <div className="intel-field"><span>Provider auth</span><b>{diagnostics.provider.authentication_configured ? 'configured' : 'missing'}</b></div>
          <div className="intel-field"><span>Provider last success</span><b>{time(diagnostics.provider.last_success_at)}</b></div>
          <div className="intel-field"><span>Provider last failure</span><b>{time(diagnostics.provider.last_failure_at)}</b></div>
          <div className="intel-field"><span>Provider last error</span><b>{diagnostics.provider.last_error ?? '—'}</b></div>
          <div className="intel-field"><span>Market</span><b>{diagnostics.market.market_status.replaceAll('_', ' ')}</b></div>
          <div className="intel-field"><span>Market freshness</span><b>{diagnostics.market.freshness}</b></div>
          <div className="intel-field"><span>History</span><b>{diagnostics.history.candle_count} / {diagnostics.history.required_candle_count}</b></div>
        </div>
      </section>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">WORKERS</p><h2>Background worker health</h2></div></div>
        <div className="status-matrix" style={{ padding: 16 }}>
          <div className="status-matrix__cell">
            <StateBadge state={marketWorker.enabled && marketWorker.running && !marketWorker.last_error ? 'healthy' : marketWorker.enabled ? 'limited' : 'disabled'} detail={marketWorker.last_error ?? undefined} />
            <span>Market data worker · {marketWorker.processing_state} · {marketWorker.consecutive_failures} failures</span>
          </div>
          <div className="status-matrix__cell">
            <StateBadge state={integrationWorker.enabled && integrationWorker.running && !integrationWorker.last_error ? 'healthy' : integrationWorker.enabled ? 'limited' : 'disabled'} detail={integrationWorker.last_error ?? undefined} />
            <span>Integration worker · {integrationWorker.consecutive_failures} failures</span>
          </div>
        </div>
      </section>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">PIPELINE</p><h2>Latest snapshot / decision / scenario</h2></div><span>{diagnostics.pipeline.status}</span></div>
        <div className="intel-grid">
          <div className="intel-field"><span>Latest snapshot</span><b>{diagnostics.pipeline.latest_snapshot ? 'present' : 'none yet'}</b></div>
          <div className="intel-field"><span>Latest decision</span><b>{diagnostics.pipeline.latest_decision ? `${diagnostics.pipeline.latest_decision.state} (${diagnostics.pipeline.latest_decision.direction})` : 'none yet'}</b></div>
          <div className="intel-field"><span>Latest scenario</span><b>{diagnostics.pipeline.latest_scenario ? 'published' : 'none active'}</b></div>
          <div className="intel-field"><span>Replay</span><b>{diagnostics.replay.status}</b></div>
        </div>
      </section>
    </div>
  )
}
