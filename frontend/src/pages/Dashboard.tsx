import { Activity, Clock3, History, RefreshCw, ShieldCheck, Signal as SignalIcon } from 'lucide-react'
import { EngineGrid } from '../components/EngineGrid'
import { MetricCard } from '../components/MetricCard'
import { SignalTable } from '../components/SignalTable'
import { useDashboard } from '../hooks/useDashboard'

export function Dashboard() {
  const { signals, engines, market, aiScore, operationalSignal, replays, loading, error, refresh } = useDashboard()
  const latest = signals[0]
  const latestReplay = replays[0]
  const ready = engines.filter((engine) => engine.state === 'ready').length
  return <div className="page">
    <header><div><p className="eyebrow">INSTITUTIONAL ANALYSIS WORKSPACE</p><h1>Gold intelligence <em>without the noise.</em></h1></div><button onClick={() => void refresh()} disabled={loading}><RefreshCw size={16} className={loading ? 'spin' : ''} />Refresh</button></header>
    {error && <div className="alert"><span>API offline</span>{error}. Dashboard will retry automatically.</div>}
    <section className="metrics" id="overview">
      <MetricCard label="Market" value={market?.is_open ? 'OPEN' : 'CLOSED'} detail={`${market?.session ?? '—'} session · XAU/USD`} icon={<Clock3 size={18} />} accent={market?.is_open ? 'green' : 'red'} />
      <MetricCard label="Operational decision" value={operationalSignal?.state.replaceAll('_', ' ').toUpperCase() ?? 'NO INTEGRATED SIGNAL'} detail={operationalSignal ? `${operationalSignal.direction.toUpperCase()} · ${operationalSignal.provider_provenance.join(', ')} · ${operationalSignal.data_quality_status}` : 'Awaiting the persisted full-system pipeline'} icon={<SignalIcon size={18} />} accent={operationalSignal?.state === 'eligible' ? 'green' : operationalSignal ? 'red' : 'gold'} />
      <MetricCard label="Confidence / risk" value={aiScore ? `${Math.round(aiScore.confidence_score)}% / ${Math.round(aiScore.market_risk_score)}%` : latest ? `${Math.round(latest.confidence * 100)}% / —` : '—'} detail={aiScore ? `Quality ${Math.round(aiScore.data_quality_score)}% · policy ${aiScore.policy_version}` : 'Evidence quality, not win probability'} icon={<Activity size={18} />} />
      <MetricCard label="System" value={`${ready}/${engines.length || '—'}`} detail="Analysis engines ready" icon={<ShieldCheck size={18} />} accent={ready === engines.length && ready > 0 ? 'green' : 'gold'} />
      <MetricCard label="Historical replay" value={latestReplay?.status.replaceAll('_', ' ').toUpperCase() ?? 'NO SESSIONS'} detail={latestReplay ? `${latestReplay.request.dataset.dataset_version} · ${latestReplay.processed_events.toLocaleString()} events · ${latestReplay.progress_percent ?? '—'}%` : 'Isolated analytical reconstruction only'} icon={<History size={18} />} accent={latestReplay?.status === 'completed' ? 'green' : latestReplay?.status === 'failed' ? 'red' : 'gold'} />
    </section>
    <section className="panel" id="signals"><div className="panel__head"><div><p className="eyebrow">SCENARIO FEED</p><h2>Current signals</h2></div><span>{signals.length} scenarios</span></div><SignalTable signals={signals} /></section>
    <section className="panel" id="engines"><div className="panel__head"><div><p className="eyebrow">PIPELINE HEALTH</p><h2>Analysis engines</h2></div><span>30s refresh</span></div><EngineGrid engines={engines} /></section>
  </div>
}
