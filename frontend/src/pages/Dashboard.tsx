import { Activity, Clock3, History, RefreshCw, ShieldCheck, Signal as SignalIcon } from 'lucide-react'
import { EngineGrid } from '../components/EngineGrid'
import { MetricCard } from '../components/MetricCard'
import { SignalTable } from '../components/SignalTable'
import { useDashboard } from '../hooks/useDashboard'

export function Dashboard() {
  const { signals, engines, market, aiScore, operationalSignal, replays, diagnostics, loading, error, refresh } = useDashboard()
  const latest = signals[0]
  const latestReplay = replays[0]
  const ready = engines.filter((engine) => engine.state === 'ready').length
  const marketValue = market?.market_status.replaceAll('_', ' ') ?? 'UNKNOWN'
  const marketDetail = market?.is_open
    ? `${market.session?.replaceAll('_', ' ') ?? 'active'} session · latest candle ${market.latest_candle_at ? new Date(market.latest_candle_at).toLocaleString() : 'unavailable'}`
    : `${market?.closure_reason?.replaceAll('_', ' ') ?? 'status unavailable'}${market?.next_expected_open_at ? ` · expected open ${new Date(market.next_expected_open_at).toLocaleString()}` : ''}`
  const latestDecision = diagnostics?.pipeline.latest_decision
  const pipelineHealthy = diagnostics?.operational_state.startsWith('HEALTHY') ?? false
  const operationalValue = !market?.is_open ? 'MARKET CLOSED'
    : operationalSignal?.state.replaceAll('_', ' ').toUpperCase()
      ?? (latestDecision ? latestDecision.state === 'eligible' ? 'QUALIFIED SIGNAL' : 'NO QUALIFIED CONFLUENCE' : undefined)
      ?? (diagnostics?.operational_state.replaceAll('_', ' ') ?? 'NO QUALIFIED CONFLUENCE')
  const operationalDetail = !market?.is_open
    ? diagnostics?.history.initialized
      ? `Historical data loaded · no live scenario until ${market?.next_expected_open_at ? new Date(market.next_expected_open_at).toLocaleString() : 'the market reopens'}`
      : `Initializing market history · ${diagnostics?.history.candle_count ?? 0} / ${diagnostics?.history.required_candle_count ?? 0} candles`
    : operationalSignal
      ? `${operationalSignal.direction.toUpperCase()} · ${operationalSignal.provider_provenance.join(', ')} · ${operationalSignal.data_quality_status}`
      : latestDecision
        ? latestDecision.blockers.map((item) => item.reason_code).join(', ') || 'Completed without a publishable live scenario'
        : pipelineHealthy ? 'Pipeline healthy; no live scenario is currently published' : 'Pipeline has not completed a persisted snapshot'

  return <div className="page">
    <header><div><p className="eyebrow">INSTITUTIONAL ANALYSIS WORKSPACE</p><h1>Gold intelligence <em>without the noise.</em></h1></div><button onClick={() => void refresh()} disabled={loading}><RefreshCw size={16} className={loading ? 'spin' : ''} />Refresh</button></header>
    {error && <div className="alert"><span>API offline</span>{error}. Dashboard will retry automatically.</div>}
    <section className="metrics" id="overview">
      <MetricCard label="Market" value={marketValue} detail={marketDetail} icon={<Clock3 size={18} />} accent={market?.is_open ? 'green' : market?.market_status === 'UNKNOWN' ? 'gold' : 'red'} />
      <MetricCard label="Operational decision" value={operationalValue} detail={operationalDetail} icon={<SignalIcon size={18} />} accent={operationalSignal?.state === 'eligible' ? 'green' : pipelineHealthy ? 'gold' : 'red'} />
      <MetricCard label="Confidence / risk" value={aiScore ? `${Math.round(aiScore.confidence_score)}% / ${Math.round(aiScore.market_risk_score)}%` : latest ? `${Math.round(latest.confidence * 100)}% / —` : '—'} detail={aiScore ? `Quality ${Math.round(aiScore.data_quality_score)}% · policy ${aiScore.policy_version}` : `History ${diagnostics?.history.candle_count ?? 0} / ${diagnostics?.history.required_candle_count ?? 0} candles`} icon={<Activity size={18} />} />
      <MetricCard label="System" value={diagnostics?.operational_state.replaceAll('_', ' ') ?? `${ready}/${engines.length || '—'}`} detail={`${ready}/${engines.length || '—'} pipeline components registered`} icon={<ShieldCheck size={18} />} accent={pipelineHealthy ? 'green' : 'gold'} />
      <MetricCard label="Historical replay" value={latestReplay?.status.replaceAll('_', ' ').toUpperCase() ?? (diagnostics?.replay.enabled ? 'NO SESSIONS CREATED' : 'REPLAY DISABLED')} detail={latestReplay ? `${latestReplay.request.dataset.dataset_version} · ${latestReplay.processed_events.toLocaleString()} events · ${latestReplay.progress_percent ?? '—'}%` : diagnostics?.replay.enabled ? 'No replay sessions have been created' : 'Replay worker is intentionally disabled'} icon={<History size={18} />} accent={latestReplay?.status === 'completed' ? 'green' : latestReplay?.status === 'failed' ? 'red' : 'gold'} />
    </section>
    <section className="panel" id="signals"><div className="panel__head"><div><p className="eyebrow">SCENARIO FEED</p><h2>Current signals</h2></div><span>{signals.length} scenarios</span></div><SignalTable signals={signals} /></section>
    <section className="panel" id="engines"><div className="panel__head"><div><p className="eyebrow">PIPELINE HEALTH</p><h2>Pipeline components</h2></div><span>30s refresh</span></div><EngineGrid engines={engines} /></section>
  </div>
}
