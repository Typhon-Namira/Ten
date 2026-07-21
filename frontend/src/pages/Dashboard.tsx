import { Activity, Clock3, History, RefreshCw, ShieldCheck, Signal as SignalIcon, Wifi, WifiOff } from 'lucide-react'
import { ChartWorkspace } from '../components/ChartWorkspace'
import { EngineGrid } from '../components/EngineGrid'
import { LiveLogPanel } from '../components/LiveLogPanel'
import { MarketIntelligencePanel } from '../components/MarketIntelligencePanel'
import { MetricCard } from '../components/MetricCard'
import { PerformancePanel } from '../components/PerformancePanel'
import { PipelineStageTracker } from '../components/PipelineStageTracker'
import { PublicationDistancePanel } from '../components/PublicationDistancePanel'
import { RejectionReasonPanel } from '../components/RejectionReasonPanel'
import { SignalTable } from '../components/SignalTable'
import { Sparkline } from '../components/widgets/Widgets'
import { ChartFocusProvider } from '../lib/ChartFocusContext'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useAiScoreHistory } from '../hooks/useAiScoreHistory'
import { useChartOverlays } from '../hooks/useChartOverlays'
import { useDashboard } from '../hooks/useDashboard'
import { useEventStream } from '../hooks/useEventStream'
import { useLiveDashboard } from '../hooks/useLiveDashboard'

export function Dashboard() {
  const { selection } = useActiveSelection()
  const { signals, engines, market, aiScore, operationalSignal, replays, diagnostics, loading, error, refresh } = useDashboard(selection.instrument, selection.timeframe)
  const { status: streamStatus, events } = useEventStream()
  const { stages, rejections, marketIntelligence, performance, lastUpdated } = useLiveDashboard(selection.instrument, selection.timeframe)
  const aiScoreHistory = useAiScoreHistory(selection.instrument, selection.timeframe)
  // Same call the chart makes for its default (unchanged) timeframe — reused here so the
  // liquidity distribution widget shows real per-pool strength data instead of the single
  // `latest_price` scalar that's all `MarketIntelligence.liquidity.state` carries.
  const { data: overlays } = useChartOverlays(selection.instrument, selection.timeframe)
  const latestReplay = replays[0]
  const ready = engines.filter((engine) => engine.state === 'ready').length
  const marketValue = market?.market_status.replaceAll('_', ' ') ?? 'UNKNOWN'
  // Sourced from `marketIntelligence` (5s poll), not `market` (30s poll) — both used to carry
  // their own independently-fetched "latest candle" timestamp, so the header and the market
  // intelligence panel below could legitimately show two different candles for up to ~25s at a
  // time even when otherwise healthy. There is only one authoritative candle-timestamp source now.
  const latestCandleAt = marketIntelligence?.latest_candle_timestamp ?? null
  const marketDetail = market?.is_open
    ? `${market.session?.replaceAll('_', ' ') ?? 'active'} session · latest candle ${latestCandleAt ? new Date(latestCandleAt).toLocaleString() : 'unavailable'}`
    : `${market?.closure_reason?.replaceAll('_', ' ') ?? 'status unavailable'}${market?.next_expected_open_at ? ` · expected open ${new Date(market.next_expected_open_at).toLocaleString()}` : ''}`
  const latestDecision = diagnostics?.pipeline.latest_decision
  const pipelineHealthy = diagnostics?.operational_state.startsWith('HEALTHY') ?? false
  const topRejection = rejections?.rejections[0]
  const operationalValue = !market?.is_open ? 'MARKET CLOSED'
    : operationalSignal?.state.replaceAll('_', ' ').toUpperCase()
      ?? (latestDecision ? latestDecision.state === 'eligible' ? 'QUALIFIED SIGNAL' : latestDecision.state.replaceAll('_', ' ').toUpperCase() : undefined)
      ?? (diagnostics?.operational_state.replaceAll('_', ' ') ?? 'AWAITING FIRST DECISION')
  const operationalDetail = !market?.is_open
    ? diagnostics?.history.initialized
      ? `Historical data loaded · no live scenario until ${market?.next_expected_open_at ? new Date(market.next_expected_open_at).toLocaleString() : 'the market reopens'}`
      : `Initializing market history · ${diagnostics?.history.candle_count ?? 0} / ${diagnostics?.history.required_candle_count ?? 0} candles`
    : operationalSignal
      ? `${operationalSignal.direction.toUpperCase()} · ${operationalSignal.provider_provenance.join(', ')} · ${operationalSignal.data_quality_status}`
      : topRejection
        ? `Failed: ${topRejection.diagnostics.filter((item) => item.status === 'failed').map((item) => item.label).join(', ') || topRejection.blockers.join(', ') || 'see rejection detail below'}`
        : latestDecision
          ? latestDecision.blockers.map((item) => item.reason_code).join(', ') || 'Completed without a publishable live scenario'
          : pipelineHealthy ? 'Pipeline healthy; no live scenario is currently published' : 'Pipeline has not completed a persisted snapshot'

  return <ChartFocusProvider><div className="page">
    <header>
      <div><p className="eyebrow">INSTITUTIONAL ANALYSIS WORKSPACE</p><h1>Gold intelligence <em>without the noise.</em></h1></div>
      <div className="header-actions">
        <span className={`stream-status stream-status--${streamStatus}`}>{streamStatus === 'open' ? <Wifi size={13} /> : <WifiOff size={13} />}{streamStatus === 'open' ? 'Live' : streamStatus}</span>
        {lastUpdated && <span className="stream-status">updated {lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>}
        <button onClick={() => void refresh()} disabled={loading}><RefreshCw size={16} className={loading ? 'spin' : ''} />Refresh</button>
      </div>
    </header>
    {error && <div className="alert"><span>API degraded</span>{error}. Showing the last known-good data; retrying automatically.</div>}
    <section className="metrics" id="overview">
      <MetricCard label="Market" value={marketValue} detail={marketDetail} icon={<Clock3 size={18} />} accent={market?.is_open ? 'green' : market?.market_status === 'UNKNOWN' ? 'gold' : 'red'} />
      <MetricCard label="Operational decision" value={operationalValue} detail={operationalDetail} icon={<SignalIcon size={18} />} accent={operationalSignal?.state === 'eligible' ? 'green' : pipelineHealthy ? 'gold' : 'red'} />
      <MetricCard label="Confidence / risk" value={aiScore ? `${Math.round(aiScore.confidence_score)}% / ${Math.round(aiScore.market_risk_score)}%` : '—'} detail={aiScore ? `Quality ${Math.round(aiScore.data_quality_score)}% · policy ${aiScore.policy_version}` : `History ${diagnostics?.history.candle_count ?? 0} / ${diagnostics?.history.required_candle_count ?? 0} candles`} icon={<Activity size={18} />} />
      <MetricCard label="System" value={diagnostics?.operational_state.replaceAll('_', ' ') ?? `${ready}/${engines.length || '—'}`} detail={`${ready}/${engines.length || '—'} pipeline components registered`} icon={<ShieldCheck size={18} />} accent={pipelineHealthy ? 'green' : 'gold'} />
      <MetricCard label="Historical replay" value={latestReplay?.status.replaceAll('_', ' ').toUpperCase() ?? (diagnostics?.replay.enabled ? 'NO SESSIONS CREATED' : 'REPLAY DISABLED')} detail={latestReplay ? `${latestReplay.request.dataset.dataset_version} · ${latestReplay.processed_events.toLocaleString()} events · ${latestReplay.progress_percent ?? '—'}%` : diagnostics?.replay.enabled ? 'No replay sessions have been created' : 'Replay worker is intentionally disabled'} icon={<History size={18} />} accent={latestReplay?.status === 'completed' ? 'green' : latestReplay?.status === 'failed' ? 'red' : 'gold'} />
    </section>

    <section className="panel panel--chart" id="chart">
      <div className="panel__head"><div><p className="eyebrow">LIVE MARKET</p><h2>{selection.instrument} chart</h2></div><span>{streamStatus === 'open' ? 'live' : streamStatus}</span></div>
      <ChartWorkspace instrument={selection.instrument} defaultTimeframe={selection.timeframe} />
    </section>

    <div className="workspace-grid">
      <div className="workspace-grid__main">
        <section className="panel" id="stages"><div className="panel__head"><div><p className="eyebrow">PIPELINE STAGES</p><h2>Live stage tracker</h2></div><span>updates every 5s</span></div><div className="panel-body"><PipelineStageTracker data={stages} /></div></section>
        <section className="panel" id="intelligence"><div className="panel__head"><div><p className="eyebrow">MARKET INTELLIGENCE</p><h2>Live market state</h2></div><span>updates every 5s</span></div><div className="panel-body"><MarketIntelligencePanel data={marketIntelligence} liquidityPools={overlays?.liquidity_pools ?? []} /></div></section>
        <section className="panel" id="signals">
          <div className="panel__head"><div><p className="eyebrow">SCENARIO FEED</p><h2>Current signals</h2></div><span>{signals.length} scenarios</span></div>
          {signals.length === 0 ? <PublicationDistancePanel intelligence={marketIntelligence} rejections={rejections} /> : <SignalTable signals={signals} />}
        </section>
        <section className="panel" id="rejections"><div className="panel__head"><div><p className="eyebrow">REJECTED SETUPS</p><h2>Why signals were rejected</h2></div><span>{rejections?.count ?? 0} recent</span></div><div className="panel-body"><RejectionReasonPanel data={rejections} /></div></section>
      </div>
      <div className="workspace-grid__rail">
        <section className="panel" id="confidence-trend">
          <div className="panel__head"><div><p className="eyebrow">TREND</p><h2>AI confidence</h2></div></div>
          <div className="panel-body">
            <Sparkline values={aiScoreHistory.map((item) => item.confidence_score).reverse()} width={260} height={60} />
          </div>
        </section>
        <section className="panel" id="engines"><div className="panel__head"><div><p className="eyebrow">PIPELINE HEALTH</p><h2>Pipeline components</h2></div><span>30s refresh</span></div><EngineGrid engines={engines} /></section>
        <section className="panel" id="performance"><div className="panel__head"><div><p className="eyebrow">PERFORMANCE</p><h2>Latency &amp; worker health</h2></div><span>updates every 5s</span></div><div className="panel-body"><PerformancePanel data={performance} /></div></section>
        <section className="panel" id="logs"><div className="panel__head"><div><p className="eyebrow">LIVE LOGS</p><h2>Pipeline event stream</h2></div><span>{events.length} events</span></div><LiveLogPanel status={streamStatus} events={events} /></section>
      </div>
    </div>
  </div></ChartFocusProvider>
}
