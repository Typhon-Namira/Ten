import { Gauge as GaugeIcon } from 'lucide-react'
import { PerformancePanel } from '../components/PerformancePanel'
import { Sparkline } from '../components/widgets/Widgets'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useLiveDashboard } from '../hooks/useLiveDashboard'
import { usePerformanceHistory } from '../hooks/usePerformanceHistory'

function nonNull(values: (number | null)[]): number[] {
  return values.filter((v): v is number => v !== null)
}

export function PerformancePage() {
  const { selection } = useActiveSelection()
  const { performance } = useLiveDashboard(selection.instrument, selection.timeframe)
  const history = usePerformanceHistory(performance)

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SYSTEM</p><h1>Performance <em>monitoring.</em></h1></div>
        <div className="page-icon"><GaugeIcon size={25} /></div>
      </header>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">TRENDS</p><h2>Live latency history</h2></div><span>last {history.length} samples</span></div>
        <div className="perf-trend-grid">
          <div className="perf-trend-grid__cell">
            <p className="intel__widget-title">Pipeline latency (ms)</p>
            <Sparkline values={nonNull(history.map((h) => h.pipelineLatency))} width={340} height={70} />
          </div>
          <div className="perf-trend-grid__cell">
            <p className="intel__widget-title">Provider latency (ms)</p>
            <Sparkline values={nonNull(history.map((h) => h.providerLatency))} width={340} height={70} />
          </div>
          <div className="perf-trend-grid__cell">
            <p className="intel__widget-title">Queue length</p>
            <Sparkline values={nonNull(history.map((h) => h.queueLength))} width={340} height={70} />
          </div>
        </div>
      </section>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">SNAPSHOT</p><h2>Current metrics</h2></div><span>updates every 5s</span></div>
        <div className="panel-body"><PerformancePanel data={performance} /></div>
      </section>
    </div>
  )
}
