import { Workflow } from 'lucide-react'
import { LiveLogPanel } from '../components/LiveLogPanel'
import { PipelineStageTracker } from '../components/PipelineStageTracker'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useEventStream } from '../hooks/useEventStream'
import { useLiveDashboard } from '../hooks/useLiveDashboard'

export function PipelinePage() {
  const { selection } = useActiveSelection()
  const { stages, lastUpdated } = useLiveDashboard(selection.instrument, selection.timeframe)
  const { status: streamStatus, events } = useEventStream()

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SYSTEM</p><h1>Pipeline <em>execution.</em></h1></div>
        <div className="page-icon"><Workflow size={25} /></div>
      </header>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">LIVE STAGE FLOW</p><h2>{selection.instrument} / {selection.timeframe}</h2></div><span>{lastUpdated ? `updated ${lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}` : 'connecting'}</span></div>
        <div className="panel-body"><PipelineStageTracker data={stages} /></div>
      </section>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">EVENT STREAM</p><h2>Every pipeline event, live</h2></div><span>{events.length} buffered</span></div>
        <LiveLogPanel status={streamStatus} events={events} />
      </section>
    </div>
  )
}
