import { Signal as SignalIcon } from 'lucide-react'
import { PublicationDistancePanel } from '../components/PublicationDistancePanel'
import { SignalTable } from '../components/SignalTable'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useDashboard } from '../hooks/useDashboard'
import { useLiveDashboard } from '../hooks/useLiveDashboard'

export function CurrentSignalsPage() {
  const { selection } = useActiveSelection()
  const { signals } = useDashboard(selection.instrument, selection.timeframe)
  const { marketIntelligence, rejections } = useLiveDashboard(selection.instrument, selection.timeframe)

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SIGNALS</p><h1>Current <em>signals.</em></h1></div>
        <div className="page-icon"><SignalIcon size={25} /></div>
      </header>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">SCENARIO FEED</p><h2>Published, analytical-only scenarios</h2></div><span>{signals.length} live</span></div>
        {signals.length === 0 ? <PublicationDistancePanel intelligence={marketIntelligence} rejections={rejections} /> : <SignalTable signals={signals} />}
      </section>
    </div>
  )
}
