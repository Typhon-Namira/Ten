import { XOctagon } from 'lucide-react'
import { RejectionReasonPanel } from '../components/RejectionReasonPanel'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useLiveDashboard } from '../hooks/useLiveDashboard'

export function RejectedSignalsPage() {
  const { selection } = useActiveSelection()
  const { rejections } = useLiveDashboard(selection.instrument, selection.timeframe)

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SIGNALS</p><h1>Rejected <em>setups.</em></h1></div>
        <div className="page-icon"><XOctagon size={25} /></div>
      </header>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">REJECTIONS</p><h2>Why signals were blocked</h2></div><span>{rejections?.count ?? 0} recent</span></div>
        <div className="panel-body"><RejectionReasonPanel data={rejections} /></div>
      </section>
    </div>
  )
}
