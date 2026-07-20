import { Signal as SignalIcon } from 'lucide-react'
import { SignalTable } from '../components/SignalTable'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useDashboard } from '../hooks/useDashboard'

export function SignalsPage() {
  const { selection } = useActiveSelection()
  const { signals, loading, error } = useDashboard(selection.instrument, selection.timeframe)
  return <div className="page module-page"><header><div><p className="eyebrow">SCENARIO INTELLIGENCE</p><h1>Signals</h1><p className="page-description">Explainable, non-executable XAU/USD scenarios with deterministic confidence.</p></div><div className="page-icon"><SignalIcon size={25} /></div></header>{error && <div className="alert">{error}</div>}<section className="panel"><div className="panel__head"><div><p className="eyebrow">SCENARIO FEED</p><h2>{loading ? 'Refreshing…' : 'Current signals'}</h2></div><span>{signals.length} scenarios</span></div><SignalTable signals={signals} /></section></div>
}
