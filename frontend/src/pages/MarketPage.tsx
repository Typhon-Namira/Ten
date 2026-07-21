import { CandlestickChart } from 'lucide-react'
import { ChartWorkspace } from '../components/ChartWorkspace'
import { MarketIntelligencePanel } from '../components/MarketIntelligencePanel'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useChartOverlays } from '../hooks/useChartOverlays'
import { useLiveDashboard } from '../hooks/useLiveDashboard'

export function MarketPage() {
  const { selection } = useActiveSelection()
  const { marketIntelligence } = useLiveDashboard(selection.instrument, selection.timeframe)
  const { data: overlays } = useChartOverlays(selection.instrument, selection.timeframe)

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">MARKET WORKSPACE</p><h1>{selection.instrument} <em>full-screen chart.</em></h1></div>
        <div className="page-icon"><CandlestickChart size={25} /></div>
      </header>
      <section className="panel panel--chart">
        <div className="panel__head"><div><p className="eyebrow">LIVE MARKET</p><h2>{selection.instrument} chart</h2></div></div>
        <ChartWorkspace instrument={selection.instrument} defaultTimeframe={selection.timeframe} />
      </section>
      <section className="panel">
        <div className="panel__head"><div><p className="eyebrow">MARKET INTELLIGENCE</p><h2>Session, structure &amp; regime</h2></div><span>updates every 5s</span></div>
        <div className="panel-body"><MarketIntelligencePanel data={marketIntelligence} liquidityPools={overlays?.liquidity_pools ?? []} /></div>
      </section>
    </div>
  )
}
