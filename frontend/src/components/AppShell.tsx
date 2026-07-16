import type { ReactNode } from 'react'
import { Activity, BrainCircuit, CalendarClock, CandlestickChart, ChartNoAxesCombined, Database, Droplets, FileClock, Gauge, Settings2, ShieldCheck, Signal, Waves } from 'lucide-react'
import { navigate } from '../router/navigation'

const items = [
  ['/', 'Dashboard', Activity],
  ['/signals', 'Signals', Signal],
  ['/market', 'Market', CandlestickChart],
  ['/smc', 'SMC', ChartNoAxesCombined],
  ['/liquidity', 'Liquidity', Droplets],
  ['/institutional-flow', 'Institutional Flow', Waves],
  ['/volume-profile', 'Volume Profile', Gauge],
  ['/economic-calendar', 'Economic Calendar', CalendarClock],
  ['/ai-analysis', 'AI Analysis', BrainCircuit],
  ['/engine-status', 'Engine Status', ShieldCheck],
  ['/logs', 'Logs', FileClock],
  ['/configuration', 'Configuration', Settings2],
] as const

export function AppShell({ currentPath, children }: { currentPath: string; children: ReactNode }) {
  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="/" onClick={(event) => { event.preventDefault(); navigate('/') }}><span>TEN</span><small>MARKET INTELLIGENCE</small></a>
      <nav>{items.map(([path, label, Icon]) => <a className={currentPath === path ? 'active' : ''} href={path} key={path} onClick={(event) => { event.preventDefault(); navigate(path) }}><Icon size={17} />{label}</a>)}</nav>
      <div className="sidebar__foot"><span><Database size={11} /> ANALYSIS ONLY</span><p>No broker or execution connection</p></div>
    </aside>
    <main>{children}<footer>Signals are analytical scenarios, not financial advice or execution instructions. Validate all market data independently.</footer></main>
  </div>
}
