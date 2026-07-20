import type { ReactNode } from 'react'
import { Activity, BrainCircuit, CalendarClock, CandlestickChart, ChartNoAxesCombined, Database, Droplets, FileClock, Gauge, Settings2, ShieldCheck, Signal, Waves } from 'lucide-react'
import { navigate } from '../router/navigation'

interface NavItem {
  path: string
  label: string
  icon: typeof Activity
}

interface NavGroup {
  label: string | null
  items: NavItem[]
}

// Grouped into the categories a trading workstation sidebar uses (Market / Signals / Analytics /
// History / Settings) instead of one flat 12-item list — every path here is a real, working page;
// nothing is a placeholder link to functionality that doesn't exist yet.
const GROUPS: NavGroup[] = [
  { label: null, items: [{ path: '/', label: 'Dashboard', icon: Activity }] },
  {
    label: 'Market',
    items: [
      { path: '/market', label: 'Market', icon: CandlestickChart },
      { path: '/smc', label: 'SMC', icon: ChartNoAxesCombined },
      { path: '/liquidity', label: 'Liquidity', icon: Droplets },
      { path: '/institutional-flow', label: 'Institutional Flow', icon: Waves },
      { path: '/volume-profile', label: 'Volume Profile', icon: Gauge },
      { path: '/economic-calendar', label: 'Economic Calendar', icon: CalendarClock },
    ],
  },
  {
    label: 'Signals',
    items: [
      { path: '/signals', label: 'Signals', icon: Signal },
      { path: '/ai-analysis', label: 'AI Analysis', icon: BrainCircuit },
    ],
  },
  { label: 'Analytics', items: [{ path: '/engine-status', label: 'Engine Status', icon: ShieldCheck }] },
  { label: 'History', items: [{ path: '/logs', label: 'Logs', icon: FileClock }] },
  { label: 'Settings', items: [{ path: '/configuration', label: 'Configuration', icon: Settings2 }] },
]

export function AppShell({ currentPath, children }: { currentPath: string; children: ReactNode }) {
  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="/" onClick={(event) => { event.preventDefault(); navigate('/') }}><span>TEN</span><small>MARKET INTELLIGENCE</small></a>
      <nav>
        {GROUPS.map((group) => (
          <div className="nav-group" key={group.label ?? '__root'}>
            {group.label && <p className="nav-group__label">{group.label}</p>}
            {group.items.map(({ path, label, icon: Icon }) => (
              <a className={currentPath === path ? 'active' : ''} href={path} key={path} onClick={(event) => { event.preventDefault(); navigate(path) }}>
                <Icon size={17} />{label}
              </a>
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar__foot"><span><Database size={11} /> ANALYSIS ONLY</span><p>No broker or execution connection</p></div>
    </aside>
    <main>{children}<footer>Signals are analytical scenarios, not financial advice or execution instructions. Validate all market data independently.</footer></main>
  </div>
}
