import type { ReactNode } from 'react'
import {
  Activity, BrainCircuit, CalendarClock, CandlestickChart, ChartNoAxesCombined, Clock3, Database, Droplets,
  FileClock, Gauge, GitBranch, History, Settings2, ShieldCheck, Signal, Stethoscope, Waves, Workflow, XOctagon,
} from 'lucide-react'
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

// Matches the institutional-terminal grouping: Market / Analysis Engines / Signals / System /
// Settings — every path here is a real, working page backed by live data, nothing is a dead link.
const GROUPS: NavGroup[] = [
  { label: null, items: [{ path: '/', label: 'Dashboard', icon: Activity }] },
  {
    label: 'Market',
    items: [
      { path: '/', label: 'Live Chart', icon: CandlestickChart },
      { path: '/market', label: 'Market Intelligence', icon: Gauge },
      { path: '/replay', label: 'Replay', icon: History },
    ],
  },
  {
    label: 'Analysis Engines',
    items: [
      { path: '/smc', label: 'SMC', icon: ChartNoAxesCombined },
      { path: '/liquidity', label: 'Liquidity', icon: Droplets },
      { path: '/volume-profile', label: 'Volume Profile', icon: Gauge },
      { path: '/institutional-flow', label: 'Institutional Flow', icon: Waves },
      { path: '/market-regime', label: 'Market Regime', icon: Waves },
      { path: '/economic-calendar', label: 'Economic Calendar', icon: CalendarClock },
      { path: '/ai-analysis', label: 'AI Scoring', icon: BrainCircuit },
    ],
  },
  {
    label: 'Signals',
    items: [
      { path: '/signals', label: 'Current Signals', icon: Signal },
      { path: '/signals/rejected', label: 'Rejected Signals', icon: XOctagon },
      { path: '/signals/history', label: 'Signal History', icon: Clock3 },
      { path: '/signals/decision-tree', label: 'Decision Tree', icon: GitBranch },
    ],
  },
  {
    label: 'System',
    items: [
      { path: '/pipeline', label: 'Pipeline', icon: Workflow },
      { path: '/performance', label: 'Performance', icon: Gauge },
      { path: '/logs', label: 'Live Logs', icon: FileClock },
      { path: '/diagnostics', label: 'Diagnostics', icon: Stethoscope },
      { path: '/engine-status', label: 'Engine Registry', icon: ShieldCheck },
    ],
  },
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
              <a className={currentPath === path ? 'active' : ''} href={path} key={`${group.label ?? 'root'}-${path}-${label}`} onClick={(event) => { event.preventDefault(); navigate(path) }}>
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
