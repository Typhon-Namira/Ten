import type { ReactNode } from 'react'
import { Activity, Gauge, Settings2, ShieldCheck, Signal } from 'lucide-react'
import { navigate } from '../router/navigation'

const NAVIGATION = [
  { path: '/', label: 'Overview', icon: Activity },
  { path: '/signals', label: 'Signals', icon: Signal },
  { path: '/performance', label: 'Performance', icon: Gauge },
  { path: '/calibration', label: 'Calibration', icon: ShieldCheck },
  { path: '/system', label: 'System', icon: Settings2 },
]

export function AppShell({ currentPath, children }: { currentPath: string; children: ReactNode }) {
  return <div className="app">
    <nav className="top-nav" aria-label="Primary navigation">
      <a className="top-nav__brand" href="/" onClick={(event) => { event.preventDefault(); navigate('/') }}>
        <span>TEN</span><small>AI MARKET INTELLIGENCE</small>
      </a>
      <div className="top-nav__links">
        {NAVIGATION.map(({ path, label, icon: Icon }) => <a
          className={currentPath === path ? 'active' : ''}
          href={path}
          key={path}
          aria-current={currentPath === path ? 'page' : undefined}
          onClick={(event) => { event.preventDefault(); navigate(path) }}
        ><Icon size={16} />{label}</a>)}
      </div>
      <span className="top-nav__scope">Analysis only</span>
    </nav>
    <main>{children}</main>
    <footer className="app-footer">Analysis and decision support only. No Broker Execution. Outcomes and probabilities are not guarantees.</footer>
  </div>
}
