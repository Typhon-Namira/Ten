import type { ReactNode } from 'react'
import { Activity, Gauge, Settings2, Target } from 'lucide-react'
import { navigate } from '../router/navigation'
import { DiagnosticsBar } from './DiagnosticsBar'

const NAVIGATION = [
  { path: '/', label: 'Forecast', icon: Activity },
  { path: '/signals', label: 'Opportunities', icon: Target },
  { path: '/performance', label: 'Performance', icon: Gauge },
  { path: '/system', label: 'System', icon: Settings2 },
]

export function AppShell({ currentPath, children }: { currentPath: string; children: ReactNode }) {
  return <div className="app">
    <nav className="top-nav" aria-label="Primary navigation">
      <a className="top-nav__brand" href="/" onClick={(event) => { event.preventDefault(); navigate('/') }}>
        <span>TEN</span><small>FUTURE MARKET INTELLIGENCE</small>
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
      <span className="top-nav__scope">30m scenario intelligence</span>
    </nav>
    <main>{children}</main>
    <DiagnosticsBar />
    <footer className="app-footer">Scenario intelligence and decision support only. No broker execution. Forecast probabilities are not guarantees.</footer>
  </div>
}
