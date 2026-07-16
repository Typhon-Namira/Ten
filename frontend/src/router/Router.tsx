import { useEffect, useState } from 'react'
import { AIAnalysisPage, ConfigurationPage, EconomicCalendarPage, EngineStatusPage, InstitutionalFlowPage, LiquidityPage, LogsPage, MarketPage, SMCPage, VolumeProfilePage } from '../pages/ModulePages'
import { Dashboard } from '../pages/Dashboard'
import { SignalsPage } from '../pages/SignalsPage'
import { AppShell } from '../components/AppShell'

const routes: Record<string, () => React.JSX.Element> = {
  '/': Dashboard,
  '/signals': SignalsPage,
  '/market': MarketPage,
  '/smc': SMCPage,
  '/liquidity': LiquidityPage,
  '/institutional-flow': InstitutionalFlowPage,
  '/volume-profile': VolumeProfilePage,
  '/economic-calendar': EconomicCalendarPage,
  '/ai-analysis': AIAnalysisPage,
  '/engine-status': EngineStatusPage,
  '/logs': LogsPage,
  '/configuration': ConfigurationPage,
}

export function Router() {
  const [path, setPath] = useState(window.location.pathname)
  useEffect(() => {
    const update = () => setPath(window.location.pathname)
    window.addEventListener('popstate', update)
    return () => window.removeEventListener('popstate', update)
  }, [])
  const Page = routes[path] ?? Dashboard
  return <AppShell currentPath={path}><Page /></AppShell>
}
