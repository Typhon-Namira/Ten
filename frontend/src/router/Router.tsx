import { useEffect, useState } from 'react'
import { AIAnalysisPage, ConfigurationPage, EngineStatusPage, InstitutionalFlowPage, LiquidityPage, LogsPage, MarketRegimePage, VolumeProfilePage } from '../pages/ModulePages'
import { Dashboard } from '../pages/Dashboard'
import { MarketPage } from '../pages/MarketPage'
import { ReplaySessionsPage } from '../pages/ReplaySessionsPage'
import { CurrentSignalsPage } from '../pages/CurrentSignalsPage'
import { RejectedSignalsPage } from '../pages/RejectedSignalsPage'
import { SignalHistoryPage } from '../pages/SignalHistoryPage'
import { PipelinePage } from '../pages/PipelinePage'
import { PerformancePage } from '../pages/PerformancePage'
import { DiagnosticsPage } from '../pages/DiagnosticsPage'
import { SMCEnginePage } from '../pages/engines/SMCEnginePage'
import { EconomicCalendarEnginePage } from '../pages/engines/EconomicCalendarEnginePage'
import { SignalDecisionTreePage } from '../pages/engines/SignalDecisionTreePage'
import { AssistantPage } from '../pages/AssistantPage'
import { AppShell } from '../components/AppShell'

const routes: Record<string, () => React.JSX.Element> = {
  '/': Dashboard,
  '/market': MarketPage,
  '/market/sessions': MarketPage,
  '/replay': ReplaySessionsPage,
  '/smc': SMCEnginePage,
  '/liquidity': LiquidityPage,
  '/volume-profile': VolumeProfilePage,
  '/institutional-flow': InstitutionalFlowPage,
  '/market-regime': MarketRegimePage,
  '/economic-calendar': EconomicCalendarEnginePage,
  '/ai-analysis': AIAnalysisPage,
  '/signals': CurrentSignalsPage,
  '/signals/rejected': RejectedSignalsPage,
  '/signals/history': SignalHistoryPage,
  '/signals/decision-tree': SignalDecisionTreePage,
  '/assistant': AssistantPage,
  '/pipeline': PipelinePage,
  '/performance': PerformancePage,
  '/logs': LogsPage,
  '/diagnostics': DiagnosticsPage,
  '/engine-status': EngineStatusPage,
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
