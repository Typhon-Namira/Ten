import { BrainCircuit, CalendarClock, CandlestickChart, ChartNoAxesCombined, Droplets, FileClock, Gauge, Settings2, ShieldCheck, Waves } from 'lucide-react'
import { EngineGrid } from '../components/EngineGrid'
import { PlaceholderPage } from '../components/PlaceholderPage'
import { useDashboard } from '../hooks/useDashboard'

export const MarketPage = () => <PlaceholderPage eyebrow="MARKET WORKSPACE" title="Market" description="Provider-normalized XAU/USD observations and session context." icon={<CandlestickChart size={25} />} capabilities={['Multi-timeframe candles', 'Session state', 'Provider provenance']} />
export const SMCPage = () => <PlaceholderPage eyebrow="STRUCTURE WORKSPACE" title="SMC / ICT" description="Versioned smart-money and market-structure feature exploration." icon={<ChartNoAxesCombined size={25} />} capabilities={['Structure events', 'Displacement zones', 'Premium / discount']} />
export const LiquidityPage = () => <PlaceholderPage eyebrow="LIQUIDITY WORKSPACE" title="Liquidity" description="Liquidity pools, session levels, and sweep evidence." icon={<Droplets size={25} />} capabilities={['Buy-side liquidity', 'Sell-side liquidity', 'Session ranges']} />
export const InstitutionalFlowPage = () => <PlaceholderPage eyebrow="FLOW WORKSPACE" title="Institutional Flow" description="Transparent OHLCV estimates with methodology provenance." icon={<Waves size={25} />} capabilities={['Volume pressure', 'Acceleration', 'Absorption probability']} />
export const VolumeProfilePage = () => <PlaceholderPage eyebrow="PROFILE WORKSPACE" title="Volume Profile" description="Session and composite price-distribution features." icon={<Gauge size={25} />} capabilities={['POC / VAH / VAL', 'HVN / LVN', 'Composite profiles']} />
export const EconomicCalendarPage = () => <PlaceholderPage eyebrow="RISK WORKSPACE" title="Economic Calendar" description="Macro-event windows and configurable risk filtering." icon={<CalendarClock size={25} />} capabilities={['Impact classification', 'Risk windows', 'Event provenance']} />
export const AIAnalysisPage = () => <PlaceholderPage eyebrow="AI WORKSPACE" title="AI Analysis" description="OpenRouter assessments consume feature snapshots only; confidence remains deterministic." icon={<BrainCircuit size={25} />} capabilities={['Feature context', 'Prompt versions', 'Reasoning provenance']} />
export const LogsPage = () => <PlaceholderPage eyebrow="OBSERVABILITY" title="Logs" description="Engine, event, and pipeline lifecycle visibility." icon={<FileClock size={25} />} capabilities={['Correlation IDs', 'Event history', 'Engine failures']} />
export const ConfigurationPage = () => <PlaceholderPage eyebrow="PLATFORM CONTROL" title="Configuration" description="Read-only view of versioned YAML settings and feature flags." icon={<Settings2 size={25} />} capabilities={['Pipeline order', 'Engine versions', 'Feature flags']} />

export function EngineStatusPage() {
  const { engines, loading, error } = useDashboard()
  return <div className="page module-page"><header><div><p className="eyebrow">PLATFORM HEALTH</p><h1>Engine Status</h1><p className="page-description">Discovered versions, compatibility contracts, dependencies, and runtime state.</p></div><div className="page-icon"><ShieldCheck size={25} /></div></header>{error && <div className="alert">{error}</div>}<section className="panel"><div className="panel__head"><div><p className="eyebrow">ENGINE REGISTRY</p><h2>{loading ? 'Refreshing…' : `${engines.length} registered engines`}</h2></div></div><EngineGrid engines={engines} /></section></div>
}
