import { BrainCircuit, Droplets, FileClock, Gauge, Settings2, ShieldCheck, Waves } from 'lucide-react'
import { EngineGrid } from '../components/EngineGrid'
import { PlaceholderPage } from '../components/PlaceholderPage'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useDashboard } from '../hooks/useDashboard'
import { EngineDetailPage } from './engines/EngineDetailPage'

export const LiquidityPage = () => (
  <EngineDetailPage eyebrow="LIQUIDITY WORKSPACE" title="Liquidity Engine" description="Liquidity pools, session levels, sweeps, raids, and target ranking." icon={<Droplets size={25} />} basePath="/liquidity" statePath="/snapshot" />
)
export const InstitutionalFlowPage = () => (
  <EngineDetailPage eyebrow="FLOW WORKSPACE" title="Institutional Flow Engine" description="Participation intensity, initiative/responsive activity, inventory, and campaign phase." icon={<Waves size={25} />} basePath="/institutional-flow" statePath="/snapshot" />
)
export const VolumeProfilePage = () => (
  <EngineDetailPage eyebrow="PROFILE WORKSPACE" title="Volume Profile Engine" description="Session and composite price-distribution features: POC, value area, volume nodes." icon={<Gauge size={25} />} basePath="/volume-profile" statePath="/snapshot" />
)
export const MarketRegimePage = () => (
  <EngineDetailPage eyebrow="REGIME WORKSPACE" title="Market Regime Engine" description="Trend, compression, volatility, strength, and regime transition history." icon={<Waves size={25} />} basePath="/market-regime" statePath="/state" />
)
export const AIAnalysisPage = () => (
  <EngineDetailPage eyebrow="AI WORKSPACE" title="AI Scoring Engine" description="Provider assessments consume feature snapshots only; confidence remains deterministic, never model-generated." icon={<BrainCircuit size={25} />} basePath="/ai-scoring" statePath="/latest" />
)
export const LogsPage = () => <PlaceholderPage eyebrow="OBSERVABILITY" title="Logs" description="Engine, event, and pipeline lifecycle visibility." icon={<FileClock size={25} />} capabilities={['Correlation IDs', 'Event history', 'Engine failures']} />
export const ConfigurationPage = () => <PlaceholderPage eyebrow="PLATFORM CONTROL" title="Configuration" description="Read-only view of versioned YAML settings and feature flags." icon={<Settings2 size={25} />} capabilities={['Pipeline order', 'Engine versions', 'Feature flags']} />

export function EngineStatusPage() {
  const { selection } = useActiveSelection()
  const { engines, loading, error } = useDashboard(selection.instrument, selection.timeframe)
  return <div className="page module-page"><header><div><p className="eyebrow">PLATFORM HEALTH</p><h1>Engine Status</h1><p className="page-description">Discovered versions, compatibility contracts, dependencies, and runtime state.</p></div><div className="page-icon"><ShieldCheck size={25} /></div></header>{error && <div className="alert">{error}</div>}<section className="panel"><div className="panel__head"><div><p className="eyebrow">ENGINE REGISTRY</p><h2>{loading ? 'Refreshing…' : `${engines.length} registered engines`}</h2></div></div><EngineGrid engines={engines} /></section></div>
}
