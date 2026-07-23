import { AlertCircle, CheckCircle2, ChevronDown, Circle, Clock3, Info, XCircle } from 'lucide-react'
import type { ReactNode } from 'react'
import type { PipelineStatus, Tone } from '../../lib/aiDashboard'

export function StatusBadge({ tone = 'neutral', children, pulse = false }: { tone?: Tone; children: ReactNode; pulse?: boolean }) {
  return <span className={`ai-badge ai-badge--${tone}${pulse ? ' ai-badge--pulse' : ''}`}>{children}</span>
}

export function SectionHeader({ eyebrow, title, action }: { eyebrow: string; title: string; action?: ReactNode }) {
  return <div className="ai-section-head">
    <div><p>{eyebrow}</p><h2>{title}</h2></div>
    {action}
  </div>
}

export function ProbabilityBar({ buy, sell, neutral }: { buy: number; sell: number; neutral: number }) {
  return <div className="probability">
    <div className="probability__bar" aria-label={`Bullish ${(buy * 100).toFixed(0)}%, bearish ${(sell * 100).toFixed(0)}%, neutral ${(neutral * 100).toFixed(0)}%`}>
      <span className="probability__buy" style={{ width: `${buy * 100}%` }} />
      <span className="probability__neutral" style={{ width: `${neutral * 100}%` }} />
      <span className="probability__sell" style={{ width: `${sell * 100}%` }} />
    </div>
    <div className="probability__legend">
      <span><i className="dot dot--buy" />Bullish <strong>{(buy * 100).toFixed(0)}%</strong></span>
      <span><i className="dot dot--neutral" />Neutral <strong>{(neutral * 100).toFixed(0)}%</strong></span>
      <span><i className="dot dot--sell" />Bearish <strong>{(sell * 100).toFixed(0)}%</strong></span>
    </div>
  </div>
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="ai-empty" role="status"><Circle size={22} /><strong>{title}</strong><p>{detail}</p></div>
}

export function ErrorState({ message }: { message: string }) {
  return <div className="ai-error" role="alert"><AlertCircle size={18} /><span>{message}</span></div>
}

export function LoadingSkeleton({ rows = 3 }: { rows?: number }) {
  return <div className="ai-skeleton" aria-label="Loading dashboard data">{Array.from({ length: rows }).map((_, index) => <span key={index} />)}</div>
}

export function DetailDrawer({ label = 'View details', children }: { label?: string; children: ReactNode }) {
  return <details className="ai-drawer">
    <summary>{label}<ChevronDown size={15} /></summary>
    <div className="ai-drawer__content">{children}</div>
  </details>
}

export function FreshnessIndicator({ stale, timestamp }: { stale: boolean; timestamp: string | null }) {
  return <StatusBadge tone={stale ? 'negative' : timestamp ? 'positive' : 'neutral'}>
    <Clock3 size={12} />{stale ? 'Data stale' : timestamp ? 'Data fresh' : 'Freshness unknown'}
  </StatusBadge>
}

export function PipelineIcon({ status }: { status: PipelineStatus }) {
  if (status === 'completed') return <CheckCircle2 aria-hidden="true" />
  if (status === 'rejected' || status === 'failed') return <XCircle aria-hidden="true" />
  if (status === 'unavailable') return <Info aria-hidden="true" />
  return <Circle aria-hidden="true" />
}

export function Metric({ label, value, detail }: { label: string; value: ReactNode; detail?: string }) {
  return <div className="ai-metric"><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>
}
