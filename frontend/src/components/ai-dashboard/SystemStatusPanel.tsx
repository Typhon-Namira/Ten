import { AlertTriangle, Database, GitCommitHorizontal, ServerCog } from 'lucide-react'
import type { DashboardSystemStatus, SystemStageStatus } from '../../types'
import { humanize } from '../../lib/aiDashboard'
import { EmptyState, SectionHeader, StatusBadge } from './Primitives'

const tones: Record<SystemStageStatus, 'positive' | 'warning' | 'negative' | 'neutral'> = {
  healthy: 'positive',
  running: 'warning',
  degraded: 'warning',
  failed: 'negative',
  disabled: 'neutral',
  blocked: 'neutral',
  stale: 'warning',
  no_data: 'neutral',
}

function bytes(value: number | null): string {
  if (value == null) return 'Not measured'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${amount.toFixed(unit < 2 ? 0 : 2)} ${units[unit]}`
}

export function SystemStatusPanel({ data }: { data: DashboardSystemStatus | null }) {
  if (!data) {
    return <section className="ai-card ai-card--wide">
      <EmptyState title="System status not yet reported" detail="Awaiting the backend-authoritative status endpoint." />
    </section>
  }
  return <>
    <section className="ai-card ai-card--wide system-map">
      <SectionHeader eyebrow="Backend authoritative" title="Pipeline timeline" action={<GitCommitHorizontal size={19} />} />
      <div className="system-map__stages">
        {data.stages.map((stage, index) => <div className={`system-stage system-stage--${stage.status}`} key={stage.id}>
          <span className="system-stage__number">{index + 1}</span>
          <div>
            <strong>{stage.label}</strong>
            <small>{humanize(stage.reason)}</small>
          </div>
          <StatusBadge tone={tones[stage.status]}>{humanize(stage.status)}</StatusBadge>
        </div>)}
      </div>
    </section>
    <section className="ai-card ai-card--wide">
      <SectionHeader eyebrow="Measured PostgreSQL footprint" title="Storage diagnostics" action={<Database size={19} />} />
      <div className="storage-overview">
        <div><span>Database size</span><strong>{bytes(data.storage.database_bytes)}</strong><small>{humanize(data.storage.reason)}</small></div>
        <div><span>Growth rate</span><strong>{data.storage.growth_bytes_per_hour == null ? 'Collecting baseline' : `${bytes(data.storage.growth_bytes_per_hour)}/hour`}</strong><small>{data.storage.projected_gb_per_day == null ? 'Second sample required' : `${data.storage.projected_gb_per_day.toFixed(2)} GB/day projected`}</small></div>
        <div><span>Retention status</span><StatusBadge tone={tones[data.storage.retention.status]}>{humanize(data.storage.retention.status)}</StatusBadge><small>{data.storage.retention.policies.length} bounded policies</small></div>
        <div><span>Storage health</span><StatusBadge tone={tones[data.storage.status]}>{humanize(data.storage.status)}</StatusBadge><small>{data.storage.circuit_retry_at ? `Retry after ${new Date(data.storage.circuit_retry_at).toLocaleTimeString()}` : 'Circuit closed'}</small></div>
      </div>
      <div className="storage-relations">
        {data.storage.largest_relations.slice(0, 8).map(relation => <div key={relation.relname}>
          <span><ServerCog size={14} />{humanize(relation.relname)}</span>
          <strong>{bytes(relation.total_bytes)}</strong>
          <small>{relation.n_live_tup.toLocaleString()} rows · {relation.n_dead_tup.toLocaleString()} dead</small>
        </div>)}
      </div>
    </section>
    <section className="ai-card ai-card--wide">
      <SectionHeader eyebrow="Actionable backend reasons" title="Failure and blocking history" action={<AlertTriangle size={19} />} />
      {data.failure_history.length === 0
        ? <EmptyState title="No active failures" detail="All enabled stages report a healthy terminal state." />
        : <div className="failure-list">{data.failure_history.map((item, index) => <div key={`${item.stage}-${index}`}>
          <StatusBadge tone={tones[item.status]}>{humanize(item.status)}</StatusBadge>
          <span><strong>{humanize(item.stage)}</strong><small>{humanize(item.reason)}</small></span>
          <time>{item.timestamp ? new Date(item.timestamp).toLocaleString() : 'No event timestamp'}</time>
        </div>)}</div>}
    </section>
  </>
}
