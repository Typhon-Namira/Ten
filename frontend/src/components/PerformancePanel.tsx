import type { PerformanceMetrics } from '../types'

function Tile({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className={`perf-tile${warn ? ' perf-tile--warn' : ''}`}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  )
}

const ms = (value: number | null) => (value === null ? '—' : `${value.toFixed(0)} ms`)
const at = (value: string | null) => (value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'never')

export function PerformancePanel({ data }: { data: PerformanceMetrics | null }) {
  if (!data) {
    return <div className="empty-state"><h3>Loading performance metrics…</h3></div>
  }
  const workerHealthy = (worker: Record<string, unknown>) => Boolean(worker.enabled) && Boolean(worker.running) && !worker.last_error
  return (
    <div className="perf-grid">
      <Tile label="Pipeline latency" value={ms(data.pipeline_latency_ms)} />
      <Tile label="Provider latency" value={ms(data.provider.last_latency_ms)} warn={!data.provider.healthy} />
      <Tile label="Last successful provider call" value={at(data.provider.last_success_at)} />
      <Tile label="Last failed provider call" value={at(data.provider.last_failure_at)} warn={Boolean(data.provider.last_error)} />
      <Tile label="Database mode" value={data.database.mode} warn={data.database.mode === 'memory'} />
      <Tile label="Queue length" value={data.queue_length === null ? '—' : String(data.queue_length)} warn={(data.queue_length ?? 0) > 25} />
      <Tile label="Market data worker" value={workerHealthy(data.workers.market_data_worker) ? 'healthy' : 'degraded'} warn={!workerHealthy(data.workers.market_data_worker)} />
      <Tile label="Integration worker" value={workerHealthy(data.workers.integration_worker) ? 'healthy' : 'degraded'} warn={!workerHealthy(data.workers.integration_worker)} />
    </div>
  )
}
