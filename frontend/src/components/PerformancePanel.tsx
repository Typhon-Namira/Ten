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
const seconds = (value: number | null) => (value === null ? '—' : `${value.toFixed(1)}s`)

export function PerformancePanel({ data }: { data: PerformanceMetrics | null }) {
  if (!data) {
    return <div className="empty-state"><h3>Loading performance metrics…</h3></div>
  }
  const workerHealthy = (worker: Record<string, unknown>) => Boolean(worker.enabled) && Boolean(worker.running) && !worker.last_error
  // `pipeline_latency_ms` (last completed cycle) and `pipeline_in_flight_ms` (current cycle,
  // still running) are two different measurements — showing whichever one is actually available
  // instead of only the former is what keeps this tile from reading "—" while the pipeline is busy.
  const pipelineLatencyLabel = data.pipeline_latency_ms !== null ? 'Pipeline latency' : data.pipeline_in_flight_ms !== null ? 'Pipeline latency (in flight)' : 'Pipeline latency'
  const pipelineLatencyValue = ms(data.pipeline_latency_ms ?? data.pipeline_in_flight_ms)
  return (
    <div className="perf-grid">
      <Tile label={pipelineLatencyLabel} value={pipelineLatencyValue} />
      <Tile label="Queue backlog age" value={seconds(data.queue_oldest_pending_age_seconds)} warn={(data.queue_oldest_pending_age_seconds ?? 0) > 60} />
      <Tile label="Provider latency" value={ms(data.provider.last_latency_ms)} warn={!data.provider.healthy} />
      <Tile label="Provider rate limit remaining" value={data.provider.provider_rate_limit_remaining === null ? '—' : String(data.provider.provider_rate_limit_remaining)} warn={(data.provider.provider_rate_limit_remaining ?? 99) === 0} />
      <Tile label="Provider backoff until" value={at(data.provider.provider_backoff_until)} warn={Boolean(data.provider.provider_backoff_until)} />
      <Tile label="Last successful provider call" value={at(data.provider.last_success_at)} />
      <Tile label="Last failed provider call" value={at(data.provider.last_failure_at)} warn={Boolean(data.provider.last_error)} />
      <Tile label="Database mode" value={data.database.mode} warn={data.database.mode === 'memory'} />
      <Tile label="Last database update" value={at(data.database.last_database_update)} />
      <Tile label="Last cache update" value={at(data.cache.last_cache_update)} />
      <Tile label="Queue length" value={data.queue_length === null ? '—' : String(data.queue_length)} warn={(data.queue_length ?? 0) > 25} />
      <Tile label="Market data worker" value={workerHealthy(data.workers.market_data_worker) ? 'healthy' : 'degraded'} warn={!workerHealthy(data.workers.market_data_worker)} />
      <Tile label="Integration worker" value={workerHealthy(data.workers.integration_worker) ? 'healthy' : 'degraded'} warn={!workerHealthy(data.workers.integration_worker)} />
    </div>
  )
}
