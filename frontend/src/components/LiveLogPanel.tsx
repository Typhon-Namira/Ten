import { useEffect, useMemo, useRef, useState } from 'react'
import { Search } from 'lucide-react'
import type { ActivityEvent } from '../types'
import type { StreamStatus } from '../hooks/useEventStream'
import { useChartFocus } from '../lib/ChartFocusContext'

function humanize(type: string): string {
  return type.replace(/([a-z])([A-Z])/g, '$1 $2')
}

function detail(event: ActivityEvent): string {
  const payload = event.payload ?? {}
  if (typeof payload.reason_code === 'string') return `Rejected: ${payload.reason_code.replaceAll('_', ' ')}`
  if (typeof payload.state === 'string') return `state = ${payload.state}`
  if (typeof payload.confidence_score === 'number') return `confidence = ${payload.confidence_score.toFixed(0)}%`
  if (typeof payload.score === 'number') return `score = ${payload.score.toFixed(2)}`
  if (typeof payload.status === 'string') return `status = ${payload.status}`
  if (typeof payload.symbol === 'string' && typeof payload.close === 'number') return `${payload.symbol} @ ${payload.close}`
  return ''
}

type Severity = 'error' | 'warning' | 'info'

const ERROR_MARKERS = ['failed', 'blocked', 'invalid', 'error', 'rejected']
const WARNING_MARKERS = ['degraded', 'invalidated', 'skipped', 'stale', 'sweep', 'grab', 'raid', 'hunt']

function severityOf(type: string): Severity {
  const lower = type.toLowerCase()
  if (ERROR_MARKERS.some((m) => lower.includes(m))) return 'error'
  if (WARNING_MARKERS.some((m) => lower.includes(m))) return 'warning'
  return 'info'
}

function LogRow({ event, expanded, onToggle }: { event: ActivityEvent; expanded: boolean; onToggle: () => void }) {
  const { focus } = useChartFocus()
  const severity = severityOf(event.type)
  const time = new Date(event.occurred_at)
  return (
    <div className={`live-log__row live-log__row--${severity}`}>
      <button className="live-log__line" onClick={() => { onToggle(); focus(Math.floor(time.getTime() / 1000), event.correlation_id) }} title="Click to expand and focus this moment on the chart">
        <span className="live-log__time">{time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
        <span className="live-log__source">{event.source}</span>
        <span>{humanize(event.type)}{event.count > 1 && <b className="live-log__count"> ×{event.count}</b>}</span>
        {detail(event) && <em>{detail(event)}</em>}
      </button>
      {expanded && (
        <div className="live-log__expanded">
          <div className="live-log__expanded-meta">
            <span>correlation <b>{event.correlation_id}</b></span>
            <span>event id <b>{event.id}</b></span>
          </div>
          <pre>{JSON.stringify(event.payload, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

/** Filterable, searchable, expandable live log — clicking any line focuses that moment on the
 * chart (see ChartFocusContext) instead of leaving the log as an isolated, disconnected panel. */
export function LiveLogPanel({ status, events }: { status: StreamStatus; events: ActivityEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [engineFilter, setEngineFilter] = useState<string>('all')
  const [severityFilter, setSeverityFilter] = useState<'all' | Severity>('all')
  const [search, setSearch] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const engines = useMemo(() => Array.from(new Set(events.map((e) => e.source))).sort(), [events])
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase()
    return events.filter((event) => {
      if (engineFilter !== 'all' && event.source !== engineFilter) return false
      if (severityFilter !== 'all' && severityOf(event.type) !== severityFilter) return false
      if (query && !`${event.type} ${event.source} ${event.correlation_id}`.toLowerCase().includes(query)) return false
      return true
    })
  }, [events, engineFilter, severityFilter, search])

  useEffect(() => {
    const node = scrollRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [filtered])

  return (
    <div className="live-log">
      <div className="live-log__toolbar">
        <div className="live-log__search">
          <Search size={12} />
          <input placeholder="Search type, engine, correlation id…" value={search} onChange={(event) => setSearch(event.target.value)} />
        </div>
        <select value={engineFilter} onChange={(event) => setEngineFilter(event.target.value)}>
          <option value="all">All engines</option>
          {engines.map((engine) => <option value={engine} key={engine}>{engine}</option>)}
        </select>
        <select value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value as 'all' | Severity)}>
          <option value="all">All severities</option>
          <option value="error">Errors</option>
          <option value="warning">Warnings</option>
          <option value="info">Info</option>
        </select>
      </div>
      <div ref={scrollRef} className="live-log__scroll">
        {filtered.length === 0 ? (
          <p className="live-log__empty">{events.length === 0 ? (status === 'open' ? 'Connected — waiting for the next pipeline event.' : 'Connecting to the live event stream…') : 'No events match the current filter.'}</p>
        ) : (
          filtered.map((event) => <LogRow event={event} key={event.id} expanded={expandedId === event.id} onToggle={() => setExpandedId((current) => (current === event.id ? null : event.id))} />)
        )}
      </div>
    </div>
  )
}
