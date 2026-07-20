import { useEffect, useRef } from 'react'
import type { ActivityEvent } from '../types'
import type { StreamStatus } from '../hooks/useEventStream'

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

export function LiveLogPanel({ status, events }: { status: StreamStatus; events: ActivityEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const node = scrollRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [events])

  return (
    <div className="live-log">
      <div ref={scrollRef} className="live-log__scroll">
        {events.length === 0 ? (
          <p className="live-log__empty">{status === 'open' ? 'Connected — waiting for the next pipeline event.' : 'Connecting to the live event stream…'}</p>
        ) : (
          events.map((event) => (
            <div key={event.id} className="live-log__line">
              <span className="live-log__time">{new Date(event.occurred_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
              <span className="live-log__source">{event.source}</span>
              <span>{humanize(event.type)}{event.count > 1 && <b className="live-log__count"> ×{event.count}</b>}</span>
              {detail(event) && <em>{detail(event)}</em>}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
