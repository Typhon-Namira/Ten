import { useEffect, useRef, useState } from 'react'
import { STREAM_URL } from '../services/api'
import type { ActivityEvent } from '../types'

export type StreamStatus = 'connecting' | 'open' | 'closed'

const MAX_ENTRIES = 200

export function useEventStream() {
  const [status, setStatus] = useState<StreamStatus>(() =>
    typeof window !== 'undefined' && typeof window.EventSource !== 'undefined' ? 'connecting' : 'closed',
  )
  const [events, setEvents] = useState<ActivityEvent[]>([])
  const seen = useRef<Set<string>>(new Set())

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.EventSource === 'undefined') return
    const source = new EventSource(STREAM_URL)
    source.onopen = () => setStatus('open')
    source.onerror = () => setStatus(source.readyState === EventSource.CONNECTING ? 'connecting' : 'closed')
    source.onmessage = (message) => {
      try {
        const entry = JSON.parse(message.data) as ActivityEvent
        if (seen.current.has(entry.id)) return
        seen.current.add(entry.id)
        if (seen.current.size > MAX_ENTRIES * 4) {
          seen.current = new Set(Array.from(seen.current).slice(-MAX_ENTRIES))
        }
        setEvents((previous) => [...previous.slice(-(MAX_ENTRIES - 1)), entry])
      } catch {
        // ignore malformed frames (e.g. comment/heartbeat lines never reach onmessage)
      }
    }
    return () => source.close()
  }, [])

  return { status, events }
}
