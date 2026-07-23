import { useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { AIReasoningDashboard } from '../types'

export function useAIReasoning(instrument: string): AIReasoningDashboard | null {
  const [value, setValue] = useState<AIReasoningDashboard | null>(null)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const next = await tenApi.latestAIReasoning(instrument)
        if (!cancelled) setValue(next)
      } catch {
        // Preserve the last server-computed AI state during an observability outage.
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 15_000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [instrument])

  return value
}
