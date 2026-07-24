import { useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import { toApiError } from '../lib/apiError'
import { recordFetchOutcome } from '../lib/diagnosticsFeed'
import type { SystemDiagnostics } from '../types'

const POLL_MS = 20_000

/** Independent of any page-specific hook — mounted once in AppShell so the always-visible
 * diagnostics bar has fresh worker/operational-state data on every route, not just on pages that
 * already happen to fetch /api/v1/system/diagnostics for their own purposes. */
export function useSystemDiagnostics(): SystemDiagnostics | null {
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostics | null>(null)
  const active = useRef(true)

  useEffect(() => {
    active.current = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const value = await tenApi.diagnostics()
        if (!active.current) return
        setDiagnostics(value)
        recordFetchOutcome('diagnostics', { ok: true })
      } catch (error) {
        if (!active.current) return
        recordFetchOutcome('diagnostics', { ok: false, error: toApiError(error) })
      }
      if (active.current) timer = window.setTimeout(() => void poll(), POLL_MS)
    }
    void poll()
    return () => {
      active.current = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  return diagnostics
}
