/**
 * Economic Calendar has a richer state space than the generic engine-state vocabulary
 * (see engineState.ts) can distinguish — the brief explicitly requires every one of these to
 * render as a visually distinct state, never collapsed onto a shared color. Two independent
 * dimensions are involved: is the PROVIDER actually working (connection_state), and where are we
 * relative to news right now (risk_window_phase / context_state) — this module keeps them
 * separate rather than forcing one flattened badge vocabulary onto both concerns.
 */

export type ConnectionState = 'connected' | 'unreachable' | 'timeout' | 'unauthorized' | 'rate_limited' | 'disabled' | 'unknown'
export type CalendarContextState =
  | 'provider_unreachable' | 'provider_timeout' | 'provider_auth_failed' | 'provider_rate_limited'
  | 'no_calendar_data' | 'no_relevant_events' | 'outside_risk_window' | 'inside_risk_window'
export type RiskWindowPhase = 'outside' | 'pre_event' | 'imminent' | 'at_event' | 'post_event' | 'cooldown' | 'overlapping' | 'unknown'

export interface StateVisual {
  label: string
  className: string
}

/** Provider Health badge — Healthy / Limited / Unavailable, one of 7 distinct connection outcomes. */
export function providerConnectionBadge(state: ConnectionState | string): StateVisual {
  const table: Record<ConnectionState, StateVisual> = {
    connected: { label: 'Healthy', className: 'econ-state--healthy' },
    rate_limited: { label: 'Limited', className: 'econ-state--limited' },
    unreachable: { label: 'Unavailable', className: 'econ-state--unavailable' },
    timeout: { label: 'Unavailable (timeout)', className: 'econ-state--timeout' },
    unauthorized: { label: 'Unavailable (auth failed)', className: 'econ-state--auth-failed' },
    disabled: { label: 'Disabled', className: 'econ-state--disabled' },
    unknown: { label: 'Unknown', className: 'econ-state--unknown' },
  }
  return table[state as ConnectionState] ?? { label: 'Unknown', className: 'econ-state--unknown' }
}

/** Dashboard trading-context badge — the state a trader actually cares about right now. Genuine
 * data unavailability always wins over phase (you can't know the risk window if you don't have
 * data); otherwise the risk-window phase drives the label, each with its own distinct color. */
export function tradingContextBadge(contextState: CalendarContextState | string, phase: RiskWindowPhase | string): StateVisual {
  const unavailable: Partial<Record<CalendarContextState, StateVisual>> = {
    provider_unreachable: { label: 'Unavailable', className: 'econ-state--unavailable' },
    provider_timeout: { label: 'Unavailable (timeout)', className: 'econ-state--timeout' },
    provider_auth_failed: { label: 'Unavailable (auth failed)', className: 'econ-state--auth-failed' },
    provider_rate_limited: { label: 'Limited', className: 'econ-state--limited' },
    no_calendar_data: { label: 'Unavailable', className: 'econ-state--unavailable' },
  }
  const fromContext = unavailable[contextState as CalendarContextState]
  if (fromContext) return fromContext

  const byPhase: Record<RiskWindowPhase, StateVisual> = {
    outside: { label: 'Outside Event Window', className: 'econ-state--outside' },
    pre_event: { label: 'Upcoming High Impact Event', className: 'econ-state--upcoming' },
    imminent: { label: 'High Risk Window', className: 'econ-state--high-risk' },
    at_event: { label: 'Live Event', className: 'econ-state--live' },
    post_event: { label: 'High Risk Window', className: 'econ-state--high-risk' },
    overlapping: { label: 'Live Event', className: 'econ-state--live' },
    cooldown: { label: 'Cooldown', className: 'econ-state--cooldown' },
    unknown: { label: 'Outside Event Window', className: 'econ-state--outside' },
  }
  return byPhase[phase as RiskWindowPhase] ?? { label: 'Outside Event Window', className: 'econ-state--outside' }
}
