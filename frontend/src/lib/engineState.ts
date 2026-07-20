/**
 * One canonical engine-state vocabulary used by every badge in the dashboard.
 *
 * The backend intentionally uses different domain-specific status vocabularies per surface —
 * `PipelineStageTracker` uses waiting/running/success/degraded/failed/skipped (a state machine),
 * `market_intelligence` uses available/unavailable booleans (a snapshot lookup result),
 * `EngineStatus` uses ready/degraded/offline (a registry health check) — each is correct for its
 * own domain and changing them would mean rewriting engine-level status semantics for a UI
 * concern. This module is the single normalization layer: every raw status string from any
 * backend surface maps through `normalizeEngineState()` before it reaches a badge, so the user
 * sees ONE consistent set of labels/colors everywhere instead of nine different vocabularies.
 */

export type CanonicalState = 'healthy' | 'running' | 'waiting' | 'limited' | 'unavailable' | 'disabled' | 'blocked' | 'failed'

export const STATE_LABEL: Record<CanonicalState, string> = {
  healthy: 'Healthy',
  running: 'Running',
  waiting: 'Waiting',
  limited: 'Limited',
  unavailable: 'Unavailable',
  disabled: 'Disabled',
  blocked: 'Blocked',
  failed: 'Failed',
}

// Every raw status string seen anywhere in the API surface maps to exactly one canonical state.
const RAW_TO_CANONICAL: Record<string, CanonicalState> = {
  // stage tracker
  waiting: 'waiting',
  running: 'running',
  success: 'healthy',
  degraded: 'limited',
  failed: 'failed',
  skipped: 'disabled',
  // engine registry (EngineStatus.state)
  ready: 'healthy',
  offline: 'unavailable',
  // market intelligence source_diagnostic status
  ok: 'healthy',
  error: 'failed',
  // economic calendar staged diagnostics
  none_relevant: 'limited',
  empty: 'limited',
  unavailable: 'unavailable',
  // signal decision state
  eligible: 'healthy',
  observe_only: 'limited',
  blocked: 'blocked',
  insufficient_evidence: 'limited',
  expired: 'disabled',
  invalid: 'failed',
  // rejection diagnostics
  passed: 'healthy',
  not_evaluated: 'waiting',
  informational: 'healthy',
  // provider/worker health
  healthy: 'healthy',
}

export function normalizeEngineState(raw: string | null | undefined, opts?: { available?: boolean }): CanonicalState {
  if (raw == null) return opts?.available === false ? 'unavailable' : 'waiting'
  return RAW_TO_CANONICAL[raw.toLowerCase()] ?? 'waiting'
}

export function stateFromAvailability(available: boolean): CanonicalState {
  return available ? 'healthy' : 'unavailable'
}
