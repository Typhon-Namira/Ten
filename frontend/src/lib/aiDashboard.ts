import type {
  AIReasoningDashboard,
  DashboardAggregate,
  DashboardStage,
  FinalSystemAction,
  ManagedSignal,
  PublishedAnalyticalSignal,
} from '../types'

export type Tone = 'positive' | 'negative' | 'neutral' | 'warning'
export type PipelineStatus = 'completed' | 'active' | 'waiting' | 'rejected' | 'failed' | 'unavailable' | 'blocked' | 'disabled' | 'running' | 'hold' | 'not_required' | 'not_applicable'

export interface PipelineStep {
  id: string
  label: string
  status: PipelineStatus
  detail: string
}

export function activeSignal(data: AIReasoningDashboard | null): ManagedSignal | null {
  return data?.managed_signals[0] ?? null
}

export function latestFinalAction(data: AIReasoningDashboard | null): FinalSystemAction | null {
  const signal = activeSignal(data)
  if (!signal) return null
  const actions = data?.final_actions[signal.signal_id] ?? []
  return actions.at(-1) ?? null
}

export function activePublication(data: AIReasoningDashboard | null): PublishedAnalyticalSignal | null {
  const signal = activeSignal(data)
  return signal ? data?.publications[signal.signal_id] ?? null : null
}

export function actionTone(action: FinalSystemAction | null): Tone {
  if (!action) return 'neutral'
  if (['rejected', 'invalidated', 'cancelled', 'expired'].includes(action.action)) return 'negative'
  if (['postponed', 'temporarily_blocked', 'monitoring'].includes(action.action)) return 'warning'
  if (action.final_direction === 'SELL' && ['approved', 'approved_with_reduced_risk', 'published'].includes(action.action)) return 'negative'
  if (action.final_direction === 'BUY' && ['approved', 'approved_with_reduced_risk', 'published'].includes(action.action)) return 'positive'
  return 'neutral'
}

export function humanize(value: string | null | undefined): string {
  if (!value) return 'Unavailable'
  return value.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase())
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const minutes = Math.round(seconds / 60)
  return minutes < 60 ? `${minutes}m` : `${(minutes / 60).toFixed(1)}h`
}

/** Turns a backend-authoritative stage's `reason` (plus whatever extra fields it carries, see
 * DashboardStage in types/index.ts) into one human-readable line — this is the only place that
 * text is assembled, so every panel that reads a stage's `detail` shows the same real reason a
 * bare reason-code humanization would otherwise hide (elapsed/retry time, or the exact
 * field/expected/received of a structured-output validation failure). */
export function stageDetail(stage: DashboardStage): string {
  const base = humanize(stage.reason)
  if (stage.status === 'running' && stage.elapsed_seconds != null) return `${base} (${formatDuration(stage.elapsed_seconds)} elapsed)`
  if (stage.status === 'pending' && stage.elapsed_seconds != null) return `${base} — ${humanize(stage.job_state)} for ${formatDuration(stage.elapsed_seconds)}`
  if (stage.status === 'blocked' && stage.retry_in_seconds != null) return `${base} — retry in ${formatDuration(stage.retry_in_seconds)}${stage.last_failure_state ? ` (${humanize(stage.last_failure_state)})` : ''}`
  if (stage.status === 'disabled' && stage.disabled_flags?.length) return `${base}: ${stage.disabled_flags.map(humanize).join(', ')}`
  if (stage.field) return `${base} — Field: ${stage.field}, Expected: ${humanize(stage.expected)}, Received: ${String(stage.received)}`
  if (stage.status === 'disabled' && stage.config_source) return `${base} (${stage.config_source})`
  return base
}

export function decisionHeadline(action: FinalSystemAction | null): string {
  if (!action) return 'NO ACTION'
  if (action.action === 'published') return `${action.final_direction} · PUBLISHED`
  if (action.action === 'approved') return `${action.final_direction} · APPROVED`
  if (action.action === 'approved_with_reduced_risk') return `${action.final_direction} · MODIFIED`
  if (action.action === 'rejected') return 'REJECTED'
  if (action.action === 'expired') return 'EXPIRED'
  if (action.action === 'monitoring') return 'MONITORING'
  if (action.action === 'no_action') return 'NO ACTION'
  if (action.action === 'postponed' || action.action === 'temporarily_blocked') return 'HOLD'
  return humanize(action.action).toUpperCase()
}

export function decisionExplanation(action: FinalSystemAction | null): string {
  if (!action) return 'No final system action is available. TEN is waiting for a complete, validated analysis cycle.'
  const blocker = action.gate_evaluations.find(item => item.status === 'failed' || item.status === 'unavailable')
  if (blocker) return `${humanize(blocker.gate_id)}: ${humanize(blocker.reason_codes[0] ?? blocker.status)}.`
  if (action.modifications.length) return action.modifications[0].exact_reason
  if (action.action === 'published') return 'The proposal passed deterministic safety checks and was published as an analytical signal.'
  return `The proposal completed deterministic review with status ${humanize(action.approval_state).toLowerCase()}.`
}

export function pipelineSteps(
  data: AIReasoningDashboard | null,
  marketAvailable: boolean,
  quantStatus: string | null,
  authoritative?: DashboardAggregate['stages'],
): PipelineStep[] {
  if (authoritative) {
    const mapping: Array<[keyof DashboardAggregate['stages'], string, string]> = [
      ['market_state', 'market', 'Market State'],
      ['quant_forecast', 'quant', 'Quant Forecast'],
      ['ai_reasoning', 'reasoning', 'AI Reasoning'],
      ['ai_proposal', 'proposal', 'AI Proposal'],
      ['guardrails', 'guardrails', 'Guardrails'],
      ['final_action', 'final', 'Final Action'],
      ['publication', 'publication', 'Publication'],
      ['monitoring', 'monitoring', 'Monitoring'],
      ['outcome', 'outcome', 'Outcome'],
    ]
    return mapping.map(([key, id, label]) => {
      const stage = authoritative[key]
      const status: PipelineStatus =
        stage.status === 'available' || stage.status === 'complete' ? (key === 'monitoring' ? 'active' : 'completed')
        : stage.status === 'failed' ? 'failed'
        : stage.status === 'degraded' || stage.status === 'partial' ? 'unavailable'
        // These pass through as their own distinct PipelineStatus values instead of collapsing
        // into "waiting" — that collapse is exactly what made a real, already-resolved outcome
        // (a provider-backoff block, a legitimate non-actionable HOLD conclusion, a disabled
        // stage) indistinguishable from "hasn't started yet".
        : stage.status === 'blocked' ? 'blocked'
        : stage.status === 'disabled' ? 'disabled'
        : stage.status === 'running' ? 'running'
        : stage.status === 'hold' ? 'hold'
        : stage.status === 'not_required' ? 'not_required'
        : stage.status === 'not_applicable' ? 'not_applicable'
        : 'waiting'
      return { id, label, status, detail: stageDetail(stage) }
    })
  }
  const forecast = data?.forecast
  const proposal = data?.proposal
  const action = latestFinalAction(data)
  const publication = activePublication(data)
  const signal = activeSignal(data)
  const history = signal ? data?.signal_histories[signal.signal_id] : null
  const rejected = action?.approval_state === 'rejected' || action?.approval_state === 'blocked'
  const failed = forecast?.status === 'failed' || forecast?.status === 'invalid'
  return [
    { id: 'market', label: 'Market State', status: marketAvailable ? 'completed' : 'unavailable', detail: marketAvailable ? 'M5 · M15' : 'Unavailable' },
    { id: 'quant', label: 'Quant Forecast', status: quantStatus === 'available' ? 'completed' : quantStatus ? 'unavailable' : 'waiting', detail: quantStatus ? humanize(quantStatus) : 'Awaiting forecast' },
    { id: 'reasoning', label: 'AI Reasoning', status: failed ? 'failed' : forecast ? 'completed' : 'waiting', detail: forecast ? humanize(forecast.status) : 'Not evaluated' },
    { id: 'proposal', label: 'AI Proposal', status: proposal ? 'completed' : failed ? 'unavailable' : 'waiting', detail: proposal ? humanize(proposal.recommended_action) : 'No proposal' },
    { id: 'guardrails', label: 'Guardrails', status: rejected ? 'rejected' : action ? 'completed' : 'waiting', detail: action ? humanize(action.approval_state) : 'Not evaluated' },
    { id: 'final', label: 'Final Action', status: action ? (rejected ? 'rejected' : 'completed') : 'waiting', detail: action ? humanize(action.action) : 'No action' },
    { id: 'publication', label: 'Publication', status: publication ? 'completed' : action?.publication_state === 'failed' ? 'failed' : 'waiting', detail: publication ? 'Analytical signal' : humanize(action?.publication_state) },
    { id: 'monitoring', label: 'Monitoring', status: signal ? 'active' : 'waiting', detail: signal ? humanize(signal.state) : 'No active signal' },
    { id: 'outcome', label: 'Outcome', status: history?.outcomes.length ? 'completed' : 'waiting', detail: history?.outcomes.length ? 'Evaluated' : 'Pending' },
  ]
}

export function numberValue(record: Record<string, unknown> | null, key: string): number | null {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
