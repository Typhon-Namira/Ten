import type {
  AIReasoningDashboard,
  FinalSystemAction,
  ManagedSignal,
  PublishedAnalyticalSignal,
} from '../types'

export type Tone = 'positive' | 'negative' | 'neutral' | 'warning'
export type PipelineStatus = 'completed' | 'active' | 'waiting' | 'rejected' | 'failed' | 'unavailable'

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

export function decisionHeadline(action: FinalSystemAction | null): string {
  if (!action) return 'NO ACTION'
  if (action.action === 'published') return `${action.final_direction} · PUBLISHED`
  if (action.action === 'approved') return `${action.final_direction} · APPROVED`
  if (action.action === 'approved_with_reduced_risk') return `${action.final_direction} · MODIFIED`
  if (action.action === 'rejected') return 'REJECTED'
  if (action.action === 'expired') return 'EXPIRED'
  if (action.action === 'monitoring') return 'MONITORING'
  if (action.action === 'no_action') return 'NO ACTION'
  if (action.action === 'postponed' || action.action === 'temporarily_blocked') return 'WAIT'
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
): PipelineStep[] {
  const forecast = data?.forecast
  const proposal = data?.proposal
  const action = latestFinalAction(data)
  const publication = activePublication(data)
  const signal = activeSignal(data)
  const history = signal ? data?.signal_histories[signal.signal_id] : null
  const rejected = action?.approval_state === 'rejected' || action?.approval_state === 'blocked'
  const failed = forecast?.status === 'failed' || forecast?.status === 'invalid'
  return [
    { id: 'market', label: 'Market State', status: marketAvailable ? 'completed' : 'unavailable', detail: marketAvailable ? 'M1 · M5 · M15' : 'Unavailable' },
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
