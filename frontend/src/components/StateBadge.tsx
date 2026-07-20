import { STATE_LABEL, type CanonicalState } from '../lib/engineState'

export function StateBadge({ state, detail }: { state: CanonicalState; detail?: string }) {
  return (
    <span className={`state-badge state-badge--${state}`} title={detail}>
      <i />{STATE_LABEL[state]}
    </span>
  )
}
