import type { StateVisual } from '../lib/economicState'

export function EconomicStateBadge({ visual, detail }: { visual: StateVisual; detail?: string }) {
  return (
    <span className={`econ-state ${visual.className}`} title={detail}>
      <i />{visual.label}
    </span>
  )
}
