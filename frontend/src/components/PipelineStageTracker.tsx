import { AlertTriangle, CheckCircle2, CircleDashed, Loader2, MinusCircle, XCircle } from 'lucide-react'
import type { PipelineStagesResponse, StageStatus } from '../types'

const ICONS: Record<StageStatus, typeof CheckCircle2> = {
  waiting: CircleDashed,
  running: Loader2,
  success: CheckCircle2,
  degraded: AlertTriangle,
  failed: XCircle,
  skipped: MinusCircle,
}

function StatusIcon({ status }: { status: StageStatus }) {
  const Icon = ICONS[status]
  return <Icon size={16} className={status === 'running' ? 'spin' : ''} />
}

const TERMINAL: StageStatus[] = ['success', 'degraded', 'failed', 'skipped']

/** A connected node-flow instead of a bare status list — every stage belongs to exactly one
 * candle_timestamp (see `stage-meta` above the flow), so this never mixes the current cycle's
 * progress with a previous or next one; the backend's boundary-keyed stage tracker guarantees
 * that identity, this component just visualizes it as a pipeline instead of a table. */
export function PipelineStageTracker({ data }: { data: PipelineStagesResponse | null }) {
  if (!data) {
    return <div className="empty-state"><h3>Connecting to pipeline…</h3><p>Waiting for the first stage report.</p></div>
  }
  if (!data.available || !data.stages) {
    return <div className="empty-state"><h3>No candle processed yet</h3><p>{data.reason ?? 'The pipeline has not run for this instrument/timeframe yet.'}</p></div>
  }
  const completedCount = data.stages.filter((s) => TERMINAL.includes(s.status)).length
  const progressPercent = Math.round((completedCount / data.stages.length) * 100)
  return (
    <div>
      <div className="stage-meta">
        <span>Candle {new Date(data.candle_timestamp!).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
        <span>{data.complete ? 'Cycle complete' : `${progressPercent}% through this cycle`}</span>
      </div>
      <div className="stage-flow">
        {data.stages.map((stage, index) => (
          <div className="stage-flow__node-wrap" key={stage.key}>
            <div className={`stage-flow__node stage-flow__node--${stage.status}`} title={stage.label}>
              <StatusIcon status={stage.status} />
            </div>
            <span className="stage-flow__label">{stage.label}</span>
            {index < data.stages.length - 1 && <div className={`stage-flow__connector ${TERMINAL.includes(stage.status) ? 'stage-flow__connector--filled' : ''}`} />}
          </div>
        ))}
      </div>
    </div>
  )
}
