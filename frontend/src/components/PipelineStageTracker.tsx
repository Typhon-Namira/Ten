import { CheckCircle2, CircleDashed, Loader2, MinusCircle, XCircle } from 'lucide-react'
import type { PipelineStagesResponse, StageStatus } from '../types'

const ICONS: Record<StageStatus, typeof CheckCircle2> = {
  waiting: CircleDashed,
  running: Loader2,
  success: CheckCircle2,
  failed: XCircle,
  skipped: MinusCircle,
}

function StatusIcon({ status }: { status: StageStatus }) {
  const Icon = ICONS[status]
  return <Icon size={15} className={status === 'running' ? 'spin' : ''} />
}

export function PipelineStageTracker({ data }: { data: PipelineStagesResponse | null }) {
  if (!data) {
    return <div className="empty-state"><h3>Connecting to pipeline…</h3><p>Waiting for the first stage report.</p></div>
  }
  if (!data.available || !data.stages) {
    return <div className="empty-state"><h3>No candle processed yet</h3><p>{data.reason ?? 'The pipeline has not run for this instrument/timeframe yet.'}</p></div>
  }
  return (
    <div>
      <div className="stage-meta">
        <span>Candle {new Date(data.candle_timestamp!).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
        <span>{data.complete ? 'Cycle complete' : 'In progress'}</span>
      </div>
      <ol className="stage-list">
        {data.stages.map((stage) => (
          <li key={stage.key} className={`stage-list__item stage-list__item--${stage.status}`}>
            <StatusIcon status={stage.status} />
            <span>{stage.label}</span>
            <b>{stage.status}</b>
          </li>
        ))}
      </ol>
    </div>
  )
}
