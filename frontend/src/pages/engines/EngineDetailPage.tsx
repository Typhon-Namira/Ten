import { type ReactNode, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useActiveSelection } from '../../hooks/useActiveSelection'
import { useEngineDetail, type EngineRecord } from '../../hooks/useEngineDetail'
import { normalizeEngineState } from '../../lib/engineState'
import { StateBadge } from '../../components/StateBadge'
import { AnimatedCounter, BarChart } from '../../components/widgets/Widgets'

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })

function isPlainObject(value: unknown): value is EngineRecord {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

/** Every array field in a snapshot is a population of domain objects (zones, pools, swings,
 * sweeps...) — counting them is exactly "how many objects did this engine produce," which is the
 * one thing every engine's wildly different domain model has in common. */
function objectCounts(record: EngineRecord | null): { label: string; value: number }[] {
  if (!record) return []
  return Object.entries(record)
    .filter((entry): entry is [string, unknown[]] => Array.isArray(entry[1]))
    .map(([label, items]) => ({ label: label.replaceAll('_', ' '), value: items.length }))
    .slice(0, 10)
}

function scalarFields(record: EngineRecord | null): { label: string; value: string }[] {
  if (!record) return []
  return Object.entries(record)
    .filter((entry): entry is [string, string | number | boolean] => ['string', 'number', 'boolean'].includes(typeof entry[1]))
    .map(([label, value]) => ({ label: label.replaceAll('_', ' '), value: typeof value === 'number' ? number.format(value) : String(value) }))
    .slice(0, 12)
}

function findConfidence(record: EngineRecord | null): number | null {
  if (!record) return null
  if (typeof record.confidence_score === 'number') return record.confidence_score <= 1 ? record.confidence_score * 100 : record.confidence_score
  const summary = record.confidence_summary
  if (isPlainObject(summary)) {
    const values = Object.values(summary).filter((v): v is number => typeof v === 'number')
    if (values.length) {
      const avg = values.reduce((a, b) => a + b, 0) / values.length
      return avg <= 1 ? avg * 100 : avg
    }
  }
  return null
}

function RawJson({ record }: { record: EngineRecord | null }) {
  const [open, setOpen] = useState(false)
  if (!record) return <p className="widget-empty">No data yet</p>
  return (
    <div className="raw-json">
      <button className="raw-json__toggle" onClick={() => setOpen((v) => !v)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />} {open ? 'Hide' : 'Show'} raw state (nothing hidden)
      </button>
      {open && <pre className="raw-json__body">{JSON.stringify(record, null, 2)}</pre>}
    </div>
  )
}

export interface EngineDetailPageProps {
  eyebrow: string
  title: string
  description: string
  icon: ReactNode
  basePath: string
  statePath?: string
  extra?: ReactNode
}

/** Reusable "input -> processing -> output" live detail page instantiated for every analysis
 * engine. Shows only telemetry the backend genuinely exposes (health/metrics/state, object
 * counts, confidence) — it does not fabricate synthetic sub-step timings the backend doesn't
 * actually measure. `extra` lets a specific engine's page append bespoke sections below this. */
export function EngineDetailPage({ eyebrow, title, description, icon, basePath, statePath = '/state', extra }: EngineDetailPageProps) {
  const { selection } = useActiveSelection()
  const { health, metrics, state, loaded } = useEngineDetail(basePath, statePath, selection.instrument, selection.timeframe)
  const healthState = normalizeEngineState(typeof health?.status === 'string' ? health.status : null, { available: state !== null })
  const confidence = findConfidence(state)
  const counts = objectCounts(state)
  const healthFields = scalarFields(health)
  const metricFields = scalarFields(metrics)

  return (
    <div className="page module-page engine-page">
      <header>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p className="page-description">{description}</p>
        </div>
        <div className="page-icon">{icon}</div>
      </header>

      <section className="panel engine-page__pipeline">
        <div className="panel__head"><div><p className="eyebrow">LIVE PIPELINE</p><h2>Input → Processing → Output</h2></div><StateBadge state={healthState} /></div>
        <div className="panel-body engine-io-flow">
          <div className="engine-io-flow__stage">
            <span className="engine-io-flow__stage-label">Input</span>
            <b>{selection.instrument} · {selection.timeframe}</b>
            <small>{loaded ? 'live candle feed' : 'connecting…'}</small>
          </div>
          <div className={`engine-io-flow__arrow ${loaded ? 'engine-io-flow__arrow--active' : ''}`} />
          <div className="engine-io-flow__stage">
            <span className="engine-io-flow__stage-label">Processing</span>
            <b><StateBadge state={healthState} /></b>
            <small>{healthFields.find((f) => f.label === 'repository mode' || f.label === 'mode')?.value ?? 'analysis engine'}</small>
          </div>
          <div className={`engine-io-flow__arrow ${state ? 'engine-io-flow__arrow--active' : ''}`} />
          <div className="engine-io-flow__stage">
            <span className="engine-io-flow__stage-label">Output</span>
            <b>{state ? <AnimatedCounter value={counts.reduce((sum, c) => sum + c.value, 0)} /> : '—'} objects</b>
            <small>{confidence === null ? 'confidence unavailable' : `${confidence.toFixed(0)}% confidence`}</small>
          </div>
        </div>
      </section>

      <div className="workspace-grid">
        <div className="workspace-grid__main">
          <section className="panel">
            <div className="panel__head"><div><p className="eyebrow">OUTPUT</p><h2>Object counts</h2></div></div>
            <div className="panel-body">
              {counts.length ? <BarChart data={counts.map((c) => ({ ...c, color: '#c4a359' }))} width={480} height={140} /> : <p className="widget-empty">No live snapshot yet for this instrument/timeframe.</p>}
            </div>
          </section>
          <section className="panel">
            <div className="panel__head"><div><p className="eyebrow">STATE</p><h2>Current snapshot fields</h2></div></div>
            <div className="intel-grid">
              {scalarFields(state).map((f) => <div className="intel-field" key={f.label}><span>{f.label}</span><b>{f.value}</b></div>)}
              {!scalarFields(state).length && <p className="widget-empty" style={{ padding: 16 }}>No scalar fields on the current state.</p>}
            </div>
            <div className="panel-body"><RawJson record={state} /></div>
          </section>
          {extra}
        </div>
        <div className="workspace-grid__rail">
          <section className="panel">
            <div className="panel__head"><div><p className="eyebrow">HEALTH</p><h2>Engine health</h2></div></div>
            <div className="intel-grid">
              {healthFields.map((f) => <div className="intel-field" key={f.label}><span>{f.label}</span><b>{f.value}</b></div>)}
            </div>
          </section>
          <section className="panel">
            <div className="panel__head"><div><p className="eyebrow">METRICS</p><h2>Runtime metrics</h2></div></div>
            <div className="intel-grid">
              {metricFields.map((f) => <div className="intel-field" key={f.label}><span>{f.label}</span><b>{f.value}</b></div>)}
              {!metricFields.length && <p className="widget-empty" style={{ padding: 16 }}>No metrics reported yet.</p>}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
