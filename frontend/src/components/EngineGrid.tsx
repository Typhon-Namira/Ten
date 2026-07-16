import { Activity, Boxes, BrainCircuit, Database, Droplets, Gauge } from 'lucide-react'
import type { EngineStatus } from '../types'

const icons = [Database, Boxes, Droplets, Activity, Gauge, BrainCircuit]

export function EngineGrid({ engines }: { engines: EngineStatus[] }) {
  return <div className="engine-grid">{engines.map((engine, index) => {
    const Icon = icons[index % icons.length]
    return <div className="engine" key={engine.name}>
      <div className="engine__icon"><Icon size={18} /></div>
      <div><strong>{engine.name.replaceAll('_', ' ')}</strong><span>v{engine.version}</span></div>
      <i className={`status-dot status-dot--${engine.state}`} title={engine.details} />
    </div>
  })}</div>
}

