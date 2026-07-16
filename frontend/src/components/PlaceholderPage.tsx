import type { ReactNode } from 'react'
import { Construction, GitBranch, Layers3 } from 'lucide-react'

interface PlaceholderPageProps {
  eyebrow: string
  title: string
  description: string
  icon: ReactNode
  capabilities: string[]
}

export function PlaceholderPage({ eyebrow, title, description, icon, capabilities }: PlaceholderPageProps) {
  return <div className="page module-page">
    <header><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="page-description">{description}</p></div><div className="page-icon">{icon}</div></header>
    <section className="module-grid">
      <article className="panel module-hero"><Construction size={20} /><div><p className="eyebrow">ARCHITECTURE READY</p><h2>Module boundary established</h2><p>This workspace is intentionally waiting for its versioned engine implementation. Contracts, routing, feature flags, and pipeline integration points are ready.</p></div></article>
      <article className="panel module-card"><GitBranch size={19} /><h3>Extension contract</h3><p>Future versions register through the engine or plugin loader and can coexist through semantic version selection.</p></article>
      <article className="panel module-card"><Layers3 size={19} /><h3>Planned capabilities</h3><ul>{capabilities.map((capability) => <li key={capability}>{capability}</li>)}</ul></article>
    </section>
  </div>
}
