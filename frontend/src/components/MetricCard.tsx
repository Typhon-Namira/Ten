import type { ReactNode } from 'react'

interface MetricCardProps {
  label: string
  value: string
  detail: string
  icon: ReactNode
  accent?: 'gold' | 'green' | 'red'
}

export function MetricCard({ label, value, detail, icon, accent = 'gold' }: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${accent}`}>
      <div className="metric-card__head"><span>{label}</span>{icon}</div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

