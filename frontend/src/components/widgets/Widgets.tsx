/**
 * Hand-rolled SVG visualization primitives — no charting dependency, so they render with zero
 * npm install. Used throughout the dashboard to replace raw text ("Confidence: 74%") with
 * professional visual widgets (a gauge needle, a trend line, a radar shape).
 */

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 180) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function arcPath(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, startAngle)
  const end = polarToCartesian(cx, cy, r, endAngle)
  const largeArc = endAngle - startAngle <= 180 ? 0 : 1
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`
}

const GAUGE_ZONES = [
  { upTo: 0.4, color: '#c26965' },
  { upTo: 0.7, color: '#c4a359' },
  { upTo: 1, color: '#59b993' },
]

/** Semicircle gauge for a 0-100 metric (confidence, risk). */
export function Gauge({ value, label, max = 100, invert = false }: { value: number | null; label: string; max?: number; invert?: boolean }) {
  const cx = 60
  const cy = 58
  const r = 46
  const fraction = value === null ? 0 : Math.max(0, Math.min(1, value / max))
  const displayFraction = invert ? 1 - fraction : fraction
  const zone = GAUGE_ZONES.find((z) => displayFraction <= z.upTo) ?? GAUGE_ZONES[GAUGE_ZONES.length - 1]
  return (
    <div className="gauge">
      <svg width={120} height={70} viewBox="0 0 120 70">
        <path d={arcPath(cx, cy, r, 0, 180)} fill="none" stroke="#20252c" strokeWidth={10} strokeLinecap="round" />
        {value !== null && <path d={arcPath(cx, cy, r, 0, fraction * 180)} fill="none" stroke={zone.color} strokeWidth={10} strokeLinecap="round" />}
      </svg>
      <b className="gauge__value">{value === null ? '—' : `${Math.round(value)}${max === 100 ? '%' : ''}`}</b>
      <span className="gauge__label">{label}</span>
    </div>
  )
}

/** Full-circle progress ring for a 0-100 completion metric (scenario readiness). */
export function RadialProgress({ value, label, size = 92 }: { value: number | null; label: string; size?: number }) {
  const r = size / 2 - 8
  const circumference = 2 * Math.PI * r
  const pct = value === null ? 0 : Math.max(0, Math.min(1, value / 100))
  const offset = circumference * (1 - pct)
  return (
    <div className="radial-progress">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#20252c" strokeWidth={7} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#c4a359"
          strokeWidth={7}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset .6s ease' }}
        />
        <text x="50%" y="48%" textAnchor="middle" dominantBaseline="central" className="radial-progress__value">{value === null ? '—' : `${Math.round(value)}%`}</text>
      </svg>
      <span className="radial-progress__label">{label}</span>
    </div>
  )
}

/** Inline trend line for a series of recent values (AI confidence history, etc). */
export function Sparkline({ values, width = 200, height = 44 }: { values: number[]; width?: number; height?: number }) {
  if (values.length < 2) {
    return <div className="sparkline sparkline--empty" style={{ width, height }}>not enough history</div>
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = values.map((v, i) => `${(i / (values.length - 1)) * width},${height - 4 - ((v - min) / range) * (height - 8)}`).join(' ')
  const last = values[values.length - 1]
  const lastPoint = points.split(' ').at(-1)
  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline points={points} fill="none" stroke="#c4a359" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
      {lastPoint && <circle cx={lastPoint.split(',')[0]} cy={lastPoint.split(',')[1]} r={2.5} fill="#c4a359" />}
      <title>{`latest: ${last.toFixed(1)}`}</title>
    </svg>
  )
}

/** Small vertical bar chart (signal state distribution, per-engine quality contribution). */
export function BarChart({ data, width = 240, height = 96 }: { data: { label: string; value: number; color?: string }[]; width?: number; height?: number }) {
  if (!data.length) return null
  const max = Math.max(...data.map((d) => d.value), 1)
  const gap = 6
  const barWidth = data.length ? width / data.length - gap : 0
  return (
    <svg className="bar-chart" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {data.map((d, i) => {
        const h = Math.max(1, (d.value / max) * (height - 18))
        const x = i * (barWidth + gap)
        return (
          <g key={d.label}>
            <rect x={x} y={height - 16 - h} width={barWidth} height={h} fill={d.color ?? '#c4a359'} rx={2} />
            <text x={x + barWidth / 2} y={height - 4} textAnchor="middle" fontSize={8} fill="#737c86">{d.label}</text>
          </g>
        )
      })}
    </svg>
  )
}

/** Horizontal intensity strip (liquidity distribution / institutional flow histogram). */
export function HeatmapStrip({ cells }: { cells: { value: number; label: string }[] }) {
  if (!cells.length) return <p className="widget-empty">No data yet</p>
  const max = Math.max(...cells.map((c) => Math.abs(c.value)), 1)
  return (
    <div className="heatmap-strip">
      {cells.map((c, i) => {
        const intensity = Math.min(1, Math.abs(c.value) / max)
        const color = c.value >= 0 ? `rgba(89,185,147,${0.15 + intensity * 0.75})` : `rgba(194,105,101,${0.15 + intensity * 0.75})`
        return <div key={i} className="heatmap-strip__cell" style={{ background: color }} title={`${c.label}: ${c.value.toFixed(2)}`} />
      })}
    </div>
  )
}

/** Radar/spider chart across N quality dimensions (signal quality composite). */
export function RadarChart({ axes, size = 160 }: { axes: { label: string; value: number }[]; size?: number }) {
  if (axes.length < 3) return null
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 26
  const n = axes.length
  const angleFor = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2
  const pointAt = (i: number, fraction: number) => {
    const angle = angleFor(i)
    return `${cx + r * fraction * Math.cos(angle)},${cy + r * fraction * Math.sin(angle)}`
  }
  const shape = axes.map((a, i) => pointAt(i, Math.max(0, Math.min(1, a.value / 100)))).join(' ')
  return (
    <svg className="radar-chart" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {[0.25, 0.5, 0.75, 1].map((ring) => (
        <polygon key={ring} points={axes.map((_, i) => pointAt(i, ring)).join(' ')} fill="none" stroke="#20252c" strokeWidth={1} />
      ))}
      {axes.map((a, i) => {
        const end = pointAt(i, 1).split(',')
        return <line key={a.label} x1={cx} y1={cy} x2={end[0]} y2={end[1]} stroke="#20252c" strokeWidth={1} />
      })}
      <polygon points={shape} fill="rgba(196,163,89,.25)" stroke="#c4a359" strokeWidth={1.5} />
      {axes.map((a, i) => {
        const angle = angleFor(i)
        const lx = cx + (r + 14) * Math.cos(angle)
        const ly = cy + (r + 14) * Math.sin(angle)
        return <text key={a.label} x={lx} y={ly} textAnchor="middle" dominantBaseline="central" fontSize={8} fill="#737c86">{a.label}</text>
      })}
    </svg>
  )
}

/** Colored horizontal timeline of discrete windows (economic risk windows, regime history). */
export function Timeline({ segments, now }: { segments: { label: string; color: string; start: number; end: number }[]; now?: number }) {
  if (!segments.length) return <p className="widget-empty">No timeline data</p>
  const min = Math.min(...segments.map((s) => s.start))
  const max = Math.max(...segments.map((s) => s.end))
  const span = max - min || 1
  return (
    <div className="timeline">
      {segments.map((s, i) => (
        <div key={i} className="timeline__segment" style={{ left: `${((s.start - min) / span) * 100}%`, width: `${Math.max(0.6, ((s.end - s.start) / span) * 100)}%`, background: s.color }} title={s.label} />
      ))}
      {now !== undefined && now >= min && now <= max && <div className="timeline__now" style={{ left: `${((now - min) / span) * 100}%` }} />}
    </div>
  )
}

/** Directional bias arrow (bullish/bearish/neutral), used wherever a text bias label used to be. */
export function BiasArrow({ direction }: { direction: string | null }) {
  const normalized = (direction ?? '').toLowerCase()
  const rotation = normalized.includes('bull') ? -45 : normalized.includes('bear') ? 45 : 90
  const color = normalized.includes('bull') ? '#59b993' : normalized.includes('bear') ? '#c26965' : '#737c86'
  return (
    <span className="bias-arrow" style={{ color }}>
      <svg width={16} height={16} viewBox="0 0 16 16" style={{ transform: `rotate(${rotation}deg)` }}>
        <path d="M8 1 L14 14 L8 10.5 L2 14 Z" fill="currentColor" />
      </svg>
      {direction ? direction.replaceAll('_', ' ') : 'neutral'}
    </span>
  )
}
