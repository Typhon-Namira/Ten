import { GitBranch, CheckCircle2, XCircle, AlertTriangle, MinusCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useActiveSelection } from '../../hooks/useActiveSelection'
import { fetchSafe } from '../../services/api'
import { StateBadge } from '../../components/StateBadge'
import { Gauge } from '../../components/widgets/Widgets'

const POLL_MS = 5_000

type RuleOutcome = 'passed' | 'failed' | 'warning' | 'not_applicable' | 'not_evaluated'

interface RuleEvaluation {
  rule_id: string
  category: string
  severity: string
  outcome: RuleOutcome
  observed_value: unknown
  threshold: unknown
  reason_code: string
}

interface SignalDecision {
  decision_id: string
  instrument: string
  timeframe: string
  direction: string
  state: string
  confidence_score: number
  market_risk_score: number
  eligibility_score: number
  as_of: string
  rules: RuleEvaluation[]
  blockers: { reason_code: string }[]
  warnings: { reason_code: string }[]
}

const OUTCOME_ICON: Record<RuleOutcome, typeof CheckCircle2> = {
  passed: CheckCircle2,
  failed: XCircle,
  warning: AlertTriangle,
  not_applicable: MinusCircle,
  not_evaluated: MinusCircle,
}

const CATEGORY_ORDER = ['confidence', 'directional_strength', 'risk', 'market_regime', 'economic_event', 'alignment', 'conflict', 'dependency_health', 'freshness', 'data_quality', 'source_integrity', 'duplicate', 'cooldown', 'reversal', 'temporal_validity']

function formatScalar(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2)
  return String(value)
}

function CategoryBranch({ category, rules }: { category: string; rules: RuleEvaluation[] }) {
  const [open, setOpen] = useState(rules.some((r) => r.outcome === 'failed'))
  const worstOutcome: RuleOutcome = rules.some((r) => r.outcome === 'failed') ? 'failed' : rules.some((r) => r.outcome === 'warning') ? 'warning' : rules.every((r) => r.outcome === 'not_evaluated' || r.outcome === 'not_applicable') ? 'not_evaluated' : 'passed'
  const Icon = OUTCOME_ICON[worstOutcome]
  return (
    <div className={`decision-tree__branch decision-tree__branch--${worstOutcome}`}>
      <button className="decision-tree__branch-head" onClick={() => setOpen((v) => !v)}>
        <Icon size={15} />
        <span>{category.replaceAll('_', ' ')}</span>
        <small>{rules.length} rule{rules.length === 1 ? '' : 's'}</small>
      </button>
      {open && (
        <ul className="decision-tree__rules">
          {rules.map((rule) => {
            const RuleIcon = OUTCOME_ICON[rule.outcome]
            return (
              <li key={rule.rule_id} className={`decision-tree__rule decision-tree__rule--${rule.outcome}`}>
                <RuleIcon size={13} />
                <span>{rule.reason_code.replaceAll('_', ' ')}</span>
                <small>observed {formatScalar(rule.observed_value)}{rule.threshold !== null && rule.threshold !== undefined ? ` · threshold ${formatScalar(rule.threshold)}` : ''}</small>
                <em>{rule.severity.replaceAll('_', ' ')}</em>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

/** Every decision explained as a tree, not a bare "Blocked" string — grouped by the same
 * `RuleCategory` the Signal Decision Engine itself evaluates against, sourced directly from the
 * decision's own persisted `rules` ledger (nothing re-derived or approximated). */
export function SignalDecisionTreePage() {
  const { selection } = useActiveSelection()
  const [decision, setDecision] = useState<SignalDecision | null>(null)
  const [notFound, setNotFound] = useState(false)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      const history = await fetchSafe<SignalDecision[]>(`/signal-decisions/history?instrument=${encodeURIComponent(selection.instrument)}&timeframe=${encodeURIComponent(selection.timeframe)}&limit=1`)
      if (cancelled) return
      const latest = history?.[0] ?? null
      setDecision(latest)
      setNotFound(!latest)
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [selection.instrument, selection.timeframe])

  const grouped = decision
    ? Object.entries(
        decision.rules.reduce<Record<string, RuleEvaluation[]>>((acc, rule) => {
          (acc[rule.category] ??= []).push(rule)
          return acc
        }, {}),
      ).sort(([a], [b]) => CATEGORY_ORDER.indexOf(a) - CATEGORY_ORDER.indexOf(b))
    : []

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SIGNAL DECISION</p><h1>Decision explainability <em>tree.</em></h1></div>
        <div className="page-icon"><GitBranch size={25} /></div>
      </header>
      {notFound && <div className="empty-state"><h3>No decision recorded yet</h3><p>The pipeline hasn't evaluated a signal decision for {selection.instrument}/{selection.timeframe} yet.</p></div>}
      {decision && (
        <>
          <section className="panel">
            <div className="panel__head">
              <div><p className="eyebrow">FINAL DECISION</p><h2>{decision.direction} · {decision.state.replaceAll('_', ' ')}</h2></div>
              <StateBadge state={decision.state === 'eligible' ? 'healthy' : decision.state === 'blocked' ? 'blocked' : 'limited'} />
            </div>
            <div className="panel-body decision-tree__summary">
              <Gauge value={decision.confidence_score} label="Confidence" />
              <Gauge value={decision.market_risk_score} label="Risk" invert />
              <Gauge value={decision.eligibility_score} label="Eligibility" />
              <div className="decision-tree__summary-text">
                <p>{decision.blockers.length ? `${decision.blockers.length} blocker(s): ${decision.blockers.map((b) => b.reason_code).join(', ')}` : 'No hard blockers.'}</p>
                <p>{decision.warnings.length ? `${decision.warnings.length} warning(s): ${decision.warnings.map((w) => w.reason_code).join(', ')}` : 'No warnings.'}</p>
                <small>as of {new Date(decision.as_of).toLocaleString()}</small>
              </div>
            </div>
          </section>
          <section className="panel">
            <div className="panel__head"><div><p className="eyebrow">DECISION TREE</p><h2>Every rule, every category</h2></div><span>{decision.rules.length} rules evaluated</span></div>
            <div className="panel-body decision-tree">
              {grouped.map(([category, rules]) => <CategoryBranch category={category} rules={rules} key={category} />)}
            </div>
          </section>
        </>
      )}
    </div>
  )
}
