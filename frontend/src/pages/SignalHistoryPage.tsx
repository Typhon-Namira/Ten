import { Clock3 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { fetchSafe } from '../services/api'
import { StateBadge } from '../components/StateBadge'
import { normalizeEngineState } from '../lib/engineState'
import type { SignalDecisionSnapshot } from '../types'

const POLL_MS = 10_000

export function SignalHistoryPage() {
  const { selection } = useActiveSelection()
  const [history, setHistory] = useState<SignalDecisionSnapshot[]>([])

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      const value = await fetchSafe<SignalDecisionSnapshot[]>(`/signal-decisions/history?instrument=${encodeURIComponent(selection.instrument)}&timeframe=${encodeURIComponent(selection.timeframe)}&limit=50`)
      if (!cancelled && value) setHistory(value)
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [selection.instrument, selection.timeframe])

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">SIGNALS</p><h1>Decision <em>history.</em></h1></div>
        <div className="page-icon"><Clock3 size={25} /></div>
      </header>
      {!history.length && <div className="empty-state"><h3>No decisions recorded yet</h3><p>The signal decision engine hasn't evaluated {selection.instrument}/{selection.timeframe} yet.</p></div>}
      {history.length > 0 && (
        <section className="panel">
          <div className="panel__head"><div><p className="eyebrow">HISTORY</p><h2>{history.length} decisions</h2></div></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Time</th><th>Direction</th><th>State</th><th>Confidence</th><th>Risk</th><th>Blockers</th></tr></thead>
              <tbody>
                {history.map((decision) => (
                  <tr key={decision.decision_id}>
                    <td>{new Date(decision.as_of).toLocaleString()}</td>
                    <td><span className={`direction direction--${decision.direction === 'bullish' ? 'long' : decision.direction === 'bearish' ? 'short' : 'neutral'}`}>{decision.direction}</span></td>
                    <td><StateBadge state={normalizeEngineState(decision.state)} /></td>
                    <td>{decision.confidence_score.toFixed(1)}%</td>
                    <td>{decision.market_risk_score.toFixed(1)}%</td>
                    <td>{decision.blockers.map((b) => b.reason_code).join(', ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
