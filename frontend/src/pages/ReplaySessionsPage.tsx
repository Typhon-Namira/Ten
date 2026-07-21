import { History } from 'lucide-react'
import { useActiveSelection } from '../hooks/useActiveSelection'
import { useDashboard } from '../hooks/useDashboard'
import { StateBadge } from '../components/StateBadge'
import { normalizeEngineState } from '../lib/engineState'

export function ReplaySessionsPage() {
  const { selection } = useActiveSelection()
  const { replays, diagnostics } = useDashboard(selection.instrument, selection.timeframe)

  return (
    <div className="page">
      <header>
        <div><p className="eyebrow">MARKET</p><h1>Replay <em>sessions.</em></h1></div>
        <div className="page-icon"><History size={25} /></div>
      </header>
      {!diagnostics?.replay.enabled && <div className="empty-state"><h3>Replay worker disabled</h3><p>Deterministic historical replay is intentionally disabled in this deployment.</p></div>}
      {diagnostics?.replay.enabled && !replays.length && <div className="empty-state"><h3>No replay sessions created</h3><p>Create a session via the replay API to backtest deterministically against historical data.</p></div>}
      {replays.length > 0 && (
        <section className="panel">
          <div className="panel__head"><div><p className="eyebrow">SESSIONS</p><h2>{replays.length} recent</h2></div></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Dataset</th><th>Status</th><th>Progress</th><th>Events</th><th>Created</th></tr></thead>
              <tbody>
                {replays.map((session) => (
                  <tr key={session.replay_id}>
                    <td><strong>{session.request.dataset.dataset_version}</strong><small>{session.request.instruments.join(', ')} · {session.request.timeframes.join(', ')}</small></td>
                    <td><StateBadge state={normalizeEngineState(session.status === 'completed' ? 'success' : session.status === 'failed' ? 'failed' : session.status === 'running' ? 'running' : 'waiting')} /></td>
                    <td>{session.progress_percent ?? '—'}%</td>
                    <td>{session.processed_events.toLocaleString()}</td>
                    <td>{new Date(session.created_at).toLocaleString()}</td>
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
