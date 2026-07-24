import { useState, useSyncExternalStore } from 'react'
import { AlertTriangle, ChevronUp, Wifi } from 'lucide-react'
import { useSystemDiagnostics } from '../hooks/useSystemDiagnostics'
import { getDiagnosticsFeedSnapshot, subscribeToDiagnosticsFeed } from '../lib/diagnosticsFeed'
import type { CanonicalState } from '../lib/engineState'
import { bootstrapEstimate, formatEtaMinutes } from '../lib/bootstrap'
import { StateBadge } from './StateBadge'

function relativeTime(date: Date | null): string {
  if (!date) return 'never'
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  return `${Math.round(minutes / 60)}h ago`
}

/** Always-visible, always-collapsed-by-default status affordance (Part 3 requirement #2). Reads
 * only data the backend already exposes via /api/v1/system/diagnostics and the per-hook fetch
 * outcomes recorded in lib/diagnosticsFeed — issues no requests of its own beyond the one polling
 * hook shared across the whole app. */
export function DiagnosticsBar() {
  const [expanded, setExpanded] = useState(false)
  const diagnostics = useSystemDiagnostics()
  const feed = useSyncExternalStore(subscribeToDiagnosticsFeed, getDiagnosticsFeedSnapshot)

  const anyErrors = feed.some((item) => item.lastError != null)
  const workerIssue = Boolean(diagnostics && (diagnostics.workers.market_data_worker.last_error || diagnostics.workers.integration_worker.last_error))
  const overallState: CanonicalState = !diagnostics
    ? 'waiting'
    : anyErrors || workerIssue
      ? 'limited'
      : diagnostics.operational_state.startsWith('HEALTHY')
        ? 'healthy'
        : 'waiting'
  const bootstrapping = diagnostics != null && !diagnostics.history.initialized
  const estimate = diagnostics ? bootstrapEstimate(diagnostics.history.candle_count, diagnostics.history.required_candle_count, diagnostics.market.timeframe) : null

  return (
    <div className={`diagnostics-bar${expanded ? ' diagnostics-bar--expanded' : ''}`}>
      <button type="button" className="diagnostics-bar__summary" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
        {anyErrors ? <AlertTriangle size={14} /> : <Wifi size={14} />}
        <span>{diagnostics ? diagnostics.operational_state.replaceAll('_', ' ') : 'Connecting…'}</span>
        {bootstrapping && estimate && <span className="diagnostics-bar__hint">Bootstrapping history — {estimate.remainingCandles} candles remaining (ETA {formatEtaMinutes(estimate.etaMinutes)})</span>}
        <StateBadge state={overallState} />
        <ChevronUp size={14} className={expanded ? '' : 'diagnostics-bar__chevron--flipped'} />
      </button>
      {expanded && (
        <div className="diagnostics-bar__detail">
          <div className="diagnostics-bar__section">
            <h4>Data sources (this session)</h4>
            {feed.length === 0 && <p className="diagnostics-bar__empty">No fetches recorded yet.</p>}
            <table>
              <thead><tr><th>Source</th><th>Last success</th><th>Last error</th></tr></thead>
              <tbody>
                {feed.map((item) => (
                  <tr key={item.source}>
                    <td>{item.label}</td>
                    <td>{relativeTime(item.lastSuccessAt)}</td>
                    <td>{item.lastError ? `${item.lastError.status ?? item.lastError.kind}: ${item.lastError.message}` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {diagnostics && (
            <div className="diagnostics-bar__section">
              <h4>Background workers</h4>
              <table>
                <thead><tr><th>Worker</th><th>State</th><th>Last success</th><th>Failures</th></tr></thead>
                <tbody>
                  <tr>
                    <td>Market data</td>
                    <td><StateBadge state={diagnostics.workers.market_data_worker.enabled && diagnostics.workers.market_data_worker.running && !diagnostics.workers.market_data_worker.last_error ? 'healthy' : diagnostics.workers.market_data_worker.enabled ? 'limited' : 'disabled'} detail={diagnostics.workers.market_data_worker.last_error ?? undefined} /></td>
                    <td>{relativeTime(diagnostics.workers.market_data_worker.last_success_at ? new Date(diagnostics.workers.market_data_worker.last_success_at) : null)}</td>
                    <td>{diagnostics.workers.market_data_worker.consecutive_failures}</td>
                  </tr>
                  <tr>
                    <td>Integration</td>
                    <td><StateBadge state={diagnostics.workers.integration_worker.enabled && diagnostics.workers.integration_worker.running && !diagnostics.workers.integration_worker.last_error ? 'healthy' : diagnostics.workers.integration_worker.enabled ? 'limited' : 'disabled'} detail={diagnostics.workers.integration_worker.last_error ?? undefined} /></td>
                    <td>{relativeTime(diagnostics.workers.integration_worker.last_success_at ? new Date(diagnostics.workers.integration_worker.last_success_at) : null)}</td>
                    <td>{diagnostics.workers.integration_worker.consecutive_failures}</td>
                  </tr>
                </tbody>
              </table>
              <p className="diagnostics-bar__note">History: {diagnostics.history.candle_count} / {diagnostics.history.required_candle_count} candles{bootstrapping && estimate ? ` — bootstrapping, ETA ${formatEtaMinutes(estimate.etaMinutes)}` : ' — complete'}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
