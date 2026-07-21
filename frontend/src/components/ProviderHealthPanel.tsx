import { EconomicStateBadge } from './EconomicStateBadge'
import { providerConnectionBadge } from '../lib/economicState'
import type { ProviderStatus } from '../types'

const time = (value: string | null) => (value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'never')
const na = (value: unknown) => (value === null || value === undefined || value === '' ? '—' : String(value))

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="provider-card__field">
      <span>{label}</span>
      <b title={value}>{value}</b>
    </div>
  )
}

/** Every field the brief asked for: name, version, URL, auth/key status, HTTP status, request
 * timeline, latency, retries, backoff, rate limit, quota, connection state, and the raw failure —
 * one card per configured provider, updating on every poll. */
function ProviderCard({ provider }: { provider: ProviderStatus }) {
  const visual = providerConnectionBadge(provider.connection_state)
  return (
    <div className="provider-card">
      <div className="provider-card__head">
        <h3>{provider.provider_name.replaceAll('_', ' ')}</h3>
        <EconomicStateBadge visual={visual} detail={provider.failure_reason ?? undefined} />
      </div>
      {provider.base_url && <p className="provider-card__url">{provider.base_url}</p>}
      <div className="provider-card__fields">
        <Field label="Version" value={na(provider.provider_version)} />
        <Field label="Mode" value={na(provider.mode)} />
        <Field label="API key" value={provider.api_key_configured ? 'Configured' : 'Not configured'} />
        <Field label="Authenticated" value={provider.authenticated ? 'Yes' : 'No'} />
        <Field label="HTTP status" value={na(provider.http_status)} />
        <Field label="Response time" value={provider.response_time_ms != null ? `${provider.response_time_ms.toFixed(0)} ms` : '—'} />
        <Field label="Last request" value={time(provider.last_request)} />
        <Field label="Last success" value={time(provider.last_success)} />
        <Field label="Last failure" value={time(provider.last_failure)} />
        <Field label="Retry count" value={String(provider.retry_count)} />
        <Field label="Backoff until" value={time(provider.backoff_until)} />
        <Field label="Rate limit remaining" value={provider.rate_limit_remaining != null ? `${provider.rate_limit_remaining}${provider.rate_limit_limit != null ? ` / ${provider.rate_limit_limit}` : ''}` : '—'} />
        <Field label="Daily quota" value={provider.daily_quota_used != null ? `${provider.daily_quota_used}${provider.daily_quota_limit != null ? ` / ${provider.daily_quota_limit}` : ''}` : '—'} />
        <Field label="Monthly quota" value={provider.monthly_quota_used != null ? `${provider.monthly_quota_used}${provider.monthly_quota_limit != null ? ` / ${provider.monthly_quota_limit}` : ''}` : '—'} />
        <Field label="Connection state" value={provider.connection_state.replaceAll('_', ' ')} />
      </div>
      {provider.raw_error && <pre className="provider-card__error">{provider.raw_error}</pre>}
    </div>
  )
}

export function ProviderHealthPanel({ providers }: { providers: ProviderStatus[] | null }) {
  if (!providers?.length) {
    return <div className="widget-empty" style={{ padding: '20px' }}>No providers configured.</div>
  }
  return (
    <div className="provider-health-grid">
      {providers.map((provider) => <ProviderCard key={provider.provider_name} provider={provider} />)}
    </div>
  )
}
