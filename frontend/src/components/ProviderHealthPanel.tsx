import { EconomicStateBadge } from './EconomicStateBadge'
import { providerConnectionBadge } from '../lib/economicState'
import type { ProviderStatus } from '../types'

const time = (value: string | null) => (value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'never')
const na = (value: unknown) => (value === null || value === undefined || value === '' ? '—' : String(value))

const SOURCE_TYPE_LABEL: Record<string, string> = {
  keyed_api: 'Keyed API',
  public_webpage: 'Public webpage',
  rss_feed: 'RSS feed',
  ics_calendar: 'ICS calendar',
  deterministic_rule: 'Deterministic rule',
  none: 'Unknown',
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="provider-card__field">
      <span>{label}</span>
      <b title={value}>{value}</b>
    </div>
  )
}

/** One card per configured provider. Fields adapt to `source_type`: a public source (the only
 * kind the economic calendar uses) never shows API-key/quota fields — there is no key and no
 * quota, so showing them would misleadingly imply a subscription that doesn't exist. */
function ProviderCard({ provider }: { provider: ProviderStatus }) {
  const visual = providerConnectionBadge(provider.connection_state)
  const isKeyed = provider.source_type === 'keyed_api'
  return (
    <div className="provider-card">
      <div className="provider-card__head">
        <h3>{provider.provider_name.replaceAll('_', ' ')}</h3>
        <EconomicStateBadge visual={visual} detail={provider.failure_reason ?? undefined} />
      </div>
      {provider.base_url && <p className="provider-card__url">{provider.base_url}</p>}
      <div className="provider-card__fields">
        <Field label="Source type" value={SOURCE_TYPE_LABEL[provider.source_type] ?? na(provider.source_type)} />
        <Field label="Version" value={na(provider.provider_version)} />
        <Field label="Mode" value={na(provider.mode)} />
        {isKeyed ? (
          <>
            <Field label="API key" value={provider.api_key_configured ? 'Configured' : 'Not configured'} />
            <Field label="Authenticated" value={provider.authenticated ? 'Yes' : 'No'} />
            <Field label="Rate limit remaining" value={provider.rate_limit_remaining != null ? `${provider.rate_limit_remaining}${provider.rate_limit_limit != null ? ` / ${provider.rate_limit_limit}` : ''}` : '—'} />
            <Field label="Daily quota" value={provider.daily_quota_used != null ? `${provider.daily_quota_used}${provider.daily_quota_limit != null ? ` / ${provider.daily_quota_limit}` : ''}` : '—'} />
            <Field label="Monthly quota" value={provider.monthly_quota_used != null ? `${provider.monthly_quota_used}${provider.monthly_quota_limit != null ? ` / ${provider.monthly_quota_limit}` : ''}` : '—'} />
          </>
        ) : (
          <>
            <Field label="API key" value="Not used — public source" />
            <Field label="Robots policy" value={na(provider.robots_policy_status)} />
            <Field label="Parser version" value={na(provider.parser_version)} />
            <Field label="Events parsed" value={String(provider.events_parsed)} />
            <Field label="Schedule extends to" value={provider.last_schedule_date ? new Date(provider.last_schedule_date).toLocaleDateString() : '—'} />
            <Field label="Cache age" value={provider.cache_age_seconds != null ? `${Math.round(provider.cache_age_seconds / 60)} min` : '—'} />
          </>
        )}
        <Field label="HTTP status" value={na(provider.http_status)} />
        <Field label="Response time" value={provider.response_time_ms != null ? `${provider.response_time_ms.toFixed(0)} ms` : '—'} />
        <Field label="Last request" value={time(provider.last_request)} />
        <Field label="Last success" value={time(provider.last_success)} />
        <Field label="Last failure" value={time(provider.last_failure)} />
        <Field label="Retry count" value={String(provider.retry_count)} />
        <Field label="Circuit breaker" value={provider.circuit_breaker_open ? `Open until ${time(provider.circuit_breaker_open_until)}` : 'Closed'} />
        {provider.last_failure_category && <Field label="Last failure category" value={provider.last_failure_category.replaceAll('_', ' ')} />}
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
