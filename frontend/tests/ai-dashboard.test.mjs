import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const source = async path => readFile(new URL(`../${path}`, import.meta.url), 'utf8')

test('primary navigation exposes only the five AI-centric destinations', async () => {
  const shell = await source('src/components/AppShell.tsx')
  for (const label of ['Overview', 'Signals', 'Performance', 'Calibration', 'System']) {
    assert.match(shell, new RegExp(`label: '${label}'`))
  }
  assert.doesNotMatch(shell, /label: '(Live|Engines|Logs|Rejections|Replay|AI Score)'/)
})

test('overview preserves the required decision hierarchy', async () => {
  const dashboard = await source('src/components/ai-dashboard/AIDashboard.tsx')
  const orderedSections = [
    '<FinalDecisionHero',
    '<DecisionPipeline',
    '<MarketStateSummary',
    '<QuantForecastSummary',
    '<AIReasoningSummary',
    '<GuardrailSummary',
    '<ActiveSignalMonitor',
    '<ValidationSummary',
    '<OperationalHealth',
  ]
  let previous = -1
  for (const section of orderedSections) {
    const position = dashboard.indexOf(section)
    assert.ok(position > previous, `${section} must appear in decision order`)
    previous = position
  }
})

test('decision copy is derived from backend action states and never implies execution', async () => {
  const model = await source('src/lib/aiDashboard.ts')
  const shell = await source('src/components/AppShell.tsx')
  for (const state of ['published', 'approved', 'monitoring', 'rejected', 'expired', 'no_action']) {
    assert.match(model, new RegExp(state))
  }
  assert.match(shell, /No Broker Execution/)
  assert.match(shell, /Analysis and decision support only/)
})

test('data hook retains last good backend values and marks missing market evidence stale', async () => {
  const hook = await source('src/hooks/useAIDashboardData.ts')
  assert.match(hook, /Promise\.allSettled/)
  assert.match(hook, /if \(results\[0\]\.status === 'fulfilled'\) setIntelligence/)
  assert.match(hook, /latest_candle_timestamp == null/)
  assert.doesNotMatch(hook, /setIntelligence\(null\)/)
})

test('theme is light, responsive, accessible, and motion-safe', async () => {
  const css = await source('src/styles.css')
  assert.match(css, /--surface:\s*#fff/i)
  assert.match(css, /@media \(max-width:\s*820px\)/)
  assert.match(css, /@media \(max-width:\s*520px\)/)
  assert.match(css, /prefers-reduced-motion:\s*reduce/)
  assert.match(css, /:focus-visible/)
  assert.doesNotMatch(css, /fonts\.googleapis\.com/)
})
