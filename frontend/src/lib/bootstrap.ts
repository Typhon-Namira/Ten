/** Approximate bar duration per timeframe, minutes — matches backend/app/engines/market_data_engine
 * Timeframe durations. Used only to turn "N candles still needed" into a human ETA; never sent to
 * the backend and never used for anything except this estimate. */
const TIMEFRAME_MINUTES: Record<string, number> = { M5: 5, M15: 15 }

export interface BootstrapEstimate {
  remainingCandles: number
  etaMinutes: number | null
}

export function bootstrapEstimate(candleCount: number, requiredCandleCount: number, timeframe: string): BootstrapEstimate {
  const remainingCandles = Math.max(0, requiredCandleCount - candleCount)
  const minutesPerCandle = TIMEFRAME_MINUTES[timeframe]
  return { remainingCandles, etaMinutes: minutesPerCandle ? remainingCandles * minutesPerCandle : null }
}

export function formatEtaMinutes(minutes: number | null): string {
  if (minutes == null) return 'unknown'
  if (minutes < 1) return 'under a minute'
  if (minutes < 60) return `~${Math.ceil(minutes)} min`
  const hours = minutes / 60
  return hours < 48 ? `~${hours.toFixed(1)} hr` : `~${Math.round(hours / 24)} days`
}
