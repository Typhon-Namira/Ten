import { useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { ActiveSelection } from '../types'

/** Backend-default fallback while `/api/v1/system/selection` is loading — matches the backend's
 * own out-of-the-box default (`settings.market_data_symbols`/`market_data_timeframes`), so the
 * dashboard renders something sensible on first paint instead of blocking. Once the real
 * response lands, every widget re-queries with whatever the backend is actually configured to
 * run, even if that differs from this fallback. */
const FALLBACK: ActiveSelection = { instrument: 'XAUUSD', timeframe: 'M15', configured_instruments: ['XAUUSD'], configured_timeframes: ['M15'] }

/** The one place the dashboard learns which (instrument, timeframe) pair the pipeline actually
 * runs. Every other hook must consume this instead of hardcoding its own default — that
 * divergence (each widget independently defaulting to "XAUUSD"/"M15") is what previously let the
 * stage tracker, market intelligence panel, and live log silently describe different candle
 * series whenever the deployment's real primary timeframe differed from the hardcoded literal. */
export function useActiveSelection(): { selection: ActiveSelection; loaded: boolean } {
  const [selection, setSelection] = useState<ActiveSelection>(FALLBACK)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    tenApi
      .selection()
      .then((value) => {
        if (!cancelled) {
          setSelection(value)
          setLoaded(true)
        }
      })
      .catch(() => {
        // Keep the fallback — every dependent hook already tolerates a failing source.
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { selection, loaded }
}
