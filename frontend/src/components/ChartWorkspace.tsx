import { useEffect, useRef, useState } from 'react'
import { useChartOverlays } from '../hooks/useChartOverlays'
import { loadLightweightCharts, type LWChartApi, type LWPriceLine, type LWSeriesApi } from '../lib/loadLightweightCharts'
import { useChartFocus } from '../lib/ChartFocusContext'
import type { ChartCandle } from '../types'

const TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']

type OverlayKey = 'structure' | 'zones' | 'liquidity' | 'volumeProfile' | 'events'

const OVERLAY_TOGGLES: { key: OverlayKey; label: string }[] = [
  { key: 'structure', label: 'Structure (BOS/CHOCH)' },
  { key: 'zones', label: 'Zones (OB/FVG)' },
  { key: 'liquidity', label: 'Liquidity' },
  { key: 'volumeProfile', label: 'POC/VAH/VAL' },
  { key: 'events', label: 'Economic events' },
]

const number = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

function zoneColor(direction: string): string {
  return direction === 'bullish' ? '#59b993' : direction === 'bearish' ? '#c26965' : '#c4a359'
}

/** The mandatory live XAU/USD chart. Candles come from the same `/chart/overlays` call as every
 * overlay, so structure/zone/liquidity annotations are always drawn against the exact candles
 * they were computed from — never a mismatched timeframe or a stale set of levels. The chart's
 * own timeframe selector is a local viewing choice (like any professional terminal); changing it
 * re-requests candles AND overlays for that timeframe together, so the two can never drift apart
 * within this one visual, even though it may differ from the dashboard's global analysis
 * timeframe shown elsewhere. */
export function ChartWorkspace({ instrument, defaultTimeframe }: { instrument: string; defaultTimeframe: string }) {
  const [timeframe, setTimeframe] = useState(defaultTimeframe)
  const { data, error: dataError } = useChartOverlays(instrument, timeframe)
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<LWChartApi | null>(null)
  const candleSeriesRef = useRef<LWSeriesApi | null>(null)
  const volumeSeriesRef = useRef<LWSeriesApi | null>(null)
  const priceLinesRef = useRef<LWPriceLine[]>([])
  const hasFitRef = useRef(false)
  const [ready, setReady] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [crosshair, setCrosshair] = useState<ChartCandle | null>(null)
  const [overlays, setOverlays] = useState<Record<OverlayKey, boolean>>({ structure: true, zones: true, liquidity: true, volumeProfile: true, events: true })
  const { focusTime } = useChartFocus()

  useEffect(() => {
    let disposed = false
    loadLightweightCharts()
      .then((LWC) => {
        if (disposed || !containerRef.current || !LWC) return
        const chart = LWC.createChart(containerRef.current, {
          layout: { background: { color: 'transparent' }, textColor: '#9ba2ab', fontFamily: "'DM Sans', sans-serif" },
          grid: { vertLines: { color: '#171a20' }, horzLines: { color: '#171a20' } },
          crosshair: { mode: LWC.CrosshairMode.Normal },
          timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#20252c' },
          rightPriceScale: { borderColor: '#20252c' },
          autoSize: true,
        })
        const candleSeries = chart.addCandlestickSeries({ upColor: '#59b993', downColor: '#c26965', borderVisible: false, wickUpColor: '#59b993', wickDownColor: '#c26965' })
        const volumeSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'volume', color: '#2a3037' })
        // Confine the volume histogram to the bottom ~18% of the pane instead of overlapping price.
        chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
        chart.subscribeCrosshairMove((raw) => {
          const param = raw as { seriesData?: Map<unknown, unknown>; time?: unknown }
          const point = param.seriesData?.get(candleSeries) as (ChartCandle & { value?: number }) | undefined
          setCrosshair(point ? { time: Number(param.time), open: point.open, high: point.high, low: point.low, close: point.close, volume: 0 } : null)
        })
        chartRef.current = chart
        candleSeriesRef.current = candleSeries
        volumeSeriesRef.current = volumeSeries
        setReady(true)
      })
      .catch((caught: Error) => setLoadError(caught.message))
    return () => {
      disposed = true
      chartRef.current?.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
      hasFitRef.current = false
    }
    // Chart instance is created once per mount; re-created only if the workspace itself remounts.
  }, [])

  // Candle + volume data — refreshed on every poll, but the view is only auto-fit once so the
  // user's own zoom/pan never gets reset by a background refresh.
  useEffect(() => {
    if (!ready || !data || !candleSeriesRef.current || !volumeSeriesRef.current) return
    candleSeriesRef.current.setData(data.candles.map((c) => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })))
    volumeSeriesRef.current.setData(data.candles.map((c) => ({ time: c.time, value: c.volume, color: c.close >= c.open ? 'rgba(89,185,147,.5)' : 'rgba(194,105,101,.5)' })))
    if (!hasFitRef.current && data.candles.length) {
      chartRef.current?.timeScale().fitContent()
      hasFitRef.current = true
    }
  }, [data, ready])

  // Every engine's plottable objects — price lines for zones/liquidity/POC/dealing-range, markers
  // for structure events and sweeps. Re-applied whenever the data or the visible-overlay toggles
  // change; never touches the candle/volume series data itself.
  useEffect(() => {
    if (!ready || !data || !candleSeriesRef.current) return
    const series = candleSeriesRef.current
    priceLinesRef.current.forEach((line) => series.removePriceLine(line))
    priceLinesRef.current = []

    if (overlays.zones) {
      for (const zone of data.zones) {
        const color = zoneColor(zone.direction)
        priceLinesRef.current.push(series.createPriceLine({ price: zone.upper, color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: zone.kind.replaceAll('_', ' ') }))
        priceLinesRef.current.push(series.createPriceLine({ price: zone.lower, color, lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: '' }))
      }
    }
    if (overlays.liquidity) {
      for (const pool of data.liquidity_pools) {
        const color = pool.side === 'buy_side' ? '#59b993' : '#c26965'
        priceLinesRef.current.push(series.createPriceLine({ price: (pool.upper + pool.lower) / 2, color, lineWidth: 1, lineStyle: 3, axisLabelVisible: true, title: `liquidity ${pool.side.replace('_side', '')}` }))
      }
      for (const level of data.equal_levels) {
        const color = level.side === 'buy_side' ? '#59b993' : '#c26965'
        priceLinesRef.current.push(series.createPriceLine({ price: level.price, color, lineWidth: 1, lineStyle: 4, axisLabelVisible: true, title: `EQ${level.side === 'buy_side' ? 'H' : 'L'} ×${level.member_count}` }))
      }
    }
    if (overlays.structure && data.dealing_range) {
      const dr = data.dealing_range
      priceLinesRef.current.push(series.createPriceLine({ price: dr.equilibrium, color: '#c4a359', lineWidth: 1, lineStyle: 0, axisLabelVisible: true, title: 'EQ' }))
      priceLinesRef.current.push(series.createPriceLine({ price: dr.premium_boundary, color: '#c26965', lineWidth: 1, lineStyle: 1, axisLabelVisible: true, title: 'premium' }))
      priceLinesRef.current.push(series.createPriceLine({ price: dr.discount_boundary, color: '#59b993', lineWidth: 1, lineStyle: 1, axisLabelVisible: true, title: 'discount' }))
    }
    if (overlays.volumeProfile && data.volume_profile) {
      const vp = data.volume_profile
      if (vp.poc !== null) priceLinesRef.current.push(series.createPriceLine({ price: vp.poc, color: '#8f7bd6', lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: 'POC' }))
      if (vp.vah !== null) priceLinesRef.current.push(series.createPriceLine({ price: vp.vah, color: '#8f7bd6', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'VAH' }))
      if (vp.val !== null) priceLinesRef.current.push(series.createPriceLine({ price: vp.val, color: '#8f7bd6', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'VAL' }))
    }

    const markers: unknown[] = []
    if (overlays.structure) {
      for (const event of data.structure_events) {
        markers.push({ time: event.time, position: event.direction === 'bullish' ? 'belowBar' : 'aboveBar', color: zoneColor(event.direction), shape: event.direction === 'bullish' ? 'arrowUp' : 'arrowDown', text: event.kind.toUpperCase() })
      }
    }
    if (overlays.liquidity) {
      for (const sweep of data.liquidity_sweeps) {
        markers.push({ time: sweep.time, position: 'inBar', color: '#c4a359', shape: 'circle', text: sweep.kind.replace('Liquidity', '') })
      }
    }
    if (overlays.events) {
      for (const event of data.economic_events) {
        const highImpact = event.importance === 'high' || event.importance === 'critical'
        markers.push({ time: event.time, position: 'aboveBar', color: highImpact ? '#c26965' : '#8a929b', shape: 'square', text: `📅 ${event.name}` })
      }
    }
    // A log entry (or any cross-linked object) that called `useChartFocus().focus(time)` gets a
    // distinct highlight marker merged in here, on top of whatever overlays are already showing —
    // "click a log line, see it on the chart" without a full object-selection subsystem.
    if (focusTime !== null) {
      markers.push({ time: focusTime, position: 'inBar', color: '#f0dfb1', shape: 'arrowDown', text: '◆ focused', size: 2 })
    }
    markers.sort((a, b) => (a as { time: number }).time - (b as { time: number }).time)
    series.setMarkers(markers)
  }, [data, ready, overlays, focusTime])

  // Cross-link: when something elsewhere on the page focuses a timestamp, recenter the visible
  // range on it instead of requiring the user to manually scroll/zoom to find it.
  useEffect(() => {
    if (focusTime === null || !ready || !chartRef.current) return
    const span = 60 * 30 // 30 candle-minutes of padding on each side, timeframe-agnostic enough for a visual nudge
    chartRef.current.timeScale().setVisibleRange({ from: focusTime - span, to: focusTime + span })
  }, [focusTime, ready])

  const latest = data?.candles.at(-1) ?? null
  const displayed = crosshair ?? latest

  return (
    <div className="chart-workspace">
      <div className="chart-toolbar">
        <div className="chart-toolbar__timeframes">
          {TIMEFRAMES.map((tf) => (
            <button key={tf} className={tf === timeframe ? 'active' : ''} onClick={() => { hasFitRef.current = false; setTimeframe(tf) }}>{tf}</button>
          ))}
        </div>
        <div className="chart-toolbar__overlays">
          {OVERLAY_TOGGLES.map(({ key, label }) => (
            <label key={key}>
              <input type="checkbox" checked={overlays[key]} onChange={(event) => setOverlays((prev) => ({ ...prev, [key]: event.target.checked }))} />
              {label}
            </label>
          ))}
        </div>
      </div>
      <div className="chart-ohlc">
        {displayed ? (
          <>
            <span>{instrument}</span>
            <span>{timeframe}</span>
            <span>O <b>{number.format(displayed.open)}</b></span>
            <span>H <b>{number.format(displayed.high)}</b></span>
            <span>L <b>{number.format(displayed.low)}</b></span>
            <span className={displayed.close >= displayed.open ? 'chart-ohlc__up' : 'chart-ohlc__down'}>C <b>{number.format(displayed.close)}</b></span>
          </>
        ) : <span>Loading candles…</span>}
        {data?.decision && (
          <span className={`chart-ohlc__decision chart-ohlc__decision--${data.decision.direction === 'bullish' ? 'up' : data.decision.direction === 'bearish' ? 'down' : 'neutral'}`}>
            AI: {data.decision.direction} · {data.decision.state.replaceAll('_', ' ')} · {data.decision.confidence.toFixed(0)}% confidence
          </span>
        )}
      </div>
      <div className="chart-canvas" ref={containerRef}>
        {loadError && <div className="chart-canvas__message">Chart library unavailable ({loadError}) — the CDN may be unreachable.</div>}
        {!loadError && !ready && <div className="chart-canvas__message chart-canvas__message--skeleton">Loading chart engine…</div>}
        {dataError && ready && <div className="chart-canvas__banner">Live candle feed degraded: {dataError}</div>}
      </div>
    </div>
  )
}
