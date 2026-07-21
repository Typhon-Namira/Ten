/**
 * Loads TradingView's lightweight-charts via a CDN <script> tag instead of an npm dependency —
 * this sandbox has no Node.js/npm to install packages or build against, so a bundled import
 * would not compile (and there is no local copy of the package to type-check against either).
 * The CDN standalone build attaches `window.LightweightCharts` (UMD global), pinned to a fixed
 * version for reproducibility — same pattern as the Google Fonts <link> already used in
 * index.html. Requires the dashboard to have internet access at runtime (fine for a
 * Railway-hosted app; callers fall back to a "chart unavailable" message if the CDN can't load).
 *
 * The types below are a minimal, hand-written surface covering only the calls this codebase
 * makes — not a full re-declaration of the library's public API — since the real `@types` are
 * unavailable without an npm install.
 */

export interface LWPriceLine {
  applyOptions(options: Record<string, unknown>): void
}

export interface LWSeriesApi {
  setData(data: unknown[]): void
  update(bar: unknown): void
  createPriceLine(options: Record<string, unknown>): LWPriceLine
  removePriceLine(line: LWPriceLine): void
  setMarkers(markers: unknown[]): void
}

export interface LWTimeScale {
  fitContent(): void
  subscribeVisibleLogicalRangeChange(callback: (range: unknown) => void): void
  setVisibleRange(range: { from: number; to: number }): void
}

export interface LWPriceScaleApi {
  applyOptions(options: Record<string, unknown>): void
}

export interface LWChartApi {
  addCandlestickSeries(options?: Record<string, unknown>): LWSeriesApi
  addHistogramSeries(options?: Record<string, unknown>): LWSeriesApi
  applyOptions(options: Record<string, unknown>): void
  priceScale(priceScaleId: string): LWPriceScaleApi
  timeScale(): LWTimeScale
  subscribeCrosshairMove(callback: (param: unknown) => void): void
  resize(width: number, height: number): void
  remove(): void
}

export interface LightweightChartsGlobal {
  createChart(container: HTMLElement, options?: Record<string, unknown>): LWChartApi
  CrosshairMode: { Normal: number; Magnet: number }
}

declare global {
  interface Window {
    LightweightCharts?: LightweightChartsGlobal
  }
}

const CDN_URL = 'https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js'

let loading: Promise<LightweightChartsGlobal> | null = null

export function loadLightweightCharts(): Promise<LightweightChartsGlobal> {
  if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts)
  if (loading) return loading
  loading = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${CDN_URL}"]`)
    const onReady = () => (window.LightweightCharts ? resolve(window.LightweightCharts) : reject(new Error('charting library did not attach to window')))
    if (existing) {
      existing.addEventListener('load', onReady)
      existing.addEventListener('error', () => reject(new Error('failed to load charting library')))
      return
    }
    const script = document.createElement('script')
    script.src = CDN_URL
    script.async = true
    script.onload = onReady
    script.onerror = () => reject(new Error('failed to load charting library'))
    document.head.appendChild(script)
  })
  return loading
}
