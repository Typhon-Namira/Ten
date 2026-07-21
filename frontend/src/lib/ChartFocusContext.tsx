import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'

interface ChartFocusValue {
  focusTime: number | null
  focusCorrelationId: string | null
  focus: (time: number, correlationId?: string) => void
}

const ChartFocusContext = createContext<ChartFocusValue | null>(null)

/** The light version of "cross-link everything": a log entry (or any object with a timestamp)
 * can call `focus(time)` and the chart, if mounted in the same page, scrolls to center that
 * moment and flashes it — without a full bidirectional object-selection system spanning every
 * page. Pages that don't render a chart simply have no listener, so calling `focus()` there is a
 * harmless no-op. */
export function ChartFocusProvider({ children }: { children: ReactNode }) {
  const [focusTime, setFocusTime] = useState<number | null>(null)
  const [focusCorrelationId, setFocusCorrelationId] = useState<string | null>(null)
  const value = useMemo<ChartFocusValue>(
    () => ({
      focusTime,
      focusCorrelationId,
      focus: (time: number, correlationId?: string) => {
        setFocusTime(time)
        setFocusCorrelationId(correlationId ?? null)
      },
    }),
    [focusTime, focusCorrelationId],
  )
  return <ChartFocusContext.Provider value={value}>{children}</ChartFocusContext.Provider>
}

/** Safe outside a provider (e.g. pages with no chart) — returns a no-op focus function. */
export function useChartFocus(): ChartFocusValue {
  const context = useContext(ChartFocusContext)
  return context ?? { focusTime: null, focusCorrelationId: null, focus: () => {} }
}
