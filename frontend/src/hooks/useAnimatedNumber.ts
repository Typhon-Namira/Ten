import { useEffect, useRef, useState } from 'react'

/** Smoothly tweens a displayed number toward `target` instead of snapping on every poll refresh —
 * used by every gauge/counter widget so a confidence score ticking from 61 to 64 reads as
 * movement, not a flicker. Falls back to the raw target while `target` is null (nothing to
 * animate toward). */
export function useAnimatedNumber(target: number | null, durationMs = 500): number | null {
  const [value, setValue] = useState<number | null>(target)
  const frameRef = useRef<number | null>(null)
  const fromRef = useRef<number | null>(target)

  useEffect(() => {
    if (target === null) {
      setValue(null)
      fromRef.current = null
      return
    }
    const from = fromRef.current ?? target
    const start = performance.now()
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    const tick = (now: number) => {
      const elapsed = Math.min(1, (now - start) / durationMs)
      const eased = 1 - (1 - elapsed) * (1 - elapsed)
      setValue(from + (target - from) * eased)
      if (elapsed < 1) {
        frameRef.current = requestAnimationFrame(tick)
      } else {
        fromRef.current = target
      }
    }
    frameRef.current = requestAnimationFrame(tick)
    return () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs])

  return value
}
