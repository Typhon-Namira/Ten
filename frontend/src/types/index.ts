export type Direction = 'long' | 'short' | 'neutral'
export type EngineState = 'ready' | 'degraded' | 'offline'

export interface Signal {
  symbol: string
  timeframe: string
  direction: Direction
  entry_zone: [number, number]
  stop_loss: number
  take_profit: number
  confidence: number
  reasoning: string[]
  risk_notes: string[]
  timestamp: string
}

export interface EngineStatus {
  name: string
  version: string
  state: EngineState
  details: string
  checked_at: string
}

export interface MarketStatus {
  symbol: string
  session: string
  is_open: boolean
  checked_at: string
  note: string
}

