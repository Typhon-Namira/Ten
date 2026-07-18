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

export type AIScoreStatus = 'ready' | 'degraded' | 'insufficient_evidence' | 'stale' | 'invalid' | 'replay'
export type DirectionalLabel = 'strong_bearish' | 'bearish' | 'slightly_bearish' | 'neutral' | 'slightly_bullish' | 'bullish' | 'strong_bullish'

export interface AIScoreSnapshot {
  snapshot_id: string
  instrument: string
  timeframe: string
  as_of: string
  policy_version: string
  directional_score: number
  directional_label: DirectionalLabel
  confidence_score: number
  market_risk_score: number
  evidence_alignment_score: number
  data_quality_score: number
  composite_score: number
  status: AIScoreStatus
  missing_sources: string[]
  degraded_sources: string[]
}

