import type { EngineStatus, MarketStatus, Signal } from '../types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`)
  if (!response.ok) {
    throw new Error(`TEN API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const tenApi = {
  signals: () => request<Signal[]>('/signals?limit=20'),
  engines: () => request<EngineStatus[]>('/engines/status'),
  market: () => request<MarketStatus>('/market/status'),
}

