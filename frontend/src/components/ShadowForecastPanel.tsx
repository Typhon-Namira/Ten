import type { QuantCalibrationReport, QuantForecastOutcome, QuantForecastResult } from '../types'

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

export function ShadowForecastPanel({ forecast, calibration, outcomes }: { forecast: QuantForecastResult | null; calibration: QuantCalibrationReport | null; outcomes: QuantForecastOutcome[] }) {
  return <div className="shadow-forecast">
    <div className="shadow-forecast__banner">SHADOW FORECAST — NOT USED FOR LIVE SIGNAL PUBLICATION</div>
    {!forecast
      ? <p className="empty-state">No shadow forecast is persisted. Shadow mode is disabled by default or a synchronized M1/M5/M15 state is not yet complete.</p>
      : <>
        <div className="shadow-forecast__meta">
          <span>{forecast.status.replaceAll('_', ' ')}</span>
          <span>{forecast.model_kind.replaceAll('_', ' ')}</span>
          <span>{forecast.calibration_status}</span>
          <span>{new Date(forecast.point_in_time).toLocaleString()}</span>
        </div>
        <div className="shadow-forecast__calibration">
          <span>Calibration samples <strong>{calibration?.sample_count ?? 0}</strong></span>
          <span>Brier <strong>{calibration?.brier_score?.toFixed(4) ?? '—'}</strong></span>
          <span>Log loss <strong>{calibration?.log_loss?.toFixed(4) ?? '—'}</strong></span>
          <span>ECE <strong>{calibration?.expected_calibration_error?.toFixed(4) ?? '—'}</strong></span>
        </div>
        {forecast.status !== 'available'
          ? <p className="empty-state">{forecast.reason_codes.join(', ') || 'Numeric output unavailable; no zero values were substituted.'}</p>
          : <div className="shadow-forecast__grid">
            {forecast.predictions.map((item) => <article key={item.horizon.horizon_id}>
              <h3>{item.horizon.candle_count} × {item.horizon.timeframe}</h3>
              <div><span>Buy</span><strong>{percent(item.buy_probability)}</strong></div>
              <div><span>Sell</span><strong>{percent(item.sell_probability)}</strong></div>
              <div><span>Neutral</span><strong>{percent(item.neutral_probability)}</strong></div>
              <div><span>Expected return</span><strong>{percent(item.expected_return)}</strong></div>
              <div><span>Expected range</span><strong>{percent(item.expected_minimum_movement)}–{percent(item.expected_maximum_movement)}</strong></div>
              <div><span>Volatility</span><strong>{percent(item.expected_volatility)}</strong></div>
              <div><span>MFE / MAE</span><strong>{percent(item.expected_mfe)} / {percent(item.expected_mae)}</strong></div>
              <div><span>TP1 / TP2</span><strong>{percent(item.tp1_probability)} / {percent(item.tp2_probability)}</strong></div>
              <div><span>SL before TP</span><strong>{percent(item.sl_before_tp_probability)}</strong></div>
              <div><span>Uncertainty</span><strong>{percent(item.uncertainty_interval.low)}–{percent(item.uncertainty_interval.high)}</strong></div>
              <div><span>Outcome</span><strong>{outcomes.find((outcome) => outcome.horizon_id === item.horizon.horizon_id)?.status.replaceAll('_', ' ') ?? 'pending'}</strong></div>
            </article>)}
          </div>}
      </>}
  </div>
}
