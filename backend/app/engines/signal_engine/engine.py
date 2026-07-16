from abc import ABC

from backend.app.engines.common import AnalysisEngine

from .config import SignalConfig
from .models import Direction, Signal, SignalInputs


class SignalEngine(AnalysisEngine[SignalInputs, Signal], ABC):
    """Central decision port. Its outputs are scenarios, never execution orders."""


class BaselineSignalEngine(SignalEngine):
    name = "signal"
    version = "1.0.0"

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()

    def analyze(self, data: SignalInputs) -> Signal:
        direction = Direction(data.ai_score.direction.value)
        advisory_confidence = data.ai_score.confidence if data.ai_score.confidence is not None else 0.0
        confidence = data.calculated_confidence if data.calculated_confidence is not None else advisory_confidence
        confidence = 0.0 if data.news_risk.no_trade else confidence
        if confidence < self.config.minimum_confidence:
            direction = Direction.NEUTRAL
        half_zone = data.average_true_range * 0.15
        entry = (data.current_price - half_zone, data.current_price + half_zone)
        if direction == Direction.LONG:
            stop = entry[0] - data.average_true_range * self.config.atr_stop_multiplier
            target = entry[1] + (entry[1] - stop) * self.config.risk_reward_ratio
        elif direction == Direction.SHORT:
            stop = entry[1] + data.average_true_range * self.config.atr_stop_multiplier
            target = entry[0] - (stop - entry[0]) * self.config.risk_reward_ratio
        else:
            stop = data.current_price
            target = data.current_price
        reasoning = [*data.ai_score.reasoning, f"SMC bias: {data.smc.bias.value}", f"Estimated flow: {data.flow.bias.value}"]
        notes = [*data.ai_score.risk_notes]
        if data.news_risk.no_trade:
            notes.append("High-impact economic-event risk window is active.")
        triggered = ["smc", "liquidity", "institutional_flow", "volume_profile", "economic_calendar", "ai_scoring", "signal"]
        from .models import SignalExplanation

        explanation = SignalExplanation(
            reasons=reasoning,
            triggered_engines=triggered,
            supporting_evidence={"smc": [data.smc.bias.value], "flow": [data.flow.bias.value]},
            confidence_breakdown=data.confidence_breakdown,
            risk_notes=notes,
            rejected_conditions=["economic_no_trade"] if data.news_risk.no_trade else [],
        )
        return Signal(symbol=data.symbol, timeframe=data.timeframe, direction=direction, entry_zone=entry, stop_loss=stop, take_profit=target, confidence=confidence, reasoning=reasoning, risk_notes=notes, explanation=explanation)
