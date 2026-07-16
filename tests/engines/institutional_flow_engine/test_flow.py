from backend.app.engines.institutional_flow_engine import BaselineInstitutionalFlowEngine
from backend.app.engines.market_data_engine import Candle


def test_estimated_flow_is_bounded_and_disclosed(candles: list[Candle]) -> None:
    score = BaselineInstitutionalFlowEngine().analyze(candles)
    assert -1 <= score.score <= 1
    assert -1 <= score.delta_estimate <= 1
    assert "not exchange order flow" in score.methodology

