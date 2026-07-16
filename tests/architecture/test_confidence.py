from backend.app.services.confidence import ConfidenceCalculator, ConfidenceConfig, ConfidenceFactors, ConfidenceWeights


def test_confidence_is_deterministic_and_ai_is_only_a_capped_bonus() -> None:
    calculator = ConfidenceCalculator(ConfidenceConfig(weights=ConfidenceWeights(smc=.25, liquidity=.2, institutional_flow=.2, volume_profile=.15, economic_risk=.15, ai_bonus=.05), maximum_ai_bonus=.03))
    factors = ConfidenceFactors(smc=1, liquidity=.5, institutional_flow=.4, volume_profile=1, economic_risk=1, ai_bonus=1)
    first = calculator.calculate(factors)
    second = calculator.calculate(factors)
    assert first == second
    assert first.breakdown["ai_bonus"] == .03
    assert 0 <= first.confidence <= 1
