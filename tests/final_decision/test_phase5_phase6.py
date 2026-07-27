"""Safety/reporting tests retained after retirement of AI proposal generation."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from backend.app.api.routes.ai_reasoning import _runtime_state
from backend.app.core.config import YamlConfigRepository
from backend.app.final_decision import (
    GuardrailPolicyConfig,
    HardGateRegistry,
    PerformanceReporter,
    ProbabilityCalibration,
    ProductionReadinessEvaluator,
)
from tests.ai_reasoning.test_ai_reasoning_lifecycle import NOW


def test_dashboard_runtime_metadata_is_observational_and_analytical_only() -> None:
    flags = {
        "ai_centric_shadow_mode": True,
        "ai_signal_publication": False,
    }
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_registry=SimpleNamespace(
                    context=SimpleNamespace(
                        feature_flags=SimpleNamespace(snapshot=lambda: flags),
                    ),
                ),
            ),
        ),
    )
    assert _runtime_state(request) == {
        "operating_profile": "shadow",
        "feature_flags": flags,
        "analytical_only": True,
        "broker_execution_available": False,
    }


def test_registry_contains_only_genuine_safety_gates() -> None:
    gate_ids = {item.gate_id for item in HardGateRegistry().all()}
    assert {
        "market_state_consistent",
        "future_data_absent",
        "market_open",
        "entry_geometry_valid",
        "absolute_risk_to_reward",
        "economic_event_blackout",
    } <= gate_ids
    forbidden = {
        "htf_disagreement",
        "weak_volume",
        "ranging_regime",
        "missing_fvg",
        "liquidity_disagreement",
    }
    assert not gate_ids.intersection(forbidden)


def test_probability_calibration_is_measured_not_claimed() -> None:
    report = ProbabilityCalibration().calculate(
        (
            {
                "actual": "BUY",
                "probabilities": {"BUY": 0.7, "SELL": 0.2, "NEUTRAL": 0.1},
                "horizon": "10_m1",
                "setup_family": "trend_continuation",
                "regime": "trend",
                "session": "london",
                "confidence_band": "high",
            },
            {
                "actual": "SELL",
                "probabilities": {"BUY": 0.4, "SELL": 0.4, "NEUTRAL": 0.2},
                "horizon": "10_m1",
                "setup_family": "trend_continuation",
                "regime": "trend",
                "session": "london",
                "confidence_band": "medium",
            },
        ),
        generated_at=NOW,
    )
    assert report.status == "measured_uncalibrated"
    assert report.brier_score is not None
    assert report.log_loss is not None
    assert report.expected_calibration_error is not None
    assert report.reliability_buckets


def test_performance_and_readiness_require_measured_samples() -> None:
    report = PerformanceReporter().build(
        period_start=NOW - timedelta(days=1),
        period_end=NOW,
        system_outcomes={
            "legacy": (),
            "quantitative_shadow": (),
            "ai_proposals": (),
            "guardrail_approved": (),
        },
        generated_at=NOW,
    )
    assert report.comparison["guardrail_approved"]["expected_value"] is None
    config = YamlConfigRepository().load_model("ai_guardrails", GuardrailPolicyConfig)
    readiness = ProductionReadinessEvaluator(config).evaluate(
        {"sample_count": 0},
        generated_at=NOW,
    )
    assert readiness.status == "not_ready"
    assert "sample_size" in readiness.blockers
    assert "profitability_not_guaranteed" in readiness.warnings


def test_only_configured_cerebras_and_groq_providers_are_present() -> None:
    provider_source = Path("backend/app/ai_reasoning/provider.py").read_text(
        encoding="utf-8"
    )
    assert "class CerebrasProvider" in provider_source
    assert "class GroqProvider" in provider_source
    assert "class AIProviderRouter" in provider_source


def test_operating_profiles_never_enable_broker_execution() -> None:
    profiles = Path("configs/ai_operating_profiles.yaml").read_text(encoding="utf-8")
    assert "safe_test:" in profiles
    assert "shadow:" in profiles
    assert "analytical_live:" in profiles
    assert "broker_execution: false" in profiles
    flags = Path("configs/feature_flags.yaml").read_text(encoding="utf-8")
    assert "ai_signal_publication: false" in flags
    assert "ai_signal_adjustments: false" in flags
