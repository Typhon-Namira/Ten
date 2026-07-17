"""Typed, versioned SMC Production 1.0 configuration."""

from hashlib import sha256

from pydantic import BaseModel, Field, model_validator

from .models import ConfirmationMethod


class SwingDetectionConfig(BaseModel):
    left_window: int = Field(default=2, ge=1, le=50)
    right_window: int = Field(default=2, ge=1, le=50)
    minimum_separation: int = Field(default=2, ge=1)
    minimum_excursion: float = Field(default=0.0, ge=0)
    atr_excursion_multiplier: float = Field(default=0.25, ge=0)
    internal_sensitivity: float = Field(default=0.5, gt=0, le=1)
    external_min_strength: float = Field(default=65, ge=0, le=100)
    equal_level_tolerance: float = Field(default=0.0005, gt=0, le=0.1)


class StructureConfig(BaseModel):
    confirmation_method: ConfirmationMethod = ConfirmationMethod.CLOSE
    minimum_break_distance: float = Field(default=0.0, ge=0)
    atr_break_multiplier: float = Field(default=0.05, ge=0)
    displacement_body_ratio: float = Field(default=0.6, ge=0, le=1)
    mss_displacement_score: float = Field(default=65, ge=0, le=100)
    require_protected_level_for_mss: bool = True


class ProcessingConfig(BaseModel):
    minimum_history: int = Field(default=5, ge=3)
    recalculation_window: int = Field(default=500, ge=10)
    batch_size: int = Field(default=5000, ge=100)
    checkpoint_interval: int = Field(default=100, ge=1)
    maximum_active_objects: int = Field(default=5000, ge=100)
    minimum_input_quality: float = Field(default=60, ge=0, le=100)


class DisplacementConfig(BaseModel):
    minimum_atr_impulse: float = Field(default=0.8, gt=0)
    strong_atr_impulse: float = Field(default=1.5, gt=0)
    minimum_body_ratio: float = Field(default=0.55, ge=0, le=1)
    minimum_efficiency: float = Field(default=0.6, ge=0, le=1)
    volume_confirmation_ratio: float = Field(default=1.2, gt=0)
    volume_lookback: int = Field(default=20, ge=2, le=500)
    invalidation_candles: int = Field(default=10, ge=1, le=500)


class ImbalanceConfig(BaseModel):
    minimum_size: float = Field(default=0.0, ge=0)
    minimum_atr_size: float = Field(default=0.1, ge=0)
    merge_tolerance_atr: float = Field(default=0.05, ge=0)
    mitigation_threshold: float = Field(default=100, gt=0, le=100)
    expiration_candles: int = Field(default=200, ge=1)
    time_decay_per_candle: float = Field(default=0.1, ge=0, le=10)
    void_minimum_candles: int = Field(default=2, ge=2, le=20)


class OrderBlockConfig(BaseModel):
    lookback: int = Field(default=12, ge=1, le=100)
    minimum_displacement_score: float = Field(default=60, ge=0, le=100)
    minimum_body_ratio: float = Field(default=0.25, ge=0, le=1)
    require_volume_confirmation: bool = False
    mitigation_threshold: float = Field(default=100, gt=0, le=100)
    expiration_candles: int = Field(default=500, ge=1)
    refine_to_body: bool = True


class DealingRangeConfig(BaseModel):
    premium_ratio: float = Field(default=0.5, gt=0, lt=1)
    discount_ratio: float = Field(default=0.5, gt=0, lt=1)
    ote_low_ratio: float = Field(default=0.62, gt=0, lt=1)
    ote_high_ratio: float = Field(default=0.79, gt=0, lt=1)
    golden_low_ratio: float = Field(default=0.618, gt=0, lt=1)
    golden_high_ratio: float = Field(default=0.65, gt=0, lt=1)
    maximum_ranges: int = Field(default=1000, ge=1)


class MultiTimeframeConfig(BaseModel):
    hierarchy: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
    parent_mapping: dict[str, str] = Field(default_factory=lambda: {"M1": "M5", "M5": "M15", "M15": "H1", "M30": "H4", "H1": "H4", "H4": "D1", "D1": "W1", "W1": "MN1"})
    alignment_weights: dict[str, float] = Field(default_factory=lambda: {"current": 0.5, "parent": 0.3, "higher": 0.2})


class SMCConfig(BaseModel):
    swing: SwingDetectionConfig = SwingDetectionConfig()
    structure: StructureConfig = StructureConfig()
    processing: ProcessingConfig = ProcessingConfig()
    displacement: DisplacementConfig = DisplacementConfig()
    imbalance: ImbalanceConfig = ImbalanceConfig()
    order_block: OrderBlockConfig = OrderBlockConfig()
    dealing_range: DealingRangeConfig = DealingRangeConfig()
    multi_timeframe: MultiTimeframeConfig = MultiTimeframeConfig()
    algorithm_version: str = "3.0.0"

    @model_validator(mode="after")
    def history_covers_pivot_window(self) -> "SMCConfig":
        required = self.swing.left_window + self.swing.right_window + 1
        if self.processing.minimum_history < required:
            raise ValueError("minimum_history must cover the complete swing window")
        if self.displacement.strong_atr_impulse < self.displacement.minimum_atr_impulse:
            raise ValueError("strong displacement threshold cannot be below minimum")
        if self.dealing_range.ote_low_ratio >= self.dealing_range.ote_high_ratio:
            raise ValueError("OTE low ratio must be below high ratio")
        return self

    @property
    def version(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return sha256(payload.encode()).hexdigest()[:16]

    @property
    def swing_window(self) -> int:
        return max(self.swing.left_window, self.swing.right_window)

    @property
    def minimum_displacement_ratio(self) -> float:
        return self.structure.displacement_body_ratio

    @property
    def equal_level_tolerance(self) -> float:
        return self.swing.equal_level_tolerance
