"""Typed, versioned SMC Milestone 2A configuration."""

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


class SMCConfig(BaseModel):
    swing: SwingDetectionConfig = SwingDetectionConfig()
    structure: StructureConfig = StructureConfig()
    processing: ProcessingConfig = ProcessingConfig()
    algorithm_version: str = "2.0.0"

    @model_validator(mode="after")
    def history_covers_pivot_window(self) -> "SMCConfig":
        required = self.swing.left_window + self.swing.right_window + 1
        if self.processing.minimum_history < required:
            raise ValueError("minimum_history must cover the complete swing window")
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
