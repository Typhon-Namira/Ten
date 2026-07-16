from pydantic import BaseModel, Field


class AIScoringConfig(BaseModel):
    model: str = "meta-llama/llama-3.3-70b-instruct"
    prompt_version: str = "signal_analysis_v1"
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=900, ge=100)

