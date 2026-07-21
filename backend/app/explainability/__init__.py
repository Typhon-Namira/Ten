"""AI Explainability layer: observes TEN's own engines and explains them in natural language.

Architectural boundary (see models.py/context.py/service.py docstrings for the how): this module
never decides anything, never feeds back into the pipeline, and never invents data. It reads
already-persisted, already-decided state from every engine and hands a strictly-grounded JSON
context to an LLM whose only job is prose synthesis over facts computed in plain Python — the
facts themselves (which engine blocked a decision, what a score's components were, what changed
since the previous candle) are never the model's own judgment.
"""

from .models import ChatTurn, EngineFact, Evidence, Explanation, ExplainabilityContext
from .service import ExplainabilityService

__all__ = ["ChatTurn", "EngineFact", "Evidence", "Explanation", "ExplainabilityContext", "ExplainabilityService"]
