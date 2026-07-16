from abc import ABC, abstractmethod

from backend.app.ai.openrouter_client.client import OpenRouterClient
from backend.app.ai.prompts.loader import PromptLoader

from .config import AIScoringConfig
from .models import ScoringContext, SignalScore


class AIScoringEngine(ABC):
    """Async contract for interchangeable structured-data scoring models."""

    @abstractmethod
    async def score(self, context: ScoringContext) -> SignalScore:
        """Score engine outputs; raw chart images are intentionally unsupported."""


class OpenRouterScoringEngine(AIScoringEngine):
    def __init__(self, client: OpenRouterClient, prompts: PromptLoader, config: AIScoringConfig | None = None) -> None:
        self.client = client
        self.prompts = prompts
        self.config = config or AIScoringConfig()

    async def score(self, context: ScoringContext) -> SignalScore:
        prompt = self.prompts.load(self.config.prompt_version)
        response = await self.client.complete_json(system_prompt=prompt, payload=context.model_dump(mode="json"), model=self.config.model, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
        response.setdefault("model", self.config.model)
        response.setdefault("prompt_version", self.config.prompt_version)
        return SignalScore.model_validate(response)

