from pathlib import Path

from backend.app.core.exceptions import ConfigurationError


class PromptLoader:
    """Loads immutable, versioned prompts by identifier."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or Path(__file__).parent

    def load(self, version: str) -> str:
        path = self.directory / f"{version}.txt"
        if not path.is_file():
            raise ConfigurationError(f"Prompt version not found: {version}")
        return path.read_text(encoding="utf-8")

