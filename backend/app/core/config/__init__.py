from .settings import Settings
from .yaml import YamlConfigRepository


def get_settings():
    from .settings import get_settings as _get_settings

    return _get_settings()


__all__ = ["Settings", "YamlConfigRepository", "get_settings"]
