from __future__ import annotations

from collections.abc import Callable, Coroutine
from enum import StrEnum
from hmac import compare_digest
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from backend.app.core.config import Settings, get_settings


class Role(StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


_RANK = {Role.VIEWER: 1, Role.OPERATOR: 2, Role.ADMIN: 3}


def require_role(minimum: Role) -> Callable[..., Coroutine[Any, Any, Role]]:
    async def dependency(x_ten_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings)) -> Role:
        if minimum == Role.VIEWER and settings.public_read_access:
            return Role.VIEWER
        role = next((Role(value) for key, value in settings.api_keys.items() if x_ten_api_key is not None and compare_digest(key, x_ten_api_key)), None)
        if role is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="valid TEN API key required")
        if _RANK[role] < _RANK[minimum]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient role")
        return role
    return dependency


require_viewer = require_role(Role.VIEWER)
require_admin = require_role(Role.ADMIN)

__all__ = ["Role", "require_admin", "require_role", "require_viewer"]

