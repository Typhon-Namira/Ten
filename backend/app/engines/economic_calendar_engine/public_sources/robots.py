"""Re-exported from the shared, engine-agnostic implementation — see `backend.app.core.net.robots`.
Kept as a module in this package so existing `from .robots import ...` call sites are unaffected.
"""

from __future__ import annotations

from backend.app.core.net.robots import RobotsPolicy, evaluate_robots_policy

__all__ = ["RobotsPolicy", "evaluate_robots_policy"]
