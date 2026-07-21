"""Engine-agnostic network-safety helpers shared by every public, keyless HTTP/HTML source
adapter across TEN (economic calendar public sources, market data public sources, ...).

Keep this package free of any engine-specific policy (domain allowlists, user agents) — those
belong in the engine that owns them and get passed in as parameters.
"""

from .robots import RobotsPolicy, evaluate_robots_policy
from .ssrf import ALLOWED_SCHEMES, UnsafePublicUrlError, assert_safe_public_url, is_private_or_reserved, resolved_addresses_are_safe

__all__ = [
    "ALLOWED_SCHEMES",
    "RobotsPolicy",
    "UnsafePublicUrlError",
    "assert_safe_public_url",
    "evaluate_robots_policy",
    "is_private_or_reserved",
    "resolved_addresses_are_safe",
]
