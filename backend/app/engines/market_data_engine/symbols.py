from .models import canonical_symbol

_PROVIDER_SYMBOLS = {
    "twelve_data": {"XAUUSD": "XAU/USD"},
    "alpha_vantage": {"XAUUSD": "XAUUSD"},
    # FMP's commodity endpoints use COMEX futures tickers, not FX-style pairs — "XAUUSD" 404s.
    # This mapping is FMP-specific; TEN's internal canonical symbol is never changed.
    "financial_modeling_prep": {"XAUUSD": "GCUSD"},
    "oanda": {"XAUUSD": "XAU_USD"},
}


def provider_symbol(provider: str, symbol: str) -> str:
    normalized = canonical_symbol(symbol)
    return _PROVIDER_SYMBOLS.get(provider, {}).get(normalized, normalized)
