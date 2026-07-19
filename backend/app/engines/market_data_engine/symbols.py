from .models import canonical_symbol

_PROVIDER_SYMBOLS = {
    "twelve_data": {"XAUUSD": "XAU/USD"},
    "alpha_vantage": {"XAUUSD": "XAUUSD"},
    "financial_modeling_prep": {"XAUUSD": "XAUUSD"},
    "oanda": {"XAUUSD": "XAU_USD"},
}


def provider_symbol(provider: str, symbol: str) -> str:
    normalized = canonical_symbol(symbol)
    return _PROVIDER_SYMBOLS.get(provider, {}).get(normalized, normalized)
