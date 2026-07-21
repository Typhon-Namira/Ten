from .models import canonical_symbol

_PROVIDER_SYMBOLS = {
    "twelve_data": {"XAUUSD": "XAU/USD"},
    "alpha_vantage": {"XAUUSD": "XAUUSD"},
    # FMP's commodity endpoints use COMEX futures tickers, not FX-style pairs — "XAUUSD" 404s.
    # This mapping is FMP-specific; TEN's internal canonical symbol is never changed.
    "financial_modeling_prep": {"XAUUSD": "GCUSD"},
    "oanda": {"XAUUSD": "XAU_USD"},
    # LBMA has no ticker at all — its endpoint returns exactly one series (the gold fix), so no
    # symbol mapping is needed; present here only for interface consistency.
    "lbma_gold_price": {"XAUUSD": "XAUUSD"},
    # Gold-token proxy instruments, not true spot XAU/USD — see the adapter docstrings.
    "kraken": {"XAUUSD": "PAXGUSD"},
    "okx": {"XAUUSD": "XAUT-USDT"},
    # Disabled-by-default legacy adapters (robots.txt-blocked, see adapters.py).
    "yahoo_finance": {"XAUUSD": "GC=F"},
    "stooq": {"XAUUSD": "xauusd"},
    "binance": {"XAUUSD": "PAXGUSDT"},
}


def provider_symbol(provider: str, symbol: str) -> str:
    normalized = canonical_symbol(symbol)
    return _PROVIDER_SYMBOLS.get(provider, {}).get(normalized, normalized)
