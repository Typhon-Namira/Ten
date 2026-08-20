from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import httpx


BASE_URL = "https://freeserv.dukascopy.com/2.0/"


def request(params: dict[str, object]) -> object:
    key = os.getenv("DUKASCOPY_API_KEY")

    if key:
        params["key"] = key

    response = httpx.get(
        BASE_URL,
        params=params,
        timeout=30.0,
    )

    response.raise_for_status()
    return response.json()


def normalized_name(item: dict[str, object]) -> str:
    return (
        str(item.get("name", ""))
        .replace("/", "")
        .replace("_", "")
        .replace("-", "")
        .upper()
    )


def main() -> None:
    instruments = request(
        {
            "path": "api/instrumentList",
            "fields": "id,name,nameLong,pipValue",
        }
    )

    print("instrument response type:", type(instruments).__name__)

    if not isinstance(instruments, list):
        print(json.dumps(instruments, indent=2)[:5000])
        raise SystemExit("Unexpected instrumentList schema")

    xau_matches = [
        item
        for item in instruments
        if isinstance(item, dict)
        and (
            normalized_name(item) == "XAUUSD"
            or "GOLD" in str(item.get("nameLong", "")).upper()
        )
    ]

    print()
    print("XAU matches:")
    print(json.dumps(xau_matches, indent=2))

    if not xau_matches:
        raise SystemExit("Could not find XAU/USD instrument")

    instrument = xau_matches[0]
    instrument_id = instrument["id"]

    end = datetime.now(UTC) - timedelta(days=1)
    start = end - timedelta(hours=3)

    candles = request(
        {
            "path": "api/historicalPrices",
            "instrument": instrument_id,
            "timeFrame": "1min",
            "count": 20,
            "start": int(start.timestamp() * 1000),
            "end": int(end.timestamp() * 1000),
            "dayStartTime": "UTC",
            "offerSide": "B",
        }
    )

    print()
    print("Historical response type:", type(candles).__name__)
    print()
    print("Historical sample:")
    print(json.dumps(candles, indent=2)[:10000])


if __name__ == "__main__":
    main()
