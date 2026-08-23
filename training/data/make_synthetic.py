from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bars", type=int, default=5000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        "2025-01-01T00:00:00Z",
        periods=args.bars,
        freq="5min",
    )

    returns = rng.normal(0.0, 0.00035, size=args.bars)

    close = 2600.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([close[0]], close[:-1]))

    intrabar = np.abs(rng.normal(0.00035, 0.00015, size=args.bars))

    high = np.maximum(open_, close) * (1.0 + intrabar)
    low = np.minimum(open_, close) * (1.0 - intrabar)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    print("created:", args.output)
    print("bars:", len(frame))


if __name__ == "__main__":
    main()
