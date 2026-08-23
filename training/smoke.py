from __future__ import annotations

import lightning as L
import pandas as pd
import pyarrow as pa
import torch

from backend.app.future_market.models import (
    FORECAST_CADENCE_SECONDS,
    FORECAST_HORIZON_SECONDS,
)

M5_SECONDS = 5 * 60

assert FORECAST_HORIZON_SECONDS == 30 * 60
assert FORECAST_CADENCE_SECONDS == M5_SECONDS
assert FORECAST_HORIZON_SECONDS // M5_SECONDS == 6

context_bars = 96
feature_count = 12
future_bars = FORECAST_HORIZON_SECONDS // M5_SECONDS

x = torch.randn(8, context_bars, feature_count)
y = torch.randn(8, future_bars)

print("TEN training smoke test")
print("-----------------------")
print("Torch:", torch.__version__)
print("Lightning:", L.__version__)
print("Pandas:", pd.__version__)
print("PyArrow:", pa.__version__)
print("CUDA:", torch.cuda.is_available())
print()
print("Input shape:", tuple(x.shape))
print("Future path shape:", tuple(y.shape))
print("Forecast horizon:", FORECAST_HORIZON_SECONDS, "seconds")
print("Forecast cadence:", FORECAST_CADENCE_SECONDS, "seconds")
print("Future M5 bars:", future_bars)

assert x.shape == (8, 96, 12)
assert y.shape == (8, 6)

print()
print("OK: TEN world-model training environment is ready.")
