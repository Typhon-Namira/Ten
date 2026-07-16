from abc import ABC

from backend.app.engines.common import AnalysisEngine
from backend.app.engines.market_data_engine import Candle

from .config import VolumeProfileConfig
from .models import PriceNode, VolumeProfileResult


class VolumeProfileAnalyzer(AnalysisEngine[list[Candle], VolumeProfileResult], ABC):
    """Contract for session or composite volume profiles."""


class BaselineVolumeProfileAnalyzer(VolumeProfileAnalyzer):
    name = "volume_profile"
    version = "1.0.0"

    def __init__(self, config: VolumeProfileConfig | None = None) -> None:
        self.config = config or VolumeProfileConfig()

    def analyze(self, data: list[Candle]) -> VolumeProfileResult:
        if not data:
            return VolumeProfileResult(observations=["No candles supplied."])
        low = min(item.low for item in data)
        high = max(item.high for item in data)
        width = (high - low) / self.config.bins
        if width == 0:
            return VolumeProfileResult(poc=low, vah=low, val=low, total_volume=sum(item.volume for item in data))
        volumes = [0.0] * self.config.bins
        for candle in data:
            typical_price = (candle.high + candle.low + candle.close) / 3
            index = min(int((typical_price - low) / width), self.config.bins - 1)
            volumes[index] += candle.volume
        poc_index = max(range(len(volumes)), key=volumes.__getitem__)
        included = {poc_index}
        target = sum(volumes) * self.config.value_area_percent
        current = volumes[poc_index]
        while current < target and len(included) < len(volumes):
            candidates = [index for index in (min(included) - 1, max(included) + 1) if 0 <= index < len(volumes)]
            chosen = max(candidates, key=volumes.__getitem__)
            included.add(chosen)
            current += volumes[chosen]
        nonzero = sorted(value for value in volumes if value > 0)
        threshold = nonzero[int((len(nonzero) - 1) * self.config.high_volume_percentile)] if nonzero else 0
        nodes = [PriceNode(price=low + (index + 0.5) * width, volume=value, kind="HVN" if value >= threshold and value > 0 else "LVN") for index, value in enumerate(volumes)]
        return VolumeProfileResult(poc=nodes[poc_index].price, vah=nodes[max(included)].price, val=nodes[min(included)].price, total_volume=sum(volumes), nodes=nodes)

