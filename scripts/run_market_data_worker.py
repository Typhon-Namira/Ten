from __future__ import annotations

import asyncio

from backend.app.engines.market_data_engine import Timeframe
from backend.app.main import app


async def main() -> None:
    async with app.router.lifespan_context(app):
        config = app.state.integration_service.config
        while True:
            for instrument in config.instruments:
                for timeframe in instrument.timeframes:
                    await app.state.market_data_service.latest(instrument.instrument_id, Timeframe(timeframe), refresh=True)
            await asyncio.sleep(config.limits.outbox_poll_seconds)


if __name__ == "__main__":
    asyncio.run(main())
