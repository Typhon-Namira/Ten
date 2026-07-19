from __future__ import annotations

import asyncio

from backend.app.integration import IntegrationWorker
from backend.app.main import app


async def main() -> None:
    async with app.router.lifespan_context(app):
        worker = IntegrationWorker(app.state.integration_service)
        try:
            await worker.run()
        finally:
            worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
