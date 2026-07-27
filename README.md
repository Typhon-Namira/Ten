# TEN

TEN is a ten-engine analytical market-intelligence platform. Full-system integration now connects production-configured Market Data through SMC, Liquidity, Volume Profile, Institutional Flow, Market Regime, Economic Calendar, AI Scoring, and Signal Decision, while Replay remains isolated. See [Full-System Integration Production 1.0](docs/FULL_SYSTEM_INTEGRATION.md) for contracts, API, security, worker topology, Railway configuration, and failure behavior.

Signals are analytical evidence summaries only. TEN does not place orders, manage positions, connect to brokerage execution, or promise trading outcomes.

TEN's Market Data Engine is the provider-neutral single source of truth for historical, realtime, and replayable market observations. It includes TwelveData, Alpha Vantage, Financial Modeling Prep, and OANDA adapters; health-ranked automatic failover; deterministic validation and quality scoring; memory/persistent caching; SQLAlchemy persistence; DST-aware sessions; metrics; events; and time-travel APIs. See [Market Data Engine](docs/MARKET_DATA_ENGINE.md).

The [SMC Engine Production 1.0](docs/SMC_ENGINE.md) provides deterministic no-lookahead structure, displacement, imbalance, institutional-zone lifecycles, dealing ranges, multi-timeframe context, durable replay, typed events, and structural features without generating trading decisions.

The [Liquidity Engine Production 1.0](docs/LIQUIDITY_ENGINE.md) owns inferred equal-level clusters, buy-side and sell-side pools, sweep classifications, session and period references, confluence, analytical target priority, replay-safe lifecycle history, and durable checkpoints. It consumes Market Data and the public SMC liquidity contract and never represents inferred liquidity as observed broker stops or order-book orders.

The [Volume Profile Engine Production 1.0](docs/VOLUME_PROFILE_ENGINE.md) owns deterministic candle-volume allocation across price, POC, Value Area, HVN/LVN, shelves, profile gaps, shapes, migration, period/anchored/composite profiles, tested references, replay, and durable checkpoints. Candle and tick volume limitations remain explicit; outputs are analytical observations rather than order-flow facts or trading instructions.

The [Market Regime Engine Production 1.0](docs/MARKET_REGIME_ENGINE.md) synthesizes time-valid Market Data, SMC, Liquidity, Volume Profile, and Institutional Flow outputs into deterministic and explainable regime context. It classifies conditions only and never emits trading instructions.

**TEN is an AI-assisted XAU/USD market analysis and signal-intelligence platform.** It combines independent price-action, liquidity, flow-estimation, volume-profile, macro-risk, and AI quality-scoring engines into explainable market scenarios presented through an internal dashboard.

TEN is not a trading bot. It has no broker connection, order execution, or Telegram integration. Signals are analytical scenarios—not financial advice.

## Foundation

- Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, async PostgreSQL
- React, TypeScript, Vite, responsive institutional-style dashboard
- Clean Architecture boundaries, dependency injection, typed engine ports
- Dynamically discovered, semantic-versioned engine registry and factory
- YAML-configurable pipeline, feature flags, event bus, and versioned feature store
- External plugin entry points for AI, market data, analysis, and notifications
- Deterministic confidence calculator; the LLM never determines confidence
- Four-account Groq provider pool with strict application-side structured validation
- Docker Compose for backend, frontend, and PostgreSQL
- pytest, Ruff, ESLint, TypeScript, and GitHub Actions quality gates

## Architecture

```mermaid
flowchart TD
  CFG[YAML + feature flags] --> REG[Dynamic engine registry]
  REG --> PM[Pipeline manager]
  MD[Provider-neutral market data] --> PM
  EC[Economic events] --> PM
  PM --> FS[(Versioned feature store)]
  PM --> EB[Typed event bus]
  FS --> AI[Deterministic AI Scoring]
  AI --> SD[Signal Decision safety policy]
  SD --> SE[Analytical scenario presentation]
  SE --> DB[(PostgreSQL-ready storage)]
  DB --> API[FastAPI REST API]
  API --> FE[React dashboard]
```

See [Architecture](docs/ARCHITECTURE.md), [Configuration](docs/CONFIGURATION.md), the [Engine Catalog](docs/ENGINES.md), [Engine Development](docs/ENGINE_DEVELOPMENT_GUIDE.md), and [Plugin Development](docs/PLUGIN_DEVELOPMENT.md).

## Quick start

### Local development

Requirements: Python 3.12+, Node.js 22+, and PostgreSQL 16 if persistence is enabled.

```bash
cp .env.example .env
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`; API documentation is at `http://localhost:8000/docs`.

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The dashboard is served at `http://localhost:5173`, the API at `http://localhost:8000`, and PostgreSQL at `localhost:5432`.

### Railway

The repository includes [`railway.json`](railway.json) and a production Dockerfile
for a single Railway service. The image builds the Vite dashboard, serves it from
FastAPI on the same public domain, and verifies `/health` before Railway marks the
deployment healthy.

1. Create a Railway service from `Typhon-Namira/Ten`.
2. Keep the service root directory at the repository root.
3. AI Scoring Production 1.0 is deterministic and requires no LLM or provider API key.
4. Add a Railway PostgreSQL service and set `TEN_DATABASE_URL` to its async
   SQLAlchemy URL, or to `${{Postgres.DATABASE_URL}}`, when durable persistence
   is enabled.
5. Set the Railway Pre-deploy Command to `python -m alembic upgrade head`.
6. Generate a public domain under the service's Networking settings.

The backend start command is:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

The React dashboard and API are served by this one process and one public domain.
No separate frontend service or `VITE_API_URL` is required for same-origin API
requests.

## Development workflow

```bash
pytest
ruff check backend tests
cd frontend && npm run lint && npm run build
```

The default local API can start without a database connection and uses in-memory signal, feature-store, and event-bus adapters so architecture work remains testable offline. Production schema changes are managed by Alembic; Railway runs `python -m alembic upgrade head` before application startup, and the application refuses to use an unversioned or outdated production schema.

## Add an engine

Create a package under `backend/app/engines` containing configuration, typed models, interface, implementation, and `registration.py`. The loader discovers it automatically; select its version and order in YAML. Keep provider DTOs outside the engine and never call another engine. Full guidance is in [ENGINE_DEVELOPMENT_GUIDE.md](docs/ENGINE_DEVELOPMENT_GUIDE.md).

## Add an AI model

TEN uses an ordered four-account Groq pool behind a provider-neutral AI boundary. Add an immutable prompt and schema version for behavior changes. Models see compact typed context only—never ORM objects, raw candles, engine objects, or chart images—and cannot bypass deterministic guardrails.

## Repository map

```text
backend/app/       API, registry, pipeline, events, features, plugins, engines, AI
frontend/src/      Routed dashboard modules, typed API client, hooks, components
configs/           Provider and engine configuration examples
data/              Local development data (ignored except marker)
docs/              Architecture, API, and engine development guidance
tests/             API and per-engine unit/validation tests
docker/            Production-oriented container definitions
.github/workflows/ Continuous integration
```

## Data integrity

The institutional-flow baseline is explicitly an **OHLCV estimate** using volume pressure, close location, acceleration, and effort-versus-result. It does not claim access to CME order flow. A future licensed provider can implement the same contract without changing downstream scoring or signal construction.
## Institutional Flow Engine

TEN includes a persistent, replay-safe Institutional Flow Engine that synthesizes typed Market Data, SMC, Liquidity, and Volume Profile evidence into explainable probabilistic inferences. It does not claim verified institutional identity and produces no trading instructions. See [docs/INSTITUTIONAL_FLOW_ENGINE.md](docs/INSTITUTIONAL_FLOW_ENGINE.md).

## Economic Calendar Engine

TEN includes a provider-neutral, persistent, revision-aware Economic Calendar Engine with deterministic event identity, explicit publication/availability semantics, point-in-time replay, instrument relevance, bounded event-risk context, typed Feature Store/Event Bus integration, and read-only APIs. It produces probabilistic context only and no trading instructions. See [docs/ECONOMIC_CALENDAR_ENGINE.md](docs/ECONOMIC_CALENDAR_ENGINE.md).

## AI Scoring Engine

TEN includes a deterministic, explainable, point-in-time-safe AI Scoring Engine that combines versioned upstream intelligence into separate direction, confidence, risk, alignment, quality, and composite dimensions. It uses no LLM in Production 1.0 and never executes trades or emits trading instructions. See [docs/AI_SCORING_ENGINE.md](docs/AI_SCORING_ENGINE.md).

## Signal Decision Engine

TEN includes a deterministic, fail-closed [Signal Decision Engine Production 1.0](docs/SIGNAL_DECISION_ENGINE.md). It loads trusted persisted AI Scoring snapshots and applies versioned integrity, freshness, evidence, risk, event, regime, conflict, dependency, duplicate, cooldown, reversal, and validity rules. Outputs are analytical states (`eligible`, `observe_only`, `blocked`, `insufficient_evidence`, `expired`, or `invalid`)—never orders or trading instructions.

## Replay Engine

TEN includes a deterministic, point-in-time-safe [Replay Engine Production 1.0](docs/REPLAY_ENGINE.md). It merges version-pinned historical sources in canonical order, advances an isolated virtual clock, checkpoints durable progress, supports distributed worker leases and safe recovery, and records semantic hashes for reproducibility comparisons. Replay reconstructs analytical behavior; it is not backtesting, P&L simulation, brokerage, or order execution.
