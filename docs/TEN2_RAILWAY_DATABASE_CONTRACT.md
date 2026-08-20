# TEN 2.0 Railway database contract

TEN accepts either of the following database variables in managed deployments:

- `TEN_DATABASE_URL` — canonical TEN variable.
- `DATABASE_URL` — Railway/PostgreSQL compatibility fallback.

`TEN_DATABASE_URL` always has precedence when both are configured.

The Alembic pre-deploy migration and the runtime container use the same precedence rule. Native Railway URLs such as `postgresql://...` or `postgres://...` are normalized to SQLAlchemy's async `postgresql+asyncpg://...` dialect before use.

This compatibility layer prevents a deploy from failing before application startup when Railway exposes only `DATABASE_URL`.
