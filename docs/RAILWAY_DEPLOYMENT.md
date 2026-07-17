# Railway deployment

The root `railway.toml` explicitly selects Railpack, starts `uvicorn backend.app.main:app` on Railway's `$PORT`, and checks `/health`. This prevents Railpack's “No start command detected” failure for TEN's package-based FastAPI layout.

TEN is a monorepo. Deploy the backend and frontend as separate Railway
services so each service has one build and runtime responsibility.

## Backend service

- Repository root: `/`
- Builder: Railpack
- Configuration source: `/railway.json`
- Start command:
  `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

Railpack detects Python from `pyproject.toml` and installs the dependencies
declared there. The explicit start command is required because the FastAPI
entry point is nested at `backend.app.main:app`, rather than a root-level
`main.py`.

Recommended variables:

```text
TEN_ENVIRONMENT=production
TEN_LOG_LEVEL=INFO
TEN_OPENROUTER_API_KEY=<secret>
TEN_CORS_ORIGINS=["https://<frontend-domain>"]
```

When PostgreSQL-backed repositories are enabled, add Railway PostgreSQL and
set `TEN_DATABASE_URL` to a `postgresql+asyncpg://` connection URL.

## Frontend service

- Root directory: `/frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Variable: `VITE_API_URL=https://<backend-domain>`

The current backend service does not serve the React bundle. Deploying only
the repository-root service exposes the REST API and OpenAPI documentation,
not the dashboard.

## Troubleshooting

`No start command detected` means Railway found Python but could not infer the
nested ASGI import. Confirm that the service is deploying the commit containing
`railway.json`, the config file path is `/railway.json`, and no dashboard start
command overrides it.

If Railway reports a Metal-builder infrastructure failure before Railpack
emits analysis output, retry the deployment. If the log includes Railpack's
`No start command detected` message, treat the start command as the primary
failure rather than the generic infrastructure diagnosis.
