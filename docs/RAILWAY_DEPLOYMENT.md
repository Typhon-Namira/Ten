# Railway deployment

TEN deploys as one Railway service and one public domain. Railway uses
`docker/backend.Dockerfile`, whose Node build stage installs the frontend
dependencies and runs the Vite production build. The resulting
`frontend/dist` directory is copied into the Python runtime image.

The repository has a pnpm lockfile but no npm package-lock, so the Docker build
uses `npm install` rather than an invalid `npm ci` invocation. The effective
frontend build commands are:

```text
cd frontend && npm install
cd frontend && npm run build
```

Railway does not override the Docker start command because Docker-service overrides
are executed without shell variable expansion. The image expands Railway's
`PORT` and then uses `exec`, leaving Uvicorn as the single production process:

```text
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

FastAPI serves `/assets` directly from `frontend/dist/assets` and returns
`frontend/dist/index.html` for allowlisted React routes. Browser navigation to
`/signals` is selected using `Accept: text/html`; JSON clients retain the
existing `/signals` API. Health, API, OpenAPI, Swagger, and ReDoc routes are
registered before the SPA fallback and are never shadowed.

Required Railway settings:

- configuration file: `/railway.json`;
- one service rooted at `/`;
- one generated public domain;
- health check: `/health`;
- no separate frontend service;
- production variables including `TEN_ENVIRONMENT=production`, database URL,
  API-key roles, provider secrets, and allowed CORS origins.

The frontend uses same-origin API requests when `VITE_API_URL` is absent or
empty. No frontend domain or backend secret is embedded in the Vite bundle.

Production startup fails with an explicit missing-dashboard error if
`frontend/dist/index.html` was not copied into the runtime image.
