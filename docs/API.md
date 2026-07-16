# HTTP API

The API is read-only in the initial foundation.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process liveness |
| GET | `/signals` | Recent generated scenarios |
| GET | `/signals/latest` | Newest scenario or 404 |
| GET | `/engines/status` | Registered engine readiness |
| GET | `/market/status` | Indicative UTC weekday/session status |

Interactive OpenAPI documentation is available at `/docs` when the backend is running.
