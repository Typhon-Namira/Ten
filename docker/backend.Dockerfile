FROM node:22-slim AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
# There is no npm package-lock, so npm install is intentionally used instead of npm ci.
RUN cd frontend && npm install
COPY frontend ./frontend
RUN cd frontend && npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY configs ./configs
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
EXPOSE 8000
CMD ["sh", "-c", "access_flag='--no-access-log'; [ \"${TEN_LOG_ACCESS_REQUESTS:-false}\" = 'true' ] && access_flag=''; exec uvicorn backend.app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" $access_flag"]

