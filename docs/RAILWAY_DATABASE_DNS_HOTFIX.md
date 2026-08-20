# Railway database DNS hotfix

If Alembic reaches `asyncpg.connect()` but fails with `socket.gaierror: [Errno -2] Name or service not known`, the database URL exists but its hostname cannot be resolved from the deployment environment.

Railway private DNS is scoped to the current project environment and uses the service name: `<service-name>.railway.internal`.

## Required Railway configuration

On the TEN application service, configure the database variable as a **reference variable** to the PostgreSQL service instead of copying or hard-coding an internal hostname.

Preferred:

```text
TEN_DATABASE_URL=${{Postgres.DATABASE_URL}}
```

or:

```text
DATABASE_URL=${{Postgres.DATABASE_URL}}
```

Replace `Postgres` with the exact Railway PostgreSQL service name.

Both TEN and PostgreSQL must exist in the same Railway project environment. If the PostgreSQL service was renamed, recreated, or moved to another environment, update the reference variable instead of preserving an old `*.railway.internal` hostname.

`TEN_DATABASE_URL` has precedence when both variables are present.

TEN intentionally does not auto-fallback to `DATABASE_PUBLIC_URL` for migrations: silently switching networks could target the wrong database and apply schema changes outside the intended environment.
