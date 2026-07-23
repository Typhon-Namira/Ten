"""Lease integration outbox work across concurrent service replicas."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0006"
down_revision: str | None = "20260723_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("integration_outbox", sa.Column("claimed_by", sa.String(64), nullable=True))
    op.add_column("integration_outbox", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_integration_outbox_claimed_by", "integration_outbox", ["claimed_by"])
    op.create_index("ix_integration_outbox_lease_expires_at", "integration_outbox", ["lease_expires_at"])
    op.create_index(
        "ix_integration_outbox_delivery_claim",
        "integration_outbox",
        ["published_at", "available_at", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_integration_outbox_delivery_claim", table_name="integration_outbox")
    op.drop_index("ix_integration_outbox_lease_expires_at", table_name="integration_outbox")
    op.drop_index("ix_integration_outbox_claimed_by", table_name="integration_outbox")
    op.drop_column("integration_outbox", "lease_expires_at")
    op.drop_column("integration_outbox", "claimed_by")
