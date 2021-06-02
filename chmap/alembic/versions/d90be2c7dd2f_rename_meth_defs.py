"""rename meth_defs

Revision ID: d90be2c7dd2f
Revises: 1f2e28f386b1
Create Date: 2020-08-03 12:15:31.637458

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = 'd90be2c7dd2f'
down_revision = '1f2e28f386b1'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if 'meth_defs' in tables:
        op.rename_table('meth_defs', 'method_defs')


def downgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if 'method_defs' in tables:
        op.rename_table('method_defs', 'meth_defs')
