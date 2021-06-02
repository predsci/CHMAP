"""delete map image assoc table

Revision ID: 1f2e28f386b1
Revises: a1efa43a441c
Create Date: 2020-07-22 13:26:03.224366

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = '1f2e28f386b1'
down_revision = 'a1efa43a441c'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if 'map_image_assoc' in tables:
        op.drop_table('map_image_assoc')


def downgrade():
    pass
