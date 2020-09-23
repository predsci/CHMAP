"""update histogram table

Revision ID: 50d8c4f5653e
Revises: d90be2c7dd2f
Create Date: 2020-09-22 15:59:36.064375

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = '50d8c4f5653e'
down_revision = 'd90be2c7dd2f'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    indices = inspector.get_indexes('histogram')
    index_list = [index['name'] for index in indices]
    if 'hist_index' not in index_list:
        op.create_index('hist_index', 'histogram', ["date_obs", "instrument", "wavelength"])
    op.alter_column('histogram', 'lat_band', type_=sa.Float, existing_type=sa.LargeBinary)


def downgrade():
    pass
