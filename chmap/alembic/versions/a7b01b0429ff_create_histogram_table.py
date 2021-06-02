"""
create histogram table

Revision ID: a7b01b0429ff
Revises: 
Create Date: 2020-07-09 10:03:15.932674

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = 'a7b01b0429ff'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if 'histogram' not in tables:
        op.create_table(
            'histogram',
            sa.Column('hist_id', sa.Integer, primary_key=True),
            sa.Column('image_id', sa.Integer, sa.ForeignKey('euv_images.image_id')),
            sa.Column('meth_id', sa.Integer, sa.ForeignKey('meth_defs.meth_id')),
            sa.Column('date_obs', sa.DateTime),
            sa.Column('instrument', sa.String(10)),
            sa.Column('wavelength', sa.Integer),
            sa.Column('n_mu_bins', sa.Integer),
            sa.Column('n_intensity_bins', sa.Integer),
            sa.Column('lat_band', sa.Float),
            sa.Column('mu_bin_edges', sa.LargeBinary),
            sa.Column('intensity_bin_edges', sa.LargeBinary),
            sa.Column('hist', sa.LargeBinary),
        )


def downgrade():
    op.drop_table('histogram')
