"""create Var Vals Map table

Revision ID: f016c47d344a
Revises: a7b01b0429ff
Create Date: 2020-07-09 10:13:44.630376

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = 'f016c47d344a'
down_revision = 'a7b01b0429ff'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if 'var_vals_map' not in tables:
        op.create_table(
            'var_vals_map',
            sa.Column('map_id', sa.Integer, sa.ForeignKey('euv_images.image_id'), primary_key=True),
            sa.Column('combo_id', sa.Integer, sa.ForeignKey('image_combos.combo_id'), primary_key=True),
            sa.Column('meth_id', sa.Integer, sa.ForeignKey('meth_defs.meth_id')),
            sa.Column('var_id', sa.Integer, sa.ForeignKey('var_defs.var_id'), primary_key=True),
            sa.Column('var_val', sa.Float),
        )


def downgrade():
    op.drop_table('var_vals_map')
