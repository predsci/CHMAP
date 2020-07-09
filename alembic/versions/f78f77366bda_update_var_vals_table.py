"""update Var Vals table

Revision ID: f78f77366bda
Revises: f016c47d344a
Create Date: 2020-07-09 10:13:57.967468

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f78f77366bda'
down_revision = 'f016c47d344a'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table('var_vals')
    op.create_table(
        'var_vals',
        sa.Column('combo_id', sa.Integer, sa.ForeignKey('image_combos.combo_id'), primary_key=True),
        sa.Column('meth_id', sa.Integer, sa.ForeignKey('meth_defs.meth_id')),
        sa.Column('var_id', sa.Integer, sa.ForeignKey('var_defs.var_id'), primary_key=True),
        sa.Column('var_val', sa.Float),
    )


def downgrade():
    op.add_column('var_vals', sa.Column('map_id', sa.Integer, sa.ForeignKey('euv_images.image_id'), primary_key=True))
