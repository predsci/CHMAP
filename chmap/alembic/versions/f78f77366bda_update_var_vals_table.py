"""update Var Vals table

Revision ID: f78f77366bda
Revises: f016c47d344a
Create Date: 2020-07-09 10:13:57.967468

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = 'f78f77366bda'
down_revision = 'f016c47d344a'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    tables = inspector.get_table_names()
    if 'var_vals' in tables:
        op.drop_table('var_vals')
    if 'var_vals' not in tables:
        op.create_table(
            'var_vals',
            sa.Column('combo_id', sa.Integer, sa.ForeignKey('image_combos.combo_id'), primary_key=True),
            sa.Column('meth_id', sa.Integer, sa.ForeignKey('meth_defs.meth_id')),
            sa.Column('var_id', sa.Integer, sa.ForeignKey('var_defs.var_id'), primary_key=True),
            sa.Column('var_val', sa.Float),
        )


def downgrade():
    pass
