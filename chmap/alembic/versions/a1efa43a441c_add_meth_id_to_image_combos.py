"""Add meth_id to Image_Combos

Revision ID: a1efa43a441c
Revises: f78f77366bda
Create Date: 2020-07-10 09:14:02.605460

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision = 'a1efa43a441c'
down_revision = 'f78f77366bda'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table('image_combos')
    op.create_table(
        'image_combos',
        sa.Column('combo_id', sa.Integer, primary_key=True),
        sa.Column('meth_id', sa.Integer, sa.ForeignKey('meth_defs.meth_id')),
        sa.Column('n_images', sa.Integer),
        sa.Column('date_mean', sa.DateTime),
        sa.Column('date_max', sa.DateTime),
        sa.Column('date_min', sa.DateTime),
    )


def downgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = inspector.get_columns('image_combos')
    column_list = [column['name'] for column in columns]
    print(column_list)
