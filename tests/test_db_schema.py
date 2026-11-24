import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from tests.db_fixture import engine  # noqa: F401 - pytest fixture

from src import db_schema
from src.schema import COLUMNS


def test_columns_match_schema():
    db_schema.check_columns()


def test_create_tables(engine):
    db_schema.create_tables(engine)
    insp = inspect(engine)
    assert sorted(insp.get_table_names()) == ["customers", "labels", "orders", "rejects"]
    for t, cols in COLUMNS.items():
        assert [c["name"] for c in insp.get_columns(t)] == cols + ["batch_id", "loaded_at"]
        assert insp.get_pk_constraint(t)["constrained_columns"] == db_schema.PRIMARY_KEYS[t]
    assert {fk["referred_table"] for fk in insp.get_foreign_keys("labels")} == {"orders", "customers"}


def test_create_is_repeatable_and_keeps_data(engine):
    db_schema.create_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO rejects (batch_id, source_table, source_row, reason, raw_values) "
                          "VALUES ('b1', 'orders', 1, 'r', '[]')"))
    db_schema.create_tables(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM rejects")).scalar_one() == 1
    db_schema.create_tables(engine, reset=True)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM rejects")).scalar_one() == 0


def test_keys_block_duplicates_and_orphans(engine):
    db_schema.create_tables(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO orders VALUES ('MY','o1','Delivery','balance',1,1,0,'2023-01-01','b1',DEFAULT),"
                          "('PH','o1','Delivery','balance',1,1,0,'2023-01-01','b1',DEFAULT)"))  # same id, 2 countries
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("INSERT INTO orders VALUES ('MY','o1','Delivery','balance',1,1,0,'2023-01-01','b1',DEFAULT)"))
    with pytest.raises(IntegrityError), engine.begin() as conn:  # label for a customer that doesn't exist
        conn.execute(text("INSERT INTO labels VALUES ('MY','o1','nobody0000',1,'b1',DEFAULT)"))
    reject = ("INSERT INTO rejects (batch_id, source_table, source_row, reason, raw_values) "
              "VALUES ('b1','orders',5,'order_value missing','[]')")
    with engine.begin() as conn:
        conn.execute(text(reject))
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text(reject))
