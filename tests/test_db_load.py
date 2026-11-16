import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from tests.db_fixture import engine  # noqa: F401 - pytest fixture

from src import db_load
from src.data_validate import REJECT_COLUMNS
from src.schema import COLUMNS


def batch(order_ids=("o1", "o2"), country="MY"):
    """A tiny cleaned batch in the canonical string formats."""
    customers = pd.DataFrame([
        [country, "c1", "True", "2", "1", "0", "13.5", "1", "2021-03-23 09:16:29"],
        [country, "c2", "False", "0", "0", "0", "0", "100", ""],  # unknown first order
    ], columns=COLUMNS["customers"])
    orders = pd.DataFrame([[country, o, "Delivery", "payment on Delivery", "5.25", "4", "0.0", "2023/04/09"]
                           for o in order_ids], columns=COLUMNS["orders"])
    labels = pd.DataFrame([[country, o, "c1", "1"] for o in order_ids], columns=COLUMNS["labels"])
    rejects = pd.DataFrame([["orders", 7, "num_items_ordered missing", '{"order_id": "o9"}']],
                           columns=REJECT_COLUMNS)
    return {"customers": customers, "orders": orders, "labels": labels}, rejects


def rows(engine, sql):
    with engine.connect() as conn:
        return conn.execute(text(sql)).all()


def new(rows):
    return {"rows": rows, "inserted": rows, "existing": 0}


def test_upsert_sql_casts_types_and_updates_non_key_columns():
    sql = db_load.upsert_sql("orders")
    assert "CAST(s.num_items_ordered AS SIGNED)" in sql
    assert "CAST(s.order_value AS DECIMAL(18,6))" in sql
    assert "STR_TO_DATE(s.order_date, :date_fmt)" in sql
    updates = sql.split("ON DUPLICATE KEY UPDATE")[1]
    assert "`batch_id` = new.`batch_id`" in updates
    assert "`order_id`" not in updates and "`country_code`" not in updates  # keys are never updated


def test_load_batch(engine):
    tables, rejects = batch()
    stats = db_load.load_batch(engine, tables, rejects, "b1")
    assert stats == {"customers": new(2), "orders": new(2), "labels": new(2), "rejects": new(1)}
    c1, c2 = rows(engine, "SELECT mobile_verified, total_payment_last_50days, first_order_datetime, batch_id "
                          "FROM customers ORDER BY customer_id")
    assert (c1[0], float(c1[1]), str(c1[2]), c1[3]) == (1, 13.5, "2021-03-23 09:16:29", "b1")
    assert c2[0] == 0 and c2[2] is None
    assert [tuple(r) for r in rows(engine, "SELECT order_id, num_items_ordered, order_date FROM orders")] == [
        ("o1", 4, pd.Timestamp("2023-04-09").date()), ("o2", 4, pd.Timestamp("2023-04-09").date())]
    assert [tuple(r) for r in rows(engine, "SELECT batch_id, source_table, source_row, reason FROM rejects")] == [
        ("b1", "orders", 7, "num_items_ordered missing")]
    assert rows(engine, "SHOW TABLES LIKE 'stg_%'") == []  # staging tables cleaned up


def test_value_that_slipped_past_cleaning_rolls_back_whole_batch(engine):
    tables, rejects = batch()
    tables["labels"].loc[1, "is_fraud"] = "maybe"
    with pytest.raises(SQLAlchemyError):
        db_load.load_batch(engine, tables, rejects, "b1")
    for t in ["customers", "orders", "labels", "rejects"]:
        assert rows(engine, f"SELECT COUNT(*) FROM {t}")[0][0] == 0
    assert rows(engine, "SHOW TABLES LIKE 'stg_%'") == []


def test_commas_quotes_and_backslashes_survive_staging(engine):
    tables, rejects = batch()
    tables["orders"].loc[0, "payment_method"] = 'odd, "quoted" \\ method'
    db_load.load_batch(engine, tables, rejects, "b1")
    assert rows(engine, "SELECT payment_method FROM orders WHERE order_id = 'o1'")[0][0] == 'odd, "quoted" \\ method'


def test_missing_columns_are_reported(engine):
    tables, rejects = batch()
    with pytest.raises(ValueError, match="cleaned orders is missing columns"):
        db_load.load_batch(engine, {**tables, "orders": tables["orders"].drop(columns="order_date")}, rejects, "b1")


def test_run_batch_validates_cleans_and_loads(engine, tmp_path):
    tables, _ = batch(order_ids=[f"o{i}" for i in range(200)])  # big enough that 1 bad row is under 1%
    for t, df in tables.items():
        df.to_csv(tmp_path / f"{t}_b2.csv", index=False)
    with open(tmp_path / "orders_b2.csv", "a") as f:
        f.write("MY,broken\n")  # malformed: set aside by the file checks
    seen = {}

    def fake_clean(raw):
        seen.update(raw)
        return raw, pd.DataFrame([["labels", 1, "customer_id not in customers", "{}"]], columns=REJECT_COLUMNS)

    stats = db_load.run_batch(engine, tmp_path, "b2", clean=fake_clean)
    assert len(seen["orders"]) == 200
    assert stats == {"customers": new(2), "orders": new(200), "labels": new(200), "rejects": new(2)}
    assert sorted(r[0] for r in rows(engine, "SELECT reason FROM rejects")) == [
        "customer_id not in customers", "malformed row: 2 fields, expected 8"]


def counts(engine):
    return {t: rows(engine, f"SELECT COUNT(*) FROM {t}")[0][0] for t in ["customers", "orders", "labels", "rejects"]}


def test_loading_the_same_batch_twice_creates_no_duplicates(engine):
    tables, rejects = batch()
    db_load.load_batch(engine, tables, rejects, "b1")
    first = counts(engine)
    stats = db_load.load_batch(engine, tables, rejects, "b1")
    assert counts(engine) == first
    assert all(st["inserted"] == 0 and st["existing"] == st["rows"] for st in stats.values())


def test_later_batch_updates_existing_rows_and_adds_new_ones(engine):
    tables, rejects = batch(order_ids=["o1", "o2"])
    db_load.load_batch(engine, tables, rejects, "b1")

    tables, rejects = batch(order_ids=["o2", "o3"])
    tables["labels"]["is_fraud"] = "0"                  # o2 relabelled by the operations team
    tables["customers"].loc[0, "num_orders_last_50days"] = "9"  # newer customer snapshot
    stats = db_load.load_batch(engine, tables, rejects, "b2")

    assert stats["orders"] == {"rows": 2, "inserted": 1, "existing": 1}
    assert stats["rejects"] == new(1)  # same row number, but a different batch
    assert [tuple(r) for r in rows(engine, "SELECT order_id, is_fraud, batch_id FROM labels ORDER BY order_id")] == [
        ("o1", 1, "b1"), ("o2", 0, "b2"), ("o3", 0, "b2")]
    assert rows(engine, "SELECT num_orders_last_50days, batch_id FROM customers WHERE customer_id = 'c1'") == [(9, "b2")]


def test_same_order_id_in_another_country_is_a_different_order(engine):
    db_load.load_batch(engine, *batch(country="MY"), "b1")
    db_load.load_batch(engine, *batch(country="PH"), "b2")
    assert counts(engine)["orders"] == 4
