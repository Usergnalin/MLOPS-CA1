"""Tests for src/data_clean.py, mainly its hand-off to the database load (for_load)."""
import json

import pandas as pd
from sqlalchemy import text

from tests.db_fixture import engine  # noqa: F401 - pytest fixture

from src import data_clean, db_load
from src.data_validate import REJECT_COLUMNS
from src.schema import COLUMNS


def raw_batch():
    """Raw tables as data_validate.read_batch gives them: strings only. One bad row per table."""
    customers = pd.DataFrame([
        ["my", "AAAAAAAAA1", "yes", "2", "1", "0", "13.5", "1", "2021-03-23 09:16:29"],
        ["MY", "aaaaaaaaa2", "False", "0", "0", "0", "0", "100", ""],          # unknown first order
        ["MY", "aaaaaaaaa3", "maybe", "0", "0", "0", "0", "1", ""],            # bad mobile_verified
    ], columns=COLUMNS["customers"])
    orders = pd.DataFrame([
        ["MY", "bbbbbbbbb1", "delivery", "Balance", "5.25", "4", "0", "2023-04-09"],
        ["MY", "bbbbbbbbb2", "Pickup", "", "7", "1", "0", "2023/Apr/10"],    # blank payment -> unknown
        ["MY", "bbbbbbbbb3", "Pickup", "balance", "-1", "1", "0", "2023-04-11"],  # negative value
    ], columns=COLUMNS["orders"])
    labels = pd.DataFrame([
        ["MY", "bbbbbbbbb1", "aaaaaaaaa1", "1"],
        ["MY", "bbbbbbbbb2", "aaaaaaaaa2", "0"],
        ["MY", "bbbbbbbbb3", "aaaaaaaaa2", "0"],                            # its order was rejected
    ], columns=COLUMNS["labels"])
    return {"customers": customers, "orders": orders, "labels": labels}


def test_for_load_gives_canonical_strings_in_schema_columns():
    cleaned, _ = data_clean.for_load(raw_batch())

    for table, df in cleaned.items():
        assert list(df.columns) == COLUMNS[table]
        assert all(isinstance(v, str) for v in df.to_numpy().ravel())
    c, o, l = cleaned["customers"], cleaned["orders"], cleaned["labels"]
    assert c["mobile_verified"].tolist() == ["True", "False"]           # db_load casts 'True' -> 1
    assert c["first_order_datetime"].tolist() == ["2021-03-23 09:16:29", ""]
    assert c["customer_id"].tolist() == ["aaaaaaaaa1", "aaaaaaaaa2"]
    assert c["country_code"].tolist() == ["MY", "MY"]
    assert o["order_date"].tolist() == ["2023/04/09", "2023/04/10"]   # schema.ORDER_DATE_FORMAT
    assert o["payment_method"].tolist() == ["balance", "unknown"]
    assert l["is_fraud"].tolist() == ["1", "0"]


def test_for_load_rejects_match_the_load_format():
    _, rejects = data_clean.for_load(raw_batch())

    assert list(rejects.columns) == REJECT_COLUMNS
    by_table = rejects.set_index("table")
    assert by_table.loc["customers", "row"] == 3
    assert "mobile_verified" in by_table.loc["customers", "reason"]
    assert "order_value below 0" in by_table.loc["orders", "reason"]
    assert "no matching clean order" in by_table.loc["labels", "reason"]
    assert json.loads(by_table.loc["orders", "raw"])["order_value"] == "-1"


def test_load_step_uses_it_by_default():
    cleaned, rejects = db_load.default_clean(raw_batch())
    db_load.check_input(cleaned, rejects)  # raises if a column the load needs is missing
    assert len(cleaned["orders"]) == 2 and len(rejects) == 3


def test_real_cleaning_loads_into_mysql(engine, tmp_path):  # skipped unless TEST_DB_URL is set
    for table, df in raw_batch().items():
        df.to_csv(tmp_path / f"{table}_b1.csv", index=False)

    stats = db_load.run_batch(engine, tmp_path, "b1")  # default clean = data_clean.for_load

    assert {t: s["inserted"] for t, s in stats.items()} == {"customers": 2, "orders": 2, "labels": 2, "rejects": 3}
    with engine.connect() as conn:
        verified = dict(conn.execute(text("SELECT customer_id, mobile_verified FROM customers")).all())
        dates = [str(d) for (d,) in conn.execute(text("SELECT order_date FROM orders ORDER BY order_id"))]
    assert verified == {"aaaaaaaaa1": 1, "aaaaaaaaa2": 0}
    assert dates == ["2023-04-09", "2023-04-10"]
