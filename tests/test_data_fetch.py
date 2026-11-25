import json

import pandas as pd
import pytest
from sqlalchemy import text

from src import data_fetch as df_
from src import db_load, db_schema
from src.data_validate import REJECT_COLUMNS
from src.schema import COLUMNS
from tests.db_fixture import engine  # noqa: F401 - pytest fixture

def add_batch(engine, batch, n, start=0, country="MY"):
    """Load n labelled orders o<start>.. through the real load step; every 4th is fraud.

    Customer c1 has no first order date.
    """
    customers = pd.DataFrame([
        [country, "c0", "True", "5", "0", "1", "120.5", "2", "2021-03-23 09:16:29"],
        [country, "c1", "False", "0", "0", "0", "0", "1", ""],
    ], columns=COLUMNS["customers"])
    orders = pd.DataFrame([[country, f"o{i}", "Delivery", "balance", "5.25", "2", "0.5", "2023/01/15"]
                           for i in range(start, start + n)], columns=COLUMNS["orders"])
    labels = pd.DataFrame([[country, f"o{i}", f"c{i % 2}", str(int(i % 4 == 0))] for i in range(start, start + n)],
                          columns=COLUMNS["labels"])
    db_load.load_batch(engine, {"customers": customers, "orders": orders, "labels": labels},
                       pd.DataFrame(columns=REJECT_COLUMNS), batch)


@pytest.fixture
def db(engine):
    db_schema.create_tables(engine)
    return engine


def test_split_is_deterministic_and_about_20_percent():
    keys = pd.DataFrame({"country_code": ["MY"] * 10_000, "order_id": [f"o{i}" for i in range(10_000)]})
    first, again = df_.is_test(keys), df_.is_test(keys.iloc[::-1].reset_index(drop=True))[::-1]
    assert (first == again).all()                        # depends only on the order, not row position
    assert 0.18 < first.mean() < 0.22
    other_country = keys.assign(country_code="PH")
    assert (df_.is_test(other_country) != first).any()  # the same order_id in another country is another order


def test_fetch_writes_typed_train_test_and_manifest(db, tmp_path):
    add_batch(db, "base", 200)
    with db.begin() as conn:  # an order without a label is not training data
        conn.execute(text("INSERT INTO orders (country_code, order_id, collect_type, payment_method, order_value, "
                          "num_items_ordered, refund_value, order_date, batch_id) "
                          "VALUES ('MY','nolabel','Delivery','balance',1,1,0,'2023-01-01','base')"))
    train, test, manifest = df_.run(tmp_path, engine=db)

    assert len(train) + len(test) == 200 and len(test) > 0
    assert set(train["order_id"]).isdisjoint(test["order_id"])
    assert manifest["batches"] == {"base": 200}
    assert manifest["order_date_min"] == manifest["order_date_max"] == "2023-01-15"
    assert manifest["train"]["rows"] == len(train) and manifest["test"]["rows"] == len(test)
    assert manifest["database"] == db.url.database

    loaded_train, loaded_test, loaded_manifest = df_.load_training_data(tmp_path)
    assert loaded_manifest == manifest
    pd.testing.assert_frame_equal(loaded_train.reset_index(drop=True), train.reset_index(drop=True))
    assert loaded_train["order_value"].dtype == "float64" and loaded_train["mobile_verified"].dtype == "bool"
    unknown = pd.concat([loaded_train, loaded_test]).query("customer_id == 'c1'")["first_order_datetime"]
    assert unknown.isna().all()  # missing first order date survives as NaT


def test_test_orders_stay_in_test_when_a_new_batch_arrives(db, tmp_path):
    add_batch(db, "base", 200)
    _, test1, _ = df_.run(tmp_path / "run1", engine=db)
    add_batch(db, "b2", 200, start=200)
    train2, test2, manifest = df_.run(tmp_path / "run2", engine=db)
    assert set(test1["order_id"]) <= set(test2["order_id"])
    assert set(test1["order_id"]).isdisjoint(train2["order_id"])
    assert manifest["batches"] == {"b2": 200, "base": 200}


def test_same_data_gives_same_hash(db, tmp_path):
    add_batch(db, "base", 200)
    a = df_.run(tmp_path / "a", engine=db)[2]
    b = df_.run(tmp_path / "b", engine=db)[2]
    assert a["data_hash"] == b["data_hash"]


def test_empty_database_gives_clear_error(db, tmp_path):
    with pytest.raises(RuntimeError, match="no labelled orders in the database"):
        df_.run(tmp_path, engine=db)


def test_load_training_data_detects_edited_or_missing_files(db, tmp_path):
    with pytest.raises(FileNotFoundError, match="run python -m src.data_fetch first"):
        df_.load_training_data(tmp_path)
    add_batch(db, "base", 200)
    df_.run(tmp_path, engine=db)
    with open(tmp_path / "test.csv", "a") as f:
        f.write("tampered\n")
    with pytest.raises(RuntimeError, match="test.csv does not match manifest.json"):
        df_.load_training_data(tmp_path)
    (tmp_path / "train.csv").unlink()
    with pytest.raises(FileNotFoundError, match="train.csv missing"):
        df_.load_training_data(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["rows"] == 200


def test_query_and_dtypes_follow_the_shared_table_design():
    assert df_.OUTPUT_COLUMNS[:3] == ["country_code", "order_id", "customer_id"]
    assert df_.OUTPUT_COLUMNS[-2:] == ["batch_id", "is_fraud"]
    for table in ("orders", "customers"):
        assert set(COLUMNS[table]) <= set(df_.OUTPUT_COLUMNS)
    assert set(df_.DTYPES) | set(df_.DATETIME_COLUMNS) == set(df_.OUTPUT_COLUMNS)
    assert df_.DTYPES["is_fraud"] == "int8" and df_.DTYPES["mobile_verified"] == "bool"


# --- incremental fetch -------------------------------------------------------------------------

def load(engine, batch, orders, customers, labels=None):
    """Load a batch. orders: [(order_id, customer_id, is_fraud)]; customers: [(customer_id, num_orders)]."""
    cust = pd.DataFrame([["MY", c, "True", str(n), "0", "0", "10.5", "1", "2021-03-23 09:16:29"] for c, n in customers],
                        columns=COLUMNS["customers"])
    ords = pd.DataFrame([["MY", o, "Delivery", "balance", "5.25", "2", "0.5", "2023/01/15"] for o, _, _ in orders],
                        columns=COLUMNS["orders"])
    labs = pd.DataFrame([["MY", o, c, str(f)] for o, c, f in (labels or orders)], columns=COLUMNS["labels"])
    db_load.load_batch(engine, {"customers": cust, "orders": ords, "labels": labs},
                       pd.DataFrame(columns=REJECT_COLUMNS), batch)


BASE = [(f"o{i:03d}", f"k{i % 20}", int(i % 4 == 0)) for i in range(200)]
BASE_CUSTOMERS = [(f"k{i}", 5) for i in range(20)]


def hashes(manifest):
    return manifest["train"]["hash"], manifest["test"]["hash"]


def test_incremental_fetches_only_the_new_batch_and_matches_a_full_fetch(db, tmp_path):
    load(db, "base", BASE, BASE_CUSTOMERS)
    first = df_.run(tmp_path / "snap", engine=db)[2]
    assert first["fetch"]["mode"] == "full" and "no usable snapshot" in first["fetch"]["reason"]

    load(db, "b2", [(f"n{i:02d}", "new0", int(i % 3 == 0)) for i in range(30)], [("new0", 1)])
    inc = df_.run(tmp_path / "snap", engine=db)[2]
    full = df_.run(tmp_path / "full", engine=db, full=True)[2]

    assert inc["fetch"] == {"mode": "incremental", "batches_fetched": ["b2"], "rows_from_database": 30,
                            "rows_from_snapshot": 200}
    assert hashes(inc) == hashes(full) and inc["data_hash"] == full["data_hash"]   # byte-identical files
    assert inc["batches"] == {"base": 200, "b2": 30}


def test_nothing_new_means_nothing_fetched(db, tmp_path):
    load(db, "base", BASE, BASE_CUSTOMERS)
    first = df_.run(tmp_path, engine=db)[2]
    again = df_.run(tmp_path, engine=db)[2]
    assert again["fetch"]["rows_from_database"] == 0 and again["fetch"]["batches_fetched"] == []
    assert hashes(again) == hashes(first)


def test_customer_updated_by_a_later_batch_refreshes_their_old_orders(db, tmp_path):
    load(db, "base", BASE, BASE_CUSTOMERS)
    df_.run(tmp_path / "snap", engine=db)
    load(db, "b2", [("n01", "k3", 1)], [("k3", 99)])          # k3's snapshot changes: 99 orders now
    inc_train, inc_test, inc = df_.run(tmp_path / "snap", engine=db)
    full = df_.run(tmp_path / "full", engine=db, full=True)[2]
    assert inc["fetch"]["mode"] == "incremental" and inc["fetch"]["rows_from_database"] == 11  # k3's 10 + n01
    assert hashes(inc) == hashes(full)
    both = pd.concat([inc_train, inc_test])
    assert (both.loc[both["customer_id"] == "k3", "num_orders_last_50days"] == 99).all()


def test_relabel_inside_an_old_batch_is_caught_and_falls_back_to_full(db, tmp_path):
    load(db, "base", BASE, BASE_CUSTOMERS)
    df_.run(tmp_path, engine=db)
    with db.begin() as conn:  # a label changes without a new batch_id: counts no longer match
        conn.execute(text("UPDATE labels SET is_fraud = 1 - is_fraud WHERE order_id = 'o001'"))
    m = df_.run(tmp_path, engine=db)[2]
    assert m["fetch"]["mode"] == "full" and "don't match the database" in m["fetch"]["reason"]


def test_refresh_rereads_the_batch_being_processed(db, tmp_path):
    load(db, "base", BASE, BASE_CUSTOMERS)
    load(db, "b2", [(f"n{i:02d}", "new0", 0) for i in range(10)], [("new0", 1)])
    df_.run(tmp_path, engine=db)
    m = df_.run(tmp_path, engine=db, refresh=["b2"])[2]
    assert m["fetch"]["batches_fetched"] == ["b2"] and m["fetch"]["rows_from_database"] == 10


def test_snapshot_from_another_database_is_not_reused(db, tmp_path):
    load(db, "base", BASE, BASE_CUSTOMERS)
    df_.run(tmp_path, engine=db)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["host"] = "some-other-server"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    m = df_.run(tmp_path, engine=db)[2]
    assert m["fetch"]["mode"] == "full" and "some-other-server" in m["fetch"]["reason"]


def test_full_fetch_flag_and_environment(db, tmp_path, monkeypatch):
    load(db, "base", BASE, BASE_CUSTOMERS)
    df_.run(tmp_path, engine=db)
    assert df_.run(tmp_path, engine=db, full=True)[2]["fetch"]["reason"] == "full fetch requested"
    monkeypatch.setenv("FETCH_MODE", "full")
    assert df_.run(tmp_path, engine=db)[2]["fetch"]["mode"] == "full"
    monkeypatch.setenv("FETCH_MODE", "incremental")
    assert df_.run(tmp_path, engine=db)[2]["fetch"]["mode"] == "incremental"
