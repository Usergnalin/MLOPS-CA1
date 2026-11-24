"""Tests for src/models/decision_tree.py. They use a local SQLite MLflow store, not the team server."""
import json

import numpy as np
import pandas as pd
import pytest

from src import data_fetch
from src.models import decision_tree as dt


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.chdir(tmp_path)  # keep artifacts out of the repo


def write_training_dir(path, n=600, seed=0):
    """A small training set in the data_fetch format; fraud depends on num_associated_customers."""
    rng = np.random.default_rng(seed)
    assoc = rng.integers(1, 10, n)
    df = pd.DataFrame({
        "country_code": rng.choice(["MY", "PH", "PK"], n), "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)], "collect_type": "Delivery",
        "payment_method": rng.choice(["balance", "credit card"], n), "order_value": rng.uniform(1, 20, n),
        "num_items_ordered": rng.integers(1, 5, n), "refund_value": rng.uniform(0, 5, n),
        "order_date": pd.Timestamp("2023-01-15"), "mobile_verified": rng.random(n) > 0.3,
        "num_orders_last_50days": rng.integers(0, 50, n), "num_cancelled_orders_last_50days": rng.integers(0, 3, n),
        "num_refund_orders_last_50days": rng.integers(0, 3, n), "total_payment_last_50days": rng.uniform(0, 500, n),
        "num_associated_customers": assoc, "first_order_datetime": pd.Timestamp("2021-03-23 09:16:29"),
        "batch_id": "base", "is_fraud": ((assoc > 6) ^ (rng.random(n) < 0.05)).astype(int),
    })
    df = data_fetch.set_types(df)
    test_mask = data_fetch.is_test(df)
    train, test = df[~test_mask], df[test_mask]
    path.mkdir(parents=True, exist_ok=True)
    th = data_fetch.write_csv(train, path / data_fetch.TRAIN_FILE)
    te = data_fetch.write_csv(test, path / data_fetch.TEST_FILE)
    manifest = {"batches": {"base": n}, "order_date_min": "2023-01-15", "order_date_max": "2023-01-15",
                "train": data_fetch.describe(train, th), "test": data_fetch.describe(test, te),
                "data_hash": f"hash-{seed}"}
    (path / data_fetch.MANIFEST_FILE).write_text(json.dumps(manifest))
    return manifest


def test_features_leave_out_refund_value_ids_and_dates():
    assert "refund_value" not in dt.FEATURES and "refund_value" in dt.EXCLUDED
    assert not {"order_id", "customer_id", "batch_id", "order_date", "first_order_datetime", "is_fraud"} & set(dt.FEATURES)


def test_run_logs_a_baseline_that_other_models_can_find(local_store, tmp_path):
    manifest = write_training_dir(tmp_path / "training")
    metrics, result = dt.run(tmp_path / "training", author="tester")

    assert metrics["test_pr_auc"] > 0.8 and metrics["test_recall"] > 0.8  # learns the planted pattern
    assert {"train_pr_auc", "train_seconds", "test_confusion_matrix"} <= set(metrics)

    import mlflow
    run = mlflow.get_run(result.run_id)
    assert run.data.tags["baseline"] == "true" and run.data.tags["model_type"] == "decision_tree"
    assert run.data.tags["dataset.data_hash"] == manifest["data_hash"]
    assert run.data.params["excluded"] == "refund_value"

    baseline = dt.baseline_metrics(manifest["data_hash"])
    assert baseline["test_pr_auc"] == pytest.approx(metrics["test_pr_auc"])
    assert set(baseline) == {"test_pr_auc", "test_recall", "test_precision", "test_f1", "test_threshold"}


def test_baseline_must_match_the_data(local_store, tmp_path):
    write_training_dir(tmp_path / "training")
    dt.run(tmp_path / "training")
    with pytest.raises(LookupError, match="no baseline run .* for data other-hash"):
        dt.baseline_metrics("other-hash")


def test_logged_model_scores_new_categories(local_store, tmp_path):
    import mlflow
    write_training_dir(tmp_path / "training")
    _, result = dt.run(tmp_path / "training")
    model = mlflow.sklearn.load_model(result.model_uri)
    _, test, _ = data_fetch.load_training_data(tmp_path / "training")
    rows = test[dt.FEATURES].head(3).copy()
    rows["payment_method"] = "a method first seen in a new batch"
    assert model.predict_proba(rows).shape == (3, 2)
