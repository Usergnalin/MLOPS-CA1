"""Tests for src/models/logistic_regression.py. They use a local SQLite MLflow store, not the team server."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from src import data_fetch
from src.models import decision_tree as dt
from src.models import logistic_regression as lr


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.chdir(tmp_path)  # keep artifacts out of the repo


@pytest.fixture
def training_dir(tmp_path):
    """A small training set in the data_fetch format; fraud depends on num_associated_customers."""
    n, rng = 800, np.random.default_rng(0)
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
    path = tmp_path / "training"
    path.mkdir()
    th = data_fetch.write_csv(train, path / data_fetch.TRAIN_FILE)
    te = data_fetch.write_csv(test, path / data_fetch.TEST_FILE)
    manifest = {"batches": {"base": n}, "order_date_min": "2023-01-15", "order_date_max": "2023-01-15",
                "train": data_fetch.describe(train, th), "test": data_fetch.describe(test, te),
                "data_hash": "hash-lr"}
    (path / data_fetch.MANIFEST_FILE).write_text(json.dumps(manifest))
    return path


def test_uses_the_baselines_features():
    # Same columns as the baseline, so the comparison is fair, and refund_value stays out (AC5).
    assert lr.FEATURES == dt.FEATURES
    assert "refund_value" not in lr.FEATURES


def test_needs_a_baseline_first(local_store, training_dir):
    with pytest.raises(LookupError, match="baseline"):
        lr.run(training_dir, experiment="test-exp")


def test_run_logs_metrics_comparison_and_model(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    metrics, comparison, result = lr.run(training_dir, experiment="test-exp", C=0.5)

    run = mlflow.get_run(result.run_id)
    for key in ("test_pr_auc", "test_recall", "test_precision", "test_f1", "train_pr_auc",
                "vs_baseline_pr_auc", "train_seconds"):
        assert key in run.data.metrics
    assert run.data.metrics["test_pr_auc"] > 0.7          # learns the planted pattern
    assert run.data.tags["model_type"] == "logistic_regression"
    assert run.data.tags["dataset.data_hash"] == "hash-lr"
    assert run.data.tags["beats_baseline"] in ("true", "false")
    assert run.data.params["excluded"] == "refund_value"
    assert run.data.params["C"] == "0.5" and run.data.params["max_iter"] == "1000"
    assert set(comparison) >= {"pr_auc", "recall", "precision", "f1", "beats_baseline"}

    # The saved model gives the same test score as the one that was logged.
    from src.evaluation import evaluate_model

    loaded = mlflow.sklearn.load_model(result.model_uri)
    _, test, _ = data_fetch.load_training_data(training_dir)
    rescored = evaluate_model(loaded, test[lr.FEATURES], test[data_fetch.TARGET])
    assert rescored["test_pr_auc"] == pytest.approx(run.data.metrics["test_pr_auc"])


def test_retraining_run_records_the_new_batch(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    _, _, normal = lr.run(training_dir, experiment="test-exp")
    _, _, retrain = lr.run(training_dir, experiment="test-exp", new_batch="base")

    assert "dataset.new_batch" not in mlflow.get_run(normal.run_id).data.tags
    assert mlflow.get_run(retrain.run_id).data.tags["dataset.new_batch"] == "base"


def test_unseen_payment_method_is_scored(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    _, _, result = lr.run(training_dir, experiment="test-exp")
    _, test, _ = data_fetch.load_training_data(training_dir)
    rows = test[lr.FEATURES].head(3).copy()
    rows["payment_method"] = "a method first seen in a new batch"
    assert mlflow.sklearn.load_model(result.model_uri).predict_proba(rows).shape == (3, 2)
