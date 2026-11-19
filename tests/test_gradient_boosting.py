"""Tests for src/models/gradient_boosting.py. They use a local SQLite MLflow store, not the team server."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from src import data_fetch
from src.models import decision_tree as dt
from src.models import gradient_boosting as gb


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
                "data_hash": "hash-gb"}
    (path / data_fetch.MANIFEST_FILE).write_text(json.dumps(manifest))
    return path


def test_uses_the_baselines_features():
    # Same columns as the baseline, so the comparison is fair, and refund_value stays out (AC5).
    assert gb.FEATURES == dt.FEATURES
    assert "refund_value" not in gb.FEATURES


def test_needs_a_baseline_first(local_store, training_dir):
    with pytest.raises(LookupError, match="baseline"):
        gb.run(training_dir, experiment="test-exp")


def test_run_logs_metrics_comparison_and_model(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    metrics, comparison, result = gb.run(training_dir, experiment="test-exp", max_iter=30)

    run = mlflow.get_run(result.run_id)
    for key in ("test_pr_auc", "test_recall", "test_precision", "test_f1", "train_pr_auc",
                "vs_baseline_pr_auc", "n_trees"):
        assert key in run.data.metrics
    assert run.data.metrics["test_pr_auc"] > 0.7          # learns the planted pattern
    assert run.data.tags["model_type"] == "gradient_boosting"
    assert run.data.tags["dataset.data_hash"] == "hash-gb"
    assert run.data.tags["beats_baseline"] in ("true", "false")
    assert run.data.params["excluded"] == "refund_value"
    assert set(comparison) >= {"pr_auc", "recall", "precision", "f1", "beats_baseline"}

    # The saved model gives the same test score as the one that was logged.
    from src.evaluation import evaluate_model

    loaded = mlflow.sklearn.load_model(result.model_uri)
    _, test, _ = data_fetch.load_training_data(training_dir)
    rescored = evaluate_model(loaded, test[gb.FEATURES], test[data_fetch.TARGET])
    assert rescored["test_pr_auc"] == pytest.approx(run.data.metrics["test_pr_auc"])
