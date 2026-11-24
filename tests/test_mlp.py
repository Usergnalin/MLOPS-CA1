"""Tests for src/models/mlp.py. They use a local SQLite MLflow store, not the team server."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from src import data_fetch
from src.models import decision_tree as dt
from src.models import mlp


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
                "data_hash": "hash-mlp"}
    (path / data_fetch.MANIFEST_FILE).write_text(json.dumps(manifest))
    return path


def test_uses_the_baselines_features():
    # Same columns as the baseline, so the comparison is fair, and refund_value stays out (AC5).
    assert mlp.FEATURES == dt.FEATURES
    assert "refund_value" not in mlp.FEATURES
    assert set(mlp.LOG_SCALED) | set(mlp.FLAGS) == set(dt.NUMERIC)


def test_threshold_for_recall_by_hand():
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.6, 0.3, 0.7, 0.2, 0.1, 0.05]
    assert mlp.threshold_for_recall(y, p, 0.75) == 0.6   # flags 0.9, 0.8, 0.7, 0.6: 3 of 4 fraud
    assert mlp.threshold_for_recall(y, p, 1.0) == 0.3    # must go down to 0.3 to catch all 4
    assert mlp.choose_threshold(y, p, 0) == pytest.approx(0.3)  # 0 = best F1 instead (P 0.8, R 1.0)


def test_needs_a_baseline_first(local_store, training_dir):
    with pytest.raises(LookupError, match="baseline"):
        mlp.run(training_dir, experiment="test-exp")


def test_run_logs_metrics_comparison_and_model_with_its_threshold(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    metrics, comparison, result = mlp.run(training_dir, experiment="test-exp", hidden=(32,), learning_rate=0.01)

    run = mlflow.get_run(result.run_id)
    for key in ("test_pr_auc", "test_recall", "test_precision", "test_f1", "train_pr_auc",
                "vs_baseline_pr_auc", "epochs", "train_seconds"):
        assert key in run.data.metrics
    assert run.data.metrics["test_pr_auc"] > 0.7          # learns the planted pattern
    assert run.data.tags["model_type"] == "mlp" and run.data.tags["dataset.data_hash"] == "hash-mlp"
    assert run.data.tags["beats_baseline"] in ("true", "false")
    assert run.data.params["excluded"] == "refund_value"
    assert run.data.params["threshold_rule"] == "recall>=0.8"
    threshold = float(run.data.params["threshold"])
    assert run.data.metrics["test_threshold"] == pytest.approx(threshold, abs=1e-6)
    assert set(comparison) >= {"pr_auc", "recall", "precision", "f1", "beats_baseline"}

    # The saved model carries its threshold: predict() flags exactly the orders at or above it.
    loaded = mlflow.sklearn.load_model(result.model_uri)
    _, test, _ = data_fetch.load_training_data(training_dir)
    X = test[mlp.FEATURES]
    proba = loaded.predict_proba(X)[:, 1]
    assert (loaded.predict(X) == (proba >= loaded.threshold)).all()
    from src.evaluation import evaluate_model
    assert evaluate_model(loaded, X, test[data_fetch.TARGET])["test_pr_auc"] == pytest.approx(
        run.data.metrics["test_pr_auc"])


def test_best_f1_rule(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    _, _, result = mlp.run(training_dir, experiment="test-exp", hidden=(16,), max_iter=30, min_recall=0)
    assert mlflow.get_run(result.run_id).data.params["threshold_rule"] == "best_f1"


def test_retraining_run_records_the_new_batch(local_store, training_dir):
    dt.run(training_dir, experiment="test-exp")
    _, _, normal = mlp.run(training_dir, experiment="test-exp", hidden=(8,), max_iter=20)
    _, _, retrain = mlp.run(training_dir, experiment="test-exp", hidden=(8,), max_iter=20, new_batch="base")
    assert "dataset.new_batch" not in mlflow.get_run(normal.run_id).data.tags
    assert mlflow.get_run(retrain.run_id).data.tags["dataset.new_batch"] == "base"


def test_fits_on_a_capped_sample_but_chooses_the_threshold_on_the_full_slice(training_dir, monkeypatch):
    seen = {}
    real_build = mlp.build_model

    def spy_build(**params):
        model = real_build(**params)
        real_fit = model.fit
        def fit(X, y):
            seen["rows"], seen["fraud_rate"] = len(X), float(y.mean())
            return real_fit(X, y)
        model.fit = fit
        return model

    monkeypatch.setattr(mlp, "build_model", spy_build)
    train, _, _ = data_fetch.load_training_data(training_dir)
    X, y = train[mlp.FEATURES], train[data_fetch.TARGET]
    mlp.fit_with_threshold(X, y, 0.8, max_rows=200, hidden=(8,), max_iter=20)
    assert seen["rows"] == 200
    assert abs(seen["fraud_rate"] - y.mean()) < 0.02                 # stratified: fraud rate kept
    mlp.fit_with_threshold(X, y, 0.8, max_rows=0, hidden=(8,), max_iter=20)
    assert seen["rows"] == len(X) - round(len(X) * mlp.THRESHOLD_FRACTION)  # 0 = every fitting row


def test_defaults_are_the_fast_settings():
    assert mlp.DEFAULTS["hidden"] == (64, 32) and mlp.DEFAULTS["max_rows"] == 300_000
    assert mlp.DEFAULTS["patience"] == 5 and mlp.DEFAULTS["batch_size"] == 4096
