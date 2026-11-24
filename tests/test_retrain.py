"""Tests for src/retrain.py and promote.run: the pipeline's retrain and promote steps."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from src import data_fetch, promote, retrain

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
                "data_hash": "hash-retrain"}
    (path / data_fetch.MANIFEST_FILE).write_text(json.dumps(manifest))
    return path



def test_baseline_always_runs_first():
    assert retrain.model_list("random_forest,decision_tree") == ["decision_tree", "random_forest"]
    assert retrain.model_list(["mlp"]) == ["decision_tree", "mlp"]
    with pytest.raises(ValueError, match="unknown model"):
        retrain.model_list("not_a_model")


def test_models_can_be_chosen_with_an_env_var(monkeypatch):
    monkeypatch.setenv("RETRAIN_MODELS", "random_forest")
    assert retrain.model_list() == ["decision_tree", "random_forest"]


def test_retrains_each_model_tagged_with_the_new_batch(local_store, training_dir, monkeypatch):
    real_import = retrain.import_model
    monkeypatch.setattr(retrain, "import_model", lambda name: None if name == "svm" else real_import(name))

    out = retrain.run("base", models="random_forest,svm", training_dir=training_dir,
                      experiment="test-exp", fetch=False)

    assert list(out["models"]) == ["decision_tree", "random_forest"]
    assert out["models"]["random_forest"] > 0.7
    assert out["skipped"] == ["svm"]
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    tags = dict(zip(runs["tags.model_type"], runs["tags.dataset.new_batch"]))
    assert tags["random_forest"] == "base"


def test_missing_baseline_stops_the_step(local_store, training_dir, monkeypatch):
    monkeypatch.setattr(retrain, "import_model", lambda name: None)
    with pytest.raises(RuntimeError, match="decision_tree"):
        retrain.run("base", models="random_forest", training_dir=training_dir, fetch=False)


def test_promote_step_reports_the_decision(monkeypatch):
    monkeypatch.setattr(promote, "select_and_promote", lambda **kw: {"promoted": True, "champion_version": "3"})
    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "http://mlflow.example.com/")

    out = promote.run("batch_02")

    assert out["promoted"] and out["model_version"] == "3"
    assert out["mlflow_url"] == "http://mlflow.example.com/#/models/fraud-detector/versions/3"
