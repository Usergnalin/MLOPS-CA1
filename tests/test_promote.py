"""Tests for src/promote.py. They use a local SQLite MLflow store, not the team server."""
import json

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import data_fetch, promote
from src.evaluation import evaluate_model
from src.tracking import log_model_run

EXP = "test-exp"


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.chdir(tmp_path)  # keep artifacts out of the repo


def write_training_dir(path, data_hash, seed=0, n=1000):
    """data_fetch-format data: num_associated_customers predicts fraud, order_value is mostly noise."""
    rng = np.random.default_rng(seed)
    assoc = rng.integers(1, 10, n)
    df = pd.DataFrame({
        "country_code": rng.choice(["MY", "PH", "PK"], n), "order_id": [f"o{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)], "collect_type": "Delivery", "payment_method": "balance",
        "order_value": rng.uniform(1, 20, n), "num_items_ordered": rng.integers(1, 5, n),
        "refund_value": 0.0, "order_date": pd.Timestamp("2023-01-15"), "mobile_verified": True,
        "num_orders_last_50days": 1, "num_cancelled_orders_last_50days": 0, "num_refund_orders_last_50days": 0,
        "total_payment_last_50days": 1.0, "num_associated_customers": assoc,
        "first_order_datetime": pd.Timestamp("2021-03-23 09:16:29"), "batch_id": "base",
        "is_fraud": ((assoc > 6) ^ (rng.random(n) < 0.1)).astype(int),
    })
    df = data_fetch.set_types(df)
    mask = data_fetch.is_test(df)
    train, test = df[~mask], df[mask]
    path.mkdir(parents=True, exist_ok=True)
    th = data_fetch.write_csv(train, path / data_fetch.TRAIN_FILE)
    te = data_fetch.write_csv(test, path / data_fetch.TEST_FILE)
    manifest = {"batches": {"base": n}, "train": data_fetch.describe(train, th),
                "test": data_fetch.describe(test, te), "data_hash": data_hash}
    (path / data_fetch.MANIFEST_FILE).write_text(json.dumps(manifest))
    return path


def log_run(training_dir, name, columns):
    """Train a logistic regression on `columns`, score it on test and log it like the model scripts do."""
    train, test, manifest = data_fetch.load_training_data(training_dir)
    model = Pipeline([("cols", ColumnTransformer([("keep", "passthrough", columns)])),
                      ("model", LogisticRegression())]).fit(train[columns], train[data_fetch.TARGET])
    metrics = evaluate_model(model, test[columns], test[data_fetch.TARGET])
    log_model_run(model, name, metrics=metrics, dataset_tags={"data_hash": manifest["data_hash"]},
                  experiment_name=EXP)
    return metrics["test_pr_auc"]


GOOD = ["num_associated_customers", "order_value"]
OK = ["num_associated_customers"]
BAD = ["order_value"]


def aliases():
    client = MlflowClient()
    return {a: client.get_model_version_by_alias(promote.REGISTERED_MODEL, a).tags["model_type"]
            for a in (promote.CHAMPION, promote.CHALLENGER, promote.RUNNER_UP, promote.PREVIOUS)
            if a in client.get_registered_model(promote.REGISTERED_MODEL).aliases}


def run_promote(training_dir, **kw):
    return promote.select_and_promote(EXP, training_dir, **kw)


def test_first_run_shortlists_top_2_and_crowns_the_best(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    scores = {name: log_run(d, name, cols) for name, cols in [("bad", BAD), ("ok", OK), ("good", GOOD)]}
    ranked = sorted(scores, key=scores.get, reverse=True)

    decision = run_promote(d)

    assert decision["promoted"] and decision["reason"] == "no champion yet"
    assert [s["model_type"] for s in decision["shortlist"]] == ranked[:2]
    assert aliases() == {"champion": ranked[0], "challenger": ranked[0], "runner_up": ranked[1]}
    assert len(MlflowClient().search_model_versions(f"name = '{promote.REGISTERED_MODEL}'")) == 2


def test_rerun_without_new_models_changes_nothing(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    log_run(d, "ok", OK)
    first = run_promote(d)
    again = run_promote(d)
    assert not again["promoted"] and again["reason"] == "the best run is already the champion"
    assert again["champion_version"] == first["champion_version"]


def test_better_model_replaces_champion_and_old_one_is_kept(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    log_run(d, "bad", BAD)
    run_promote(d)
    assert aliases()["champion"] == "bad"

    log_run(d, "ok", OK)
    decision = run_promote(d)

    assert decision["promoted"] and "beats" in decision["reason"]
    assert decision["champion_pr_auc"] < decision["challenger_pr_auc"]
    assert aliases()["champion"] == "ok" and aliases()["previous_champion"] == "bad"


def test_worse_model_does_not_replace_champion(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    log_run(d, "ok", OK)
    run_promote(d)

    # New data: the old champion is re-scored on it; a weak new model must not win.
    d2 = write_training_dir(tmp_path / "t2", "hash-2", seed=1)
    log_run(d2, "bad", BAD)
    decision = run_promote(d2)

    assert not decision["promoted"] and "does not beat" in decision["reason"]
    assert aliases()["champion"] == "ok"


def test_min_improvement_blocks_small_gains(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    log_run(d, "bad", BAD)
    run_promote(d)
    log_run(d, "ok", OK)
    assert not run_promote(d, min_improvement=1.0)["promoted"]
    assert aliases()["champion"] == "bad"


def test_dry_run_changes_nothing(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    log_run(d, "ok", OK)
    decision = run_promote(d, dry_run=True)
    assert decision["promoted"] and decision["dry_run"]
    assert MlflowClient().search_registered_models() == []


def test_only_runs_on_the_current_data_count(local_store, tmp_path):
    old = write_training_dir(tmp_path / "old", "hash-old")
    log_run(old, "good", GOOD)
    new = write_training_dir(tmp_path / "new", "hash-new", seed=1)
    log_run(new, "bad", BAD)

    decision = run_promote(new)
    assert [s["model_type"] for s in decision["shortlist"]] == ["bad"]


def test_no_runs_gives_clear_error(local_store, tmp_path):
    d = write_training_dir(tmp_path / "t", "hash-1")
    log_run(d, "ok", OK)
    empty = write_training_dir(tmp_path / "e", "hash-none", seed=2)
    with pytest.raises(LookupError, match="train the models first"):
        run_promote(empty)
