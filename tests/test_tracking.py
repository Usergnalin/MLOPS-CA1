"""Tests for src/tracking.py. They use a local SQLite MLflow store, not the AWS server."""

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src import tracking


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # keep artifacts out of the repo
    return uri


@pytest.fixture
def fitted_model():
    X = pd.DataFrame({"amount": [10.0, 200.0, 15.0, 500.0, 12.0, 450.0], "items": [1, 5, 2, 7, 1, 6]})
    y = np.array([0, 1, 0, 1, 0, 1])
    return LogisticRegression().fit(X, y), X


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://mlflow.example.com/#/", "http://mlflow.example.com"),
        ("http://mlflow.example.com/", "http://mlflow.example.com"),
        ("  http://mlflow.example.com  ", "http://mlflow.example.com"),
        ("sqlite:///mlflow.db", "sqlite:///mlflow.db"),
    ],
)
def test_clean_uri(raw, expected):
    assert tracking._clean_uri(raw) == expected


def test_missing_uri_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore the real .env
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        tracking.get_tracking_uri()


def test_split_metrics():
    scalars, others = tracking._split_metrics(
        {"f1": 0.5, "recall": np.float64(0.7), "confusion_matrix": np.array([[5, 1], [2, 3]]), "bad": float("nan")}
    )
    assert scalars == {"f1": 0.5, "recall": 0.7}
    assert others == {"confusion_matrix": [[5, 1], [2, 3]], "bad": "nan"}


def test_log_model_run_records_everything(local_store, fitted_model):
    model, X = fitted_model

    result = tracking.log_model_run(
        model=model,
        model_name="logistic_regression",
        params=model.get_params(),
        metrics={"pr_auc": 0.9, "recall": 0.8, "precision": 0.75, "f1": 0.77, "confusion_matrix": [[3, 0], [1, 2]]},
        dataset_tags={"batch_id": "batch_01", "train_rows": len(X)},
        experiment_name="test-experiment",
        input_example=X.head(2),
    )

    run = MlflowClient().get_run(result.run_id)
    assert run.data.params["C"] == "1.0"
    assert run.data.metrics == {"pr_auc": 0.9, "recall": 0.8, "precision": 0.75, "f1": 0.77}
    assert run.data.tags["model_type"] == "logistic_regression"
    assert run.data.tags["dataset.batch_id"] == "batch_01"
    assert run.data.tags["dataset.train_rows"] == "6"
    assert result.run_url is None  # only built for an http server

    artifacts = [a.path for a in MlflowClient().list_artifacts(result.run_id, "evaluation")]
    assert "evaluation/extra_results.json" in artifacts

    import mlflow

    loaded = mlflow.sklearn.load_model(result.model_uri)
    assert list(loaded.predict(X)) == list(model.predict(X))


@pytest.mark.parametrize("model_cls", [GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier])
def test_tree_models_can_be_logged_and_loaded(local_store, fitted_model, model_cls):
    # MLflow 3's default "skops" format rejects tree models; the helper must still log them.
    _, X = fitted_model
    y = np.array([0, 1, 0, 1, 0, 1])
    model = model_cls(random_state=0).fit(X, y)

    result = tracking.log_model_run(model, model_cls.__name__, experiment_name="test-experiment")

    import mlflow

    loaded = mlflow.sklearn.load_model(result.model_uri)
    assert list(loaded.predict(X)) == list(model.predict(X))
