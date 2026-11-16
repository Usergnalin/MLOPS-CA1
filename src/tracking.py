"""Shared MLflow logging helper (#26, PBI-04 AC1).

Every training script uses this so that each model is recorded the same way:
its settings (params), its test-set results (metrics), the fitted model, and
tags saying which data it was trained on.

Example
-------
    from src.tracking import log_model_run

    result = log_model_run(
        model=clf,
        model_name="gradient_boosting",
        params=clf.get_params(),
        metrics={"pr_auc": 0.81, "recall": 0.74, "precision": 0.69, "f1": 0.71},
        dataset_tags={"batch_id": "batch_03", "train_rows": 12000},
        input_example=X_test.head(5),
    )
    print(result.run_url)

The MLflow server address comes from the MLFLOW_TRACKING_URI environment
variable (put it in your local .env file).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from numbers import Number
from typing import Any, Mapping

import mlflow

DEFAULT_EXPERIMENT = "fraud-detection"

# MLflow only accepts param values up to this length.
_MAX_PARAM_LENGTH = 6000


@dataclass(frozen=True)
class RunResult:
    """What a logged run returns, so callers can link to it or load the model."""

    run_id: str
    experiment_id: str
    model_uri: str
    run_url: str | None


def get_tracking_uri() -> str:
    """Return the MLflow server URL from the environment (or the local .env file).

    A URL copied from the browser often ends in "/#/" or "/"; that part only
    belongs to the web page, so it is removed.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional at runtime
        pass

    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Add it to your .env file, e.g.\n"
            "MLFLOW_TRACKING_URI=http://<your-eb-env>.<region>.elasticbeanstalk.com"
        )
    return _clean_uri(uri)


def _clean_uri(uri: str) -> str:
    uri = uri.strip()
    if uri.endswith("#/"):
        uri = uri[:-2]
    elif uri.endswith("#"):
        uri = uri[:-1]
    return uri.rstrip("/") if uri.startswith("http") else uri


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT, tracking_uri: str | None = None) -> str:
    """Point MLflow at the team server and select (or create) the experiment.

    Returns the experiment id.
    """
    mlflow.set_tracking_uri(_clean_uri(tracking_uri) if tracking_uri else get_tracking_uri())
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def log_model_run(
    model: Any,
    model_name: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    dataset_tags: Mapping[str, Any] | None = None,
    *,
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
    input_example: Any = None,
    tracking_uri: str | None = None,
) -> RunResult:
    """Log one trained model to MLflow and return where it was stored.

    - params: the model's settings (e.g. ``model.get_params()``).
    - metrics: test-set results. Numbers are logged as metrics; anything else
      (such as a confusion matrix) is saved as a JSON file in the run.
    - dataset_tags: which data was used (batch id, date range, row counts...).
      Each key is stored as a ``dataset.<key>`` tag.
    - tags: any other tags; ``model_type`` is always set to ``model_name``.
    """
    experiment_id = setup_mlflow(experiment_name, tracking_uri)
    _allow_emoji_output()

    with mlflow.start_run(run_name=run_name or model_name) as run:
        mlflow.set_tags({"model_type": model_name, **_stringify(tags or {})})
        mlflow.set_tags({f"dataset.{k}": v for k, v in _stringify(dataset_tags or {}).items()})

        if params:
            mlflow.log_params(_clean_params(params))

        scalar_metrics, other_results = _split_metrics(metrics or {})
        if scalar_metrics:
            mlflow.log_metrics(scalar_metrics)
        if other_results:
            mlflow.log_dict(other_results, "evaluation/extra_results.json")

        # cloudpickle instead of MLflow 3's default "skops" format: skops rejects
        # tree models (Decision Tree, Random Forest, Gradient Boosting, XGBoost)
        # unless every internal type is listed by hand. Models are only loaded
        # from the team's own server, so pickling is safe here.
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    return RunResult(
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        model_uri=model_info.model_uri,
        run_url=_run_url(experiment_id, run.info.run_id),
    )


def _allow_emoji_output() -> None:
    """MLflow prints "🏃 View run ..." when a run ends. On Windows, when output is
    redirected (pipeline logs, piped terminals) that crashes with UnicodeEncodeError,
    so unprintable characters are replaced instead."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _clean_params(params: Mapping[str, Any]) -> dict[str, str]:
    """MLflow params must be short strings; long or nested values are trimmed."""
    cleaned = {}
    for key, value in params.items():
        text = value if isinstance(value, str) else _to_json(value)
        cleaned[str(key)] = text[:_MAX_PARAM_LENGTH]
    return cleaned


def _split_metrics(metrics: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Separate plain numbers (MLflow metrics) from everything else."""
    scalars: dict[str, float] = {}
    others: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Number) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                scalars[str(key)] = number
                continue
        others[str(key)] = _jsonable(value)
    return scalars, others


def _stringify(values: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): v if isinstance(v, str) else _to_json(v) for k, v in values.items()}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _to_json(value: Any) -> str:
    try:
        return json.dumps(_jsonable(value), default=str)
    except (TypeError, ValueError):
        return str(value)


def _run_url(experiment_id: str, run_id: str) -> str | None:
    uri = mlflow.get_tracking_uri()
    if not uri.startswith("http"):
        return None
    return f"{uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
