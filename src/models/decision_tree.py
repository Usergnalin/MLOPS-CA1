"""Decision tree base model: the baseline every other model is compared with (#17).

Trains a deliberately simple model on the shared training data (src.data_fetch), scores it with
the shared evaluation module (src.evaluation) and logs it to MLflow (src.tracking) tagged
baseline=true, so the other models can fetch its test-set metrics with baseline_metrics().

The pipeline only uses raw columns: one-hot encoded categories plus numbers as they are (trees
need no scaling). class_weight="balanced" makes the rare fraud class count as much as the rest.
Feature engineering is left to the members' own models (PBI-13).

Usage:
    python -m src.models.decision_tree
    python -m src.models.decision_tree --max-depth 12 --min-samples-leaf 20

In another model's script:
    from src.models.decision_tree import baseline_metrics
    baseline = baseline_metrics(manifest["data_hash"])   # test_* metrics of the baseline run
    compare_to_baseline(metrics, baseline)              # src.evaluation
"""
import argparse
import getpass
import logging
import time

import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.evaluation import MAIN_METRICS, evaluate_model, format_metrics
from src.tracking import DEFAULT_EXPERIMENT, log_model_run, setup_mlflow

log = logging.getLogger("decision_tree")

MODEL_NAME = "decision_tree"
RANDOM_STATE = 42
BASELINE_TAG = "baseline"

CATEGORICAL = ["country_code", "collect_type", "payment_method"]
NUMERIC = [
    "order_value", "num_items_ordered", "mobile_verified",
    "num_orders_last_50days", "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
    "total_payment_last_50days", "num_associated_customers",
]
# Left out on purpose. refund_value is the money refunded on this order, which is only known after
# it ships; fraud here means ordering and then claiming a refund, so using it would leak the answer
# (PBI-01 AC5). The feature audit (#15) decides the final list; change it here if it says otherwise.
# IDs, dates and batch_id are not features.
EXCLUDED = ["refund_value"]
FEATURES = CATEGORICAL + NUMERIC


def build_model(max_depth=8, min_samples_leaf=50):
    preprocess = ColumnTransformer(
        [
            # handle_unknown="ignore": a payment method first seen in a new batch won't crash scoring.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", preprocess), ("model", tree)])


def dataset_tags(manifest):
    """Which data a run used, from the data_fetch manifest (logged as dataset.<key> tags)."""
    return {
        "data_hash": manifest["data_hash"],
        "batches": ",".join(manifest["batches"]),
        "order_date_min": manifest["order_date_min"],
        "order_date_max": manifest["order_date_max"],
        "train_rows": manifest["train"]["rows"],
        "test_rows": manifest["test"]["rows"],
    }


def run(training_dir=TRAINING_DIR, max_depth=8, min_samples_leaf=50, experiment=DEFAULT_EXPERIMENT,
        run_name=None, author=None):
    """Train, evaluate and log the baseline. Returns (metrics, RunResult)."""
    train, test, manifest = load_training_data(training_dir)
    model = build_model(max_depth, min_samples_leaf)

    start = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    train_seconds = time.perf_counter() - start

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])
    # Train metrics next to test metrics make over- or underfitting visible.
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds

    result = log_model_run(
        model, MODEL_NAME,
        params={"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "class_weight": "balanced",
                "random_state": RANDOM_STATE, "features": ",".join(FEATURES), "excluded": ",".join(EXCLUDED)},
        metrics=metrics,
        dataset_tags=dataset_tags(manifest),
        experiment_name=experiment,
        run_name=run_name,
        tags={BASELINE_TAG: "true", "author": author or getpass.getuser()},
        input_example=test[FEATURES].head(5),
    )
    log.info(format_metrics(metrics))
    log.info(format_metrics(metrics, prefix="train"))
    log.info("logged baseline run %s %s", result.run_id, result.run_url or "")
    return metrics, result


def baseline_metrics(data_hash, experiment=DEFAULT_EXPERIMENT):
    """Test-set metrics of the latest baseline run trained on the same data (same data_hash).

    Comparing against a baseline trained on other data would be meaningless, so this fails if
    there is none: run python -m src.models.decision_tree first.
    """
    setup_mlflow(experiment)
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.{BASELINE_TAG} = 'true' and tags.`dataset.data_hash` = '{data_hash}'",
        order_by=["start_time DESC"], max_results=1,
    )
    if runs.empty:
        raise LookupError(f"no baseline run in MLflow experiment {experiment!r} for data {data_hash[:12]} - "
                          "run python -m src.models.decision_tree first")
    row = runs.iloc[0]
    return {f"test_{m}": float(row[f"metrics.test_{m}"]) for m in MAIN_METRICS + ["threshold"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: decision_tree)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.max_depth, args.min_samples_leaf, args.experiment, args.run_name, args.author)
