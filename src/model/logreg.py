"""Logistic Regression fraud model (#64).

Trains a class-balanced Logistic Regression classifier on the shared training data
(src.data_fetch), evaluates it against the Decision Tree baseline (src.models.decision_tree),
and logs parameters, metrics, and dataset traceability tags to MLflow (src.tracking).

Usage:
    python -m src.models.logistic_regression
    python -m src.models.logistic_regression --C 0.5 --max-iter 1000
    python -m src.models.logistic_regression --new-batch batch2
"""
import argparse
import getpass
import logging
import time

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.dataset_tags import dataset_tags
from src.evaluation import compare_to_baseline, evaluate_model, format_metrics
from src.models.decision_tree import baseline_metrics
from src.tracking import DEFAULT_EXPERIMENT, log_model_run

log = logging.getLogger("logistic_regression")

MODEL_NAME = "logistic_regression"
RANDOM_STATE = 42

CATEGORICAL = ["country_code", "collect_type", "payment_method"]
NUMERIC = [
    "order_value", "num_items_ordered", "mobile_verified",
    "num_orders_last_50days", "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
    "total_payment_last_50days", "num_associated_customers",
]
# refund_value is post-shipment and leaks the target (PBI-01 AC5).
EXCLUDED = ["refund_value"]
FEATURES = CATEGORICAL + NUMERIC


def build_model(C=1.0, max_iter=1000):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer(
        [
            ("categorical", cat_pipe, CATEGORICAL),
            ("numeric", num_pipe, NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", clf)])


def run(
    training_dir=TRAINING_DIR,
    C=1.0,
    max_iter=1000,
    experiment=DEFAULT_EXPERIMENT,
    run_name=None,
    author=None,
    new_batch=None,
):
    """Train, evaluate against baseline, and log Logistic Regression. Returns (metrics, RunResult)."""
    train, test, manifest = load_training_data(training_dir)
    tags_for_dataset = dataset_tags(manifest, new_batch=new_batch)

    model = build_model(C=C, max_iter=max_iter)

    start = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    train_seconds = time.perf_counter() - start

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds

    try:
        baseline = baseline_metrics(manifest["data_hash"], experiment=experiment)
        metrics |= compare_to_baseline(metrics, baseline)
    except LookupError as exc:
        log.warning("Skipping baseline comparison: %s", exc)

    result = log_model_run(
        model,
        MODEL_NAME,
        params={
            "C": C,
            "max_iter": max_iter,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "features": ",".join(FEATURES),
            "excluded": ",".join(EXCLUDED),
        },
        metrics=metrics,
        dataset_tags=tags_for_dataset,
        experiment_name=experiment,
        run_name=run_name,
        tags={"author": author or getpass.getuser()},
        input_example=test[FEATURES].head(5),
    )
    log.info(format_metrics(metrics))
    log.info(format_metrics(metrics, prefix="train"))
    log.info("logged run %s %s", result.run_id, result.run_url or "")
    return metrics, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: logistic_regression)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    parser.add_argument("--new-batch", default=None, help="batch name that triggered retraining")
    args = parser.parse_args()
    setup_logging()
    run(
        training_dir=args.training_dir,
        C=args.C,
        max_iter=args.max_iter,
        experiment=args.experiment,
        run_name=args.run_name,
        author=args.author,
        new_batch=args.new_batch,
    )