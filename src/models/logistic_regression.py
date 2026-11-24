"""Logistic Regression fraud model, evaluated against the decision tree baseline (#64).

Trains a class-balanced Logistic Regression classifier on the shared training data
(src.data_fetch), scores it with the shared evaluation module (src.evaluation), compares it with
the baseline run on the same data (src.models.decision_tree) and logs parameters, metrics and
dataset traceability tags to MLflow (src.tracking).

A linear model is sensitive to the scale of its inputs, so numbers are standardised; categories
are one-hot encoded. Missing values are imputed (median / most frequent) so a gap in a new batch
can't crash scoring.

Features are the baseline's (same columns, same leak exclusions), so the comparison is fair.

Usage:
    python -m src.models.decision_tree          # the baseline must exist for the same data first
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
from src.evaluation import MAIN_METRICS, compare_to_baseline, evaluate_model, format_metrics
from src.models.decision_tree import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, baseline_metrics
from src.tracking import DEFAULT_EXPERIMENT, log_model_run

log = logging.getLogger("logistic_regression")

MODEL_NAME = "logistic_regression"
RANDOM_STATE = 42
SOLVER = "lbfgs"
DEFAULTS = {"C": 1.0, "max_iter": 1000}


def build_preprocess():
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a payment method first seen in a new batch won't crash scoring.
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        [("categorical", categorical, CATEGORICAL), ("numeric", numeric, NUMERIC)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(C=1.0, max_iter=1000):
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", solver=SOLVER,
                             random_state=RANDOM_STATE)
    return Pipeline([("preprocess", build_preprocess()), ("model", clf)])


def train_and_log(model, model_name, params, training_dir, experiment, run_name, author, new_batch, logger):
    """Shared by the linear models: fit, evaluate, compare with the baseline, log.

    Returns (metrics, comparison, RunResult).
    """
    train, test, manifest = load_training_data(training_dir)
    # Fail before the training run if there is nothing to compare with.
    baseline = baseline_metrics(manifest["data_hash"], experiment)

    start = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    train_seconds = time.perf_counter() - start
    logger.info("trained in %.1fs", train_seconds)

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])
    # Train next to test makes over- or underfitting visible.
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds

    comparison = compare_to_baseline(metrics, baseline)
    for m in MAIN_METRICS:
        metrics[f"vs_baseline_{m}"] = comparison[m]["change"]

    result = log_model_run(
        model, model_name,
        params={**params, "class_weight": "balanced", "random_state": RANDOM_STATE,
                "features": ",".join(FEATURES), "excluded": ",".join(EXCLUDED)},
        metrics=metrics,
        dataset_tags=dataset_tags(manifest, new_batch=new_batch),
        experiment_name=experiment,
        run_name=run_name,
        tags={"author": author or getpass.getuser(),
              "beats_baseline": str(comparison["beats_baseline"]).lower()},
        input_example=test[FEATURES].head(5),
    )
    logger.info(format_metrics(metrics))
    logger.info(format_metrics(metrics, prefix="train"))
    logger.info(format_metrics(baseline) + "  <- baseline")
    logger.info("beats baseline (higher recall and precision): %s", comparison["beats_baseline"])
    logger.info("logged run %s %s", result.run_id, result.run_url or "")
    return metrics, comparison, result


def run(training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT, run_name=None, author=None, new_batch=None,
        **hyperparams):
    """Train, evaluate, compare with the baseline and log. Returns (metrics, comparison, RunResult).

    new_batch: when retraining (#24), the batch that triggered it; logged as dataset.new_batch.
    """
    params = {**DEFAULTS, **{k: v for k, v in hyperparams.items() if v is not None}}
    return train_and_log(build_model(**params), MODEL_NAME, {**params, "solver": SOLVER}, training_dir,
                         experiment, run_name, author, new_batch, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, help="inverse regularisation strength (default 1.0)")
    parser.add_argument("--max-iter", type=int, help="solver iterations (default 1000)")
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: logistic_regression)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    parser.add_argument("--new-batch", help="when retraining: the batch that triggered it (logged as dataset.new_batch)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.experiment, args.run_name, args.author, new_batch=args.new_batch,
        C=args.C, max_iter=args.max_iter)
