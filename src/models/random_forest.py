"""Random Forest model, evaluated against the decision tree baseline (#65).

Trains on the shared training data (src.data_fetch), scores it with the shared evaluation module
(src.evaluation), compares it with the baseline run on the same data (src.models.decision_tree)
and logs everything to MLflow (src.tracking).

A random forest averages many decision trees, each grown on a random sample of the orders and
choosing among a random subset of the columns at every split, so it is less likely than a single
tree to memorise noise. class_weight="balanced_subsample" re-balances fraud against non-fraud
within each tree's own sample. Each tree sees a quarter of the training orders (max_samples=0.25),
which keeps training to seconds on ~760k orders and adds variety between trees.

Settings were chosen on a validation split of the training data (not the test set): smaller
leaves (min_samples_leaf=10) raised PR-AUC while keeping both recall and precision above the
baseline at the default 0.5 threshold.

Features are the baseline's (same columns, same leak exclusions), so the comparison is fair.

Usage:
    python -m src.models.decision_tree          # the baseline must exist for the same data first
    python -m src.models.random_forest
    python -m src.models.random_forest --n-estimators 200 --min-samples-leaf 5
"""
import argparse
import getpass
import logging
import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.dataset_tags import dataset_tags
from src.evaluation import compare_to_baseline, evaluate_model, format_metrics
from src.models.decision_tree import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, baseline_metrics
from src.tracking import DEFAULT_EXPERIMENT, log_model_run

log = logging.getLogger("random_forest")

MODEL_NAME = "random_forest"
RANDOM_STATE = 42
N_JOBS = 8  # threads share one copy of the data, so this costs CPU, not memory
DEFAULTS = {"n_estimators": 100, "min_samples_leaf": 10, "max_samples": 0.25, "max_features": "sqrt",
            "max_depth": None}


def build_model(n_estimators=100, min_samples_leaf=10, max_samples=0.25, max_features="sqrt", max_depth=None):
    preprocess = ColumnTransformer(
        [
            # handle_unknown="ignore": a payment method first seen in a new batch won't crash scoring.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),  # trees need no scaling
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    forest = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_samples=max_samples,
        max_features=max_features, max_depth=max_depth, class_weight="balanced_subsample",
        n_jobs=N_JOBS, random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", forest)])


def run(training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT, run_name=None, author=None, new_batch=None,
        **hyperparams):
    """Train, evaluate, compare with the baseline and log. Returns (metrics, comparison, RunResult).

    new_batch: when retraining (#24), the batch that triggered it; logged as dataset.new_batch.
    """
    params = {**DEFAULTS, **{k: v for k, v in hyperparams.items() if v is not None}}
    train, test, manifest = load_training_data(training_dir)
    # Fail before the training run if there is nothing to compare with.
    baseline = baseline_metrics(manifest["data_hash"], experiment)

    model = build_model(**params)
    start = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    train_seconds = time.perf_counter() - start
    log.info("trained %d trees in %.1fs", params["n_estimators"], train_seconds)

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])
    # Train next to test makes overfitting visible (a forest of deep trees can memorise).
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds

    comparison = compare_to_baseline(metrics, baseline)
    for m in ("pr_auc", "recall", "precision", "f1"):
        metrics[f"vs_baseline_{m}"] = comparison[m]["change"]

    result = log_model_run(
        model, MODEL_NAME,
        params={**params, "class_weight": "balanced_subsample", "random_state": RANDOM_STATE,
                "features": ",".join(FEATURES), "excluded": ",".join(EXCLUDED)},
        metrics=metrics,
        dataset_tags=dataset_tags(manifest, new_batch=new_batch),
        experiment_name=experiment,
        run_name=run_name,
        tags={"author": author or getpass.getuser(),
              "beats_baseline": str(comparison["beats_baseline"]).lower()},
        input_example=test[FEATURES].head(5),
    )
    log.info(format_metrics(metrics))
    log.info(format_metrics(metrics, prefix="train"))
    log.info(format_metrics(baseline) + "  <- baseline")
    log.info("beats baseline (higher recall and precision): %s", comparison["beats_baseline"])
    log.info("logged run %s %s", result.run_id, result.run_url or "")
    return metrics, comparison, result


def max_features_arg(value):
    """--max-features accepts sqrt, log2, a fraction like 0.5, or a count like 4."""
    if value in ("sqrt", "log2"):
        return value
    return float(value) if "." in value else int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-estimators", type=int, help="number of trees")
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--max-samples", type=float, help="share of the training orders each tree sees")
    parser.add_argument("--max-features", type=max_features_arg, help="columns considered per split")
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: random_forest)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    parser.add_argument("--new-batch", help="when retraining: the batch that triggered it (logged as dataset.new_batch)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.experiment, args.run_name, args.author, new_batch=args.new_batch,
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, max_samples=args.max_samples,
        max_features=args.max_features, max_depth=args.max_depth)
