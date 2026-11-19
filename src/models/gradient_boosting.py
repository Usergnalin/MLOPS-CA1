"""Gradient Boosting model, evaluated against the decision tree baseline (#61).

Trains on the shared training data (src.data_fetch), scores it with the shared evaluation module
(src.evaluation), compares it with the baseline run on the same data (src.models.decision_tree)
and logs everything to MLflow (src.tracking).

Uses scikit-learn's HistGradientBoostingClassifier: the same gradient boosting idea as
GradientBoostingClassifier, but it bins the numbers first, so it trains on ~760k orders in
seconds to minutes instead of hours, and it stops adding trees once a held-out part of the
training data stops improving (early stopping). class_weight="balanced" makes the rare fraud
class count as much as the rest, like the baseline.

Features are the baseline's (same columns, same leak exclusions), so the comparison is fair.

Usage:
    python -m src.models.decision_tree          # the baseline must exist for the same data first
    python -m src.models.gradient_boosting
    python -m src.models.gradient_boosting --learning-rate 0.05 --max-iter 500 --max-leaf-nodes 63
"""
import argparse
import getpass
import logging
import time

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.evaluation import compare_to_baseline, evaluate_model, format_metrics
from src.models.decision_tree import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, baseline_metrics, dataset_tags
from src.tracking import DEFAULT_EXPERIMENT, log_model_run

log = logging.getLogger("gradient_boosting")

MODEL_NAME = "gradient_boosting"
RANDOM_STATE = 42
DEFAULTS = {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 50,
            "l2_regularization": 0.0}


def build_model(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=0.0):
    preprocess = ColumnTransformer(
        [
            # Dense output: HistGradientBoosting doesn't take sparse input. Few categories, so it's small.
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        scoring="loss", random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", boosting)])


def run(training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT, run_name=None, author=None, **hyperparams):
    """Train, evaluate, compare with the baseline and log. Returns (metrics, comparison, RunResult)."""
    params = {**DEFAULTS, **{k: v for k, v in hyperparams.items() if v is not None}}
    train, test, manifest = load_training_data(training_dir)
    # Fail before the long training run if there is nothing to compare with.
    baseline = baseline_metrics(manifest["data_hash"], experiment)

    model = build_model(**params)
    start = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    train_seconds = time.perf_counter() - start
    trees = int(model.named_steps["model"].n_iter_)
    log.info("trained %d boosting rounds in %.1fs", trees, train_seconds)

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds
    metrics["n_trees"] = trees

    comparison = compare_to_baseline(metrics, baseline)
    for m in ("pr_auc", "recall", "precision", "f1"):
        metrics[f"vs_baseline_{m}"] = comparison[m]["change"]

    result = log_model_run(
        model, MODEL_NAME,
        params={**params, "class_weight": "balanced", "early_stopping": True, "validation_fraction": 0.1,
                "n_iter_no_change": 20, "random_state": RANDOM_STATE,
                "features": ",".join(FEATURES), "excluded": ",".join(EXCLUDED)},
        metrics=metrics,
        dataset_tags=dataset_tags(manifest),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-iter", type=int, help="maximum boosting rounds (early stopping may use fewer)")
    parser.add_argument("--max-leaf-nodes", type=int)
    parser.add_argument("--min-samples-leaf", type=int)
    parser.add_argument("--l2-regularization", type=float)
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: gradient_boosting)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.experiment, args.run_name, args.author,
        learning_rate=args.learning_rate, max_iter=args.max_iter, max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf, l2_regularization=args.l2_regularization)
