"""XGBoost model, evaluated against the decision tree baseline (#62).

Trains on the shared training data (src.data_fetch), scores it with the shared evaluation module
(src.evaluation), compares it with the baseline run on the same data (src.models.decision_tree)
and logs everything to MLflow (src.tracking).

Fraud is rare (~11%), so scale_pos_weight = non-fraud / fraud rows makes each fraud order count as
much as the rest together, the XGBoost equivalent of the baseline's class_weight="balanced".
Early stopping watches a 10% validation part of the training data (never the test set) and stops
adding trees when its log-loss stops improving.

Features are the baseline's (same columns, same leak exclusions), so the comparison is fair.

(Named xgboost_model.py, not xgboost.py, so it doesn't hide the xgboost package on import.)

Usage:
    python -m src.models.decision_tree          # the baseline must exist for the same data first
    python -m src.models.xgboost_model
    python -m src.models.xgboost_model --learning-rate 0.05 --n-estimators 1000 --max-depth 8
"""
import argparse
import getpass
import logging
import time

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.evaluation import compare_to_baseline, evaluate_model, format_metrics
from src.dataset_tags import dataset_tags
from src.models.decision_tree import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, baseline_metrics
from src.tracking import DEFAULT_EXPERIMENT, log_model_run

log = logging.getLogger("xgboost_model")

MODEL_NAME = "xgboost"
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.1
EARLY_STOPPING_ROUNDS = 30
DEFAULTS = {"learning_rate": 0.1, "n_estimators": 500, "max_depth": 6, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8}


def build_preprocess():
    return ColumnTransformer(
        [
            # handle_unknown="ignore": a payment method first seen in a new batch won't crash scoring.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_classifier(scale_pos_weight, learning_rate=0.1, n_estimators=500, max_depth=6, min_child_weight=5,
                     subsample=0.8, colsample_bytree=0.8):
    return XGBClassifier(
        learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
        min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight, objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, tree_method="hist", n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def fit(train, **params):
    """Fit preprocessing + XGBoost with early stopping. Returns a fitted Pipeline."""
    fit_part, val_part = train_test_split(train, test_size=VALIDATION_FRACTION, stratify=train[TARGET],
                                          random_state=RANDOM_STATE)
    y_fit = fit_part[TARGET]
    scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

    preprocess = build_preprocess().fit(fit_part[FEATURES])
    classifier = build_classifier(scale_pos_weight, **params)
    classifier.fit(preprocess.transform(fit_part[FEATURES]), y_fit,
                   eval_set=[(preprocess.transform(val_part[FEATURES]), val_part[TARGET])], verbose=False)
    # Both steps are already fitted; the Pipeline just chains them for predict_proba and saving.
    return Pipeline([("preprocess", preprocess), ("model", classifier)]), scale_pos_weight


def run(training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT, run_name=None, author=None, new_batch=None,
        **hyperparams):
    """Train, evaluate, compare with the baseline and log. Returns (metrics, comparison, RunResult).

    new_batch: when retraining (#24), the batch that triggered it; logged as dataset.new_batch.
    """
    params = {**DEFAULTS, **{k: v for k, v in hyperparams.items() if v is not None}}
    train, test, manifest = load_training_data(training_dir)
    # Fail before the long training run if there is nothing to compare with.
    baseline = baseline_metrics(manifest["data_hash"], experiment)

    start = time.perf_counter()
    model, scale_pos_weight = fit(train, **params)
    train_seconds = time.perf_counter() - start
    trees = int(model.named_steps["model"].best_iteration) + 1
    log.info("trained in %.1fs, best round %d of max %d", train_seconds, trees, params["n_estimators"])

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds
    metrics["n_trees"] = trees

    comparison = compare_to_baseline(metrics, baseline)
    for m in ("pr_auc", "recall", "precision", "f1"):
        metrics[f"vs_baseline_{m}"] = comparison[m]["change"]

    result = log_model_run(
        model, MODEL_NAME,
        params={**params, "scale_pos_weight": round(scale_pos_weight, 4), "tree_method": "hist",
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "validation_fraction": VALIDATION_FRACTION,
                "random_state": RANDOM_STATE, "features": ",".join(FEATURES), "excluded": ",".join(EXCLUDED)},
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-estimators", type=int, help="maximum trees (early stopping may use fewer)")
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-child-weight", type=float)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--colsample-bytree", type=float)
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: xgboost)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    parser.add_argument("--new-batch", help="when retraining: the batch that triggered it (logged as dataset.new_batch)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.experiment, args.run_name, args.author, new_batch=args.new_batch,
        learning_rate=args.learning_rate, n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_child_weight=args.min_child_weight, subsample=args.subsample, colsample_bytree=args.colsample_bytree)
