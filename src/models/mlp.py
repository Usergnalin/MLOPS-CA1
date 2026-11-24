"""Multi-layer perceptron (neural network) model, evaluated against the decision tree baseline (#18).

Trains on the shared training data (src.data_fetch), scores it with the shared evaluation module
(src.evaluation), compares it with the baseline run on the same data (src.models.decision_tree)
and logs everything to MLflow (src.tracking).

Unlike a tree, an MLP is sensitive to the scale of its inputs, so the heavy-tailed counts and
amounts (a few customers have thousands of orders) are log1p'd and standardised first;
categories are one-hot encoded. Early stopping holds out 10% of the fitting rows and stops once
their score hasn't improved for PATIENCE epochs.

Speed. The whole pipeline has to run in 3 minutes (PBI-11), so the network is fitted on a random,
fraud-rate-preserving sample of at most MAX_ROWS orders, with (64, 32) hidden units and large
batches. On the base batch this trains in seconds instead of ~1 minute for ~0.016 less PR-AUC
(validation 0.547 vs 0.563 for (128, 64) on every row). --max-rows 0 fits on every row.

Threshold. An MLPClassifier has no class weights, so its probabilities are calibrated to the
~11% fraud rate and 0.5 flags too few orders. The threshold is chosen on a slice of the training
data (THRESHOLD_FRACTION, never the test set) and saved inside the logged model, so predict()
uses it. Rule (default): flag enough orders to catch at least MIN_RECALL of the fraud, and among
those thresholds take the highest, i.e. the fewest false alarms. The baseline catches ~79%, so
0.80 makes "catches more fraud" (PBI-01 AC1) the aim and precision (AC2) the thing to improve.
--min-recall 0 instead picks the threshold with the best F1.

Features are the baseline's (same columns, same leak exclusions), so the comparison is fair.

Usage:
    python -m src.models.decision_tree          # the baseline must exist for the same data first
    python -m src.models.mlp
    python -m src.models.mlp --hidden 128 64 --max-rows 0 --min-recall 0.85   # slower, slightly better
"""
import argparse
import getpass
import logging
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import FixedThresholdClassifier, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.dataset_tags import dataset_tags
from src.evaluation import compare_to_baseline, evaluate_model, format_metrics, tune_threshold
from src.models.decision_tree import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, baseline_metrics
from src.tracking import DEFAULT_EXPERIMENT, log_model_run

log = logging.getLogger("mlp")

MODEL_NAME = "mlp"
RANDOM_STATE = 42
THRESHOLD_FRACTION = 0.15  # of the training rows, held out to choose the threshold
FLAGS = ["mobile_verified"]                          # already 0/1
LOG_SCALED = [c for c in NUMERIC if c not in FLAGS]  # counts and amounts: log1p, then standardise
DEFAULTS = {"hidden": (64, 32), "alpha": 1e-4, "learning_rate": 3e-3, "batch_size": 4096, "max_iter": 200,
            "patience": 5, "max_rows": 300_000,
            "min_recall": 0.80}


def build_model(hidden=(64, 32), alpha=1e-4, learning_rate=3e-3, batch_size=4096, max_iter=200, patience=5):
    preprocess = ColumnTransformer(
        [
            # handle_unknown="ignore": a payment method first seen in a new batch won't crash scoring.
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("log_scaled", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), LOG_SCALED),
            ("flags", "passthrough", FLAGS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden), alpha=alpha, learning_rate_init=learning_rate,
        batch_size=batch_size, max_iter=max_iter, early_stopping=True, n_iter_no_change=patience,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocess), ("model", mlp)])


def threshold_for_recall(y_true, y_score, min_recall):
    """Highest threshold that still flags at least min_recall of the fraud (fewest false alarms)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    reachable = np.where(recall[:-1] >= min_recall)[0]  # recall falls as the threshold rises
    return float(thresholds[reachable[-1]])


def choose_threshold(y_true, y_score, min_recall):
    return threshold_for_recall(y_true, y_score, min_recall) if min_recall else tune_threshold(y_true, y_score)


def fit_with_threshold(X, y, min_recall=0.80, max_rows=300_000, **params):
    """Fit on most of the training rows (at most max_rows of them), choose the threshold on the rest.

    Returns (model with the threshold baked in, threshold, fitted pipeline).
    """
    X_fit, X_thr, y_fit, y_thr = train_test_split(X, y, test_size=THRESHOLD_FRACTION, stratify=y,
                                                  random_state=RANDOM_STATE)
    if max_rows and len(X_fit) > max_rows:
        # Stratified, so the sample keeps the fraud rate; the threshold slice stays full size.
        X_fit, _, y_fit, _ = train_test_split(X_fit, y_fit, train_size=max_rows, stratify=y_fit,
                                              random_state=RANDOM_STATE)
    pipeline = build_model(**params).fit(X_fit, y_fit)
    threshold = choose_threshold(y_thr, pipeline.predict_proba(X_thr)[:, 1], min_recall)
    # FrozenEstimator: fit() below only records the classes, the MLP is not retrained.
    model = FixedThresholdClassifier(FrozenEstimator(pipeline), threshold=threshold).fit(X_thr, y_thr)
    return model, threshold, pipeline


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
    min_recall = params.pop("min_recall")
    model, threshold, pipeline = fit_with_threshold(train[FEATURES], train[TARGET], min_recall, **params)
    train_seconds = time.perf_counter() - start
    mlp = pipeline.named_steps["model"]
    log.info("trained %d epochs in %.1fs, threshold %.3f", mlp.n_iter_, train_seconds, threshold)

    metrics = evaluate_model(model, test[FEATURES], test[TARGET])  # uses the model's threshold
    metrics |= evaluate_model(model, train[FEATURES], train[TARGET], prefix="train")
    metrics["train_seconds"] = train_seconds
    metrics["epochs"] = int(mlp.n_iter_)

    comparison = compare_to_baseline(metrics, baseline)
    for m in ("pr_auc", "recall", "precision", "f1"):
        metrics[f"vs_baseline_{m}"] = comparison[m]["change"]

    result = log_model_run(
        model, MODEL_NAME,
        params={**params, "hidden": ",".join(map(str, params["hidden"])), "min_recall": min_recall,
                "threshold_rule": f"recall>={min_recall}" if min_recall else "best_f1",
                "threshold": round(threshold, 6), "threshold_fraction": THRESHOLD_FRACTION,
                "early_stopping": True, "random_state": RANDOM_STATE,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", help="hidden layer sizes, e.g. 128 64")
    parser.add_argument("--alpha", type=float, help="L2 regularisation")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-iter", type=int, help="max epochs (early stopping usually ends sooner)")
    parser.add_argument("--patience", type=int, help="stop after this many epochs without improvement (default 5)")
    parser.add_argument("--max-rows", type=int, help="fit on at most this many orders (default 300000; 0 = all)")
    parser.add_argument("--min-recall", type=float,
                        help="share of fraud the threshold must catch (default 0.80); 0 = best F1 instead")
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: mlp)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    parser.add_argument("--new-batch", help="when retraining: the batch that triggered it (logged as dataset.new_batch)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.experiment, args.run_name, args.author, new_batch=args.new_batch,
        hidden=tuple(args.hidden) if args.hidden else None, alpha=args.alpha, learning_rate=args.learning_rate,
        batch_size=args.batch_size, max_iter=args.max_iter, patience=args.patience, max_rows=args.max_rows,
        min_recall=args.min_recall)
