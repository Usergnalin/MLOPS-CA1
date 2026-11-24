"""Shared evaluation for every fraud model, so all members' results are computed the same way.

Models are judged on PR-AUC, recall, precision and F1, not accuracy (PBI-01 AC3): with ~11% fraud,
a model that flags nothing is already ~89% accurate. Every model gives each order a fraud
probability, and orders at or above a threshold are flagged (AC4).

    from src.data_fetch import TARGET, load_training_data
    from src.evaluation import evaluate_model, compare_to_baseline, tune_threshold

    metrics = evaluate_model(model, X_test, y_test)                 # keys like "test_pr_auc"
    metrics |= evaluate_model(model, X_train, y_train, prefix="train")  # for overfitting checks
    log_model_run(model, "random_forest", metrics=metrics, ...)    # src.tracking

    compare_to_baseline(metrics, baseline_metrics)   # does it beat the decision tree? (AC1, AC2)

Metric keys (with prefix "test"):
    test_pr_auc      area under the precision-recall curve (average precision); the main ranking
    test_roc_auc     area under the ROC curve
    test_recall      share of fraud orders that were flagged (AC1)
    test_precision   share of flagged orders that really were fraud (AC2)
    test_f1          balance of recall and precision
    test_tp / test_fp / test_fn / test_tn   confusion matrix counts
    test_confusion_matrix   [[tn, fp], [fn, tp]], saved by src.tracking as a JSON file in the run
    test_threshold, test_rows, test_fraud_rate
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5
# Higher is better for all of these; PR-AUC first because it is the team's ranking metric.
MAIN_METRICS = ["pr_auc", "recall", "precision", "f1"]


def _check(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true has {len(y_true)} rows but y_score has {len(y_score)}")
    if len(y_true) == 0:
        raise ValueError("nothing to evaluate: no rows")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must be 0 (not fraud) or 1 (fraud)")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true needs both fraud and non-fraud orders to compute PR-AUC")
    if np.isnan(y_score).any() or y_score.min() < 0 or y_score.max() > 1:
        raise ValueError("y_score must be fraud probabilities between 0 and 1")
    return y_true, y_score


def flag(y_score, threshold=DEFAULT_THRESHOLD):
    """Orders flagged as fraud: probability at or above the threshold (1) or not (0)."""
    return (np.asarray(y_score, dtype=float) >= threshold).astype(int)


def evaluate(y_true, y_score, threshold=DEFAULT_THRESHOLD, prefix="test"):
    """Metrics for fraud probabilities y_score against true labels y_true (1 = fraud)."""
    y_true, y_score = _check(y_true, y_score)
    y_pred = flag(y_score, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
        "rows": len(y_true),
        "fraud_rate": float(y_true.mean()),
    }
    return {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}


def fraud_probability(model, X):
    """Probability of fraud (class 1) for each row of X."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} has no predict_proba(); every model must output a fraud "
                        "probability (e.g. SVC(probability=True) or wrap it in CalibratedClassifierCV)")
    classes = list(getattr(model, "classes_", [0, 1]))
    return model.predict_proba(X)[:, classes.index(1)]


def evaluate_model(model, X, y, threshold=None, prefix="test"):
    """evaluate() for a fitted model. threshold defaults to the model's own (see tune_threshold)."""
    if threshold is None:
        threshold = getattr(model, "threshold", DEFAULT_THRESHOLD)
    return evaluate(y, fraud_probability(model, X), threshold, prefix)


def tune_threshold(y_true, y_score, metric="f1"):
    """Threshold that maximises F1 on the given rows.

    Use rows the model was not trained on and that are not the test set (e.g. a validation split
    carved out of train); tuning on test would make the test scores look better than they are.
    """
    if metric != "f1":
        raise ValueError("only metric='f1' is supported")
    y_true, y_score = _check(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])  # the last curve point has no threshold


def compare_to_baseline(metrics, baseline, prefix="test"):
    """How a model's metrics compare with the baseline model's (both from evaluate()).

    Returns {"<metric>": {"model", "baseline", "change"}, ..., "beats_baseline": bool}.
    beats_baseline means higher recall (AC1) and higher precision (AC2) than the baseline.
    """
    result = {}
    for m in MAIN_METRICS:
        key = f"{prefix}_{m}" if prefix else m
        if key not in metrics or key not in baseline:
            raise KeyError(f"{key} missing - pass the output of evaluate() with prefix={prefix!r}")
        result[m] = {"model": metrics[key], "baseline": baseline[key], "change": metrics[key] - baseline[key]}
    result["beats_baseline"] = result["recall"]["change"] > 0 and result["precision"]["change"] > 0
    return result


def format_metrics(metrics, prefix="test"):
    """One log line, e.g. 'test: pr_auc=0.8578 recall=0.7937 precision=0.7800 f1=0.7868 (threshold 0.39)'."""
    p = f"{prefix}_" if prefix else ""
    main = " ".join(f"{m}={metrics[p + m]:.4f}" for m in MAIN_METRICS)
    return f"{prefix or 'metrics'}: {main} (threshold {metrics[p + 'threshold']:.2f})"
