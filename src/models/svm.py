"""Support Vector Machine (SVM) fraud model, evaluated against the decision tree baseline (#63).

A class-balanced linear SVM (LinearSVC) on the same preprocessing as the Logistic Regression
(src.models.logistic_regression): one-hot categories, imputed and standardised numbers. An SVM
outputs a distance from its boundary, not a probability, so it is wrapped in
CalibratedClassifierCV (3 folds), which turns that distance into the fraud probability every
model must give (src.evaluation). A linear kernel keeps training fast on ~760k orders.

Features are the baseline's (same columns, same leak exclusions), so the comparison is fair.

Usage:
    python -m src.models.decision_tree          # the baseline must exist for the same data first
    python -m src.models.svm
    python -m src.models.svm --C 0.5 --max-iter 2000
    python -m src.models.svm --new-batch batch2
"""
import argparse
import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import TRAINING_DIR, setup_logging
from src.models.decision_tree import EXCLUDED, FEATURES  # noqa: F401 - same features as the baseline
from src.models.logistic_regression import RANDOM_STATE, build_preprocess, train_and_log
from src.tracking import DEFAULT_EXPERIMENT

log = logging.getLogger("svm")

MODEL_NAME = "svm"
CALIBRATION_FOLDS = 3
DEFAULTS = {"C": 1.0, "max_iter": 2000}


def build_model(C=1.0, max_iter=2000):
    svc = LinearSVC(C=C, max_iter=max_iter, class_weight="balanced", random_state=RANDOM_STATE)
    clf = CalibratedClassifierCV(svc, cv=CALIBRATION_FOLDS)
    return Pipeline([("preprocess", build_preprocess()), ("model", clf)])


def run(training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT, run_name=None, author=None, new_batch=None,
        **hyperparams):
    """Train, evaluate, compare with the baseline and log. Returns (metrics, comparison, RunResult).

    new_batch: when retraining (#24), the batch that triggered it; logged as dataset.new_batch.
    """
    params = {**DEFAULTS, **{k: v for k, v in hyperparams.items() if v is not None}}
    return train_and_log(build_model(**params), MODEL_NAME,
                         {**params, "calibration": f"sigmoid, cv={CALIBRATION_FOLDS}"}, training_dir,
                         experiment, run_name, author, new_batch, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, help="inverse regularisation strength (default 1.0)")
    parser.add_argument("--max-iter", type=int, help="solver iterations (default 2000)")
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--run-name", help="MLflow run name (default: svm)")
    parser.add_argument("--author", help="author tag (default: OS user name)")
    parser.add_argument("--new-batch", help="when retraining: the batch that triggered it (logged as dataset.new_batch)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir, args.experiment, args.run_name, args.author, new_batch=args.new_batch,
        C=args.C, max_iter=args.max_iter)
