import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.svm import SVC

from src import evaluation as ev
from src import tracking

# 3 fraud, 3 not fraud. At threshold 0.5: flagged = orders 0, 2, 4 -> tp=2 (0, 4), fp=1 (2), fn=1 (1), tn=2.
Y = [1, 1, 0, 0, 1, 0]
P = [0.9, 0.4, 0.6, 0.1, 0.8, 0.2]


def test_metrics_by_hand():
    m = ev.evaluate(Y, P)
    assert (m["test_tp"], m["test_fp"], m["test_fn"], m["test_tn"]) == (2, 1, 1, 2)
    assert m["test_confusion_matrix"] == [[2, 1], [1, 2]]
    assert m["test_recall"] == pytest.approx(2 / 3)
    assert m["test_precision"] == pytest.approx(2 / 3)
    assert m["test_f1"] == pytest.approx(2 / 3)
    assert m["test_pr_auc"] == pytest.approx(average_precision_score(Y, P))
    assert m["test_roc_auc"] == pytest.approx(8 / 9)  # 8 of the 9 fraud/non-fraud pairs ranked correctly
    assert (m["test_rows"], m["test_fraud_rate"], m["test_threshold"]) == (6, 0.5, 0.5)
    assert not any("accuracy" in k for k in m)  # AC3: not judged on accuracy


def test_threshold_and_prefix():
    strict = ev.evaluate(Y, P, threshold=0.85, prefix="train")
    assert (strict["train_tp"], strict["train_fp"]) == (1, 0)  # only 0.9 is flagged
    assert strict["train_precision"] == 1.0 and strict["train_recall"] == pytest.approx(1 / 3)
    assert ev.flag([0.5, 0.49], 0.5).tolist() == [1, 0]  # at the threshold counts as flagged
    assert "pr_auc" in ev.evaluate(Y, P, prefix=None)


def test_nothing_flagged_gives_zero_not_an_error():
    m = ev.evaluate(Y, P, threshold=0.99)
    assert m["test_tp"] == m["test_fp"] == 0 and m["test_precision"] == 0 and m["test_f1"] == 0


@pytest.mark.parametrize("y, p, message", [
    ([1, 0], [0.5], "y_true has 2 rows but y_score has 1"),
    ([], [], "no rows"),
    ([0, 0, 0], [0.1, 0.2, 0.3], "needs both fraud and non-fraud"),
    ([0, 2], [0.1, 0.2], "must be 0 (not fraud) or 1 (fraud)"),
    ([0, 1], [0.1, 1.5], "probabilities between 0 and 1"),
    ([0, 1], [0.1, float("nan")], "probabilities between 0 and 1"),
])
def test_clear_errors(y, p, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        ev.evaluate(y, p)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=4000, weights=[0.89], random_state=0)
    return X[:3000], y[:3000], X[3000:], y[3000:]


def test_evaluate_model_and_its_own_threshold(data):
    X_tr, y_tr, X_te, y_te = data
    model = LogisticRegression().fit(X_tr, y_tr)
    m = ev.evaluate_model(model, X_te, y_te)
    assert m["test_threshold"] == 0.5 and 0.5 < m["test_pr_auc"] <= 1
    tuned = FixedThresholdClassifier(model, threshold=0.3).fit(X_tr, y_tr)
    assert ev.evaluate_model(tuned, X_te, y_te)["test_threshold"] == 0.3  # uses the model's threshold
    assert ev.evaluate_model(tuned, X_te, y_te, threshold=0.7)["test_threshold"] == 0.7


def test_models_without_probabilities_are_rejected(data):
    X_tr, y_tr, X_te, y_te = data
    with pytest.raises(TypeError, match="SVC has no predict_proba"):
        ev.evaluate_model(SVC().fit(X_tr[:300], y_tr[:300]), X_te, y_te)


def test_probability_column_follows_classes_order():
    class Reversed:  # classes_ in the order [1, 0]
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9]])
    assert ev.fraud_probability(Reversed(), None).tolist() == [0.8, 0.1]


def test_tune_threshold_maximises_f1(data):
    X_tr, y_tr, X_val, y_val = data
    p = LogisticRegression().fit(X_tr, y_tr).predict_proba(X_val)[:, 1]
    t = ev.tune_threshold(y_val, p)
    best = max(f1_score(y_val, p >= c) for c in np.unique(p))
    assert f1_score(y_val, p >= t) == pytest.approx(best)


def test_compare_to_baseline():
    base = ev.evaluate(Y, P)
    better = ev.evaluate([1, 1, 0, 0, 1, 0], [0.9, 0.7, 0.3, 0.1, 0.8, 0.2])
    worse_precision = ev.evaluate(Y, [0.9, 0.6, 0.6, 0.6, 0.8, 0.2])
    c = ev.compare_to_baseline(better, base)
    assert c["beats_baseline"] and c["recall"]["change"] == pytest.approx(1 / 3)
    assert not ev.compare_to_baseline(worse_precision, base)["beats_baseline"]  # more recall, less precision
    with pytest.raises(KeyError, match="test_pr_auc missing"):
        ev.compare_to_baseline({}, base)


def test_metrics_split_cleanly_for_mlflow():
    scalars, others = tracking._split_metrics(ev.evaluate(Y, P))
    assert set(others) == {"test_confusion_matrix"}  # saved as JSON in the run
    assert {"test_pr_auc", "test_recall", "test_precision", "test_f1", "test_tp"} <= set(scalars)


def test_format_metrics():
    line = ev.format_metrics(ev.evaluate(Y, P))
    assert line.startswith("test: pr_auc=") and "recall=0.6667 precision=0.6667 f1=0.6667" in line
    assert line.endswith("(threshold 0.50)")
