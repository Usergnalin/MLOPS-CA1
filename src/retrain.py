"""Pipeline step 'retrain': fetch the latest data and retrain every model on it (PBI-11, #24).

1. src.data_fetch writes a fresh train / test snapshot from MySQL (all loaded batches). By default
   only new batches and the batch being processed are read from the database (incremental);
   --full-fetch / FETCH_MODE=full re-reads everything.
2. The decision tree baseline is trained first: every other model compares itself with it.
3. Each model in MODELS is trained and logged to MLflow, tagged dataset.new_batch=<postfix>
   so the registry shows which batch triggered the retraining.

A model whose file isn't in this checkout yet (e.g. its PR is still open) is skipped with a
warning; a model that fails stops the step. The promote step then picks the champion.

    python -m src.retrain --postfix batch_02
    python -m src.retrain --models decision_tree,random_forest      # a subset, e.g. to save time
    RETRAIN_MODELS=decision_tree,xgboost_model python src/pipeline.py

In the pipeline (src/pipeline.py) this is STEP_MODULES["retrain"]: run(postfix) returns
{"mlflow_urls": [...], "models": {model: test_pr_auc}, "skipped": [...]}.
"""
import argparse
import importlib
import inspect
import logging
import os
import time

from src import data_fetch
from src.config import TRAINING_DIR, setup_logging
from src.tracking import DEFAULT_EXPERIMENT

log = logging.getLogger("retrain")

BASELINE = "decision_tree"
# Modules under src/models/, trained in this order (the baseline must come first).
MODELS = [BASELINE, "logistic_regression", "svm", "gradient_boosting", "xgboost_model", "random_forest", "mlp"]


def model_list(models=None):
    """The models to train: the argument, else $RETRAIN_MODELS, else MODELS. Baseline always first."""
    if models is None:
        models = os.environ.get("RETRAIN_MODELS") or MODELS
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {', '.join(MODELS)}")
    return [BASELINE] + [m for m in models if m != BASELINE]


def import_model(name):
    """src.models.<name>, or None if that file isn't in this checkout."""
    module = f"src.models.{name}"
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module:
            return None
        raise  # the model exists but one of its imports is missing: show that


def train(module, new_batch, training_dir, experiment):
    """Run one model script's run(); returns (its test metrics, its src.tracking RunResult)."""
    kwargs = {"training_dir": training_dir, "experiment": experiment}
    if "new_batch" in inspect.signature(module.run).parameters:  # the baseline has no new_batch
        kwargs["new_batch"] = new_batch
    out = module.run(**kwargs)
    return out[0], out[-1]  # every model returns (metrics, ..., RunResult)


def run(postfix="base", models=None, training_dir=TRAINING_DIR, experiment=DEFAULT_EXPERIMENT, fetch=True,
        full_fetch=None):
    """Fetch the latest data and retrain the models. Returns the summary dict described above.

    full_fetch: None follows $FETCH_MODE (default incremental); True re-reads the whole database.
    """
    names = model_list(models)
    if fetch:
        # The batch being processed is always re-read, so a re-loaded batch is never stale.
        data_fetch.run(training_dir, full=full_fetch, refresh=[postfix])

    urls, scores, skipped = [], {}, []
    for name in names:
        module = import_model(name)
        if module is None:
            if name == BASELINE:
                raise RuntimeError("src/models/decision_tree.py is missing: every model needs the baseline")
            log.warning("skipping %s: src/models/%s.py is not in this checkout", name, name)
            skipped.append(name)
            continue
        start = time.perf_counter()
        metrics, result = train(module, postfix, training_dir, experiment)
        scores[name] = metrics.get("test_pr_auc")
        if result.run_url:
            urls.append(result.run_url)
        log.info("%-20s done in %.1fs", name, time.perf_counter() - start)
    return {"mlflow_urls": urls, "models": scores, "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postfix", default="base", help="the batch that triggered the retraining")
    parser.add_argument("--models", help=f"comma-separated subset of: {', '.join(MODELS)}")
    parser.add_argument("--no-fetch", action="store_true", help="re-use the snapshot in data/training")
    parser.add_argument("--full-fetch", action="store_true", help="re-read all data from the database")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    args = parser.parse_args()
    setup_logging()
    run(args.postfix, args.models, experiment=args.experiment, fetch=not args.no_fetch,
        full_fetch=True if args.full_fetch else None)
