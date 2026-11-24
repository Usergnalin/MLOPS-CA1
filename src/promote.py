"""Pick the production model: shortlist the top 2 model types and keep the best as @champion (#27, #28).

Run after the models of a pipeline run have been trained and logged (PBI-04 AC2, AC3):

1. Candidates are the finished runs in the experiment that were trained on the current data
   (same dataset.data_hash as the data_fetch manifest) and have a test PR-AUC.
2. Only the best run of each model type counts (retraining the same model twice can't fill both
   places). The top 2 model types by test_pr_auc are registered as versions of one registered model in the MLflow
   Model Registry and shortlisted: the best gets alias @challenger, the second @runner_up.
3. The challenger is compared with the current @champion. The champion may have been trained on
   older data, so it is re-scored on the *current* test set first; both are then judged on the
   same rows. The challenger becomes @champion only if its PR-AUC is higher (by at least
   --min-improvement). The replaced champion keeps alias @previous_champion, so it can be
   restored. If there is no champion yet, the challenger becomes the first one.

Every decision is written as tags on the model versions, so the registry shows why a model is
(or isn't) in production.

Usage:
    python -m src.promote                 # shortlist + promote if better
    python -m src.promote --dry-run       # show what would happen, change nothing

In code (e.g. the pipeline flow, #47):
    from src.promote import select_and_promote
    decision = select_and_promote()
"""
import argparse
import logging
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import TRAINING_DIR, setup_logging
from src.data_fetch import TARGET, load_training_data
from src.evaluation import evaluate_model
from src.tracking import DEFAULT_EXPERIMENT, setup_mlflow

log = logging.getLogger("promote")

REGISTERED_MODEL = "fraud-detector"
RANKING_METRIC = "test_pr_auc"
CHAMPION, CHALLENGER, RUNNER_UP, PREVIOUS = "champion", "challenger", "runner_up", "previous_champion"
SHORTLIST_SIZE = 2


def candidate_runs(data_hash, experiment=DEFAULT_EXPERIMENT):
    """Finished runs trained on data_hash, best test PR-AUC first (a pandas DataFrame)."""
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=(f"attributes.status = 'FINISHED' and tags.`dataset.data_hash` = '{data_hash}' "
                       f"and metrics.{RANKING_METRIC} > -1"),
        order_by=[f"metrics.{RANKING_METRIC} DESC", "attributes.start_time DESC"],
    )
    return runs


def best_per_model_type(runs):
    """Keep each model type's best run (runs come sorted best first); untagged runs count on their own."""
    if runs.empty:
        return runs
    model_type = runs.get("tags.model_type", runs["run_id"]).fillna(runs["run_id"])
    return runs[~model_type.duplicated()]


def logged_model_uri(run_id, experiment_id):
    """models:/<id> URI of the model logged by a run (src.tracking logs exactly one)."""
    models = mlflow.search_logged_models(experiment_ids=[experiment_id],
                                         filter_string=f"source_run_id = '{run_id}'", output_format="list")
    if not models:
        raise LookupError(f"run {run_id} has no logged model")
    return f"models:/{models[0].model_id}"


def register(client, run, registered_model):
    """The registry version for a run's model, registering it if needed."""
    run_id = run["run_id"]
    for version in client.search_model_versions(f"name = '{registered_model}' and run_id = '{run_id}'"):
        return version
    version = mlflow.register_model(logged_model_uri(run_id, run["experiment_id"]), registered_model, tags={
        "model_type": run.get("tags.model_type") or "",
        "data_hash": run["tags.dataset.data_hash"],
        RANKING_METRIC: f"{run[f'metrics.{RANKING_METRIC}']:.6f}",
    })
    log.info("registered %s as %s version %s", run.get("tags.model_type"), registered_model, version.version)
    return version


def alias_version(client, registered_model, alias):
    try:
        return client.get_model_version_by_alias(registered_model, alias)
    except MlflowException:
        return None


def rescore(version, test):
    """Test PR-AUC of a registered model version on the given (current) test rows."""
    model = mlflow.sklearn.load_model(f"models:/{version.name}/{version.version}")
    features = list(getattr(model, "feature_names_in_", test.columns.drop(TARGET)))
    return evaluate_model(model, test[features], test[TARGET])[RANKING_METRIC]


def select_and_promote(experiment=DEFAULT_EXPERIMENT, training_dir=TRAINING_DIR, registered_model=REGISTERED_MODEL,
                       min_improvement=0.0, dry_run=False):
    """Shortlist the top 2 runs on the current data and promote the best if it beats @champion.

    Returns a dict describing the decision (also logged).
    """
    _, test, manifest = load_training_data(training_dir)
    data_hash = manifest["data_hash"]
    setup_mlflow(experiment)
    client = MlflowClient()

    runs = best_per_model_type(candidate_runs(data_hash, experiment))
    if runs.empty:
        raise LookupError(f"no finished runs with {RANKING_METRIC} for data {data_hash[:12]} in {experiment!r} - "
                          "train the models first")
    shortlist = runs.head(SHORTLIST_SIZE)
    for rank, (_, run) in enumerate(shortlist.iterrows(), start=1):
        log.info("shortlist #%d: %-18s %s=%.4f  run %s", rank, run.get("tags.model_type"), RANKING_METRIC,
                 run[f"metrics.{RANKING_METRIC}"], run["run_id"])

    best = shortlist.iloc[0]
    decision = {
        "data_hash": data_hash,
        "shortlist": [{"run_id": r["run_id"], "model_type": r.get("tags.model_type"),
                       RANKING_METRIC: float(r[f"metrics.{RANKING_METRIC}"])} for _, r in shortlist.iterrows()],
        "challenger_pr_auc": float(best[f"metrics.{RANKING_METRIC}"]),
        "champion_pr_auc": None,
        "promoted": False,
        "dry_run": dry_run,
    }

    champion = alias_version(client, registered_model, CHAMPION)
    if champion is not None and champion.run_id == best["run_id"]:
        decision["reason"] = "the best run is already the champion"
    else:
        if champion is not None:
            decision["champion_pr_auc"] = rescore(champion, test)
            log.info("current champion: version %s (%s), %s on current test data = %.4f", champion.version,
                     champion.tags.get("model_type"), RANKING_METRIC, decision["champion_pr_auc"])
            better = decision["challenger_pr_auc"] > decision["champion_pr_auc"] + min_improvement
            decision["reason"] = (f"challenger {decision['challenger_pr_auc']:.4f} "
                                  f"{'beats' if better else 'does not beat'} champion "
                                  f"{decision['champion_pr_auc']:.4f} (+{min_improvement} needed)")
        else:
            better = True
            decision["reason"] = "no champion yet"
        decision["promoted"] = better

    log.info("decision: %s -> %s", decision["reason"], "PROMOTE" if decision["promoted"] else "keep champion")
    if dry_run:
        return decision

    # Register and alias the shortlist (#27).
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = [register(client, run, registered_model) for _, run in shortlist.iterrows()]
    for alias, version in zip([CHALLENGER, RUNNER_UP], versions):
        client.set_registered_model_alias(registered_model, alias, version.version)
        client.set_model_version_tag(registered_model, version.version, "shortlisted_at", now)

    # Promote (#28).
    if decision["promoted"]:
        new = versions[0]
        if champion is not None:
            client.set_registered_model_alias(registered_model, PREVIOUS, champion.version)
            client.set_model_version_tag(registered_model, champion.version, "replaced_at", now)
        client.set_registered_model_alias(registered_model, CHAMPION, new.version)
        client.set_model_version_tag(registered_model, new.version, "promoted_at", now)
        client.set_model_version_tag(registered_model, new.version, "promotion_reason", decision["reason"])
        decision["champion_version"] = new.version
        log.info("@%s is now %s version %s (%s)", CHAMPION, registered_model, new.version,
                 best.get("tags.model_type"))
    else:
        decision["champion_version"] = champion.version
    return decision



def run(postfix=None, **kwargs):
    """Pipeline step 'promote' (src/pipeline.py): select_and_promote() in the pipeline's summary format.

    postfix is not needed (the candidates are the runs on the current data) but is part of the step
    contract. Returns {"promoted", "model_version", "mlflow_url", "decision"}.
    """
    decision = select_and_promote(**kwargs)
    uri = mlflow.get_tracking_uri()
    version = decision.get("champion_version")
    url = None
    if uri.startswith("http") and version is not None:
        url = f"{uri.rstrip('/')}/#/models/{kwargs.get('registered_model', REGISTERED_MODEL)}/versions/{version}"
    return {"promoted": decision["promoted"], "model_version": version, "mlflow_url": url, "decision": decision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--training-dir", default=TRAINING_DIR, help="data written by src.data_fetch")
    parser.add_argument("--registered-model", default=REGISTERED_MODEL, help="Model Registry name")
    parser.add_argument("--min-improvement", type=float, default=0.0,
                        help="PR-AUC a challenger must gain over the champion (default 0)")
    parser.add_argument("--dry-run", action="store_true", help="show the decision without changing the registry")
    args = parser.parse_args()
    setup_logging()
    select_and_promote(args.experiment, args.training_dir, args.registered_model, args.min_improvement, args.dry_run)
