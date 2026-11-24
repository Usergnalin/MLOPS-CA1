"""PBI-11: one command that runs the whole pipeline, times every step, and publishes a team report.

    python src/pipeline.py                         # clean -> load -> retrain -> promote
    python src/pipeline.py --postfix batch_02      # a new raw drop: data/raw/<table>_batch_02.csv
    python src/pipeline.py --skip clean,load       # re-use data that is already loaded
    python src/pipeline.py --only retrain,promote
    python src/pipeline.py --check                 # validate step modules without running them
    python src/pipeline.py --time-budget 180 --strict-budget   # exit code 2 if over 3 minutes

The flow runs as a Prefect flow when Prefect is installed (`pip install prefect`), so runs and
Markdown summary artifacts appear in the Prefect UI. Without Prefect it runs as plain Python with
the same behaviour.

Team-accessible reporting (AC4)
-------------------------------
At the end of every run (including failed runs), the pipeline:
  1. Prints a structured summary to stdout (step timings, cleaned/rejected rows, loaded rows,
     promotion status, and all MLflow links).
  2. Writes JSON and Markdown reports to `data/reports/` (`pipeline_report_<postfix>_<ts>.{json,md}`
     and `latest_<postfix>.{json,md}`).
  3. Logs a summary run + Markdown/JSON artifacts to the shared MLflow server when
     `MLFLOW_TRACKING_URI` is set (disable with `--no-mlflow-report`).
  4. Publishes a Prefect Markdown artifact when running under Prefect.
  5. Posts the summary to `PIPELINE_WEBHOOK_URL` (Slack / Teams / Discord) if configured.

Step contract
-------------
Every step except `clean` lives in its own module under `src/` and exposes one function:

    def run(postfix: str) -> dict | None

The returned dict is optional. These keys are picked up for the end-of-run summary:

    load      {"rows_loaded": {"customers": 899000, "orders": 949000, "labels": 948000}}
    retrain   {"mlflow_urls": ["http://.../#/experiments/1/runs/abc", ...]}   (or "mlflow_url": "...")
    promote   {"mlflow_url": "...", "promoted": True, "model_version": 3}

The module for each step is listed in `STEP_MODULES` below. Change a name there if a step file is
named differently.
"""
from __future__ import annotations

import argparse
import getpass
import importlib
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `python src/pipeline.py`: the step modules import each other as src.<module>
    sys.path.insert(0, str(ROOT))
DEFAULT_REPORT_DIR = ROOT / "data" / "reports"
DEFAULT_MLFLOW_PIPELINE_EXPERIMENT = "pipeline-runs"

try:  # run as a script from src/ (like the notebook) ...
    from config import setup_logging
except ModuleNotFoundError as exc:  # ... or imported as src.pipeline (like the tests)
    if exc.name == "config":
        try:
            from src.config import setup_logging
        except ModuleNotFoundError:
            setup_logging = None
    else:
        setup_logging = None

if setup_logging is None:
    def setup_logging(level=logging.INFO):
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

try:
    from prefect import flow, task
    from prefect.artifacts import create_markdown_artifact

    PREFECT_AVAILABLE = True
except ImportError:  # no Prefect: decorators are no-ops and the pipeline runs as plain Python
    PREFECT_AVAILABLE = False

    def task(fn=None, **_kwargs):
        return (lambda f: f) if fn is None else fn

    flow = task
    create_markdown_artifact = None

log = logging.getLogger("pipeline")

STEPS = ("clean", "load", "retrain", "promote")  # run in this order
STEP_MODULES = {  # step -> module under src/ that defines run(postfix)
    "clean": "data_clean",
    "load": "data_load",
    "retrain": "retrain",
    "promote": "promote",
}
DEFAULT_TIME_BUDGET_SECONDS = 180.0  # PBI-11 AC2 target for the whole run (3 minutes)


class PipelineError(RuntimeError):
    """A step failed or is missing. `.result` holds partial run state for reporting."""

    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result or {}


# ---------------------------------------------------------------------------
# Finding and running steps
# ---------------------------------------------------------------------------
def _import(module: str):
    """Import src.<module> (package style), or `module` as a top-level module (script style)."""
    for name in (f"src.{module}", module):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name not in (name, module, "src"):
                raise  # the step exists but one of ITS imports is missing: surface that error
    raise ModuleNotFoundError(module, name=module)


def _clean_step(postfix: str) -> dict:
    result = _import(STEP_MODULES["clean"]).run(postfix=postfix)  # data_clean.run returns CleanResult
    return {"report": result.report}


def _load_step(name: str) -> Callable[[str], dict | None]:
    """Return the callable that runs one step. Raises PipelineError if unavailable."""
    module = STEP_MODULES[name]
    try:
        mod = _import(module)
    except ModuleNotFoundError as exc:
        if exc.name in (module, f"src.{module}") or str(exc) == module:
            raise PipelineError(
                f"step '{name}': src/{module}.py not found "
                f"(it must define run(postfix); see the docstring of src/pipeline.py)"
            ) from None
        raise PipelineError(
            f"step '{name}': missing dependency '{exc.name}' required by src/{module}.py "
            f"(run: pip install -r requirements.txt)"
        ) from None
    if name == "clean":
        return _clean_step
    if not callable(getattr(mod, "run", None)):
        raise PipelineError(f"step '{name}': src/{module}.py has no run(postfix) function")
    return mod.run


@task(task_run_name="{name}")
def execute_step(name: str, postfix: str) -> dict:
    return _load_step(name)(postfix) or {}


def select_steps(only: list[str] | None = None, skip: list[str] | tuple[str, ...] = ()) -> list[str]:
    unknown = [s for s in list(only or []) + list(skip) if s not in STEPS]
    if unknown:
        raise PipelineError(f"unknown step(s) {unknown}. Choose from: {', '.join(STEPS)}")
    selected = [s for s in STEPS if (not only or s in only) and s not in skip]
    if not selected:
        raise PipelineError("no steps selected to run")
    return selected


def preflight_check(only: list[str] | None = None, skip: list[str] | None = None) -> list[str]:
    """Verify every selected step module exists and exposes run(postfix)."""
    names = select_steps(only, skip or [])
    for name in names:
        _load_step(name)
    return names


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------
@flow(name="fraud-pipeline")
def pipeline(
    postfix: str = "base",
    only: list[str] | None = None,
    skip: list[str] | None = None,
    time_budget: float = DEFAULT_TIME_BUDGET_SECONDS,
    report_dir: Path | str | None = DEFAULT_REPORT_DIR,
    publish_mlflow: bool = True,
    mlflow_experiment: str | None = None,
    webhook_url: str | None = None,
) -> dict:
    """Run selected steps in order, time them, and publish team-accessible reports.

    Returns a summary dict. Raises PipelineError if a step is missing or fails.
    """
    names = preflight_check(only, skip)
    started_at = _utc_now()
    result: dict[str, Any] = {
        "postfix": postfix,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "steps": [],
        "outputs": {},
        "time_budget": time_budget,
        "total_seconds": 0.0,
        "within_budget": True,
        "report_files": {},
        "report_urls": {},
    }

    start = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        log.info("step '%s' started (postfix=%s)", name, postfix)
        try:
            out = execute_step(name, postfix)
        except Exception as exc:
            elapsed = _since(t0)
            result["steps"].append({"name": name, "status": "failed", "seconds": elapsed, "error": str(exc)})
            result["total_seconds"] = _since(start)
            result["within_budget"] = result["total_seconds"] <= time_budget
            result["finished_at"] = _utc_now()
            result["status"] = "failed"
            result["failed_step"] = name
            log.error("step '%s' failed after %.2f s: %s", name, elapsed, exc)
            _publish_reports(result, report_dir, publish_mlflow, mlflow_experiment, webhook_url)
            raise PipelineError(f"step '{name}' failed: {exc}", result) from exc

        elapsed = _since(t0)
        result["steps"].append({"name": name, "status": "ok", "seconds": elapsed})
        result["outputs"][name] = out
        log.info("step '%s' finished in %.2f s", name, elapsed)

    result["total_seconds"] = _since(start)
    result["within_budget"] = result["total_seconds"] <= time_budget
    result["finished_at"] = _utc_now()
    result["status"] = "ok"
    log.info(
        "pipeline finished in %.2f s (budget %.0f s, %s)",
        result["total_seconds"],
        time_budget,
        "within budget" if result["within_budget"] else "OVER BUDGET",
    )
    _publish_reports(result, report_dir, publish_mlflow, mlflow_experiment, webhook_url)
    return result


def _since(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# End-of-run summaries & team-accessible reporting (AC4)
# ---------------------------------------------------------------------------
def mlflow_links(result_or_outputs: dict) -> list[tuple[str, str]]:
    """Return (source, url) pairs for every MLflow link reported by steps or the pipeline summary."""
    outputs = result_or_outputs.get("outputs", result_or_outputs)
    links: list[tuple[str, str]] = []
    for step, out in outputs.items():
        if not isinstance(out, dict):
            continue
        for url in [out.get("mlflow_url")] + list(out.get("mlflow_urls") or []):
            if url:
                links.append((step, str(url)))
    pipeline_mlflow = (result_or_outputs.get("report_urls") or {}).get("mlflow")
    if pipeline_mlflow:
        links.append(("pipeline", str(pipeline_mlflow)))
    return links


def format_summary(result: dict) -> str:
    """Format a readable plain-text summary for terminal output."""
    out = result.get("outputs", {})
    lines = [
        "",
        f"=== Pipeline summary (postfix: {result.get('postfix')}) ===",
        "",
        f"{'step':<10}{'status':<9}{'seconds':>9}",
    ]
    lines += [f"{s['name']:<10}{s['status']:<9}{s['seconds']:>9.1f}" for s in result.get("steps", [])]

    total, budget = result.get("total_seconds", 0.0), result.get("time_budget")
    if budget and not result.get("failed_step"):
        verdict = "within budget" if total <= budget else "OVER BUDGET"
        lines.append(f"{'total':<10}{'':<9}{total:>9.1f}   (budget {budget:.0f} s, {verdict})")
    else:
        lines.append(f"{'total':<10}{'':<9}{total:>9.1f}")

    report = out.get("clean", {}).get("report")
    if report:
        lines += ["", "Rows (cleaning):"]
        for table, r in report.items():
            lines.append(
                f"  {table:<10} read {r['rows_in']:>9,}   clean {r['rows_clean']:>9,}   "
                f"rejected {r['rows_rejected']:>7,} ({r['pct_rejected']:.2f}%)"
            )

    loaded = out.get("load", {}).get("rows_loaded")
    if loaded:
        lines += ["", "Rows loaded into MySQL:"]
        lines += [f"  {table:<10} {n:>9,}" for table, n in loaded.items()]

    promote = out.get("promote", {})
    if "promoted" in promote:
        version = f" (model version {promote['model_version']})" if promote.get("model_version") else ""
        lines += [
            "",
            "Promotion: "
            + (
                "new model promoted" + version
                if promote["promoted"]
                else "kept the current model (new model was not better)"
            ),
        ]

    links = mlflow_links(result)
    if links:
        lines += ["", "MLflow:"] + [f"  {step:<10} {url}" for step, url in links]

    report_files = result.get("report_files") or {}
    if report_files:
        lines += ["", "Saved reports:"] + [f"  {kind:<10} {path}" for kind, path in report_files.items()]

    if result.get("failed_step"):
        failed = next(s for s in result["steps"] if s["status"] == "failed")
        lines += ["", f"FAILED at step '{result['failed_step']}': {failed.get('error')}"]

    return "\n".join(lines)


def format_markdown_summary(result: dict) -> str:
    """Format a Markdown report suitable for MLflow artifacts, Prefect UI, and team sharing."""
    out = result.get("outputs", {})
    status = str(result.get("status", "unknown")).upper()
    total = result.get("total_seconds", 0.0)
    budget = result.get("time_budget", DEFAULT_TIME_BUDGET_SECONDS)
    verdict = "within budget" if total <= budget else "OVER BUDGET"

    lines = [
        f"# Fraud Pipeline Run Report (`{result.get('postfix')}`)",
        "",
        f"- **Status:** {status}",
        f"- **Started (UTC):** {result.get('started_at') or 'n/a'}",
        f"- **Finished (UTC):** {result.get('finished_at') or 'n/a'}",
        f"- **Total Time:** {total:.2f} s / {budget:.0f} s ({verdict})",
        "",
        "## Step Timings",
        "",
        "| Step | Status | Seconds |",
        "|---|---|---:|",
    ]
    for s in result.get("steps", []):
        lines.append(f"| `{s['name']}` | {s['status']} | {s['seconds']:.2f} |")

    report = out.get("clean", {}).get("report")
    if report:
        lines += [
            "",
            "## Data Cleaning & Rejected Rows",
            "",
            "| Table | Rows Read | Rows Clean | Rows Rejected | Rejected % |",
            "|---|---:|---:|---:|---:|",
        ]
        for table, r in report.items():
            lines.append(
                f"| `{table}` | {r['rows_in']:,} | {r['rows_clean']:,} | "
                f"{r['rows_rejected']:,} | {r['pct_rejected']:.2f}% |"
            )

    loaded = out.get("load", {}).get("rows_loaded")
    if loaded:
        lines += [
            "",
            "## Rows Loaded into MySQL",
            "",
            "| Table | Rows Loaded |",
            "|---|---:|",
        ]
        for table, n in loaded.items():
            lines.append(f"| `{table}` | {n:,} |")

    promote = out.get("promote", {})
    if "promoted" in promote:
        version = f" (version `{promote['model_version']}`)" if promote.get("model_version") else ""
        decision = f"New model promoted{version}" if promote["promoted"] else "Kept current model"
        lines += ["", "## Model Promotion", "", f"- **Decision:** {decision}"]

    links = mlflow_links(result)
    if links:
        lines += ["", "## MLflow Links", ""]
        for step, url in links:
            lines.append(f"- **{step}:** [{url}]({url})")

    if result.get("failed_step"):
        failed = next(s for s in result["steps"] if s["status"] == "failed")
        lines += ["", "## Failure", "", f"- **Step:** `{result['failed_step']}`", f"- **Error:** {failed.get('error')}"]

    return "\n".join(lines) + "\n"


def save_report_files(result: dict, report_dir: Path | str) -> dict[str, str]:
    """Write timestamped and latest JSON/Markdown reports to `report_dir`."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    postfix = result.get("postfix", "base")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = out_dir / f"pipeline_report_{postfix}_{stamp}.json"
    md_path = out_dir / f"pipeline_report_{postfix}_{stamp}.md"
    latest_json = out_dir / f"latest_{postfix}.json"
    latest_md = out_dir / f"latest_{postfix}.md"

    md_text = format_markdown_summary(result)
    json_text = json.dumps(result, indent=2, default=str)

    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def publish_to_mlflow(result: dict, experiment_name: str | None = None) -> str | None:
    """Log a team-accessible pipeline summary run and report artifacts to MLflow.

    Returns the clickable MLflow run URL (or None if MLflow is unconfigured/unavailable).
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return None

    try:
        import mlflow
    except ImportError:
        log.debug("mlflow not installed; skipping team summary run logging")
        return None

    clean_uri = uri[:-2] if uri.endswith("#/") else (uri[:-1] if uri.endswith("#") else uri)
    clean_uri = clean_uri.rstrip("/") if clean_uri.startswith("http") else clean_uri
    exp_name = experiment_name or os.environ.get("MLFLOW_PIPELINE_EXPERIMENT") or DEFAULT_MLFLOW_PIPELINE_EXPERIMENT

    mlflow.set_tracking_uri(clean_uri)
    experiment = mlflow.set_experiment(exp_name)

    metrics: dict[str, float] = {
        "total_seconds": float(result.get("total_seconds", 0.0)),
        "time_budget_seconds": float(result.get("time_budget", DEFAULT_TIME_BUDGET_SECONDS)),
        "within_budget": 1.0 if result.get("within_budget") else 0.0,
    }
    for s in result.get("steps", []):
        metrics[f"step.{s['name']}.seconds"] = float(s["seconds"])

    outputs = result.get("outputs", {})
    for table, r in (outputs.get("clean", {}).get("report") or {}).items():
        metrics[f"clean.{table}.rows_in"] = float(r["rows_in"])
        metrics[f"clean.{table}.rows_clean"] = float(r["rows_clean"])
        metrics[f"clean.{table}.rows_rejected"] = float(r["rows_rejected"])
        metrics[f"clean.{table}.pct_rejected"] = float(r["pct_rejected"])

    for table, n in (outputs.get("load", {}).get("rows_loaded") or {}).items():
        metrics[f"load.{table}.rows_loaded"] = float(n)

    promote = outputs.get("promote", {})
    if "promoted" in promote:
        metrics["promote.promoted"] = 1.0 if promote["promoted"] else 0.0

    tags = {
        "pipeline.postfix": str(result.get("postfix")),
        "pipeline.status": str(result.get("status")),
        "pipeline.triggered_by": getpass.getuser(),
    }
    if result.get("failed_step"):
        tags["pipeline.failed_step"] = str(result["failed_step"])
    for idx, (step, url) in enumerate(mlflow_links(result)):
        tags[f"mlflow_link.{idx}.{step}"] = url

    run_name = f"pipeline-{result.get('postfix')}-{result.get('started_at') or _utc_now()}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "postfix": str(result.get("postfix")),
            "steps": ",".join(s["name"] for s in result.get("steps", [])),
            "time_budget_seconds": str(result.get("time_budget")),
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_text(format_markdown_summary(result), "pipeline_summary.md")
        mlflow.log_dict(result, "pipeline_report.json")
        run_id = run.info.run_id

    if clean_uri.startswith("http"):
        return f"{clean_uri}/#/experiments/{experiment.experiment_id}/runs/{run_id}"
    return f"mlflow://experiments/{experiment.experiment_id}/runs/{run_id}"


def publish_to_prefect(result: dict) -> None:
    """Publish a Markdown artifact to the Prefect UI when running inside a Prefect flow."""
    if not PREFECT_AVAILABLE or create_markdown_artifact is None:
        return
    try:
        create_markdown_artifact(
            key=f"fraud-pipeline-{result.get('postfix', 'base')}",
            markdown=format_markdown_summary(result),
            description=f"Pipeline run summary for postfix '{result.get('postfix')}'",
        )
    except Exception as exc:
        log.debug("Prefect artifact skipped: %s", exc)


def publish_to_webhook(result: dict, webhook_url: str | None = None) -> bool:
    """POST the run summary to a team webhook (Slack / Teams / Discord) if configured."""
    url = (webhook_url or os.environ.get("PIPELINE_WEBHOOK_URL") or "").strip()
    if not url:
        return False
    summary = format_summary(result).strip()
    payload = json.dumps({"text": summary, "content": summary}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return 200 <= resp.status < 300


def _publish_reports(
    result: dict,
    report_dir: Path | str | None,
    publish_mlflow: bool,
    mlflow_experiment: str | None,
    webhook_url: str | None,
) -> None:
    if publish_mlflow:
        try:
            url = publish_to_mlflow(result, experiment_name=mlflow_experiment)
            if url:
                result["report_urls"]["mlflow"] = url
        except Exception as exc:
            log.warning("Could not log pipeline summary to MLflow: %s", exc)

    if report_dir:
        try:
            result["report_files"] = save_report_files(result, report_dir)
        except Exception as exc:
            log.warning("Could not write local report files: %s", exc)

    publish_to_prefect(result)

    try:
        if publish_to_webhook(result, webhook_url=webhook_url):
            log.info("Posted pipeline summary to webhook")
    except Exception as exc:
        log.warning("Could not post pipeline summary to webhook: %s", exc)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--postfix", default="base", help="raw drop to process: data/raw/<table>_<postfix>.csv")
    p.add_argument("--only", type=_csv, help=f"comma-separated steps to run ({', '.join(STEPS)})")
    p.add_argument("--skip", type=_csv, default=[], help="comma-separated steps to skip")
    p.add_argument("--check", action="store_true", help="verify selected step modules exist and exit")
    p.add_argument(
        "--time-budget",
        type=float,
        default=DEFAULT_TIME_BUDGET_SECONDS,
        help="target run time in seconds (default: %(default)s)",
    )
    p.add_argument("--strict-budget", action="store_true", help="exit with code 2 if the budget is exceeded")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"directory for JSON/Markdown run reports (default: {DEFAULT_REPORT_DIR})",
    )
    p.add_argument("--no-mlflow-report", action="store_true", help="skip logging the pipeline summary run to MLflow")
    p.add_argument(
        "--mlflow-experiment",
        default=None,
        help=f"MLflow experiment for pipeline summary runs (default: {DEFAULT_MLFLOW_PIPELINE_EXPERIMENT})",
    )
    p.add_argument("--webhook-url", default=None, help="optional webhook URL for team notifications")
    args = p.parse_args(argv)

    setup_logging()

    if args.check:
        try:
            names = preflight_check(args.only, args.skip)
        except PipelineError as exc:
            log.error("%s", exc)
            return 1
        print(f"Preflight OK: {', '.join(names)}")
        return 0

    try:
        result = pipeline(
            postfix=args.postfix,
            only=args.only,
            skip=args.skip,
            time_budget=args.time_budget,
            report_dir=args.report_dir,
            publish_mlflow=not args.no_mlflow_report,
            mlflow_experiment=args.mlflow_experiment,
            webhook_url=args.webhook_url,
        )
    except PipelineError as exc:
        if exc.result:
            print(format_summary(exc.result))
        else:
            log.error("%s", exc)
        return 1

    print(format_summary(result))
    if not result["within_budget"]:
        log.warning("run took %.0f s, budget is %.0f s", result["total_seconds"], args.time_budget)
        return 2 if args.strict_budget else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())