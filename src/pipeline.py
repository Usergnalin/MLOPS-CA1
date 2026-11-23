"""PBI-11: one command that runs the whole pipeline, times every step and prints a summary.

    python src/pipeline.py                         # clean -> load -> retrain -> promote
    python src/pipeline.py --postfix batch_02      # a new raw drop: data/raw/<table>_batch_02.csv
    python src/pipeline.py --skip clean,load       # re-use data that is already loaded
    python src/pipeline.py --only retrain,promote
    python src/pipeline.py --time-budget 180 --strict-budget   # exit code 2 if it takes longer

The flow is a Prefect flow when Prefect is installed (`pip install prefect`), so runs show up in the
Prefect UI. Without Prefect it runs as plain Python with the same behaviour.

Step contract
-------------
Every step except `clean` lives in its own module under src/ and exposes one function:

    def run(postfix: str) -> dict | None

The returned dict is optional. These keys are picked up for the end-of-run summary:

    load      {"rows_loaded": {"customers": 899000, "orders": 949000, "labels": 948000}}
    retrain   {"mlflow_urls": ["http://.../#/experiments/1/runs/abc", ...]}   (or "mlflow_url": "...")
    promote   {"mlflow_url": "...", "promoted": True, "model_version": 3}

The module for each step is listed in STEP_MODULES below. Change a name there if the real file is
called something else.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from typing import Callable

try:  # run as a script from src/ (like the notebook) ...
    from config import setup_logging
except ModuleNotFoundError as exc:  # ... or imported as src.pipeline (like the tests)
    if exc.name != "config":
        raise
    from src.config import setup_logging

try:
    from prefect import flow, task
except ImportError:  # no Prefect: the decorators do nothing and the pipeline runs as plain Python
    def task(fn=None, **_kwargs):
        return (lambda f: f) if fn is None else fn

    flow = task

log = logging.getLogger("pipeline")

STEPS = ("clean", "load", "retrain", "promote")  # run in this order
STEP_MODULES = {  # step -> module under src/ that has run(postfix)
    "clean": "data_clean",
    "load": "data_load",
    "retrain": "retrain",
    "promote": "promote",
}
DEFAULT_TIME_BUDGET_SECONDS = 180.0  # PBI-11 target for the whole run (3 minutes)


class PipelineError(RuntimeError):
    """A step failed or is missing. `.result` holds what finished, for the summary."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result or {}


# ---------------------------------------------------------------------------
# Finding and running steps
# ---------------------------------------------------------------------------
def _import(module: str):
    """Import `module` as a top-level module (script style) or as src.<module> (package style)."""
    for name in (module, f"src.{module}"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name not in (name, module, "src"):
                raise  # the step exists but one of ITS imports is missing: show that error
    raise ModuleNotFoundError(module)


def _clean_step(postfix: str) -> dict:
    result = _import(STEP_MODULES["clean"]).run(postfix=postfix)  # data_clean.run returns a CleanResult
    return {"report": result.report}


def _load_step(name: str) -> Callable[[str], dict | None]:
    """Return the function that runs one step. Raises PipelineError if the step is not available."""
    module = STEP_MODULES[name]
    try:
        mod = _import(module)
    except ModuleNotFoundError:
        raise PipelineError(f"step '{name}': src/{module}.py not found "
                            f"(it must define run(postfix); see the docstring of src/pipeline.py)") from None
    if name == "clean":
        return _clean_step
    if not callable(getattr(mod, "run", None)):
        raise PipelineError(f"step '{name}': src/{module}.py has no run(postfix) function")
    return mod.run


@task(task_run_name="{name}")
def execute_step(name: str, postfix: str) -> dict:
    return _load_step(name)(postfix) or {}


def select_steps(only=None, skip=()) -> list[str]:
    unknown = [s for s in list(only or []) + list(skip) if s not in STEPS]
    if unknown:
        raise PipelineError(f"unknown step(s) {unknown}. Choose from: {', '.join(STEPS)}")
    return [s for s in STEPS if (not only or s in only) and s not in skip]


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------
@flow(name="fraud-pipeline")
def pipeline(postfix: str = "base", only: list | None = None, skip: list | None = None,
             time_budget: float = DEFAULT_TIME_BUDGET_SECONDS) -> dict:
    """Run the selected steps in order. Returns a summary dict. Raises PipelineError if a step fails."""
    names = select_steps(only, skip or [])
    result = {"postfix": postfix, "steps": [], "outputs": {}, "time_budget": time_budget}

    for name in names:  # check everything exists BEFORE spending minutes on step 1
        _load_step(name)

    start = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        log.info("step '%s' started", name)
        try:
            out = execute_step(name, postfix)
        except Exception as exc:
            result["steps"].append({"name": name, "status": "failed", "seconds": _since(t0), "error": str(exc)})
            result["total_seconds"] = _since(start)
            result["failed_step"] = name
            raise PipelineError(f"step '{name}' failed: {exc}", result) from exc
        result["steps"].append({"name": name, "status": "ok", "seconds": _since(t0)})
        result["outputs"][name] = out
        log.info("step '%s' finished in %.1f s", name, result["steps"][-1]["seconds"])

    result["total_seconds"] = _since(start)
    result["within_budget"] = result["total_seconds"] <= time_budget
    return result


def _since(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


# ---------------------------------------------------------------------------
# End-of-run summary
# ---------------------------------------------------------------------------
def mlflow_links(outputs: dict) -> list[tuple[str, str]]:
    """(step, url) for every MLflow link a step reported."""
    links = []
    for step, out in outputs.items():
        for url in ([out.get("mlflow_url")] + list(out.get("mlflow_urls") or [])):
            if url:
                links.append((step, url))
    return links


def format_summary(result: dict) -> str:
    out = result.get("outputs", {})
    lines = ["", f"=== Pipeline summary (postfix: {result.get('postfix')}) ===", "",
             f"{'step':<10}{'status':<9}{'seconds':>9}"]
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
            lines.append(f"  {table:<10} read {r['rows_in']:>9,}   clean {r['rows_clean']:>9,}   "
                         f"rejected {r['rows_rejected']:>7,} ({r['pct_rejected']:.2f}%)")
    loaded = out.get("load", {}).get("rows_loaded")
    if loaded:
        lines += ["", "Rows loaded into MySQL:"]
        lines += [f"  {table:<10} {n:>9,}" for table, n in loaded.items()]

    promote = out.get("promote", {})
    if "promoted" in promote:
        version = f" (model version {promote['model_version']})" if promote.get("model_version") else ""
        lines += ["", "Promotion: " + ("new model promoted" + version if promote["promoted"]
                                       else "kept the current model (new model was not better)")]
    links = mlflow_links(out)
    if links:
        lines += ["", "MLflow:"] + [f"  {step:<8} {url}" for step, url in links]
    if result.get("failed_step"):
        failed = next(s for s in result["steps"] if s["status"] == "failed")
        lines += ["", f"FAILED at step '{result['failed_step']}': {failed.get('error')}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--postfix", default="base", help="raw drop to process: data/raw/<table>_<postfix>.csv")
    p.add_argument("--only", type=_csv, help=f"comma-separated steps to run ({', '.join(STEPS)})")
    p.add_argument("--skip", type=_csv, default=[], help="comma-separated steps to skip")
    p.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET_SECONDS,
                   help="target run time in seconds (default: %(default)s)")
    p.add_argument("--strict-budget", action="store_true", help="exit with code 2 if the budget is exceeded")
    args = p.parse_args(argv)

    setup_logging()
    try:
        result = pipeline(postfix=args.postfix, only=args.only, skip=args.skip, time_budget=args.time_budget)
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
