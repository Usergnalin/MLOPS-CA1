"""Tests for src/pipeline.py (PBI-11 AC1–AC4).

Runs with either `pytest` or `python -m unittest discover -s tests`.
Uses fake step modules and temporary directories, so no MySQL or AWS server is needed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import pipeline as pl  # noqa: E402


def _make_fake_modules(fail_step: str | None = None, delay: float = 0.0):
    """Build in-memory fake step modules matching the pipeline step contract."""
    clean_mod = types.ModuleType("data_clean")

    def clean_run(postfix="base"):
        if delay:
            time.sleep(delay)
        if fail_step == "clean":
            raise RuntimeError("bad raw CSV")
        return types.SimpleNamespace(
            report={
                "customers": {"rows_in": 1000, "rows_clean": 995, "rows_rejected": 5, "pct_rejected": 0.5},
                "orders": {"rows_in": 2000, "rows_clean": 1990, "rows_rejected": 10, "pct_rejected": 0.5},
                "labels": {"rows_in": 1990, "rows_clean": 1985, "rows_rejected": 5, "pct_rejected": 0.25},
            }
        )

    clean_mod.run = clean_run

    load_mod = types.ModuleType("data_load")

    def load_run(postfix="base"):
        if delay:
            time.sleep(delay)
        if fail_step == "load":
            raise RuntimeError("MySQL connection refused")
        return {"rows_loaded": {"customers": 995, "orders": 1990, "labels": 1985}}

    load_mod.run = load_run

    retrain_mod = types.ModuleType("retrain")

    def retrain_run(postfix="base"):
        if delay:
            time.sleep(delay)
        if fail_step == "retrain":
            raise RuntimeError("training diverged")
        return {
            "mlflow_urls": [
                "http://mlflow.example.com/#/experiments/1/runs/run_logreg",
                "http://mlflow.example.com/#/experiments/1/runs/run_svm",
            ]
        }

    retrain_mod.run = retrain_run

    promote_mod = types.ModuleType("promote")

    def promote_run(postfix="base"):
        if delay:
            time.sleep(delay)
        if fail_step == "promote":
            raise RuntimeError("registry unavailable")
        return {
            "promoted": True,
            "model_version": 4,
            "mlflow_url": "http://mlflow.example.com/#/experiments/1/runs/run_promote",
        }

    promote_mod.run = promote_run

    return {
        "data_clean": clean_mod,
        "data_load": load_mod,
        "retrain": retrain_mod,
        "promote": promote_mod,
    }


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.report_dir = Path(self.tmpdir.name) / "reports"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _patch_imports(self, modules):
        def fake_import(name):
            if name in modules:
                return modules[name]
            raise ModuleNotFoundError(name)

        return mock.patch.object(pl, "_import", side_effect=fake_import)

    def test_ac1_ac2_ac4_full_pipeline_success_and_reports(self):
        modules = _make_fake_modules()
        with self._patch_imports(modules):
            result = pl.pipeline(
                postfix="batch_01",
                time_budget=180.0,
                report_dir=self.report_dir,
                publish_mlflow=False,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["name"] for s in result["steps"]], ["clean", "load", "retrain", "promote"])
        self.assertTrue(result["within_budget"])
        self.assertLess(result["total_seconds"], 180.0)

        # AC4: console summary includes timings, rejected rows, loaded rows, promotion, MLflow links
        summary = pl.format_summary(result)
        self.assertIn("within budget", summary)
        self.assertIn("rejected", summary)
        self.assertIn("Rows loaded into MySQL:", summary)
        self.assertIn("new model promoted (model version 4)", summary)
        self.assertIn("http://mlflow.example.com/#/experiments/1/runs/run_logreg", summary)
        self.assertIn("http://mlflow.example.com/#/experiments/1/runs/run_promote", summary)

        # AC4: persisted JSON and Markdown report files exist and contain structured outputs
        json_path = Path(result["report_files"]["json"])
        md_path = Path(result["report_files"]["markdown"])
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue((self.report_dir / "latest_batch_01.json").exists())
        self.assertTrue((self.report_dir / "latest_batch_01.md").exists())

        saved = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["outputs"]["load"]["rows_loaded"]["orders"], 1990)
        self.assertIn("## Data Cleaning & Rejected Rows", md_path.read_text(encoding="utf-8"))

    def test_ac2_strict_budget_exit_code_2(self):
        modules = _make_fake_modules(delay=0.05)
        with self._patch_imports(modules):
            exit_code = pl.main(
                [
                    "--postfix", "base",
                    "--time-budget", "0.01",
                    "--strict-budget",
                    "--report-dir", str(self.report_dir),
                    "--no-mlflow-report",
                ]
            )
        self.assertEqual(exit_code, 2)

    def test_preflight_fails_fast_when_step_missing(self):
        modules = _make_fake_modules()
        del modules["data_load"]
        with self._patch_imports(modules):
            with self.assertRaises(pl.PipelineError) as ctx:
                pl.pipeline(postfix="base", report_dir=self.report_dir, publish_mlflow=False)
        self.assertIn("src/data_load.py not found", str(ctx.exception))

    def test_check_flag_validates_without_running(self):
        modules = _make_fake_modules(fail_step="clean")
        with self._patch_imports(modules):
            exit_code = pl.main(["--check"])
        self.assertEqual(exit_code, 0)

    def test_step_failure_still_writes_report_and_returns_1(self):
        modules = _make_fake_modules(fail_step="load")
        with self._patch_imports(modules):
            exit_code = pl.main(
                [
                    "--postfix", "batch_02",
                    "--report-dir", str(self.report_dir),
                    "--no-mlflow-report",
                ]
            )
        self.assertEqual(exit_code, 1)
        latest = json.loads((self.report_dir / "latest_batch_02.json").read_text(encoding="utf-8"))
        self.assertEqual(latest["status"], "failed")
        self.assertEqual(latest["failed_step"], "load")
        self.assertEqual(len(latest["steps"]), 2)

    def test_only_and_skip_step_selection(self):
        self.assertEqual(pl.select_steps(only=["retrain", "promote"]), ["retrain", "promote"])
        self.assertEqual(pl.select_steps(skip=["clean", "load"]), ["retrain", "promote"])
        with self.assertRaises(pl.PipelineError):
            pl.select_steps(only=["unknown_step"])

    def test_webhook_notification_payload(self):
        modules = _make_fake_modules()
        fake_resp = mock.MagicMock()
        fake_resp.__enter__.return_value.status = 200
        with self._patch_imports(modules), mock.patch("urllib.request.urlopen", return_value=fake_resp) as mocked_open:
            pl.pipeline(
                postfix="base",
                report_dir=self.report_dir,
                publish_mlflow=False,
                webhook_url="https://hooks.example.com/services/test",
            )
        self.assertEqual(mocked_open.call_count, 1)
        req = mocked_open.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertIn("Pipeline summary", body["text"])


if __name__ == "__main__":
    unittest.main()