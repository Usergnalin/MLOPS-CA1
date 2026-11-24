# MLOps CA1: fraud detection pipeline

Cleans the raw customers / orders / labels data, loads it into MySQL, retrains the fraud model,
logs it to MLflow and promotes it if it is better. One command runs all of it.

```bash
python src/pipeline.py
```

## Contents
1. [What you need](#1-what-you-need)
2. [Setup from scratch](#2-setup-from-scratch)
3. [Get the data](#3-get-the-data)
4. [Run the pipeline](#4-run-the-pipeline)
5. [What each step does](#5-what-each-step-does)
6. [Project layout](#6-project-layout)
7. [Tests](#7-tests)
8. [Troubleshooting](#8-troubleshooting)

## 1. What you need
- Python
- A MySQL database you can write to (host, user, password, an empty database name)
- The team MLflow server URL (an AWS Elastic Beanstalk app built from the `Dockerfile`)
- The raw CSV files (section 3)

## 2. Setup from scratch
```bash
git clone https://github.com/Usergnalin/MLOPS-CA1.git
cd MLOPS-CA1

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install prefect                  # runs the pipeline as a Prefect flow (optional: it also runs without)

cp .env.example .env                 # Windows: copy .env.example .env
```
Fill in `.env` (it is gitignored, never commit it):

| Variable | Meaning |
|---|---|
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | MySQL connection. `DB_PORT` is optional (default 3306). `DB_NAME` may only contain letters, digits and underscores. |
| `MLFLOW_TRACKING_URI` | Address of the MLflow server, e.g. `http://<your-eb-env>.<region>.elasticbeanstalk.com` |

Variables already set in your shell take precedence over `.env`.

### MLflow server (only if you are setting up a new one)
`Dockerfile` builds the server. It needs these environment variables on Elastic Beanstalk:
`MLFLOW_BACKEND_STORE_URI`, `MLFLOW_ARTIFACTS_DESTINATION`, `MLFLOW_ALLOWED_HOSTS`. It listens on port 8000.

## 3. Get the data
Put the three raw files in `data/raw/`, named `<table>_<postfix>.csv`. The first drop uses the postfix `base`:

```
data/raw/customers_base.csv
data/raw/orders_base.csv
data/raw/labels_base.csv
```
<!-- TODO: add where the CSVs are downloaded from -->
Everything under `data/` is generated or downloaded and is gitignored.

## 4. Run the pipeline
```bash
python src/pipeline.py                          # all steps on data/raw/*_base.csv
python src/pipeline.py --postfix batch_02       # a new drop: data/raw/*_batch_02.csv
python src/pipeline.py --skip clean,load        # re-use data that is already loaded
python src/pipeline.py --only retrain,promote
python src/pipeline.py --time-budget 180 --strict-budget    # exit code 2 if it takes longer than 3 minutes
```
Before running anything the pipeline checks that every step it is about to run exists, so a mistake
fails in a second and not after minutes of work.

At the end it prints a summary: time per step against the 3-minute budget, rows read / cleaned /
rejected, rows loaded into MySQL, whether a new model was promoted, and the MLflow links. Example:

```
=== Pipeline summary (postfix: base) ===

step      status     seconds
clean     ok            38.2
load      ok            ...
total                   ...   (budget 180 s, within budget)

Rows (cleaning):
  orders     read   949,889   clean   949,000   rejected     889 (0.09%)
...
MLflow:
  retrain  http://<server>/#/experiments/1/runs/<run id>
```
Exit codes: `0` success, `1` a step failed or is missing, `2` over the time budget with `--strict-budget`.

If Prefect is installed, each run also appears in the Prefect UI (`prefect server start`).

## 5. What each step does
| Step | Module | Does |
|---|---|---|
| `clean` | `src/data_clean.py` | Fixes the problems found in `src/eda.ipynb` (case, mixed date formats, duplicates, orphan labels). Writes `data/clean/<table>_<postfix>.csv`, a `..._rejects_...csv` with the reason for every rejected row, and `cleaning_report_<postfix>.json`. Nothing is dropped silently. |
| `load` | `src/data_load.py` | Loads the clean files and their rejects into MySQL (`src/db_load.py`). Re-running a batch changes nothing. |
| `retrain` | `src/retrain.py` | Fetches the latest data from MySQL (`src/data_fetch.py`), trains the decision tree baseline, then every model in `src/models/`, and logs each to MLflow tagged with the batch. Set `RETRAIN_MODELS=decision_tree,random_forest` (or `python -m src.retrain --models ...`) to train only some. |
| `promote` | `src/promote.py` | Shortlists the best 2 model types on the current data and promotes the best as `@champion` in the MLflow Model Registry if it beats the current champion. |

Each step module (except `clean`) must define `run(postfix) -> dict | None`. The keys the summary reads
are listed at the top of `src/pipeline.py`. The module names are set in `STEP_MODULES` in the same file.

Run a single step on its own, for example `python src/data_clean.py --postfix base`,
`python -m src.data_load --postfix base`, `python -m src.retrain --postfix base` or `python -m src.promote`.

## 6. Project layout
```
src/
  config.py        paths, MySQL engine, MLflow URI, logging (shared by every script)
  data_download.py download a batch from its URLs
  data_validate.py file and column checks
  data_clean.py    cleaning rules (step: clean)
  data_load.py     load the clean files into MySQL (step: load), using db_load.py / db_schema.py
  data_fetch.py    labelled training data from MySQL with a fixed train / test split
  retrain.py       retrain every model on the latest data (step: retrain)
  promote.py       pick the champion in the Model Registry (step: promote)
  evaluation.py    shared metrics (PR-AUC, recall, precision, F1)
  dataset_tags.py  which data a run used, as MLflow tags
  tracking.py      shared MLflow logging helper
  pipeline.py      the one-command flow
  eda.ipynb        data checks that the cleaning rules come from
  models/          one file per model: decision_tree.py (baseline), logistic_regression.py, svm.py,
                   gradient_boosting.py, xgboost_model.py, random_forest.py, mlp.py
tests/             pytest tests
data/              raw, clean, eda output (gitignored)
Dockerfile         MLflow server image
```

## 7. Tests
```bash
pytest
```
The tests use fake steps and a local SQLite MLflow store, so they need no AWS. The database tests are
skipped unless `TEST_DB_URL` points at a MySQL server where you may create databases
(e.g. `mysql+pymysql://user:password@127.0.0.1:3306`, never the team database).

## 8. Troubleshooting
| Message | Fix |
|---|---|
| `missing environment variables: ... (see .env.example)` | Fill in `.env` (section 2). |
| `step 'load': src/data_load.py not found` | That step is not in your checkout yet. Pull the latest `main`, or run with `--skip load`. |
| `.../data/clean/customers_<postfix>.csv not found - run the clean step first` | `--skip clean` needs the clean files of that postfix in `data/clean/`. Run without `--skip clean`. |
| `LOAD DATA LOCAL INFILE failed` | The MySQL server must allow `local_infile` (on RDS: set `local_infile=1` in the DB parameter group). |
| `skipping svm: src/models/svm.py is not in this checkout` | Not an error: that model's PR is not merged yet. The other models still train. |
| `file not found: data/raw/customers_base.csv` | The raw CSVs are missing or the postfix is wrong (section 3). |
| `too many rejected rows (limit 5.0%)` | The raw file has a new problem. Check `data/clean/*_rejects_*.csv` and re-run `eda.ipynb` on the new drop. |
| `ModuleNotFoundError: prefect` | Not an error: the pipeline runs without Prefect. Install it to get the UI. |
