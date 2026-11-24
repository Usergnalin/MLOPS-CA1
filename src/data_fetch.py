"""Fetch the labelled training data from the database and split it into train / test files.

Pulls every labelled order of every batch loaded so far (orders + labels + customers), and is
run once per pipeline run. Every training script reads the same files through
load_training_data(), so all models in a run are trained and scored on identical rows and can be
compared fairly.

Writes to --training-dir (default data/training/):
    train.csv       training rows (one row per labelled order, joined with its customer)
    test.csv        held-out rows used to score every model
    manifest.json   split rule, row counts, fraud rates, batches, date range, file hashes

The split is hash-based: an order goes to test when a hash of its (country_code, order_id) falls
in the lowest TEST_PERCENT of buckets. The assignment depends only on the order itself, so it
never changes between runs; when new batches arrive, old test orders stay in test and new
orders are split the same way. No order that was ever used for testing leaks into training.

Usage:
    python -m src.data_fetch
    python -m src.data_fetch --training-dir X

In a training script:
    from src.data_fetch import TARGET, load_training_data
    train, test, manifest = load_training_data()   # columns already typed
"""
import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from src.config import TRAINING_DIR, make_engine, setup_logging
from src.db_schema import COLUMN_TYPES, PRIMARY_KEYS
from src.schema import COLUMNS

log = logging.getLogger("data_fetch")

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
MANIFEST_FILE = "manifest.json"
FETCH_CHUNKSIZE = 100_000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

TARGET = "is_fraud"

# Split rule. Changing either value reshuffles which orders are test orders, so agree on it
# as a team and leave it fixed.
SPLIT_KEY = ["country_code", "order_id"]
TEST_PERCENT = 20

# The query and dtypes are built from the shared table design (schema.py / db_schema.py), so a
# column added there shows up here without editing this file. Output columns, in order:
#     order key, customer_id, the other order columns, the customer columns, batch_id, is_fraud
ORDER_KEY = PRIMARY_KEYS["orders"]            # (country_code, order_id): labels join orders on it
CUSTOMER_KEY = PRIMARY_KEYS["customers"]      # (customer_id,): labels join customers on it
ORDER_COLUMNS = [c for c in COLUMNS["orders"] if c not in ORDER_KEY]
CUSTOMER_COLUMNS = [c for c in COLUMNS["customers"] if c not in CUSTOMER_KEY and c not in ORDER_KEY]
SELECT_COLUMNS = (
    [f"o.{c}" for c in ORDER_KEY] + [f"l.{c}" for c in CUSTOMER_KEY] + [f"o.{c}" for c in ORDER_COLUMNS]
    + [f"c.{c}" for c in CUSTOMER_COLUMNS] + ["o.batch_id", f"l.{TARGET}"]
)


def join_on(left, right, key):
    return " AND ".join(f"{right}.{c} = {left}.{c}" for c in key)


QUERY = (
    "SELECT\n    " + ",\n    ".join(SELECT_COLUMNS) + "\n"
    "FROM orders AS o\n"
    f"JOIN labels AS l\n    ON {join_on('o', 'l', ORDER_KEY)}\n"
    f"JOIN customers AS c\n    ON {join_on('l', 'c', CUSTOMER_KEY)}\n"
)
# Same joins, only for the progress bar total.
COUNT_QUERY = f"SELECT COUNT(*) FROM ({QUERY}) AS q"
QUERY += "ORDER BY " + ", ".join(f"o.{c}" for c in ORDER_KEY) + "\n"  # stable row order, so files hash the same


def pandas_dtype(column, ddl):
    """pandas dtype for a MySQL column type from db_schema.COLUMN_TYPES."""
    if column == TARGET:
        return "int8"
    if ddl.startswith("BOOLEAN"):
        return "bool"
    if ddl.startswith(("INT", "TINYINT")):
        return "int64"
    if ddl.startswith("DECIMAL"):
        return "float64"  # MySQL DECIMAL arrives as Python Decimal
    if ddl.startswith(("DATE", "DATETIME")):
        return None       # handled by DATETIME_COLUMNS
    return "string"


OUTPUT_COLUMNS = [c.split(".", 1)[1] for c in SELECT_COLUMNS]
_TYPES = {**COLUMN_TYPES["customers"], **COLUMN_TYPES["labels"], **COLUMN_TYPES["orders"],
          "batch_id": "VARCHAR(64)"}
# Column types of the training data. Applied after the fetch and again when a CSV is loaded,
# so training scripts always get the same dtypes.
DTYPES = {c: pandas_dtype(c, _TYPES[c]) for c in OUTPUT_COLUMNS if pandas_dtype(c, _TYPES[c])}
DATETIME_COLUMNS = [c for c in OUTPUT_COLUMNS if _TYPES[c].startswith(("DATE", "DATETIME"))]  # may hold NaT
DATETIME_DTYPE = "datetime64[us]"  # fixed, so fetched and loaded data have identical dtypes


def set_types(df):
    for c in DATETIME_COLUMNS:
        df[c] = pd.to_datetime(df[c]).astype(DATETIME_DTYPE)
    return df.astype(DTYPES)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_test(df):
    """True for test orders. Uses md5 (not Python's hash(), which changes every process)."""
    keys = df[SPLIT_KEY[0]].str.cat(df[SPLIT_KEY[1:]], sep="|")
    buckets = [int.from_bytes(hashlib.md5(k.encode()).digest()[:8], "big") % 100 for k in keys]
    return np.array(buckets, dtype=int) < TEST_PERCENT


def fetch(engine):
    with engine.connect() as conn:
        total = conn.execute(text(COUNT_QUERY)).scalar_one()
    if not total:
        raise RuntimeError("no labelled orders in the database - load a batch first (src.db_load)")
    chunks = []
    with (engine.connect().execution_options(stream_results=True) as conn,
          tqdm(total=total, desc="fetch", unit="row", unit_scale=True) as bar):
        for chunk in pd.read_sql(text(QUERY), conn, chunksize=FETCH_CHUNKSIZE):
            chunks.append(chunk)
            bar.update(len(chunk))
    return set_types(pd.concat(chunks, ignore_index=True))


def describe(df, digest):
    return {
        "rows": len(df),
        "fraud_rows": int(df[TARGET].sum()),
        "fraud_rate": round(float(df[TARGET].mean()), 6),
        "hash": digest,
    }


def build_manifest(df, train, test, train_hash, test_hash, engine):
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": engine.url.database,
        "host": engine.url.host,
        "rows": len(df),
        "fraud_rate": round(float(df[TARGET].mean()), 6),
        "batches": {str(k): int(v) for k, v in df["batch_id"].value_counts().sort_index().items()},
        # Date range of the orders, for tagging retraining runs (#25).
        "order_date_min": df["order_date"].min().date().isoformat(),
        "order_date_max": df["order_date"].max().date().isoformat(),
        "split": {"method": "md5 hash of key, bucket % 100 < test_percent",
                  "key": SPLIT_KEY, "test_percent": TEST_PERCENT},
        "train": describe(train, train_hash),
        "test": describe(test, test_hash),
        # One hash for the whole snapshot, for tagging MLflow runs.
        "data_hash": hashlib.sha256((train_hash + test_hash).encode()).hexdigest(),
        "columns": list(df.columns),
    }


def write_atomic(path, write):
    """Write to a temp file then rename, so a reader never sees a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


def write_csv(df, path):
    write_atomic(path, lambda p: df.to_csv(p, index=False, date_format=DATETIME_FORMAT,
                                           lineterminator="\n"))
    return file_hash(path)


def run(training_dir=TRAINING_DIR, engine=None):
    """Fetch and split. Returns (train, test, manifest)."""
    start = time.perf_counter()
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)

    own_engine = engine is None
    engine = engine or make_engine()
    try:
        df = fetch(engine)
    finally:
        if own_engine:
            engine.dispose()
    log.info("fetched %d rows in %.1fs", len(df), time.perf_counter() - start)

    test_mask = is_test(df)
    train, test = df[~test_mask], df[test_mask]
    if train.empty or test.empty or train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise RuntimeError("train and test must both be non-empty and contain both classes")

    with tqdm(total=2, desc="write snapshot", unit="file") as bar:
        train_hash = write_csv(train, training_dir / TRAIN_FILE)
        bar.update()
        test_hash = write_csv(test, training_dir / TEST_FILE)
        bar.update()
    manifest = build_manifest(df, train, test, train_hash, test_hash, engine)
    write_atomic(training_dir / MANIFEST_FILE,
                 lambda p: p.write_text(json.dumps(manifest, indent=2) + "\n"))

    for part in ("train", "test"):
        log.info("%-5s rows=%9d  fraud_rate=%.4f", part, manifest[part]["rows"], manifest[part]["fraud_rate"])
    log.info("batches=%s  hash=%s", manifest["batches"], manifest["data_hash"][:12])
    log.info("wrote %s in %.1fs", training_dir, time.perf_counter() - start)
    return train, test, manifest


def load_training_data(training_dir=TRAINING_DIR):
    """For training scripts: returns (train, test, manifest) from the latest fetch."""
    training_dir = Path(training_dir)
    manifest_path = training_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(f"no training data in {training_dir} - run python -m src.data_fetch first")
    manifest = json.loads(manifest_path.read_text())

    frames = []
    for part, name in (("train", TRAIN_FILE), ("test", TEST_FILE)):
        path = training_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - re-run python -m src.data_fetch")
        if file_hash(path) != manifest[part]["hash"]:
            raise RuntimeError(f"{name} does not match {MANIFEST_FILE} (hash differs); "
                               "re-run python -m src.data_fetch")
        frames.append(pd.read_csv(path, dtype=DTYPES, parse_dates=DATETIME_COLUMNS,
                                  date_format=DATETIME_FORMAT).astype({c: DATETIME_DTYPE for c in DATETIME_COLUMNS}))
    return frames[0], frames[1], manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR,
                        help="output folder (default: %(default)s)")
    args = parser.parse_args()
    setup_logging()
    run(args.training_dir)
