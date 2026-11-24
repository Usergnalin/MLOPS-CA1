"""Pipeline step 'load': insert the files the clean step wrote into MySQL (PBI-11).

Reads   data/clean/<table>_<postfix>.csv           cleaned rows (src.data_clean)
        data/clean/<table>_rejects_<postfix>.csv   rejected rows with their reasons
and loads them with src.db_load.load_batch: one transaction per batch, upserted on the primary
keys, so re-running a batch changes nothing. The clean step writes ISO dates and 1/0 booleans;
they are turned into the formats the load expects (src/schema.py) here.

    python -m src.data_load --postfix base      # after python src/data_clean.py --postfix base

In the pipeline (src/pipeline.py) this is STEP_MODULES["load"]: run(postfix) returns
{"rows_loaded": {table: rows}, "stats": db_load's stats}.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src import data_clean, db_load
from src.config import CLEAN_DIR, make_engine, setup_logging
from src.db_schema import create_tables
from src.schema import COLUMNS, FIRST_ORDER_DATETIME_FORMAT, ORDER_DATE_FORMAT, TABLE_NAMES

log = logging.getLogger("data_load")

# Formats data_clean.write_outputs uses (OUTPUT_DATE_FORMAT) -> formats db_load expects.
DATE_COLUMNS = {
    "order_date": ("%Y-%m-%d", ORDER_DATE_FORMAT),
    "first_order_datetime": ("%Y-%m-%d %H:%M:%S", FIRST_ORDER_DATETIME_FORMAT),
}
BOOLEANS = {"1": "True", "0": "False", "": ""}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the clean step first (python src/data_clean.py)")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def to_load_format(table, df):
    """A cleaned file (strings) in src/schema.py's columns and value formats."""
    missing = [c for c in COLUMNS[table] if c not in df.columns]
    if missing:
        raise ValueError(f"cleaned {table} file is missing columns {missing}")
    out = df[COLUMNS[table]].copy()
    for column, (written, expected) in DATE_COLUMNS.items():
        if column in out:
            parsed = pd.to_datetime(out[column].where(out[column] != ""), format=written)
            out[column] = parsed.dt.strftime(expected).fillna("")
    if "mobile_verified" in out:
        bad = ~out["mobile_verified"].isin(BOOLEANS)
        if bad.any():
            raise ValueError(f"cleaned {table}: mobile_verified must be 1/0, got {out['mobile_verified'][bad].iloc[0]!r}")
        out["mobile_verified"] = out["mobile_verified"].map(BOOLEANS)
    return out


def read_clean(postfix, clean_dir=CLEAN_DIR):
    """({table: DataFrame in load format}, rejects DataFrame) from the clean step's files."""
    clean_dir = Path(clean_dir)
    tables, rejects = {}, []
    for table in TABLE_NAMES:
        tables[table] = to_load_format(table, read_csv(clean_dir / f"{table}_{postfix}.csv"))
        rejects_path = clean_dir / f"{table}_rejects_{postfix}.csv"
        if rejects_path.exists():
            rejects.append(data_clean.load_rejects(table, read_csv(rejects_path)))
    rejects = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(columns=db_load.REJECT_COLUMNS)
    return tables, rejects


def run(postfix="base", clean_dir=CLEAN_DIR, engine=None):
    """Load data/clean/*_<postfix>.csv into MySQL. Returns {"rows_loaded": {...}, "stats": {...}}."""
    tables, rejects = read_clean(postfix, clean_dir)
    own_engine = engine is None
    engine = engine or make_engine()
    try:
        create_tables(engine)  # no-op when the tables exist
        stats = db_load.load_batch(engine, tables, rejects, batch_id=postfix)
    finally:
        if own_engine:
            engine.dispose()
    return {"rows_loaded": {t: s["rows"] for t, s in stats.items() if t in TABLE_NAMES}, "stats": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postfix", default="base", help="loads data/clean/<table>_<postfix>.csv")
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    args = parser.parse_args()
    setup_logging()
    run(args.postfix, args.clean_dir)
