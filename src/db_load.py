"""Load one batch into MySQL: check the raw files, clean them, and insert the result.

run_batch() is the whole step for a batch already in data/raw/:
    data_validate.read_batch   check files and columns, set aside malformed rows
    clean                      the cleaning module (PBI-02): fixes values, rejects rows it can't fix
    load_batch                 insert customers, orders, labels and all rejected rows

load_batch() never cleans anything itself. Each cleaned table is staged as text with
LOAD DATA LOCAL INFILE (fast for ~1M rows) and MySQL casts the values to their column types on
insert, so a value that slipped past cleaning fails loudly instead of being silently coerced.
All inserts of a batch are one transaction: a batch is loaded completely or not at all.

In code:
    from src.db_load import run_batch
    stats = run_batch(engine, raw_dir, postfix)
"""
import json
import logging
import tempfile
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src import data_validate
from src.db_schema import COLUMN_TYPES, REJECTS_TABLE, create_tables, q
from src.schema import COLUMNS, FIRST_ORDER_DATETIME_FORMAT, ORDER_DATE_FORMAT, TABLE_NAMES

log = logging.getLogger("db_load")

STAGING_PREFIX = "stg_"
REJECT_COLUMNS = data_validate.REJECT_COLUMNS


def mysql_fmt(py_fmt):
    """strftime format -> MySQL STR_TO_DATE format (for these formats only minutes differ)."""
    return py_fmt.replace("%M", "%i")


# SELECT expression that casts each staged text column `s.<col>` to its column type.
CASTS = {
    "mobile_verified": "(s.mobile_verified = 'True')",
    "first_order_datetime": "STR_TO_DATE(NULLIF(s.first_order_datetime, ''), :datetime_fmt)",  # '' = unknown
    "order_date": "STR_TO_DATE(s.order_date, :date_fmt)",
}


def cast_expr(column, ddl):
    if column in CASTS:
        return CASTS[column]
    if ddl.startswith(("INT", "TINYINT")):
        return f"CAST(s.{column} AS SIGNED)"
    if ddl.startswith("DECIMAL"):
        return f"CAST(s.{column} AS {ddl.split()[0]})"
    return f"s.{column}"


def insert_sql(table):
    """INSERT ... SELECT from the staging table, casting every column to its type."""
    cols = COLUMNS[table] + ["batch_id"]
    exprs = [cast_expr(c, COLUMN_TYPES[table][c]) for c in COLUMNS[table]] + [":batch_id"]
    select = ",\n    ".join(f"{e} AS {q(c)}" for c, e in zip(cols, exprs))
    return (f"INSERT INTO {q(table)} ({', '.join(q(c) for c in cols)})\n"
            f"SELECT * FROM (\n  SELECT\n    {select}\n  FROM {q(STAGING_PREFIX + table)} AS s\n) AS new")


REJECTS_INSERT = (f"INSERT INTO {q(REJECTS_TABLE)} (batch_id, source_table, source_row, reason, raw_values) "
                  "VALUES (:batch_id, :table, :row, :reason, :raw)")


def count_rows(conn, table):
    return conn.execute(text(f"SELECT COUNT(*) FROM {q(table)}")).scalar_one()


def stage(engine, table, df, tmp_dir):
    """Copy one cleaned table into a TEXT staging table via a temporary CSV + LOAD DATA LOCAL INFILE."""
    path = Path(tmp_dir) / f"{table}.csv"
    df[COLUMNS[table]].to_csv(path, index=False, header=False, lineterminator="\n")
    stg = q(STAGING_PREFIX + table)
    cols = ", ".join(q(c) for c in COLUMNS[table])
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {stg}"))
        conn.execute(text(f"CREATE TABLE {stg} ({', '.join(f'{q(c)} TEXT' for c in COLUMNS[table])}) "
                          "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"))
        try:
            # ESCAPED BY '' keeps backslashes literal; blank fields stay '' (meaning unknown).
            conn.execute(text(
                f"LOAD DATA LOCAL INFILE :path INTO TABLE {stg} CHARACTER SET utf8mb4 "
                "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' ESCAPED BY '' "
                f"LINES TERMINATED BY '\\n' ({cols})"
            ), {"path": str(path.resolve())})
        except SQLAlchemyError as e:
            raise RuntimeError(
                f"LOAD DATA LOCAL INFILE failed for {table}: {getattr(e, 'orig', e)}. The engine needs "
                "connect_args={'local_infile': True}, and the server must allow it (on RDS: set "
                "local_infile=1 in the DB parameter group)."
            ) from e


def check_input(tables, rejects):
    for t in TABLE_NAMES:
        missing = set(COLUMNS[t]) - set(tables[t].columns)
        if missing:
            raise ValueError(f"cleaned {t} is missing columns {sorted(missing)}")
    missing = set(REJECT_COLUMNS) - set(rejects.columns)
    if missing:
        raise ValueError(f"rejects are missing columns {sorted(missing)}")


def load_batch(engine, tables, rejects, batch_id):
    """Insert one cleaned batch and its rejected rows. Returns {table: rows inserted}.

    tables:  {table: DataFrame of strings in the canonical formats from schema.py}
    rejects: DataFrame with REJECT_COLUMNS (table, row, reason, raw)
    """
    check_input(tables, rejects)
    create_tables(engine)
    params = {"batch_id": batch_id, "datetime_fmt": mysql_fmt(FIRST_ORDER_DATETIME_FORMAT),
              "date_fmt": mysql_fmt(ORDER_DATE_FORMAT)}
    stats, staged = {}, []
    try:
        # Staging uses DDL (an implicit commit in MySQL), so it happens before the transaction.
        with tempfile.TemporaryDirectory() as tmp_dir:
            for t in TABLE_NAMES:
                staged.append(STAGING_PREFIX + t)
                t0 = time.perf_counter()
                stage(engine, t, tables[t], tmp_dir)
                log.info("staged   %-9s %9d rows in %5.1fs", t, len(tables[t]), time.perf_counter() - t0)

        # Everything below commits together or not at all; parents first because of foreign keys.
        with engine.begin() as conn:
            for t in TABLE_NAMES:
                t0 = time.perf_counter()
                sql = insert_sql(t)
                before = count_rows(conn, t)
                conn.execute(text(sql), {k: v for k, v in params.items() if f":{k}" in sql})
                stats[t] = count_rows(conn, t) - before
                log.info("inserted %-9s %9d rows in %5.1fs", t, stats[t], time.perf_counter() - t0)
            records = [{"batch_id": batch_id, "table": r.table, "row": int(r.row), "reason": r.reason,
                        "raw": r.raw if isinstance(r.raw, str) else json.dumps(r.raw)}
                       for r in rejects[REJECT_COLUMNS].itertuples(index=False)]
            if records:
                conn.execute(text(REJECTS_INSERT), records)
            stats[REJECTS_TABLE] = len(records)
            log.info("inserted %-9s %9d rows", REJECTS_TABLE, len(records))
    finally:
        try:
            with engine.begin() as conn:
                for stg in staged:
                    conn.execute(text(f"DROP TABLE IF EXISTS {q(stg)}"))
        except SQLAlchemyError:
            log.warning("could not drop staging tables %s", staged)
    return stats


def default_clean(tables):
    """The team cleaning module (PBI-02): data_clean.clean_all(customers, orders, labels).

    Expected to return (cleaned {table: DataFrame}, rejects DataFrame with REJECT_COLUMNS, ...).
    """
    from src import data_clean  # imported here so loading doesn't depend on it when a clean function is given
    cleaned, rejects, *_ = data_clean.clean_all(tables["customers"], tables["orders"], tables["labels"])
    return cleaned, rejects


def run_batch(engine, raw_dir, postfix, clean=default_clean):
    """Check, clean and load the batch data/raw/<table>_<postfix>.csv. Returns load_batch's stats.

    Stops with data_validate.BatchValidationError (nothing loaded) if the files are unusable.
    """
    start = time.perf_counter()
    raw, malformed = data_validate.read_batch(raw_dir, postfix)
    cleaned, rejects = clean(raw)
    all_rejects = pd.concat([malformed, rejects[REJECT_COLUMNS]], ignore_index=True)
    stats = load_batch(engine, cleaned, all_rejects, batch_id=postfix)
    log.info("loaded batch '%s' in %.1fs", postfix, time.perf_counter() - start)
    return stats
