"""Check a raw batch's files and columns before anything is cleaned or loaded.

Stops with one clear error that lists every problem found in the batch, when:
    - a file is missing or empty, or is not valid UTF-8 text
    - the header is missing a required column, or names a column twice
    - more than MAX_MALFORMED_FRACTION of a file's rows have the wrong number of fields

A handful of malformed rows does not stop the batch: those rows are returned as rejects with a
reason, so one bad line in a new batch can't block everything else from loading. Extra columns
are ignored with a warning. Column order in the file doesn't matter.

Usage:
    python -m src.data_validate --postfix batch2              # checks data/raw/*_batch2.csv
    python -m src.data_validate --postfix batch2 --raw-dir X

In code:
    from src.data_validate import read_batch
    tables, rejects = read_batch(raw_dir, postfix)   # all values as strings, columns in schema order
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import pandas as pd

from src.schema import COLUMNS, TABLE_NAMES, file_name

log = logging.getLogger("data_validate")

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MAX_MALFORMED_FRACTION = 0.01  # above this a file is treated as broken, not as having a few bad rows
REJECT_COLUMNS = ["table", "row", "reason", "raw"]  # row = data row number, 1 = first row after the header


class BatchValidationError(ValueError):
    """Raised with every problem in the batch, one per line."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__("batch failed validation:\n" + "\n".join(f"  - {p}" for p in problems))


def check_header(header, table, name):
    """Return (problems, warnings) for one file's header."""
    problems, warnings = [], []
    expected = COLUMNS[table]
    duplicated = sorted({c for c in header if header.count(c) > 1})
    if duplicated:
        problems.append(f"{name}: column(s) named more than once: {', '.join(duplicated)}")
    missing = [c for c in expected if c not in header]
    if missing:
        problems.append(f"{name}: missing column(s) {', '.join(missing)} (expected {', '.join(expected)})")
    extra = [c for c in header if c not in expected]
    if extra:
        warnings.append(f"{name}: ignoring unexpected column(s) {', '.join(extra)}")
    return problems, warnings


def read_file(path, table):
    """Read one raw CSV as strings. Returns (DataFrame in schema column order, rejects, problems)."""
    name = path.name
    no_rows = pd.DataFrame(columns=COLUMNS[table]), pd.DataFrame(columns=REJECT_COLUMNS)
    if not path.is_file():
        return *no_rows, [f"{name}: file not found in {path.parent}"]
    if path.stat().st_size == 0:
        return *no_rows, [f"{name}: file is empty"]

    try:
        # utf-8-sig accepts an optional BOM (Excel adds one); newline="" is what the csv module wants.
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    except UnicodeDecodeError as e:
        return *no_rows, [f"{name}: not valid UTF-8 text (bad byte at position {e.start}) - "
                          "is it really a CSV export?"]
    except csv.Error as e:
        return *no_rows, [f"{name}: not a readable CSV ({e})"]

    if not rows or not any(rows[0]):
        return *no_rows, [f"{name}: no header row"]
    header = [c.strip() for c in rows[0]]
    problems, warnings = check_header(header, table, name)
    for w in warnings:
        log.warning(w)
    if problems:
        return *no_rows, problems

    good, bad = [], []
    for i, row in enumerate(rows[1:], 1):
        if len(row) == len(header):
            good.append(row)
        elif any(field.strip() for field in row):  # blank lines are skipped silently
            bad.append({"table": table, "row": i,
                        "reason": f"malformed row: {len(row)} fields, expected {len(header)}",
                        "raw": json.dumps(row)})
    if not good:
        return *no_rows, [f"{name}: header only, no data rows"]
    if len(bad) > MAX_MALFORMED_FRACTION * (len(good) + len(bad)):
        lines = ", ".join(str(b["row"] + 1) for b in bad[:5])  # +1: file line number includes the header
        return *no_rows, [f"{name}: {len(bad)} of {len(good) + len(bad)} rows have the wrong number of fields "
                          f"(e.g. lines {lines}) - the file looks corrupted or uses a different delimiter"]
    if bad:
        log.warning("%s: %d malformed row(s) set aside as rejects", name, len(bad))

    df = pd.DataFrame(good, columns=header, dtype=str)[COLUMNS[table]]
    return df, pd.DataFrame(bad, columns=REJECT_COLUMNS), []


def read_batch(raw_dir, postfix):
    """Check and read all files of a batch.

    Returns ({table: DataFrame of strings}, rejects DataFrame). Raises BatchValidationError
    listing every problem if any file can't be used, so nothing is loaded from a broken batch.
    """
    raw_dir = Path(raw_dir)
    tables, rejects, problems = {}, [], []
    for table in TABLE_NAMES:
        df, bad, file_problems = read_file(raw_dir / file_name(table, postfix), table)
        tables[table] = df
        rejects.append(bad)
        problems += file_problems
    if problems:
        raise BatchValidationError(problems)
    for table, df in tables.items():
        log.info("checked %-9s %9d rows", table, len(df))
    return tables, pd.concat(rejects, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--postfix", default="base", help="batch postfix, e.g. base -> customers_base.csv")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder with the raw CSVs (default: %(default)s)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        _, rejects = read_batch(args.raw_dir, args.postfix)
    except BatchValidationError as e:
        parser.exit(1, f"error: {e}\n")
    log.info("batch '%s' is OK (%d malformed rows set aside)", args.postfix, len(rejects))
