"""Clean the raw customers / orders / labels CSVs.

Reads   data/raw/<table>_<postfix>.csv        (RAW_DIR in config.py)
Writes  data/clean/<table>_<postfix>.csv      rows that passed every rule (CLEAN_DIR in config.py)
        data/clean/<table>_rejects_<postfix>.csv   rows that did not, with the reason(s)
        data/clean/cleaning_report_<postfix>.json  counts per rule, for diffing runs

Every rule here comes from the findings in eda.ipynb (section 15). Nothing is dropped silently:
a row is either cleaned, or written to the rejects file with the reason it failed.

Usage (from the repo root or from src/):
    python src/data_clean.py                      # POSTFIX "base"
    python src/data_clean.py --postfix batch_01
    python src/data_clean.py --raw-dir other/folder --clean-dir other/out

From other code (e.g. a Prefect flow or a test):
    from src.data_clean import run, clean_all
    result = run(postfix="base")                  # reads + cleans + writes
    result = clean_all({"customers": df1, "orders": df2, "labels": df3})   # in memory only

Output conventions (the loader / training code can rely on these):
    * ids are lower-case 10-char hex, country_code is upper-case, dates are ISO (YYYY-MM-DD)
    * booleans (mobile_verified) are written as 1 / 0, is_fraud is 1 / 0
    * missing values are written as an empty field (--na-rep to change, e.g. '\\N' for LOAD DATA)
    * orders and labels are keyed by (country_code, order_id): order_id alone is NOT unique
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:  # run as a script from src/ (like the notebook) ...
    from config import CLEAN_DIR, RAW_DIR, setup_logging
except ModuleNotFoundError as exc:  # ... or imported as src.data_clean (like the tests)
    if exc.name != "config":
        raise
    from src.config import CLEAN_DIR, RAW_DIR, setup_logging

log = logging.getLogger("data_clean")

# ---------------------------------------------------------------------------
# Rules (edit here when the schema or the business rules change)
# ---------------------------------------------------------------------------
COUNTRIES = ["BD", "MY", "PH", "PK", "TH"]
ID_PATTERN = r"[0-9a-f]{10}"
NULL_TOKENS = {"", "nan", "null", "none", "n/a", "na", "-", "?", "nat", "missing"}

# Column order of each cleaned table (same as the raw files).
SCHEMA = {
    "customers": ["country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
                  "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
                  "total_payment_last_50days", "num_associated_customers", "first_order_datetime"],
    "orders": ["country_code", "order_id", "collect_type", "payment_method", "order_value",
               "num_items_ordered", "refund_value", "order_date"],
    "labels": ["country_code", "order_id", "customer_id", "is_fraud"],
}
TABLES = tuple(SCHEMA)

TRUE_VALUES = {"true", "yes"}
FALSE_VALUES = {"false", "no"}
COLLECT_TYPES = {"delivery": "Delivery", "pickup": "Pickup"}
# Lower-cased payment_method values merged into one label. Extend after checking with the data owner.
PAYMENT_METHOD_MAP = {"on delivery": "payment on delivery"}
PAYMENT_METHOD_UNKNOWN = "unknown"  # the 21.5k blank MY values are kept, as an explicit category

# Dates are parsed with these explicit formats only (never free inference: it swaps day and month).
ORDER_DATE_FORMATS = ["%Y/%m/%d", "%Y-%m-%d", "%Y/%b/%d", "%Y-%b-%d"]
DATETIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"]
MIN_PLAUSIBLE_DATE = pd.Timestamp("2005-01-01")

# num_associated_customers piles up at exactly 100/101, which looks like an upstream cap.
ASSOCIATED_CUSTOMERS_CAP = 100

OUTPUT_DATE_FORMAT = {"customers": "%Y-%m-%d %H:%M:%S", "orders": "%Y-%m-%d"}
DEFAULT_MAX_REJECT_PCT = 5.0  # run() reports failure if a table loses more than this share of its rows


class CleaningError(RuntimeError):
    """The input cannot be cleaned at all (missing file or missing required column)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class _Rejects:
    """Collects, per row, the reasons it is rejected. A row with no reason is clean."""

    def __init__(self, index):
        self.reasons = pd.Series("", index=index, dtype=object)
        self.counts: dict[str, int] = {}  # reason -> rows that failed that check
        self.notes: dict[str, int] = {}   # things worth knowing that do NOT reject the row

    def flag(self, mask, reason):
        mask = pd.Series(mask, index=self.reasons.index).fillna(False).astype(bool)
        n = int(mask.sum())
        if n:
            self.counts[reason] = self.counts.get(reason, 0) + n
            cur = self.reasons[mask]
            self.reasons[mask] = np.where(cur == "", reason, cur + "; " + reason)

    def note(self, name, mask):
        n = int(pd.Series(mask, index=self.reasons.index).fillna(False).astype(bool).sum())
        if n:
            self.notes[name] = n

    @property
    def live(self):
        return self.reasons.eq("")


@dataclass
class TableResult:
    clean: pd.DataFrame  # every input row (aligned to raw's index); keep only rows where rj.live
    rj: _Rejects


def _prep(raw: pd.DataFrame, table: str) -> pd.DataFrame:
    """Normalise headers, strip text, drop control characters, turn null-looking tokens into NA."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in SCHEMA[table] if c not in df.columns]
    if missing:
        raise CleaningError(f"{table}: missing required column(s) {missing}")
    extra = [c for c in df.columns if c not in SCHEMA[table]]
    if extra:
        log.warning("%s: ignoring unexpected column(s) %s", table, extra)
    df = df[SCHEMA[table]].astype(str)
    for c in df.columns:
        s = df[c].str.replace(r"[\x00-\x1f\x7f]", "", regex=True)
        s = s.str.strip().str.replace(r"\s+", " ", regex=True)
        df[c] = s.mask(s.str.lower().isin(NULL_TOKENS))
    return df


def _country(df, rj):
    out = df["country_code"].str.upper()
    rj.flag(~out.isin(COUNTRIES), "country_code missing or not in " + "/".join(COUNTRIES))
    return out


def _id(df, col, rj):
    out = df[col].str.lower()
    rj.flag(~out.str.fullmatch(ID_PATTERN, na=False), f"{col} missing or malformed")
    return out


def _number(s, name, rj, minimum=None, integer=False):
    """Parse a numeric column. Bad values reject the row (they are never imputed)."""
    x = pd.to_numeric(s, errors="coerce")
    rj.flag(s.isna(), f"{name} missing")
    rj.flag(s.notna() & x.isna(), f"{name} not numeric")
    inf = np.isinf(x)
    rj.flag(inf, f"{name} infinite")
    x = x.mask(inf)
    if integer:
        frac = x.notna() & (x % 1 != 0)
        rj.flag(frac, f"{name} not a whole number")
        x = x.mask(frac)  # '4.0' is fine, '4.5' is not
    if minimum is not None:
        rj.flag(x < minimum, f"{name} below {minimum}")
    return x.astype("Int64") if integer else x.astype("float64")


def _dates(s, formats):
    """Try each explicit format in turn. Returns (parsed, present_but_unparseable)."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    remaining = s.notna()
    for fmt in formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s.where(remaining), format=fmt, errors="coerce")
        hit = remaining & parsed.notna()
        out[hit] = parsed[hit]
        remaining = remaining & ~hit
    return out, remaining


def _plausible_max():
    return pd.Timestamp.now().normalize() + pd.Timedelta(days=1)


def _dedupe(out, rj, key_cols, key_name):
    """Exact duplicates: keep the first copy. Same key but different content: reject every copy."""
    # Only rows that share a key can be duplicates, so compare full rows within those (much faster).
    live = rj.live
    shares_key = pd.Series(False, index=out.index)
    shares_key[live] = out.loc[live, key_cols].duplicated(keep=False)
    exact = pd.Series(False, index=out.index)
    exact[shares_key] = out[shares_key].duplicated(keep="first")
    rj.flag(exact, "exact duplicate of an earlier row")
    live = rj.live
    conflict = pd.Series(False, index=out.index)
    conflict[live] = out.loc[live, key_cols].duplicated(keep=False)
    rj.flag(conflict, f"{key_name} reused with different content")


def _lookup(series, keys):
    """series.reindex(keys) as a Series aligned to keys' index (safe when `series` is empty)."""
    return pd.Series(series.reindex(keys.values).values, index=keys.index)


def _key(df):
    """One string per (country_code, order_id) so rows can be matched across tables."""
    return df["country_code"].astype(str) + "|" + df["order_id"].astype(str)


# ---------------------------------------------------------------------------
# Per-table cleaning
# ---------------------------------------------------------------------------
def clean_customers(raw: pd.DataFrame, add_flags: bool = True) -> TableResult:
    df = _prep(raw, "customers")
    rj = _Rejects(df.index)
    out = pd.DataFrame(index=df.index)

    out["country_code"] = _country(df, rj)
    out["customer_id"] = _id(df, "customer_id", rj)

    mv = df["mobile_verified"].str.lower()
    out["mobile_verified"] = pd.Series(
        np.select([mv.isin(TRUE_VALUES), mv.isin(FALSE_VALUES)], [1, 0], default=-1), index=df.index)
    rj.flag(out["mobile_verified"] < 0, "mobile_verified not true/false/yes/no")
    out["mobile_verified"] = out["mobile_verified"].where(out["mobile_verified"] >= 0).astype("Int8")

    out["num_orders_last_50days"] = _number(df["num_orders_last_50days"], "num_orders_last_50days", rj, 0, True)
    out["num_cancelled_orders_last_50days"] = _number(
        df["num_cancelled_orders_last_50days"], "num_cancelled_orders_last_50days", rj, 0, True)
    out["num_refund_orders_last_50days"] = _number(
        df["num_refund_orders_last_50days"], "num_refund_orders_last_50days", rj, 0, True)
    out["total_payment_last_50days"] = _number(df["total_payment_last_50days"], "total_payment_last_50days", rj, 0)
    out["num_associated_customers"] = _number(df["num_associated_customers"], "num_associated_customers", rj, 1, True)

    # first_order_datetime is not needed to use the row, so a missing or bad value becomes NULL (no imputing).
    first, unparseable = _dates(df["first_order_datetime"], DATETIME_FORMATS)
    implausible = first.notna() & ((first < MIN_PLAUSIBLE_DATE) | (first > _plausible_max()))
    rj.note("first_order_datetime unparseable, set to NULL", unparseable)
    rj.note("first_order_datetime implausible, set to NULL", implausible)
    out["first_order_datetime"] = first.mask(implausible)

    # Row rules that EDA marked as errors.
    n_orders = out["num_orders_last_50days"]
    rj.flag(out["num_cancelled_orders_last_50days"] > n_orders, "cancelled orders exceed total orders")
    rj.flag(out["num_refund_orders_last_50days"] > n_orders, "refunded orders exceed total orders")
    # Warnings only: counted in the report, rows kept.
    rj.note("payment > 0 but no orders", (n_orders == 0) & (out["total_payment_last_50days"] > 0))
    rj.note("orders > 0 but payment is 0", (n_orders > 0) & (out["total_payment_last_50days"] == 0))
    rj.note("first_order_datetime empty", df["first_order_datetime"].isna())

    if add_flags:
        out["first_order_datetime_missing"] = out["first_order_datetime"].isna().astype("Int8")
        out["num_associated_customers_censored"] = (
            out["num_associated_customers"].ge(ASSOCIATED_CUSTOMERS_CAP).fillna(False).astype("Int8"))

    _dedupe(out, rj, ["customer_id"], "customer_id")
    return TableResult(out, rj)


def clean_orders(raw: pd.DataFrame) -> TableResult:
    df = _prep(raw, "orders")
    rj = _Rejects(df.index)
    out = pd.DataFrame(index=df.index)

    out["country_code"] = _country(df, rj)
    out["order_id"] = _id(df, "order_id", rj)

    ct = df["collect_type"].str.lower().map(COLLECT_TYPES)
    rj.flag(ct.isna(), "collect_type not delivery/pickup")
    out["collect_type"] = ct

    pm = df["payment_method"].str.lower().replace(PAYMENT_METHOD_MAP)
    rj.note("payment_method blank, set to 'unknown'", pm.isna())
    out["payment_method"] = pm.fillna(PAYMENT_METHOD_UNKNOWN)

    out["order_value"] = _number(df["order_value"], "order_value", rj, 0)
    out["num_items_ordered"] = _number(df["num_items_ordered"], "num_items_ordered", rj, 1, True)
    out["refund_value"] = _number(df["refund_value"], "refund_value", rj, 0)

    order_date, unparseable = _dates(df["order_date"], ORDER_DATE_FORMATS)
    rj.flag(unparseable, "order_date not in a known format")
    rj.flag(order_date.notna() & ((order_date < MIN_PLAUSIBLE_DATE) | (order_date > _plausible_max())),
            "order_date outside plausible range")
    rj.flag(df["order_date"].isna(), "order_date missing")
    out["order_date"] = order_date

    # Warnings only. EDA: confirm the meaning of refund_value with the data owner before adding a rule.
    rj.note("refund_value > order_value", out["refund_value"] > out["order_value"] + 1e-9)
    rj.note("order_value is 0", out["order_value"] == 0)

    _dedupe(out, rj, ["country_code", "order_id"], "(country_code, order_id)")
    return TableResult(out, rj)


def clean_labels(raw: pd.DataFrame) -> TableResult:
    df = _prep(raw, "labels")
    rj = _Rejects(df.index)
    out = pd.DataFrame(index=df.index)

    out["country_code"] = _country(df, rj)
    out["order_id"] = _id(df, "order_id", rj)
    out["customer_id"] = _id(df, "customer_id", rj)

    fraud = pd.to_numeric(df["is_fraud"], errors="coerce")
    rj.flag(~fraud.isin([0, 1]), "is_fraud not 0/1")
    out["is_fraud"] = fraud.where(fraud.isin([0, 1])).astype("Int8")

    _dedupe(out, rj, ["country_code", "order_id"], "(country_code, order_id)")
    return TableResult(out, rj)


# ---------------------------------------------------------------------------
# Cross-table rules
# ---------------------------------------------------------------------------
def link_tables(customers: TableResult, orders: TableResult, labels: TableResult) -> None:
    """Reject labels that point at nothing, and count the cross-table oddities EDA found.

    Orphans are quarantined (written to the labels rejects file), not dropped silently.
    """
    c, o, l = customers.clean, orders.clean, labels.clean

    order_keys = set(_key(o[orders.rj.live]))
    live = labels.rj.live
    labels.rj.flag(live & ~_key(l).isin(order_keys),
                   "no matching clean order for (country_code, order_id)")
    customer_ids = set(c.loc[customers.rj.live, "customer_id"])
    live = labels.rj.live
    labels.rj.flag(live & ~l["customer_id"].isin(customer_ids), "customer_id not found in customers")

    # --- warnings (rows are kept) ---
    lv = l[labels.rj.live]
    ov = o[orders.rj.live]
    cv = c[customers.rj.live]
    orders.rj.note("order has no label", ~_key(ov).isin(set(_key(lv))))
    customers.rj.note("customer has no label", ~cv["customer_id"].isin(set(lv["customer_id"])))

    cust_country = cv.drop_duplicates("customer_id").set_index("customer_id")["country_code"]
    labels.rj.note("label country differs from customer country",
                   _lookup(cust_country, lv["customer_id"]).ne(lv["country_code"]))
    order_date = ov.set_index(_key(ov))["order_date"]
    first_order = cv.drop_duplicates("customer_id").set_index("customer_id")["first_order_datetime"]
    od = _lookup(order_date, _key(lv))
    fo = _lookup(first_order, lv["customer_id"])
    labels.rj.note("order dated before the customer's first order", od < fo.dt.normalize())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class CleanResult:
    clean: dict = field(default_factory=dict)    # table -> cleaned DataFrame
    rejects: dict = field(default_factory=dict)  # table -> rejected raw rows + reason
    report: dict = field(default_factory=dict)   # json-serialisable summary


def clean_all(raw: dict, add_flags: bool = True) -> CleanResult:
    """Clean all three tables in memory. `raw` maps table name -> DataFrame read as strings."""
    absent = [t for t in TABLES if t not in raw]
    if absent:
        raise CleaningError(f"missing table(s): {absent}")

    results = {
        "customers": clean_customers(raw["customers"], add_flags=add_flags),
        "orders": clean_orders(raw["orders"]),
        "labels": clean_labels(raw["labels"]),
    }
    link_tables(results["customers"], results["orders"], results["labels"])

    out = CleanResult()
    for name, res in results.items():
        keep = res.rj.live
        out.clean[name] = res.clean[keep].reset_index(drop=True)
        rej = raw[name].loc[~keep].copy()
        rej.insert(0, "reject_reason", res.rj.reasons[~keep].values)
        rej.insert(0, "raw_row", rej.index + 1)  # 1 = first data row of the raw file
        out.rejects[name] = rej.reset_index(drop=True)
        out.report[name] = {
            "rows_in": int(len(raw[name])),
            "rows_clean": int(keep.sum()),
            "rows_rejected": int((~keep).sum()),
            "pct_rejected": round(100 * float((~keep).mean()), 4) if len(keep) else 0.0,
            "reject_reasons": dict(sorted(res.rj.counts.items(), key=lambda kv: -kv[1])),
            "kept_but_noted": dict(sorted(res.rj.notes.items(), key=lambda kv: -kv[1])),
        }
    return out


def read_raw_table(path: Path) -> pd.DataFrame:
    """Read a CSV as strings (no type guessing, so formatting problems stay visible)."""
    if not path.exists():
        raise CleaningError(f"file not found: {path}")
    kwargs = dict(dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig",
                  encoding_errors="replace")
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        # Some rows have the wrong number of fields: skip them but say so (slower python parser).
        skipped = []
        df = pd.read_csv(path, engine="python", on_bad_lines=lambda row: skipped.append(len(row)), **kwargs)
        log.warning("%s: skipped %d row(s) with the wrong number of fields", path.name, len(skipped))
        return df


def write_outputs(result: CleanResult, clean_dir: Path, postfix: str, na_rep: str = "") -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name, df in result.clean.items():
        path = clean_dir / f"{name}_{postfix}.csv"
        df.to_csv(path, index=False, na_rep=na_rep, date_format=OUTPUT_DATE_FORMAT.get(name))
        log.info("%-10s %9d clean rows -> %s", name, len(df), path)
    for name, df in result.rejects.items():
        path = clean_dir / f"{name}_rejects_{postfix}.csv"
        df.to_csv(path, index=False)
        if len(df):
            log.warning("%-10s %9d rejected rows -> %s", name, len(df), path)
    report_path = clean_dir / f"cleaning_report_{postfix}.json"
    report_path.write_text(json.dumps(result.report, indent=2))
    log.info("report -> %s", report_path)


def run(postfix: str = "base", raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR,
        na_rep: str = "", add_flags: bool = True) -> CleanResult:
    """Read data/raw/<table>_<postfix>.csv, clean, write to clean_dir. Returns the result."""
    raw = {}
    for name in TABLES:
        path = Path(raw_dir) / f"{name}_{postfix}.csv"
        raw[name] = read_raw_table(path)
        log.info("%-10s %9d rows read from %s", name, len(raw[name]), path)
    result = clean_all(raw, add_flags=add_flags)
    write_outputs(result, Path(clean_dir), postfix, na_rep=na_rep)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--postfix", default="base", help="reads <table>_<postfix>.csv (default: base)")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR, help=f"default: {RAW_DIR}")
    p.add_argument("--clean-dir", type=Path, default=CLEAN_DIR, help=f"default: {CLEAN_DIR}")
    p.add_argument("--na-rep", default="", help="how missing values are written (default: empty field)")
    p.add_argument("--no-flags", action="store_true", help="do not add the extra *_missing / *_censored columns")
    p.add_argument("--max-reject-pct", type=float, default=DEFAULT_MAX_REJECT_PCT,
                   help="exit with code 2 if a table loses more than this %% of its rows (default: %(default)s)")
    args = p.parse_args(argv)

    setup_logging()
    try:
        result = run(args.postfix, args.raw_dir, args.clean_dir, args.na_rep, add_flags=not args.no_flags)
    except CleaningError as exc:
        log.error("%s", exc)
        return 1

    too_many = {t: r["pct_rejected"] for t, r in result.report.items() if r["pct_rejected"] > args.max_reject_pct}
    if too_many:
        log.error("too many rejected rows (limit %.1f%%): %s", args.max_reject_pct, too_many)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
