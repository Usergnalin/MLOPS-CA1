"""Download one batch of raw CSVs (customers, orders, labels) from URLs into data/raw/.

A new batch arrives as three publicly accessible URLs. Each file is saved as
data/raw/<table>_<postfix>.csv, the name every later step expects.

The table and postfix are read from each URL's file name (e.g. .../orders_batch2.csv), so the
URLs can be given in any order. If the URLs don't end in such a name (e.g. a share link), give
each table's URL and the postfix explicitly.

Each file is written to a .part file first and renamed when complete, so an interrupted download
never leaves a half-written CSV behind for the next step.

Usage:
    python -m src.data_download https://host/customers_b2.csv https://host/orders_b2.csv https://host/labels_b2.csv
    python -m src.data_download --postfix b2 --customers URL --orders URL --labels URL
    python -m src.data_download ... --raw-dir some/folder
"""
import argparse
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

log = logging.getLogger("data_download")

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TABLE_NAMES = ["customers", "orders", "labels"]
FILE_NAME = re.compile(r"^(?P<table>customers|orders|labels)_(?P<postfix>\w+)\.csv$")

TIMEOUT = (10, 60)  # seconds: connect, and between bytes received
RETRIES = 3
CHUNK_SIZE = 1 << 20


class DownloadError(RuntimeError):
    pass


def parse_url(url):
    """Return (table, postfix) from a URL ending in <table>_<postfix>.csv."""
    name = unquote(Path(urlparse(url).path).name)
    m = FILE_NAME.match(name)
    if not m:
        raise DownloadError(
            f"cannot tell the table and postfix from {url!r}: its file name should look like "
            "customers_<postfix>.csv; otherwise pass --postfix with --customers/--orders/--labels")
    return m["table"], m["postfix"]


def resolve_batch(urls=(), postfix=None, customers=None, orders=None, labels=None):
    """Turn the user's input into (postfix, {table: url}) for exactly one complete batch."""
    explicit = {"customers": customers, "orders": orders, "labels": labels}
    if any(explicit.values()):
        if urls:
            raise DownloadError("give either positional URLs or --customers/--orders/--labels, not both")
        missing = [t for t, u in explicit.items() if not u]
        if missing:
            raise DownloadError(f"missing URL for {', '.join(missing)}")
        if not postfix:
            try:
                postfixes = {parse_url(u)[1] for u in explicit.values()}
            except DownloadError:
                postfixes = set()
            if len(postfixes) != 1:
                raise DownloadError("--postfix is required when the URLs don't share one <table>_<postfix>.csv name")
            postfix = postfixes.pop()
        return postfix, explicit

    if len(urls) != len(TABLE_NAMES):
        raise DownloadError(f"expected {len(TABLE_NAMES)} URLs (customers, orders, labels), got {len(urls)}")
    by_table, postfixes = {}, set()
    for url in urls:
        table, url_postfix = parse_url(url)
        if table in by_table:
            raise DownloadError(f"two URLs for {table}: {by_table[table]} and {url}")
        by_table[table] = url
        postfixes.add(url_postfix)
    if postfix is None:
        if len(postfixes) != 1:
            raise DownloadError(f"the URLs are from different batches: postfixes {sorted(postfixes)}")
        postfix = postfixes.pop()
    return postfix, {t: by_table[t] for t in TABLE_NAMES}


def download_file(url, dest, session=None):
    """Stream url to dest atomically, retrying connection errors and 5xx responses."""
    session = session or requests.Session()
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                if r.status_code >= 500:
                    raise requests.HTTPError(f"server error {r.status_code}", response=r)
                if r.status_code != 200:
                    # 4xx won't fix itself: stop now with the status in the message.
                    raise DownloadError(f"{url}: HTTP {r.status_code} {r.reason}")
                if "text/html" in r.headers.get("Content-Type", ""):
                    raise DownloadError(
                        f"{url} returned an HTML page, not a CSV - is it a share/login page "
                        "instead of a direct download link?")
                size = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(CHUNK_SIZE):
                        f.write(chunk)
                        size += len(chunk)
            if size == 0:
                raise DownloadError(f"{url}: downloaded file is empty")
            os.replace(part, dest)
            return size
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            part.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise DownloadError(f"{url}: failed after {RETRIES} attempts ({e})") from e
            wait = 2 ** attempt
            log.warning("%s: %s - retrying in %ds (%d/%d)", url, e, wait, attempt, RETRIES)
            time.sleep(wait)
        except BaseException:
            part.unlink(missing_ok=True)
            raise


def run(urls=(), postfix=None, raw_dir=RAW_DIR, customers=None, orders=None, labels=None):
    """Download one batch. Returns (postfix, {table: saved path})."""
    postfix, by_table = resolve_batch(urls, postfix, customers, orders, labels)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {t: raw_dir / f"{t}_{postfix}.csv" for t in by_table}

    start = time.perf_counter()
    with requests.Session() as session, ThreadPoolExecutor(len(by_table)) as pool:
        futures = {t: pool.submit(download_file, by_table[t], paths[t], session) for t in by_table}
        sizes = {t: f.result() for t, f in futures.items()}
    for t in by_table:
        log.info("downloaded %-9s %8.1f MB -> %s", t, sizes[t] / 1e6, paths[t])
    log.info("downloaded batch '%s' in %.1fs", postfix, time.perf_counter() - start)
    return postfix, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*", help="the three URLs, named <table>_<postfix>.csv, any order")
    parser.add_argument("--customers", help="URL of the customers file")
    parser.add_argument("--orders", help="URL of the orders file")
    parser.add_argument("--labels", help="URL of the labels file")
    parser.add_argument("--postfix", help="batch postfix (default: taken from the file names)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="where to save the files (default: %(default)s)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        run(args.urls, args.postfix, args.raw_dir, args.customers, args.orders, args.labels)
    except DownloadError as e:
        parser.exit(1, f"error: {e}\n")
