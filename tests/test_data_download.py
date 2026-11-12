import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src import data_download as dd
from src.data_download import DownloadError

CSV = b"country_code,order_id\nMY,abc\n"


@pytest.fixture
def server(monkeypatch):
    """Local HTTP server. routes: path -> list of (status, content_type, body), one per request."""
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)  # no real back-off waits in tests
    routes, hits = {}, {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits[self.path] = hits.get(self.path, 0) + 1
            replies = routes.get(self.path, [(404, "text/plain", b"not found")])
            status, ctype, body = replies[min(hits[self.path], len(replies)) - 1]
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    yield base, routes, hits
    httpd.shutdown()


def serve_batch(routes, postfix="b2"):
    for t in dd.TABLE_NAMES:
        routes[f"/{t}_{postfix}.csv"] = [(200, "text/csv", CSV)]


def test_parse_url():
    assert dd.parse_url("https://x.com/data/orders_batch2.csv?dl=1") == ("orders", "batch2")
    with pytest.raises(DownloadError, match="cannot tell the table"):
        dd.parse_url("https://x.com/download?id=123")


def test_resolve_batch_any_order():
    urls = ["h/labels_b2.csv", "h/customers_b2.csv", "h/orders_b2.csv"]
    postfix, by_table = dd.resolve_batch(urls)
    assert postfix == "b2"
    assert list(by_table) == ["customers", "orders", "labels"]


@pytest.mark.parametrize("urls, message", [
    (["h/customers_b2.csv", "h/orders_b2.csv"], "expected 3 URLs"),
    (["h/customers_b2.csv", "h/orders_b2.csv", "h/orders_b2.csv"], "two URLs for orders"),
    (["h/customers_b2.csv", "h/orders_b2.csv", "h/labels_b3.csv"], "different batches"),
])
def test_resolve_batch_rejects_bad_sets(urls, message):
    with pytest.raises(DownloadError, match=message):
        dd.resolve_batch(urls)


def test_resolve_batch_explicit_needs_postfix_for_share_links():
    links = dict(customers="h/a?id=1", orders="h/a?id=2", labels="h/a?id=3")
    with pytest.raises(DownloadError, match="--postfix is required"):
        dd.resolve_batch(**links)
    assert dd.resolve_batch(postfix="b2", **links)[0] == "b2"
    with pytest.raises(DownloadError, match="missing URL for labels"):
        dd.resolve_batch(postfix="b2", customers="h/a", orders="h/b")


def test_downloads_batch(server, tmp_path):
    base, routes, _ = server
    serve_batch(routes)
    postfix, paths = dd.run([f"{base}/{t}_b2.csv" for t in dd.TABLE_NAMES], raw_dir=tmp_path)
    assert postfix == "b2"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["customers_b2.csv", "labels_b2.csv", "orders_b2.csv"]
    assert paths["orders"].read_bytes() == CSV


def test_retries_server_errors(server, tmp_path):
    base, routes, hits = server
    routes["/f.csv"] = [(503, "text/plain", b"busy"), (200, "text/csv", CSV)]
    dd.download_file(f"{base}/f.csv", tmp_path / "f.csv")
    assert hits["/f.csv"] == 2
    assert (tmp_path / "f.csv").read_bytes() == CSV


def test_gives_up_after_retries_and_leaves_no_file(server, tmp_path):
    base, routes, hits = server
    routes["/f.csv"] = [(500, "text/plain", b"boom")]
    with pytest.raises(DownloadError, match="failed after 3 attempts"):
        dd.download_file(f"{base}/f.csv", tmp_path / "f.csv")
    assert hits["/f.csv"] == dd.RETRIES
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reply, message", [
    ((404, "text/plain", b"nope"), "HTTP 404"),
    ((200, "text/html; charset=utf-8", b"<html>login</html>"), "HTML page"),
    ((200, "text/csv", b""), "empty"),
])
def test_clear_errors_and_no_partial_file(server, tmp_path, reply, message):
    base, routes, hits = server
    routes["/f.csv"] = [reply]
    with pytest.raises(DownloadError, match=message):
        dd.download_file(f"{base}/f.csv", tmp_path / "f.csv")
    assert hits["/f.csv"] == 1  # client errors are not retried
    assert list(tmp_path.iterdir()) == []


def test_existing_file_survives_failed_download(server, tmp_path):
    base, routes, _ = server
    dest = tmp_path / "f.csv"
    dest.write_bytes(b"previous good copy")
    routes["/f.csv"] = [(404, "text/plain", b"nope")]
    with pytest.raises(DownloadError):
        dd.download_file(f"{base}/f.csv", dest)
    assert dest.read_bytes() == b"previous good copy"
