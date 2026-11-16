"""Tests for src/secret_scan.py, plus the check itself: the repository must contain no hard-coded secrets."""
import subprocess
import sys

import pytest

from src import secret_scan


def test_repository_has_no_hardcoded_secrets():
    findings = secret_scan.scan()
    assert not findings, "hard-coded connection details found:\n" + "\n".join(map(str, findings))


@pytest.mark.parametrize("line, rule", [
    ('DB_HOST = "ubuntuserver"', "hostname literal"),
    ('create_engine(host="mydb.internal")', "hostname literal"),
    ('MLFLOW = "http://ubuntuserver:5000"', "server URL with port"),
    ('ip = "192.168.1.31"', "IP address"),
    ('url = "team-db.abc123.ap-southeast-1.rds.amazonaws.com"', "AWS endpoint"),
    ('url = "mlflow-env.eba-xyz.us-east-1.elasticbeanstalk.com"', "AWS endpoint"),
    ('password = "hunter2"', "password/secret literal"),
    ('{"api_key": "abc123def"}', "password/secret literal"),
    ('"mysql+pymysql://admin:Sup3rS3cret@db/x"', "credentials in URL"),
    ('key = "AKIAABCDEFGHIJKLMNOP"', "AWS access key"),
    ("token = ghp_" + "a" * 36, "GitHub token"),
])
def test_catches(line, rule):
    assert rule in [name for name, _ in secret_scan.scan_line(line)]


@pytest.mark.parametrize("line", [
    'url = "mysql+pymysql://user:password@127.0.0.1:3306"',  # placeholder credentials, local host
    'MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]',
    'password = os.environ.get("DB_PASSWORD")',
    'host, user, password = require_env("DB_HOST", "DB_USER", "DB_PASSWORD")',
    'print("numpy 1.26.4")',
    'see https://github.com/Usergnalin/MLOPS-CA1/issues/39',
    'server = "http://localhost:5000"',
    'ip = "10.0.0.1"  # secret-scan: allow',
    'version = "999.1.2.3"',  # not a valid IP
])
def test_allows(line):
    assert secret_scan.scan_line(line) == []


def test_report_hides_secret_values(tmp_path):
    (tmp_path / "app.py").write_text('url = "mysql://admin:Sup3rS3cret@db.example.org:3306"\npassword = "hunter2"\n')
    report = "\n".join(map(str, secret_scan.scan(tmp_path, ["app.py"])))
    assert "Sup3rS3cret" not in report and "hunter2" not in report
    assert "app.py:1: server URL with port" in report and "app.py:2: password/secret literal" in report


def test_env_example_must_have_empty_secret_values(tmp_path):
    (tmp_path / ".env.example").write_text("# MySQL\nDB_HOST=\nDB_PORT=3306\nDB_PASSWORD=abc\n")
    findings = secret_scan.scan(tmp_path, [".env.example"])
    assert [(f.line, f.rule) for f in findings] == [(4, ".env.example has a value")]
    assert "abc" not in str(findings[0])


def test_scans_files_git_would_commit_and_flags_a_committed_env(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / ".gitignore").write_text("data/\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dump.csv").write_text("host,192.168.1.31\n")  # ignored by git: not scanned
    (tmp_path / "notebook.ipynb").write_text('{"outputs": ["connected to 192.168.1.31"]}')  # untracked: scanned
    (tmp_path / ".env").write_text("DB_PASSWORD=x\n")
    git("add", "-f", ".env")
    found = {(f.path, f.rule) for f in secret_scan.scan(tmp_path)}
    assert found == {("notebook.ipynb", "IP address"), (".env", ".env must not be committed")}


def test_cli():
    result = subprocess.run([sys.executable, "-m", "src.secret_scan"], cwd=secret_scan.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
