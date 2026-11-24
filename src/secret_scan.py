"""Scan the repository for hard-coded hostnames, IP addresses, passwords and keys.

Connection details belong in environment variables / .env (see config.py and .env.example),
never in committed files. This scan fails if any tracked file contains:
    - an IP address (other than localhost)
    - a hostname given as a literal (host="db1") or a URL to a non-local server:port
    - an AWS endpoint (RDS, Elastic Beanstalk, S3, ...)
    - a password, secret, token or API key assigned a literal value
    - credentials inside a URL (scheme://user:password@host)
    - an AWS access key or GitHub token
    - a value for a secret setting in .env.example, or a committed .env file

It scans the files git tracks (plus new files not yet ignored), so data/, .venv/ and .env are
skipped the same way git skips them. Notebooks are scanned including their saved outputs.

A line that is a deliberate example can be allowed with the marker `secret-scan: allow` on it.

Usage:
    python -m src.secret_scan          # exit code 1 and a list of findings if anything is found
"""
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW_MARKER = "secret-scan: allow"
# The scanner's own patterns and test samples would always match.
SELF_FILES = {"src/secret_scan.py", "tests/test_secret_scan.py"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".pkl", ".joblib", ".parquet", ".zip"}

# Values that are obviously placeholders, not real credentials.
PLACEHOLDERS = {"", "password", "pass", "passwd", "secret", "token", "changeme", "xxx", "***", "...",
                "<password>", "your_password", "your-password"}
LOCAL_IPS = {"127.0.0.1", "0.0.0.0"}
PLACEHOLDER_HOSTS = {"localhost", "host", "hostname", "your-host", "<host>", "example.com", "server"} | LOCAL_IPS
# .env.example keys that may have a value because they are not secrets.
ENV_EXAMPLE_ALLOWED = {"DB_PORT"}

RULES = [
    ("IP address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
    ("AWS endpoint", re.compile(r"[\w.-]+\.(?:rds|elasticbeanstalk|s3|ec2|compute)[\w.-]*\.amazonaws\.com"
                                r"|[\w.-]+\.elasticbeanstalk\.com", re.I)),
    ("password/secret literal", re.compile(
        r"""(?:password|passwd|pwd|secret|token|api_?key|access_?key)\w*["']?\s*[:=]\s*["']([^"'\s]*)["']""", re.I)),
    ("hostname literal", re.compile(r"""\b(?:db_?)?host(?:name)?["']?\s*[:=]\s*["']([^"'\s]+)["']""", re.I)),
    ("server URL with port", re.compile(r"\w+://(?:[^/\s@'\"]+@)?([\w.-]+):\d{2,5}\b")),
    ("credentials in URL", re.compile(r"\w+://[^/\s:@'\"]+:([^/\s@'\"]+)@")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_\w{22,})\b")),
]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    text: str

    def __str__(self):
        return f"{self.path}:{self.line}: {self.rule}: {self.text}"


def is_valid_ip(s):
    return all(0 <= int(p) <= 255 for p in s.split("."))


# The report hides values matching these rules, so a real secret isn't copied into logs.
MASKED_RULES = {"password/secret literal", "credentials in URL", "AWS access key", "GitHub token"}


def redact(line):
    for name, rx in RULES:
        if name in MASKED_RULES:
            line = rx.sub(lambda m: m.group(0).replace(m.group(1) if rx.groups else m.group(0), "<hidden>"), line)
    return line


def scan_line(line):
    """Return [(rule name, match)] for each rule this line breaks."""
    if ALLOW_MARKER in line:
        return []
    hits = []
    for name, rx in RULES:
        for m in rx.finditer(line):
            if name == "IP address" and (m.group(0) in LOCAL_IPS or not is_valid_ip(m.group(0))):
                continue
            if name in ("password/secret literal", "credentials in URL") and m.group(1).lower() in PLACEHOLDERS:
                continue
            if name in ("hostname literal", "server URL with port") and m.group(1).lower() in PLACEHOLDER_HOSTS:
                continue
            hits.append((name, m))
            break
    return hits


def scan_env_example(path, rel):
    findings = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#") and value.strip() and key.strip() not in ENV_EXAMPLE_ALLOWED:
            findings.append(Finding(rel, i, ".env.example has a value", f"{key.strip()}=<hidden>"))
    return findings


def scan_file(path, rel):
    if path.suffix.lower() in SKIP_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable: not source code
    findings = []
    for i, line in enumerate(text.splitlines(), 1):
        for rule, _ in scan_line(line):
            findings.append(Finding(rel, i, rule, redact(line).strip()[:120]))
    if path.name == ".env.example":
        findings += scan_env_example(path, rel)
    return findings


def repo_files(root=ROOT):
    """Files git tracks plus untracked files that aren't ignored."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         cwd=root, capture_output=True, text=True, check=True).stdout
    return sorted({p for p in out.split("\0") if p})


def scan(root=ROOT, files=None):
    root = Path(root)
    files = repo_files(root) if files is None else files
    findings = []
    for rel in files:
        path = root / rel
        if Path(rel).name == ".env":
            findings.append(Finding(rel, 0, ".env must not be committed", "add it to .gitignore"))
        elif path.is_file() and rel not in SELF_FILES:
            findings += scan_file(path, rel)
    return findings


if __name__ == "__main__":
    found = scan()
    for f in found:
        print(f)
    if found:
        print(f"\n{len(found)} possible hard-coded secret(s). Move them to .env (see .env.example), "
              f"or mark a deliberate example with '{ALLOW_MARKER}'.", file=sys.stderr)
        sys.exit(1)
    print("no hard-coded hostnames, IP addresses or passwords found")
