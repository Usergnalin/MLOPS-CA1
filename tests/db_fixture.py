import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@pytest.fixture
def engine():
    """Engine on a fresh, throwaway MySQL database; dropped again after the test.

    Set TEST_DB_URL to a server where you may create databases, e.g.
    mysql+pymysql://user:password@127.0.0.1:3306 . Never point it at the team database.
    """
    url = os.environ.get("TEST_DB_URL")
    if not url:
        pytest.skip("TEST_DB_URL not set - skipping database tests")
    name = f"ca1test_{uuid.uuid4().hex[:8]}"
    server = create_engine(url)
    with server.begin() as conn:
        conn.execute(text(f"CREATE DATABASE `{name}` CHARACTER SET utf8mb4"))
    eng = create_engine(make_url(url).set(database=name), connect_args={"local_infile": True})
    try:
        yield eng
    finally:
        eng.dispose()
        with server.begin() as conn:
            conn.execute(text(f"DROP DATABASE `{name}`"))
        server.dispose()
