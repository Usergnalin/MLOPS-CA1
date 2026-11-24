"""Tests for src/data_load.py: the pipeline's load step reads the clean step's files."""
import pandas as pd
from sqlalchemy import text

from tests.db_fixture import engine  # noqa: F401 - pytest fixture
from tests.test_data_clean import raw_batch

from src import data_clean, data_load
from src.data_validate import REJECT_COLUMNS


def write_clean(tmp_path, postfix="b1"):
    raw_dir, clean_dir = tmp_path / "raw", tmp_path / "clean"
    raw_dir.mkdir()
    for table, df in raw_batch().items():
        df.to_csv(raw_dir / f"{table}_{postfix}.csv", index=False)
    data_clean.run(postfix, raw_dir, clean_dir)  # the pipeline's clean step
    return clean_dir


def test_clean_files_come_back_in_the_same_format_as_in_memory_cleaning(tmp_path):
    clean_dir = write_clean(tmp_path)

    tables, rejects = data_load.read_clean("b1", clean_dir)
    expected_tables, expected_rejects = data_clean.for_load(raw_batch())

    for table, df in tables.items():
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_tables[table].reset_index(drop=True))
    assert list(rejects.columns) == REJECT_COLUMNS
    assert sorted(rejects["reason"]) == sorted(expected_rejects["reason"])


def test_missing_clean_files_say_to_run_the_clean_step(tmp_path):
    try:
        data_load.read_clean("nope", tmp_path)
    except FileNotFoundError as e:
        assert "run the clean step first" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_run_loads_into_mysql(engine, tmp_path):  # skipped unless TEST_DB_URL is set
    clean_dir = write_clean(tmp_path)

    out = data_load.run("b1", clean_dir, engine=engine)
    again = data_load.run("b1", clean_dir, engine=engine)  # safe to repeat

    assert out["rows_loaded"] == {"customers": 2, "orders": 2, "labels": 2}
    assert again["stats"]["orders"]["inserted"] == 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM rejects")).scalar_one() == 3
