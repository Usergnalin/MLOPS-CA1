import pytest

from src import data_validate as dv
from src.data_validate import BatchValidationError
from src.schema import COLUMNS, TABLE_NAMES

GOOD_ROW = {
    "customers": ["MY", "c238e72e73", "True", "2", "1", "0", "13.19", "1", "2021-03-23 09:16:29"],
    "orders": ["MY", "c285aa06a5", "Delivery", "balance", "5.38", "4.0", "0.0", "2023/04/09"],
    "labels": ["MY", "c285aa06a5", "c238e72e73", "0"],
}


def write(path, header, rows, **kw):
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", **kw)


def write_batch(tmp_path, postfix="b2", n=200, skip=()):
    for t in TABLE_NAMES:
        if t not in skip:
            write(tmp_path / f"{t}_{postfix}.csv", COLUMNS[t], [GOOD_ROW[t]] * n)


def problems(tmp_path, postfix="b2"):
    with pytest.raises(BatchValidationError) as e:
        dv.read_batch(tmp_path, postfix)
    return e.value.problems


def test_good_batch(tmp_path):
    write_batch(tmp_path)
    tables, rejects = dv.read_batch(tmp_path, "b2")
    assert {t: len(df) for t, df in tables.items()} == {"customers": 200, "orders": 200, "labels": 200}
    assert rejects.empty
    assert tables["orders"]["num_items_ordered"].iloc[0] == "4.0"  # values stay strings for cleaning


def test_column_order_extra_columns_bom_and_header_spaces(tmp_path, caplog):
    write_batch(tmp_path)
    header = ["note", " is_fraud", "customer_id", "order_id", "country_code "]
    write(tmp_path / "labels_b2.csv", header, [["x", "1", "c1", "o1", "MY"]], encoding="utf-8-sig")
    tables, _ = dv.read_batch(tmp_path, "b2")
    assert list(tables["labels"].columns) == COLUMNS["labels"]
    assert tables["labels"].iloc[0].tolist() == ["MY", "o1", "c1", "1"]
    assert "ignoring unexpected column(s) note" in caplog.text


def test_every_problem_is_reported_at_once(tmp_path):
    write_batch(tmp_path, skip=["customers"])
    write(tmp_path / "orders_b2.csv", [c for c in COLUMNS["orders"] if c != "order_date"], [])
    (tmp_path / "labels_b2.csv").write_text("")
    found = problems(tmp_path)
    assert len(found) == 3
    assert "customers_b2.csv: file not found" in found[0]
    assert "orders_b2.csv: missing column(s) order_date" in found[1]
    assert "labels_b2.csv: file is empty" in found[2]


@pytest.mark.parametrize("content, message", [
    (b"\n\n", "no header row"),
    (",".join(COLUMNS["labels"]).encode() + b"\n", "header only"),
    (",".join(COLUMNS["labels"] + ["is_fraud"]).encode() + b"\nMY,o,c,1,1\n", "more than once: is_fraud"),
    (b"\xff\xfe" + "country_code".encode("utf-16-le"), "not valid UTF-8"),
])
def test_broken_files(tmp_path, content, message):
    write_batch(tmp_path)
    (tmp_path / "labels_b2.csv").write_bytes(content)
    assert message in problems(tmp_path)[0]


def test_few_malformed_rows_become_rejects(tmp_path):
    write_batch(tmp_path)
    rows = [GOOD_ROW["labels"]] * 200
    rows[9] = ["MY", "o1", "c1"]                # too few fields
    rows[49] = ["MY", "o2", "c2", "1", "extra"]  # too many
    write(tmp_path / "labels_b2.csv", COLUMNS["labels"], rows + [[""]])  # plus a blank line
    tables, rejects = dv.read_batch(tmp_path, "b2")
    assert len(tables["labels"]) == 198
    assert rejects["row"].tolist() == [10, 50]
    assert rejects["reason"].tolist() == ["malformed row: 3 fields, expected 4",
                                          "malformed row: 5 fields, expected 4"]
    assert rejects["raw"].iloc[0] == '["MY", "o1", "c1"]'


def test_many_malformed_rows_stop_the_batch(tmp_path):
    write_batch(tmp_path)
    (tmp_path / "orders_b2.csv").write_text(
        ";".join(COLUMNS["orders"]) + "\n" + "\n".join(";".join(GOOD_ROW["orders"]) for _ in range(50)))
    assert "missing column(s)" in problems(tmp_path)[0]  # wrong delimiter: whole header is one column

    write(tmp_path / "orders_b2.csv", COLUMNS["orders"], [GOOD_ROW["orders"]] * 90 + [["MY", "x"]] * 10)
    assert "10 of 100 rows have the wrong number of fields" in problems(tmp_path)[0]


def test_cli_exit_code_and_message(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    result = subprocess.run([sys.executable, "-m", "src.data_validate", "--postfix", "b2", "--raw-dir", str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 1
    assert "customers_b2.csv: file not found" in result.stderr
