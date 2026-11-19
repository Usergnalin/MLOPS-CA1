"""Tests for src/dataset_tags.py."""
import pytest

from src.dataset_tags import dataset_tags

MANIFEST = {
    "data_hash": "abc123",
    "batches": {"base": 900, "batch2": 100},
    "order_date_min": "2022-11-12",
    "order_date_max": "2023-06-30",
    "train": {"rows": 800},
    "test": {"rows": 200},
}


def test_tags_come_from_the_manifest():
    assert dataset_tags(MANIFEST) == {
        "data_hash": "abc123",
        "batches": "base,batch2",
        "order_date_min": "2022-11-12",
        "order_date_max": "2023-06-30",
        "train_rows": 800,
        "test_rows": 200,
    }


def test_retraining_run_records_the_new_batch():
    assert dataset_tags(MANIFEST, new_batch="batch2")["new_batch"] == "batch2"


def test_new_batch_must_be_in_the_data():
    with pytest.raises(ValueError, match="batch3"):
        dataset_tags(MANIFEST, new_batch="batch3")
