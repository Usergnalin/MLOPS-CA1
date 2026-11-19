"""Which data a model run used, as MLflow tags, so every run can be traced to its batches (#25).

Every model script passes these to src.tracking.log_model_run(dataset_tags=...), which stores
each one as a dataset.<key> tag. They come from the manifest written by src.data_fetch, so they
describe exactly the train / test files the model was trained and scored on (PBI-03 AC2).

    from src.data_fetch import load_training_data
    from src.dataset_tags import dataset_tags

    train, test, manifest = load_training_data()
    log_model_run(model, "my_model", ..., dataset_tags=dataset_tags(manifest))

A retraining run (#24) also says which batch triggered it:
    dataset_tags(manifest, new_batch="batch2")

Tags:
    data_hash          one hash for the whole train / test snapshot; equal hash = identical data
    batches            every batch the data contains, e.g. "base,batch2"
    order_date_min     earliest order date in the data
    order_date_max     latest order date in the data
    train_rows         rows the model was trained on
    test_rows          rows it was scored on
    new_batch          (retraining only) the batch that triggered this run
"""


def dataset_tags(manifest, new_batch=None):
    """Tags describing the data in a data_fetch manifest (logged as dataset.<key> by src.tracking)."""
    batches = manifest["batches"]
    if new_batch is not None and new_batch not in batches:
        raise ValueError(f"new_batch {new_batch!r} is not in the data (batches: {', '.join(batches)})")
    tags = {
        "data_hash": manifest["data_hash"],
        "batches": ",".join(batches),
        "order_date_min": manifest["order_date_min"],
        "order_date_max": manifest["order_date_max"],
        "train_rows": manifest["train"]["rows"],
        "test_rows": manifest["test"]["rows"],
    }
    if new_batch is not None:
        tags["new_batch"] = new_batch
    return tags
