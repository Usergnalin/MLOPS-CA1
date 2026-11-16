"""MySQL tables for the cleaned batches and for the rows that were rejected.

customers, orders and labels mirror the batch files (same columns, see schema.py) with proper
types. Every row also records the batch that last wrote it (batch_id) and when (loaded_at).
rejects keeps every row that could not be loaded, with the reason, so nothing is dropped silently.

Keys:
    customers  customer_id                   (the cleaning step de-duplicates customers on it)
    orders     country_code, order_id        (order_id is reused across countries)
    labels     country_code, order_id        -> orders; customer_id -> customers
    rejects    batch_id, source_table, source_row, reason

In code:
    from src.db_schema import create_tables
    create_tables(engine)               # creates missing tables, leaves existing ones alone
    create_tables(engine, reset=True)   # drops and recreates all of them (deletes the data)
"""
import logging

from sqlalchemy import text

from src.schema import COLUMNS, TABLE_NAMES

log = logging.getLogger("db_schema")

REJECTS_TABLE = "rejects"

# MySQL type of every batch column, in schema.COLUMNS order (checked by check_columns()).
COLUMN_TYPES = {
    "customers": {
        "country_code": "CHAR(2) NOT NULL",
        "customer_id": "CHAR(10) NOT NULL",
        "mobile_verified": "BOOLEAN NOT NULL",
        "num_orders_last_50days": "INT NOT NULL",
        "num_cancelled_orders_last_50days": "INT NOT NULL",
        "num_refund_orders_last_50days": "INT NOT NULL",
        "total_payment_last_50days": "DECIMAL(18,6) NOT NULL",
        "num_associated_customers": "INT NOT NULL",
        "first_order_datetime": "DATETIME NULL",  # unknown for some customers; not imputed
    },
    "orders": {
        "country_code": "CHAR(2) NOT NULL",
        "order_id": "CHAR(10) NOT NULL",
        "collect_type": "VARCHAR(20) NOT NULL",
        "payment_method": "VARCHAR(100) NOT NULL",
        "order_value": "DECIMAL(18,6) NOT NULL",
        "num_items_ordered": "INT NOT NULL",
        "refund_value": "DECIMAL(18,6) NOT NULL",
        "order_date": "DATE NOT NULL",
    },
    "labels": {
        "country_code": "CHAR(2) NOT NULL",
        "order_id": "CHAR(10) NOT NULL",
        "customer_id": "CHAR(10) NOT NULL",
        "is_fraud": "TINYINT(1) NOT NULL",
    },
}

PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "orders": ["country_code", "order_id"],
    "labels": ["country_code", "order_id"],
}

EXTRA = {
    "customers": [],
    "orders": ["INDEX `idx_orders_order_date` (`order_date`)"],
    "labels": [
        "INDEX `idx_labels_customer` (`customer_id`)",
        "CONSTRAINT `fk_labels_order` FOREIGN KEY (`country_code`, `order_id`) "
        "REFERENCES `orders` (`country_code`, `order_id`)",
        "CONSTRAINT `fk_labels_customer` FOREIGN KEY (`customer_id`) REFERENCES `customers` (`customer_id`)",
    ],
}

BATCH_COLUMNS = [
    "`batch_id` VARCHAR(64) NOT NULL",
    "`loaded_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
]

REJECTS_DDL = f"""CREATE TABLE IF NOT EXISTS `{REJECTS_TABLE}` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `batch_id` VARCHAR(64) NOT NULL,
  `source_table` VARCHAR(20) NOT NULL,
  `source_row` INT NOT NULL COMMENT 'data row number in the raw file, 1 = first row after the header',
  `reason` VARCHAR(255) NOT NULL,
  `raw_values` TEXT NOT NULL COMMENT 'the raw row as JSON',
  `rejected_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_rejects_row` (`batch_id`, `source_table`, `source_row`, `reason`),
  INDEX `idx_rejects_reason` (`reason`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""


def q(name):
    return f"`{name}`"


def check_columns():
    for t in TABLE_NAMES:
        if list(COLUMN_TYPES[t]) != COLUMNS[t]:
            raise RuntimeError(f"COLUMN_TYPES['{t}'] is out of sync with schema.COLUMNS['{t}']")


def create_table_sql(table):
    lines = [f"{q(c)} {ddl}" for c, ddl in COLUMN_TYPES[table].items()]
    lines += BATCH_COLUMNS
    lines += ["PRIMARY KEY (" + ", ".join(q(c) for c in PRIMARY_KEYS[table]) + ")", *EXTRA[table]]
    return (f"CREATE TABLE IF NOT EXISTS {q(table)} (\n  " + ",\n  ".join(lines)
            + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")


def create_tables(engine, reset=False):
    """Create the batch tables and the rejects table if they don't exist yet."""
    check_columns()
    with engine.begin() as conn:
        if reset:
            for t in [REJECTS_TABLE, *reversed(TABLE_NAMES)]:  # children first because of foreign keys
                conn.execute(text(f"DROP TABLE IF EXISTS {q(t)}"))
            log.warning("dropped tables %s", ", ".join([*TABLE_NAMES, REJECTS_TABLE]))
        for t in TABLE_NAMES:  # parents first
            conn.execute(text(create_table_sql(t)))
        conn.execute(text(REJECTS_DDL))
