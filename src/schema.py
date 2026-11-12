"""The raw batch format every step agrees on: file names, tables and their columns.

Matches the dataset description in the CA1 brief (Appendix A1). A batch is three CSV files named
<table>_<postfix>.csv with exactly these columns (order in the file doesn't matter).
"""
TABLE_NAMES = ["customers", "orders", "labels"]  # parents before children

COLUMNS = {
    "customers": [
        "country_code", "customer_id", "mobile_verified", "num_orders_last_50days",
        "num_cancelled_orders_last_50days", "num_refund_orders_last_50days",
        "total_payment_last_50days", "num_associated_customers", "first_order_datetime",
    ],
    "orders": [
        "country_code", "order_id", "collect_type", "payment_method",
        "order_value", "num_items_ordered", "refund_value", "order_date",
    ],
    "labels": ["country_code", "order_id", "customer_id", "is_fraud"],
}


def file_name(table, postfix):
    return f"{table}_{postfix}.csv"
