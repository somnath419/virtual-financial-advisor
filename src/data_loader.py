import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load transaction data from a CSV file.

    Args:
        path (str): File path to the CSV file containing transaction data.

    Returns:
        pd.DataFrame: DataFrame with parsed ``date`` column, containing
                      transaction data with columns such as ``user_id``,
                      ``amount``, ``category``, ``payment_method``,
                      ``merchant``, and ``description``.
    """
    return pd.read_csv(path, parse_dates=["date"])


def load_user_ids(path: str) -> list:
    """
    Return a sorted list of unique user IDs from the transaction CSV.

    Reads only the ``user_id`` column so the full file is not parsed twice
    when the caller only needs to populate a user-selection widget.

    Args:
        path (str): File path to the CSV file containing transaction data.

    Returns:
        list[str]: Alphabetically sorted list of unique user ID strings.
    """
    df = pd.read_csv(path, usecols=["user_id"])
    return sorted(df["user_id"].unique().tolist())
