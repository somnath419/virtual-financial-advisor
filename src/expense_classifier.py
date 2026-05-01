def classify_transaction(amount: float) -> str:
    """
    Classify a transaction as Income, Expense, or Neutral based on its amount.

    Simple rule-based classification that categorizes a transaction solely on
    the sign of its numeric value.

    Args:
        amount (float): Transaction amount. Positive values represent income,
                        negative values represent expenses, and zero is neutral.

    Returns:
        str: One of:
            - ``"Income"``  – amount is strictly positive.
            - ``"Expense"`` – amount is strictly negative.
            - ``"Neutral"`` – amount is zero.
    """
    if amount > 0:
        return "Income"
    elif amount < 0:
        return "Expense"
    return "Neutral"
