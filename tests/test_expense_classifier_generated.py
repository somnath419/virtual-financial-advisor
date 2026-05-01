"""
Auto-generated tests for src/expense_classifier.py
Covers: classify_transaction – happy path, edge cases, boundary values.
"""
import pytest
from src.expense_classifier import classify_transaction


# ── Happy path ────────────────────────────────────────────────────────────────

def test_classify_transaction_positive_amount_returns_income():
    """Positive amounts should be classified as Income."""
    assert classify_transaction(500.0) == "Income"


def test_classify_transaction_negative_amount_returns_expense():
    """Negative amounts should be classified as Expense."""
    assert classify_transaction(-200.0) == "Expense"


def test_classify_transaction_zero_returns_neutral():
    """Zero amount should be classified as Neutral."""
    assert classify_transaction(0) == "Neutral"


# ── Boundary values ───────────────────────────────────────────────────────────

def test_classify_transaction_very_small_positive():
    """Smallest positive float should still be Income."""
    assert classify_transaction(0.01) == "Income"


def test_classify_transaction_very_small_negative():
    """Smallest negative float should still be Expense."""
    assert classify_transaction(-0.01) == "Expense"


def test_classify_transaction_large_income():
    """Very large positive amounts should be Income."""
    assert classify_transaction(1_000_000) == "Income"


def test_classify_transaction_large_expense():
    """Very large negative amounts should be Expense."""
    assert classify_transaction(-1_000_000) == "Expense"


# ── Return type ───────────────────────────────────────────────────────────────

def test_classify_transaction_return_type_is_str():
    """Return value must always be a string."""
    assert isinstance(classify_transaction(100), str)
    assert isinstance(classify_transaction(-100), str)
    assert isinstance(classify_transaction(0), str)


# ── Parametrised cases ────────────────────────────────────────────────────────

@pytest.mark.parametrize("amount,expected", [
    (1,       "Income"),
    (9999,    "Income"),
    (-1,      "Expense"),
    (-9999,   "Expense"),
    (0,       "Neutral"),
    (0.0,     "Neutral"),
])
def test_classify_transaction_parametrized(amount, expected):
    """Parametrised coverage of income / expense / neutral categories."""
    assert classify_transaction(amount) == expected

