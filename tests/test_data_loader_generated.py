"""
Auto-generated tests for src/data_loader.py
Covers: load_data, load_user_ids — happy path, edge cases, file I/O.
"""
import os
import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import load_data, load_user_ids


# ── Shared fixture ────────────────────────────────────────────────────────────

DATA_PATH = str(
    Path(__file__).parent.parent / "data" / "virtual_financial_advisor_data.csv"
)


@pytest.fixture
def sample_csv(tmp_path):
    """Write a minimal CSV and return its path."""
    csv = tmp_path / "transactions.csv"
    csv.write_text(
        "user_id,amount,category,date\n"
        "user_1,5000,salary,2023-01-05\n"
        "user_1,-1200,rent,2023-01-01\n"
        "user_2,3000,salary,2023-01-05\n"
        "user_3,-400,groceries,2023-01-10\n",
        encoding="utf-8",
    )
    return str(csv)


# ─────────────────────────────────────────────────────────────────────────────
# load_data
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadData:

    def test_returns_dataframe(self, sample_csv):
        """load_data must return a pandas DataFrame."""
        df = load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, sample_csv):
        """All four CSV rows should be loaded."""
        df = load_data(sample_csv)
        assert len(df) == 4

    def test_date_column_parsed(self, sample_csv):
        """date column must be parsed as datetime."""
        df = load_data(sample_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_expected_columns_present(self, sample_csv):
        """DataFrame must contain user_id, amount, category, date."""
        df = load_data(sample_csv)
        for col in ("user_id", "amount", "category", "date"):
            assert col in df.columns, f"Missing column: {col}"

    def test_file_not_found_raises(self):
        """Non-existent file path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")

    @pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Real CSV not available")
    def test_loads_real_csv(self):
        """Integration smoke-test: real CSV loads without error."""
        df = load_data(DATA_PATH)
        assert not df.empty
        assert "user_id" in df.columns

    def test_amount_column_numeric(self, sample_csv):
        """amount column must contain numeric values."""
        df = load_data(sample_csv)
        assert pd.api.types.is_numeric_dtype(df["amount"])


# ─────────────────────────────────────────────────────────────────────────────
# load_user_ids
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadUserIds:

    def test_returns_list(self, sample_csv):
        """load_user_ids must return a list."""
        result = load_user_ids(sample_csv)
        assert isinstance(result, list)

    def test_returns_unique_users(self, sample_csv):
        """Returned list must contain no duplicates."""
        result = load_user_ids(sample_csv)
        assert len(result) == len(set(result))

    def test_returns_sorted_users(self, sample_csv):
        """Returned list must be sorted alphabetically."""
        result = load_user_ids(sample_csv)
        assert result == sorted(result)

    def test_correct_user_ids(self, sample_csv):
        """Returned IDs must match the unique user_ids in the CSV."""
        result = load_user_ids(sample_csv)
        assert set(result) == {"user_1", "user_2", "user_3"}

    def test_file_not_found_raises(self):
        """Non-existent file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_user_ids("/nonexistent/path/data.csv")

    @pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Real CSV not available")
    def test_loads_user_ids_from_real_csv(self):
        """Integration: all user_ids from the real CSV are non-empty strings."""
        result = load_user_ids(DATA_PATH)
        assert len(result) > 0
        assert all(isinstance(uid, str) and uid for uid in result)

    def test_single_user_csv(self, tmp_path):
        """CSV with one user must return a list with exactly one entry."""
        csv = tmp_path / "single.csv"
        csv.write_text("user_id,amount\nuser_only,100\nuser_only,200\n", encoding="utf-8")
        result = load_user_ids(str(csv))
        assert result == ["user_only"]


