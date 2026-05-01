# tests/scenario_simulation_test.py
"""
Unit tests for src/scenario_simulation.py
"""
import pytest
import datetime as dt
import pandas as pd

from src.scenario_simulation import (
    simulate_savings_increase,
    simulate_expense_reduction,
    simulate_investment_growth,
)


@pytest.fixture
def sample_df():
    """
    Minimal transaction DataFrame for a single user:
      - Annual income  : $60,000  (5 x $1,000 salary entries to keep maths simple)
      - Annual expenses: $36,000  (6 x $500 expense entries)
    Monthly income  : $5,000
    Monthly expenses: $3,000
    Savings rate    : (60000 - 36000) / 60000 = 0.40
    """
    records = [
        # Income entries (positive amounts)
        {"user_id": "user_1", "amount": 60_000.0, "category": "Salary",
         "date": dt.date(2023, 1, 5)},
        # Expense entries (negative amounts)
        {"user_id": "user_1", "amount": -12_000.0, "category": "Rent",
         "date": dt.date(2023, 1, 1)},
        {"user_id": "user_1", "amount": -6_000.0, "category": "Groceries",
         "date": dt.date(2023, 1, 10)},
        {"user_id": "user_1", "amount": -6_000.0, "category": "Transport",
         "date": dt.date(2023, 1, 15)},
        {"user_id": "user_1", "amount": -6_000.0, "category": "Dining",
         "date": dt.date(2023, 1, 20)},
        {"user_id": "user_1", "amount": -6_000.0, "category": "Entertainment",
         "date": dt.date(2023, 1, 25)},
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# simulate_savings_increase
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateSavingsIncrease:

    def test_returns_expected_keys(self, sample_df):
        """Result must contain all three expected keys."""
        result = simulate_savings_increase(sample_df, "user_1", 0.10, 12)
        assert "current_savings_rate" in result
        assert "extra_savings" in result
        assert "projected_total" in result

    def test_extra_savings_calculation(self, sample_df):
        """
        Monthly income = 60000/12 = 5000.
        10% extra savings for 12 months = 5000 * 0.10 * 12 = 6000.
        """
        result = simulate_savings_increase(sample_df, "user_1", 0.10, 12)
        assert result["extra_savings"] == pytest.approx(6000.0, rel=1e-3)

    def test_savings_rate_is_correct(self, sample_df):
        """Savings rate = (60000 - 36000) / 60000 = 0.40."""
        result = simulate_savings_increase(sample_df, "user_1", 0.10, 12)
        assert result["current_savings_rate"] == pytest.approx(0.40, rel=1e-3)

    def test_extra_savings_scales_with_months(self, sample_df):
        """Doubling months should double extra_savings."""
        r6 = simulate_savings_increase(sample_df, "user_1", 0.10, 6)
        r12 = simulate_savings_increase(sample_df, "user_1", 0.10, 12)
        assert pytest.approx(r12["extra_savings"], rel=1e-3) == r6["extra_savings"] * 2

    def test_zero_increase_gives_zero_extra_savings(self, sample_df):
        """0% increase should yield $0 extra savings."""
        result = simulate_savings_increase(sample_df, "user_1", 0.0, 12)
        assert result["extra_savings"] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_user_returns_zero_extra_savings(self, sample_df):
        """An unknown user has $0 income, so extra savings should be $0."""
        result = simulate_savings_increase(sample_df, "unknown_user", 0.10, 12)
        assert result["extra_savings"] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_expense_reduction
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateExpenseReduction:

    def test_returns_expected_keys(self, sample_df):
        result = simulate_expense_reduction(sample_df, "user_1", 0.15, 12)
        for key in ("current_monthly_expenses", "monthly_savings_gained",
                    "total_savings_gained", "new_savings_rate"):
            assert key in result

    def test_monthly_expenses_calculation(self, sample_df):
        """Monthly expenses = 36000 / 12 = 3000."""
        result = simulate_expense_reduction(sample_df, "user_1", 0.15, 12)
        assert result["current_monthly_expenses"] == pytest.approx(3000.0, rel=1e-3)

    def test_monthly_savings_gained(self, sample_df):
        """15% of $3000/month = $450/month saved."""
        result = simulate_expense_reduction(sample_df, "user_1", 0.15, 12)
        assert result["monthly_savings_gained"] == pytest.approx(450.0, rel=1e-3)

    def test_total_savings_gained(self, sample_df):
        """$450/month × 12 months = $5400 total savings gained."""
        result = simulate_expense_reduction(sample_df, "user_1", 0.15, 12)
        assert result["total_savings_gained"] == pytest.approx(5400.0, rel=1e-3)

    def test_new_savings_rate_improves(self, sample_df):
        """Cutting expenses must always increase the savings rate."""
        original = simulate_savings_increase(sample_df, "user_1", 0.0, 1)
        result = simulate_expense_reduction(sample_df, "user_1", 0.15, 12)
        assert result["new_savings_rate"] > original["current_savings_rate"]

    def test_zero_reduction_preserves_savings_rate(self, sample_df):
        """0% reduction → savings rate unchanged."""
        original = simulate_savings_increase(sample_df, "user_1", 0.0, 1)
        result = simulate_expense_reduction(sample_df, "user_1", 0.0, 12)
        assert result["new_savings_rate"] == pytest.approx(
            original["current_savings_rate"], rel=1e-3
        )


# ─────────────────────────────────────────────────────────────────────────────
# simulate_investment_growth
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateInvestmentGrowth:

    def test_returns_expected_keys(self, sample_df):
        result = simulate_investment_growth(sample_df, "user_1", 0.07, 10)
        for key in ("annual_investable_savings", "projected_portfolio_value",
                    "total_interest_earned"):
            assert key in result

    def test_annual_investable_savings(self, sample_df):
        """Net savings = income − expenses = 60000 − 36000 = 24000."""
        result = simulate_investment_growth(sample_df, "user_1", 0.07, 10)
        assert result["annual_investable_savings"] == pytest.approx(24_000.0, rel=1e-3)

    def test_zero_return_equals_principal(self, sample_df):
        """At 0% return the portfolio equals contributions only."""
        result = simulate_investment_growth(sample_df, "user_1", 0.0, 5)
        assert result["projected_portfolio_value"] == pytest.approx(
            result["annual_investable_savings"] * 5, rel=1e-3
        )
        assert result["total_interest_earned"] == pytest.approx(0.0, abs=1.0)

    def test_positive_interest_with_positive_return(self, sample_df):
        """A positive return rate must generate positive interest."""
        result = simulate_investment_growth(sample_df, "user_1", 0.07, 10)
        assert result["total_interest_earned"] > 0

    def test_longer_horizon_gives_higher_value(self, sample_df):
        """A 20-year horizon should yield more than a 10-year horizon."""
        r10 = simulate_investment_growth(sample_df, "user_1", 0.07, 10)
        r20 = simulate_investment_growth(sample_df, "user_1", 0.07, 20)
        assert r20["projected_portfolio_value"] > r10["projected_portfolio_value"]

    def test_higher_return_gives_higher_value(self, sample_df):
        """Higher annual return rate should produce a larger portfolio."""
        r_low = simulate_investment_growth(sample_df, "user_1", 0.05, 10)
        r_high = simulate_investment_growth(sample_df, "user_1", 0.10, 10)
        assert r_high["projected_portfolio_value"] > r_low["projected_portfolio_value"]

