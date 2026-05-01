"""
Auto-generated tests for src/scenario_simulation.py
Covers: simulate_savings_increase, simulate_expense_reduction,
        simulate_investment_growth — happy path, edge cases, boundary values.
"""
import pytest
import pandas as pd

from src.scenario_simulation import (
    simulate_savings_increase,
    simulate_expense_reduction,
    simulate_investment_growth,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def user_df():
    """
    Minimal DataFrame: user_1 earns $60,000/yr, spends $36,000/yr.
    savings_rate = (60000 - 36000) / 60000 = 0.4
    monthly_income   = 5000
    monthly_expenses = 3000
    annual_savings   = 24000
    """
    records = [
        {"user_id": "user_1", "amount":  60000.0, "category": "salary"},
        {"user_id": "user_1", "amount": -36000.0, "category": "expenses"},
    ]
    return pd.DataFrame(records)


@pytest.fixture
def zero_income_df():
    """User with no income at all."""
    return pd.DataFrame([{"user_id": "user_0", "amount": -500.0, "category": "expenses"}])


@pytest.fixture
def zero_expense_df():
    """User with income but zero expenses."""
    return pd.DataFrame([{"user_id": "user_x", "amount": 12000.0, "category": "salary"}])


# ─────────────────────────────────────────────────────────────────────────────
# simulate_savings_increase
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateSavingsIncrease:

    def test_returns_required_keys(self, user_df):
        """Result dict must contain the three expected keys."""
        result = simulate_savings_increase(user_df, "user_1", 0.10, 12)
        assert set(result.keys()) == {"current_savings_rate", "extra_savings", "projected_total"}

    def test_extra_savings_calculation(self, user_df):
        """
        monthly_income=5000, increase_pct=0.10, months=12
        extra_savings = 5000 * 0.10 * 12 = 6000
        """
        result = simulate_savings_increase(user_df, "user_1", 0.10, 12)
        assert result["extra_savings"] == pytest.approx(6000.0, rel=1e-3)

    def test_projected_total_equals_extra_savings(self, user_df):
        """projected_total is currently the same as extra_savings."""
        result = simulate_savings_increase(user_df, "user_1", 0.10, 12)
        assert result["projected_total"] == result["extra_savings"]

    def test_current_savings_rate_correct(self, user_df):
        """Savings rate for user_1 should be 0.4."""
        result = simulate_savings_increase(user_df, "user_1", 0.10, 12)
        assert result["current_savings_rate"] == pytest.approx(0.4, rel=1e-3)

    def test_zero_increase_pct(self, user_df):
        """Saving 0% more yields zero extra savings."""
        result = simulate_savings_increase(user_df, "user_1", 0.0, 12)
        assert result["extra_savings"] == 0.0

    def test_zero_months(self, user_df):
        """Zero months yields zero extra savings."""
        result = simulate_savings_increase(user_df, "user_1", 0.10, 0)
        assert result["extra_savings"] == 0.0

    def test_result_values_are_floats(self, user_df):
        """All numeric values must be Python floats (not numpy types)."""
        result = simulate_savings_increase(user_df, "user_1", 0.10, 12)
        assert isinstance(result["extra_savings"], float)

    @pytest.mark.parametrize("pct,months,expected_extra", [
        (0.05,  6,  1500.0),   # 5000 * 0.05 * 6
        (0.20, 12, 12000.0),   # 5000 * 0.20 * 12
        (0.10,  3,  1500.0),   # 5000 * 0.10 * 3
    ])
    def test_parametrized_extra_savings(self, user_df, pct, months, expected_extra):
        """Parametrised check of extra_savings arithmetic."""
        result = simulate_savings_increase(user_df, "user_1", pct, months)
        assert result["extra_savings"] == pytest.approx(expected_extra, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_expense_reduction
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateExpenseReduction:

    def test_returns_required_keys(self, user_df):
        """Result dict must contain all four expected keys."""
        result = simulate_expense_reduction(user_df, "user_1", 0.15, 12)
        assert set(result.keys()) == {
            "current_monthly_expenses",
            "monthly_savings_gained",
            "total_savings_gained",
            "new_savings_rate",
        }

    def test_current_monthly_expenses(self, user_df):
        """monthly_expenses = total_expenses / 12 = 36000/12 = 3000."""
        result = simulate_expense_reduction(user_df, "user_1", 0.15, 12)
        assert result["current_monthly_expenses"] == pytest.approx(3000.0, rel=1e-3)

    def test_monthly_savings_gained(self, user_df):
        """monthly_savings_gained = 3000 * 0.15 = 450."""
        result = simulate_expense_reduction(user_df, "user_1", 0.15, 12)
        assert result["monthly_savings_gained"] == pytest.approx(450.0, rel=1e-3)

    def test_total_savings_gained(self, user_df):
        """total_savings_gained = 450 * 12 = 5400."""
        result = simulate_expense_reduction(user_df, "user_1", 0.15, 12)
        assert result["total_savings_gained"] == pytest.approx(5400.0, rel=1e-3)

    def test_new_savings_rate_increases(self, user_df):
        """Cutting expenses should yield a higher savings rate."""
        original = simulate_expense_reduction(user_df, "user_1", 0.0, 12)["new_savings_rate"]
        improved = simulate_expense_reduction(user_df, "user_1", 0.15, 12)["new_savings_rate"]
        assert improved >= original

    def test_zero_reduction(self, user_df):
        """0% reduction means zero monthly savings gained."""
        result = simulate_expense_reduction(user_df, "user_1", 0.0, 12)
        assert result["monthly_savings_gained"] == 0.0
        assert result["total_savings_gained"] == 0.0

    def test_zero_income_does_not_crash(self, zero_income_df):
        """User with no income should not raise ZeroDivisionError."""
        result = simulate_expense_reduction(zero_income_df, "user_0", 0.10, 6)
        assert result["new_savings_rate"] == 0.0

    @pytest.mark.parametrize("reduction,months,expected_total", [
        (0.10, 12, 3600.0),   # 3000*0.10*12
        (0.20,  6, 3600.0),   # 3000*0.20*6
        (0.50, 12, 18000.0),  # 3000*0.50*12
    ])
    def test_parametrized_total_savings_gained(self, user_df, reduction, months, expected_total):
        result = simulate_expense_reduction(user_df, "user_1", reduction, months)
        assert result["total_savings_gained"] == pytest.approx(expected_total, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_investment_growth
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateInvestmentGrowth:

    def test_returns_required_keys(self, user_df):
        """Result dict must contain the three expected keys."""
        result = simulate_investment_growth(user_df, "user_1", 0.07, 10)
        assert set(result.keys()) == {
            "annual_investable_savings",
            "projected_portfolio_value",
            "total_interest_earned",
        }

    def test_annual_investable_savings(self, user_df):
        """annual_savings = 60000 - 36000 = 24000."""
        result = simulate_investment_growth(user_df, "user_1", 0.07, 10)
        assert result["annual_investable_savings"] == pytest.approx(24000.0, rel=1e-3)

    def test_projected_value_greater_than_principal(self, user_df):
        """Projected value must exceed principal when rate > 0."""
        result = simulate_investment_growth(user_df, "user_1", 0.07, 10)
        principal = result["annual_investable_savings"] * 10
        assert result["projected_portfolio_value"] > principal

    def test_interest_earned_equals_difference(self, user_df):
        """total_interest_earned == projected_portfolio_value - principal."""
        result = simulate_investment_growth(user_df, "user_1", 0.07, 10)
        principal = result["annual_investable_savings"] * 10
        assert result["total_interest_earned"] == pytest.approx(
            result["projected_portfolio_value"] - principal, rel=1e-3
        )

    def test_zero_return_rate(self, user_df):
        """At 0% return, projected value equals annual_savings * years."""
        result = simulate_investment_growth(user_df, "user_1", 0.0, 10)
        assert result["projected_portfolio_value"] == pytest.approx(
            result["annual_investable_savings"] * 10, rel=1e-3
        )
        assert result["total_interest_earned"] == pytest.approx(0.0, abs=1e-2)

    def test_zero_years(self, user_df):
        """0 years of investment yields 0 projected value."""
        result = simulate_investment_growth(user_df, "user_1", 0.07, 0)
        assert result["projected_portfolio_value"] == pytest.approx(0.0, abs=1e-2)

    @pytest.mark.parametrize("rate,years", [
        (0.05, 5),
        (0.10, 20),
        (0.07, 15),
    ])
    def test_higher_rate_or_years_increases_value(self, user_df, rate, years):
        """Higher rate or more years should produce a larger portfolio value."""
        result = simulate_investment_growth(user_df, "user_1", rate, years)
        principal = result["annual_investable_savings"] * years
        if rate > 0 and years > 0:
            assert result["projected_portfolio_value"] >= principal

