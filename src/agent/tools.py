from langchain_core.tools import tool

from src.trend_detection import financial_health_metrics, detect_risky_patterns
from src.scenario_simulation import (
    simulate_savings_increase,
    simulate_expense_reduction,
    simulate_investment_growth,
)


def _df():
    """Lazy accessor for the shared DataFrame (avoids circular imports)."""
    from src.agent.agent_core import get_df

    return get_df()


@tool
def analyze_financial_health(user_id: str) -> dict:
    """
    Analyze income, expenses, and savings rate for a given user.

    Fetches the user's complete transaction history and returns comprehensive
    financial health metrics, including total income, total expenses, savings
    rate, and a category-wise expense breakdown.

    Args:
        user_id (str): Unique identifier for the user (e.g. ``"user_1"``).

    Returns:
        dict:
            - total_income (float): Sum of all positive (income) transactions.
            - total_expenses (float): Sum of all negative (expense) transactions.
            - savings_rate (float): Ratio of net savings to income (0 – 1).
            - category_spend (dict): Expense amounts keyed by category name.
    """
    return financial_health_metrics(_df(), user_id)


@tool
def check_risky_patterns(user_id: str) -> list:
    """
    Detect risky financial patterns for a given user.

    Identifies overspending, low savings rate, idle savings, and unusually
    large individual expenses using statistical and rule-based heuristics.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        list[dict]: A list of risk findings, each with keys:
            - type (str): Risk category (e.g. ``"overspending"``).
            - severity (str): ``"high"``, ``"medium"``, or ``"low"``.
            - message (str): Human-readable description.
            - details (dict): Supporting numeric evidence.
    """
    return detect_risky_patterns(_df(), user_id)


@tool
def simulate_saving_more(user_id: str) -> dict:
    """
    Simulate the impact of saving 10 % more of monthly income for 12 months.

    Useful as a quick "what-if" baseline to show the compounding benefit of
    a modest increase in monthly savings discipline.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        dict:
            - current_savings_rate (float): User's current savings rate.
            - extra_savings (float): Extra amount saved over the 12-month window.
            - projected_total (float): Same as extra_savings (annual figure).
    """
    return simulate_savings_increase(_df(), user_id, 0.10, 12)


@tool
def simulate_expense_cut(user_id: str) -> dict:
    """
    Simulate the impact of cutting monthly expenses by 15 % for 12 months.

    Models the savings improvement if the user trims discretionary spending
    by 15 % and maintains that behaviour for a full year.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        dict:
            - current_monthly_expenses (float): Baseline average monthly expenses.
            - monthly_savings_gained (float): Extra saved per month after the cut.
            - total_savings_gained (float): Total extra savings over 12 months.
            - new_savings_rate (float): Estimated savings rate after the cut.
    """
    return simulate_expense_reduction(_df(), user_id, 0.15, 12)


@tool
def simulate_investment(user_id: str) -> dict:
    """
    Project portfolio growth by investing annual net savings at 7 % over 10 years.

    Uses compound-interest future-value math to estimate how much the user's
    current net annual savings would grow if invested in a diversified index.

    Args:
        user_id (str): Unique identifier for the user.

    Returns:
        dict:
            - annual_investable_savings (float): Net annual savings available to invest.
            - projected_portfolio_value (float): Estimated value after 10 years.
            - total_interest_earned (float): Growth earned on top of contributions.
    """
    return simulate_investment_growth(_df(), user_id, 0.07, 10)
