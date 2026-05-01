from src.trend_detection import financial_health_metrics


def simulate_savings_increase(df, user_id: str, increase_pct: float, months: int) -> dict:
    """
    Simulate the impact of increasing monthly savings by a specified percentage.

    Calculates how much extra savings a user would accumulate by saving an
    additional percentage of their monthly income over a given time period.

    Args:
        df (pd.DataFrame): Transaction data containing user financial records.
        user_id (str): Unique identifier for the user.
        increase_pct (float): Percentage increase in savings (e.g. 0.1 → 10 %).
        months (int): Number of months to simulate the increased savings.

    Returns:
        dict:
            - current_savings_rate (float): User's current savings rate.
            - extra_savings (float): Additional savings accumulated over the period.
            - projected_total (float): Extra savings expressed as an annual figure.
    """
    metrics = financial_health_metrics(df, user_id)
    monthly_income = metrics["total_income"] / 12
    extra_savings = monthly_income * increase_pct * months

    return {
        "current_savings_rate": metrics["savings_rate"],
        "extra_savings": round(float(extra_savings), 2),
        "projected_total": round(float(extra_savings), 2),
    }


def simulate_expense_reduction(df, user_id: str, reduction_pct: float, months: int) -> dict:
    """
    Simulate the impact of reducing monthly expenses by a specified percentage.

    Estimates the additional savings a user would accumulate if they cut their
    monthly expenses by the given percentage over the simulation period.

    Args:
        df (pd.DataFrame): Transaction data containing user financial records.
        user_id (str): Unique identifier for the user.
        reduction_pct (float): Fraction of monthly expenses to cut (e.g. 0.15 → 15 %).
        months (int): Number of months to simulate the reduced expenses.

    Returns:
        dict:
            - current_monthly_expenses (float): Average monthly expense amount.
            - monthly_savings_gained (float): Extra saved each month.
            - total_savings_gained (float): Total extra savings over the period.
            - new_savings_rate (float): Estimated savings rate after reduction.
    """
    metrics = financial_health_metrics(df, user_id)
    monthly_expenses = metrics["total_expenses"] / 12
    monthly_income = metrics["total_income"] / 12

    monthly_savings_gained = monthly_expenses * reduction_pct
    total_savings_gained = monthly_savings_gained * months

    new_expenses = monthly_expenses - monthly_savings_gained
    new_savings_rate = (
        round((monthly_income - new_expenses) / monthly_income, 2)
        if monthly_income
        else 0.0
    )

    return {
        "current_monthly_expenses": round(float(monthly_expenses), 2),
        "monthly_savings_gained": round(float(monthly_savings_gained), 2),
        "total_savings_gained": round(float(total_savings_gained), 2),
        "new_savings_rate": new_savings_rate,
    }


def simulate_investment_growth(df, user_id: str, annual_return_rate: float, years: int) -> dict:
    """
    Project investment growth from current savings using compound interest.

    Assumes the user invests their entire annual net savings at a fixed annual
    return rate compounded annually for the given number of years.

    Args:
        df (pd.DataFrame): Transaction data containing user financial records.
        user_id (str): Unique identifier for the user.
        annual_return_rate (float): Annual return rate as a decimal (e.g. 0.07 → 7 %).
        years (int): Number of years for the investment projection.

    Returns:
        dict:
            - annual_investable_savings (float): Net savings available to invest per year.
            - projected_portfolio_value (float): Estimated portfolio value after the period.
            - total_interest_earned (float): Interest/growth earned on top of principal.
    """
    metrics = financial_health_metrics(df, user_id)
    annual_savings = metrics["total_income"] - metrics["total_expenses"]

    # Future Value of a growing annuity (annual contributions, compounded yearly)
    if annual_return_rate == 0:
        projected_value = annual_savings * years
    else:
        projected_value = annual_savings * (
            ((1 + annual_return_rate) ** years - 1) / annual_return_rate
        )

    principal = annual_savings * years
    interest_earned = projected_value - principal

    return {
        "annual_investable_savings": round(float(annual_savings), 2),
        "projected_portfolio_value": round(float(projected_value), 2),
        "total_interest_earned": round(float(interest_earned), 2),
    }
