from typing import List, Dict, Any, Optional
import pandas as pd


def financial_health_metrics(df: pd.DataFrame, user_id: str) -> dict:
    """
    Calculate comprehensive financial health metrics for a specific user.

    Analyzes transaction data to compute income, expenses, savings rate,
    and category-wise spending patterns.

    Args:
        df (pd.DataFrame): Transaction data containing columns for ``user_id``,
                           ``amount``, and ``category``.
        user_id (str): Unique identifier for the user to analyze.

    Returns:
        dict: Dictionary containing:
            - total_income (float): Sum of all income transactions (positive amounts).
            - total_expenses (float): Sum of all expense transactions (negative amounts).
            - savings_rate (float): Ratio of savings to income (0 to 1).
            - category_spend (dict): Mapping of expense categories to amounts spent.
    """
    user_df = df[df["user_id"] == user_id]

    income = user_df[user_df["amount"] > 0]["amount"].sum()
    expenses = abs(user_df[user_df["amount"] < 0]["amount"].sum())
    savings_rate = (income - expenses) / income if income else 0

    category_spend = (
        user_df[user_df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .abs()
        .to_dict()
    )

    return {
        "total_income": round(income, 2),
        "total_expenses": round(expenses, 2),
        "savings_rate": round(savings_rate, 2),
        "category_spend": category_spend,
    }


def detect_risky_patterns(
    df: pd.DataFrame,
    user_id: str,
    thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Detect risky financial patterns for a given user.

    Checks for:
    - **Overspending**: expenses exceed income or the expense/income ratio
      is above ``overspending_margin``.
    - **Low savings rate**: savings rate is below ``min_savings_rate``.
    - **Idle savings**: income is healthy but the savings rate is
      consistently low (below 75 % of ``min_savings_rate``).
    - **Unusually large expenses**: individual transactions that are
      statistical outliers (> mean + ``large_expense_sigma`` × std) *or*
      exceed the absolute ``large_expense_amount`` threshold.

    Args:
        df (pd.DataFrame): Transactions with columns ``user_id``, ``amount``,
                           ``category``, and optional ``date``.
        user_id (str): User identifier to analyze.
        thresholds (dict, optional): Tunables:
            - ``min_savings_rate`` (float): Minimum acceptable savings rate.
              Default ``0.2``.
            - ``overspending_margin`` (float): Expense/income ratio above
              which spending is flagged as close to income. Default ``0.95``.
            - ``large_expense_sigma`` (float): Std-dev multiplier for outlier
              detection. Default ``2.5``.
            - ``large_expense_amount`` (float): Absolute floor that always
              triggers an outlier flag. Default ``1000.0``.
            - ``idle_income_floor`` (float): Minimum income to apply the
              idle-savings heuristic. Default ``2000.0``.

    Returns:
        List[Dict[str, Any]]: A list of findings, each with keys ``type``,
        ``severity``, ``message``, and ``details``.
    """
    thresholds = thresholds or {}
    min_savings_rate    = float(thresholds.get("min_savings_rate",    0.2))
    overspending_margin = float(thresholds.get("overspending_margin", 0.95))
    large_expense_sigma  = float(thresholds.get("large_expense_sigma",  2.5))
    large_expense_amount = float(thresholds.get("large_expense_amount", 1000.0))
    idle_income_floor    = float(thresholds.get("idle_income_floor",    2000.0))

    findings: List[Dict[str, Any]] = []
    user_df = df[df["user_id"] == user_id].copy()
    if user_df.empty:
        return findings

    metrics      = financial_health_metrics(df, user_id)
    income       = metrics["total_income"]
    expenses     = metrics["total_expenses"]
    savings_rate = metrics["savings_rate"]

    # ── Overspending checks ──────────────────────────────────────────────────
    if expenses > income:
        findings.append({
            "type":     "overspending",
            "severity": "high",
            "message":  "Expenses exceed income.",
            "details":  {"income": income, "expenses": expenses, "savings_rate": savings_rate},
        })
    elif income and (expenses / income) >= overspending_margin:
        findings.append({
            "type":     "overspending",
            "severity": "medium",
            "message":  "Expenses are close to income.",
            "details":  {"income": income, "expenses": expenses, "ratio": round(expenses / income, 2)},
        })

    # ── Low savings rate ─────────────────────────────────────────────────────
    if income > 0 and savings_rate < min_savings_rate:
        findings.append({
            "type":     "low_savings_rate",
            "severity": "medium",
            "message":  "Savings rate below target.",
            "details":  {"savings_rate": savings_rate, "target": min_savings_rate},
        })

    # ── Idle savings: decent income, very low savings rate ───────────────────
    if income >= idle_income_floor and savings_rate < (min_savings_rate * 0.75):
        findings.append({
            "type":     "idle_savings",
            "severity": "low",
            "message":  "Income is healthy but savings rate is consistently low.",
            "details":  {"income": income, "savings_rate": savings_rate},
        })

    # ── Unusually large expenses: outliers by sigma or absolute threshold ────
    expenses_series = user_df[user_df["amount"] < 0]["amount"].abs()
    if not expenses_series.empty:
        mu    = float(expenses_series.mean())
        sigma = float(expenses_series.std(ddof=0))
        dynamic_cutoff = mu + large_expense_sigma * (sigma if sigma > 0 else mu)
        cutoff = max(dynamic_cutoff, large_expense_amount)

        large_txns = user_df[
            (user_df["amount"] < 0) & (user_df["amount"].abs() >= cutoff)
        ].copy()

        for _, row in large_txns.iterrows():
            findings.append({
                "type":     "unusual_large_expense",
                "severity": "medium",
                "message":  "Unusually large expense detected.",
                "details":  {
                    "amount":   round(abs(float(row["amount"])), 2),
                    "category": row.get("category", None),
                    "cutoff":   round(cutoff, 2),
                    "date":     row.get("date", None),
                },
            })

    return findings
