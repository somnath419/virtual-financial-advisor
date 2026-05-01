"""
Auto-generated tests for src/trend_detection.py
Covers: financial_health_metrics, detect_risky_patterns
— comprehensive happy path, edge cases, boundary values, parametrised scenarios.
"""
import pytest
import pandas as pd

from src.trend_detection import financial_health_metrics, detect_risky_patterns


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_df():
    """user_1: income=5000, expenses=1750 → savings_rate=0.65."""
    return pd.DataFrame([
        {"user_id": "user_1", "amount":  5000.0, "category": "salary"},
        {"user_id": "user_1", "amount": -1200.0, "category": "rent"},
        {"user_id": "user_1", "amount":  -400.0, "category": "groceries"},
        {"user_id": "user_1", "amount":  -150.0, "category": "transport"},
    ])


@pytest.fixture
def overspending_df():
    """user_2: expenses > income → high-severity overspending."""
    return pd.DataFrame([
        {"user_id": "user_2", "amount":  3000.0, "category": "salary"},
        {"user_id": "user_2", "amount": -2500.0, "category": "rent"},
        {"user_id": "user_2", "amount":  -600.0, "category": "dining"},
    ])


@pytest.fixture
def close_to_income_df():
    """user_3: expenses = 97% of income → medium-severity overspending."""
    return pd.DataFrame([
        {"user_id": "user_3", "amount":  4000.0, "category": "salary"},
        {"user_id": "user_3", "amount": -3880.0, "category": "rent"},
    ])


@pytest.fixture
def idle_savings_df():
    """user_4: high income but very low savings rate → idle_savings."""
    return pd.DataFrame([
        {"user_id": "user_4", "amount":  6000.0, "category": "salary"},
        {"user_id": "user_4", "amount": -5800.0, "category": "shopping"},
    ])


@pytest.fixture
def large_expense_df():
    """user_6: 10 regular $80 expenses + 1 x $1500 luxury → unusual_large_expense."""
    records = [
        {"user_id": "user_6", "amount":  5000.0, "category": "salary"},
    ] + [
        {"user_id": "user_6", "amount": -80.0, "category": "groceries"}
        for _ in range(10)
    ] + [
        {"user_id": "user_6", "amount": -1500.0, "category": "luxury"},
    ]
    return pd.DataFrame(records)


@pytest.fixture
def empty_df():
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=["user_id", "amount", "category"])


def find(findings, type_):
    """Return first finding matching *type_*, or None."""
    return next((f for f in findings if f["type"] == type_), None)


# ─────────────────────────────────────────────────────────────────────────────
# financial_health_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestFinancialHealthMetrics:

    def test_returns_required_keys(self, healthy_df):
        """Result must contain total_income, total_expenses, savings_rate, category_spend."""
        result = financial_health_metrics(healthy_df, "user_1")
        assert set(result.keys()) == {"total_income", "total_expenses", "savings_rate", "category_spend"}

    def test_total_income(self, healthy_df):
        result = financial_health_metrics(healthy_df, "user_1")
        assert result["total_income"] == pytest.approx(5000.0)

    def test_total_expenses(self, healthy_df):
        result = financial_health_metrics(healthy_df, "user_1")
        # abs(-1200 - 400 - 150) = 1750
        assert result["total_expenses"] == pytest.approx(1750.0)

    def test_savings_rate(self, healthy_df):
        """savings_rate = (5000 - 1750) / 5000 = 0.65."""
        result = financial_health_metrics(healthy_df, "user_1")
        assert result["savings_rate"] == pytest.approx(0.65)

    def test_category_spend_keys(self, healthy_df):
        """category_spend must contain all expense categories."""
        result = financial_health_metrics(healthy_df, "user_1")
        assert set(result["category_spend"].keys()) == {"rent", "groceries", "transport"}

    def test_category_spend_values_positive(self, healthy_df):
        """All category_spend values must be positive (abs)."""
        result = financial_health_metrics(healthy_df, "user_1")
        for cat, val in result["category_spend"].items():
            assert val > 0, f"Category '{cat}' has non-positive value {val}"

    def test_zero_income_savings_rate(self):
        """User with no income should have savings_rate=0, not raise ZeroDivisionError."""
        df = pd.DataFrame([{"user_id": "u", "amount": -200.0, "category": "food"}])
        result = financial_health_metrics(df, "u")
        assert result["savings_rate"] == 0
        assert result["total_income"] == 0.0

    def test_unknown_user_returns_zeros(self, healthy_df):
        """Non-existent user should return zeros and empty category_spend."""
        result = financial_health_metrics(healthy_df, "ghost_user")
        assert result["total_income"] == 0.0
        assert result["total_expenses"] == 0.0
        assert result["savings_rate"] == 0
        assert result["category_spend"] == {}

    def test_income_only_no_expenses(self):
        """User with only income should have 100% savings rate."""
        df = pd.DataFrame([{"user_id": "u", "amount": 1000.0, "category": "salary"}])
        result = financial_health_metrics(df, "u")
        assert result["savings_rate"] == pytest.approx(1.0)
        assert result["total_expenses"] == 0.0

    def test_multiple_users_isolation(self):
        """Metrics for user_A should not be polluted by user_B transactions."""
        df = pd.DataFrame([
            {"user_id": "user_A", "amount":  2000.0, "category": "salary"},
            {"user_id": "user_B", "amount": 50000.0, "category": "salary"},
            {"user_id": "user_A", "amount": -1000.0, "category": "rent"},
        ])
        result = financial_health_metrics(df, "user_A")
        assert result["total_income"] == pytest.approx(2000.0)


# ─────────────────────────────────────────────────────────────────────────────
# detect_risky_patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectRiskyPatterns:

    # ── healthy user ──────────────────────────────────────────────────────────

    def test_healthy_user_no_findings(self, healthy_df):
        """User with good finances should produce no findings."""
        findings = detect_risky_patterns(healthy_df, "user_1")
        assert findings == []

    # ── overspending: high severity ───────────────────────────────────────────

    def test_detects_high_severity_overspending(self, overspending_df):
        """expenses > income → high severity overspending."""
        findings = detect_risky_patterns(overspending_df, "user_2")
        f = find(findings, "overspending")
        assert f is not None
        assert f["severity"] == "high"
        assert f["message"] == "Expenses exceed income."

    def test_overspending_details_contain_income_and_expenses(self, overspending_df):
        findings = detect_risky_patterns(overspending_df, "user_2")
        f = find(findings, "overspending")
        assert "income" in f["details"]
        assert "expenses" in f["details"]

    # ── overspending: medium severity ─────────────────────────────────────────

    def test_detects_medium_severity_overspending(self, close_to_income_df):
        """expenses/income ≥ 0.95 → medium severity overspending."""
        findings = detect_risky_patterns(close_to_income_df, "user_3")
        f = find(findings, "overspending")
        assert f is not None
        assert f["severity"] == "medium"
        assert f["message"] == "Expenses are close to income."

    def test_medium_overspending_details_contain_ratio(self, close_to_income_df):
        findings = detect_risky_patterns(close_to_income_df, "user_3")
        f = find(findings, "overspending")
        assert "ratio" in f["details"]
        assert f["details"]["ratio"] >= 0.95

    # ── low savings rate ──────────────────────────────────────────────────────

    def test_detects_low_savings_rate(self, healthy_df):
        """Setting a very high min_savings_rate should trigger low_savings_rate."""
        findings = detect_risky_patterns(healthy_df, "user_1", {"min_savings_rate": 0.9})
        f = find(findings, "low_savings_rate")
        assert f is not None
        assert f["details"]["target"] == 0.9

    def test_low_savings_rate_message(self, healthy_df):
        findings = detect_risky_patterns(healthy_df, "user_1", {"min_savings_rate": 0.9})
        f = find(findings, "low_savings_rate")
        assert f["message"] == "Savings rate below target."

    def test_no_low_savings_rate_when_rate_sufficient(self, healthy_df):
        """User_1's savings_rate=0.65 > 0.2 default, so low_savings_rate must NOT fire."""
        findings = detect_risky_patterns(healthy_df, "user_1")
        assert find(findings, "low_savings_rate") is None

    # ── idle savings ──────────────────────────────────────────────────────────

    def test_detects_idle_savings(self, idle_savings_df):
        """High income + very low savings rate → idle_savings."""
        thresholds = {"idle_income_floor": 5000.0, "min_savings_rate": 0.2}
        findings = detect_risky_patterns(idle_savings_df, "user_4", thresholds)
        f = find(findings, "idle_savings")
        assert f is not None
        assert f["severity"] == "low"

    def test_idle_savings_details_income_and_rate(self, idle_savings_df):
        thresholds = {"idle_income_floor": 5000.0, "min_savings_rate": 0.2}
        findings = detect_risky_patterns(idle_savings_df, "user_4", thresholds)
        f = find(findings, "idle_savings")
        assert "income" in f["details"]
        assert "savings_rate" in f["details"]

    def test_no_idle_savings_below_income_floor(self, idle_savings_df):
        """Raising income floor above user's income → idle_savings should NOT fire."""
        thresholds = {"idle_income_floor": 999999.0}
        findings = detect_risky_patterns(idle_savings_df, "user_4", thresholds)
        assert find(findings, "idle_savings") is None

    # ── unusual large expense ─────────────────────────────────────────────────

    def test_detects_unusual_large_expense(self, large_expense_df):
        """user_6's $1500 luxury txn is a statistical outlier."""
        findings = detect_risky_patterns(large_expense_df, "user_6")
        f = find(findings, "unusual_large_expense")
        assert f is not None
        assert f["severity"] == "medium"
        assert f["details"]["amount"] == 1500.0
        assert f["details"]["category"] == "luxury"

    def test_unusual_large_expense_detail_keys(self, large_expense_df):
        """unusual_large_expense must expose amount, category, cutoff, date."""
        findings = detect_risky_patterns(large_expense_df, "user_6")
        f = find(findings, "unusual_large_expense")
        for key in ("amount", "category", "cutoff", "date"):
            assert key in f["details"], f"Missing key: {key}"

    def test_unusual_large_expense_amount_exceeds_cutoff(self, large_expense_df):
        """Flagged amount must always be >= reported cutoff."""
        findings = detect_risky_patterns(large_expense_df, "user_6")
        f = find(findings, "unusual_large_expense")
        assert f["details"]["amount"] >= f["details"]["cutoff"]

    def test_raising_absolute_floor_suppresses_flag(self, large_expense_df):
        """Setting large_expense_amount=2000 should suppress the $1500 flag."""
        findings = detect_risky_patterns(large_expense_df, "user_6", {"large_expense_amount": 2000.0})
        assert find(findings, "unusual_large_expense") is None

    def test_lowering_sigma_flags_smaller_outliers(self):
        """With sigma=1.0, a $1200 expense among {100,150} should be flagged."""
        df = pd.DataFrame([
            {"user_id": "u", "amount":  4000.0, "category": "salary"},
            {"user_id": "u", "amount":  -100.0, "category": "groceries"},
            {"user_id": "u", "amount":  -150.0, "category": "utilities"},
            {"user_id": "u", "amount": -1200.0, "category": "luxury"},
        ])
        findings = detect_risky_patterns(df, "u", {"large_expense_sigma": 1.0})
        assert find(findings, "unusual_large_expense") is not None

    def test_only_one_large_expense_flagged(self, large_expense_df):
        """user_6 has exactly one outlier txn; only one finding expected."""
        flagged = [f for f in detect_risky_patterns(large_expense_df, "user_6")
                   if f["type"] == "unusual_large_expense"]
        assert len(flagged) == 1

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_empty_dataframe_returns_empty_list(self, empty_df):
        """Empty DataFrame should return []."""
        assert detect_risky_patterns(empty_df, "user_1") == []

    def test_non_existent_user_returns_empty_list(self, healthy_df):
        """User not in DataFrame should return []."""
        assert detect_risky_patterns(healthy_df, "ghost") == []

    def test_findings_are_list_of_dicts(self, overspending_df):
        """Output must always be a list of dicts."""
        findings = detect_risky_patterns(overspending_df, "user_2")
        assert isinstance(findings, list)
        for f in findings:
            assert isinstance(f, dict)

    def test_finding_has_required_top_level_keys(self, overspending_df):
        """Each finding must have type, severity, message, details."""
        findings = detect_risky_patterns(overspending_df, "user_2")
        for f in findings:
            for key in ("type", "severity", "message", "details"):
                assert key in f, f"Missing top-level key: {key}"

    def test_custom_overspending_margin(self):
        """Setting overspending_margin=0.5 flags a user spending 60% of income."""
        df = pd.DataFrame([
            {"user_id": "u", "amount":  1000.0, "category": "salary"},
            {"user_id": "u", "amount":  -600.0, "category": "rent"},
        ])
        findings = detect_risky_patterns(df, "u", {"overspending_margin": 0.5})
        f = find(findings, "overspending")
        assert f is not None

    def test_user_with_only_income_no_findings(self):
        """User with income but zero expenses should produce no findings."""
        df = pd.DataFrame([
            {"user_id": "u", "amount": 5000.0, "category": "salary"},
        ])
        findings = detect_risky_patterns(df, "u")
        assert findings == []

    @pytest.mark.parametrize("user_id", ["user_2"])
    def test_parametrized_overspending_users(self, overspending_df, user_id):
        """Parametrised check that expected overspending users are flagged."""
        findings = detect_risky_patterns(overspending_df, user_id)
        assert find(findings, "overspending") is not None

