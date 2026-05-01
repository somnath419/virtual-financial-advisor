# tests/trend_detection_test.py
import pytest
import pandas as pd
import datetime as dt

# Assuming functions are in 'src/trend_detection.py'
from src.trend_detection import detect_risky_patterns

@pytest.fixture
def sample_df():
    """Provides a sample DataFrame with transactions for multiple users."""
    records = [
        # User 1: Healthy financial behavior
        {"user_id": "user_1", "amount": 5000,  "category": "salary",    "type": "income",  "date": dt.date(2023, 1, 5)},
        {"user_id": "user_1", "amount": -1200, "category": "rent",       "type": "expense", "date": dt.date(2023, 1, 1)},
        {"user_id": "user_1", "amount": -400,  "category": "groceries",  "type": "expense", "date": dt.date(2023, 1, 10)},
        {"user_id": "user_1", "amount": -150,  "category": "transport",  "type": "expense", "date": dt.date(2023, 1, 15)},

        # User 2: Overspending (expenses > income)
        {"user_id": "user_2", "amount": 3000,  "category": "salary",    "type": "income",  "date": dt.date(2023, 1, 5)},
        {"user_id": "user_2", "amount": -2500, "category": "rent",       "type": "expense", "date": dt.date(2023, 1, 1)},
        {"user_id": "user_2", "amount": -600,  "category": "dining",     "type": "expense", "date": dt.date(2023, 1, 20)},

        # User 3: Unusual large expense — only 3 transactions, so sigma approach alone
        #         won't fire with default large_expense_sigma=2.5; use custom sigma in tests.
        {"user_id": "user_3", "amount": 4000,  "category": "salary",    "type": "income",  "date": dt.date(2023, 1, 5)},
        {"user_id": "user_3", "amount": -100,  "category": "groceries",  "type": "expense", "date": dt.date(2023, 1, 10)},
        {"user_id": "user_3", "amount": -150,  "category": "utilities",  "type": "expense", "date": dt.date(2023, 1, 12)},
        {"user_id": "user_3", "amount": -1200, "category": "luxury",     "type": "expense", "date": dt.date(2023, 1, 25)},

        # User 4: Idle savings (high income, low savings rate)
        {"user_id": "user_4", "amount": 6000,  "category": "salary",    "type": "income",  "date": dt.date(2023, 1, 5)},
        {"user_id": "user_4", "amount": -5800, "category": "shopping",   "type": "expense", "date": dt.date(2023, 1, 18)},

        # User 5: No records (non-existent user ID)

        # User 6: Unusual large expense — 10 regular ~80 expenses + 1 large 1,500 expense.
        # With default thresholds: μ≈209, σ≈408 → dynamic_cutoff≈1,230 → cutoff=1,230.
        # $1,500 ≥ 1,230 → unusual_large_expense fires WITHOUT any threshold override.
        {"user_id": "user_6", "amount": 5000,  "category": "salary",    "type": "income",  "date": dt.date(2023, 1, 5)},
        {"user_id": "user_6", "amount": -80,   "category": "groceries",  "type": "expense", "date": dt.date(2023, 1, 1)},
        {"user_id": "user_6", "amount": -80,   "category": "groceries",  "type": "expense", "date": dt.date(2023, 1, 2)},
        {"user_id": "user_6", "amount": -80,   "category": "transport",  "type": "expense", "date": dt.date(2023, 1, 3)},
        {"user_id": "user_6", "amount": -80,   "category": "dining",     "type": "expense", "date": dt.date(2023, 1, 6)},
        {"user_id": "user_6", "amount": -80,   "category": "utilities",  "type": "expense", "date": dt.date(2023, 1, 8)},
        {"user_id": "user_6", "amount": -80,   "category": "groceries",  "type": "expense", "date": dt.date(2023, 1, 9)},
        {"user_id": "user_6", "amount": -80,   "category": "dining",     "type": "expense", "date": dt.date(2023, 1, 11)},
        {"user_id": "user_6", "amount": -80,   "category": "transport",  "type": "expense", "date": dt.date(2023, 1, 13)},
        {"user_id": "user_6", "amount": -80,   "category": "groceries",  "type": "expense", "date": dt.date(2023, 1, 16)},
        {"user_id": "user_6", "amount": -80,   "category": "utilities",  "type": "expense", "date": dt.date(2023, 1, 19)},
        {"user_id": "user_6", "amount": -1500, "category": "luxury",     "type": "expense", "date": dt.date(2023, 1, 28)},
    ]
    df = pd.DataFrame(records)
    # Ensure correct types for calculation
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float)
    return df

# ── helpers ───────────────────────────────────────────────────────────────────

def find_finding(findings, type_):
    """Return the first finding matching *type_*, or None."""
    return next((f for f in findings if f.get("type") == type_), None)

# ── existing tests ────────────────────────────────────────────────────────────

def test_no_risks_detected_for_healthy_user(sample_df):
    """A user with healthy finances should have no risky patterns."""
    findings = detect_risky_patterns(sample_df, "user_1")
    assert findings == []

def test_detects_high_severity_overspending(sample_df):
    """Detects when a user's expenses exceed their income."""
    findings = detect_risky_patterns(sample_df, "user_2")
    overspending = find_finding(findings, "overspending")
    assert overspending is not None
    assert overspending["severity"] == "high"
    assert overspending["message"] == "Expenses exceed income."

def test_detects_idle_savings(sample_df):
    """Detects when a user with high income has a very low savings rate."""
    thresholds = {"idle_income_floor": 5000.0, "min_savings_rate": 0.2}
    findings = detect_risky_patterns(sample_df, "user_4", thresholds=thresholds)
    idle_savings = find_finding(findings, "idle_savings")
    assert idle_savings is not None
    assert idle_savings["message"] == "Income is healthy but savings rate is consistently low."
    assert idle_savings["details"]["income"] == 6000.0

def test_detects_low_savings_rate(sample_df):
    """Detects when savings rate is below the minimum target."""
    thresholds = {"min_savings_rate": 0.8}
    findings = detect_risky_patterns(sample_df, "user_1", thresholds=thresholds)
    low_savings = find_finding(findings, "low_savings_rate")
    assert low_savings is not None
    assert low_savings["message"] == "Savings rate below target."
    assert low_savings["details"]["target"] == 0.8

def test_handles_non_existent_user(sample_df):
    """Ensures an empty list is returned for a user_id not in the DataFrame."""
    findings = detect_risky_patterns(sample_df, "user_5")
    assert findings == []

def test_handles_empty_dataframe():
    """Ensures the function works correctly with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["user_id", "amount", "category", "type", "date"])
    findings = detect_risky_patterns(empty_df, "user_1")
    assert findings == []

# ── unusual large expense tests ───────────────────────────────────────────────

def test_detects_unusual_large_expense_with_default_thresholds(sample_df):
    """
    user_6 has 10 regular 80 expenses and one 1,500 luxury transaction.

    Math (ddof=0):
        μ  ≈ 209   σ  ≈ 408
        dynamic_cutoff = 209 + 2.5×408 ≈ 1,229
        cutoff = max(1,229 , 1,000)    = 1,229
        $1,500 ≥ 1,229  →  flagged ✓

    No threshold overrides – exercises the default behaviour end-to-end.
    """
    findings = detect_risky_patterns(sample_df, "user_6")
    finding  = find_finding(findings, "unusual_large_expense")

    assert finding is not None
    assert finding["severity"] == "medium"
    assert finding["message"]  == "Unusually large expense detected."
    assert finding["details"]["amount"]   == 1500.0
    assert finding["details"]["category"] == "luxury"


def test_unusual_large_expense_finding_contains_required_detail_keys(sample_df):
    """
    Every unusual_large_expense finding must expose amount, category,
    cutoff, and date so downstream code and the UI can render them.
    """
    findings = detect_risky_patterns(sample_df, "user_6")
    finding  = find_finding(findings, "unusual_large_expense")

    assert finding is not None
    for key in ("amount", "category", "cutoff", "date"):
        assert key in finding["details"], f"Missing detail key: '{key}'"


def test_unusual_large_expense_amount_exceeds_cutoff(sample_df):
    """
    The flagged transaction amount must always be >= the reported cutoff.
    This validates that the detection logic and the reported evidence are consistent.
    """
    findings = detect_risky_patterns(sample_df, "user_6")
    finding  = find_finding(findings, "unusual_large_expense")

    assert finding is not None
    assert finding["details"]["amount"] >= finding["details"]["cutoff"]


def test_only_one_large_expense_flagged_for_user_6(sample_df):
    """
    user_6 has exactly one transaction above the cutoff ($1,500).
    The other 10 regular $80 expenses must NOT produce extra findings.
    """
    findings = [f for f in detect_risky_patterns(sample_df, "user_6")
                if f["type"] == "unusual_large_expense"]
    assert len(findings) == 1


def test_detects_unusual_large_expense_with_custom_sigma(sample_df):
    """
    user_3 has only 3 expenses ($100, $150, $1,200).
    With default large_expense_sigma=2.5 the cutoff (~$1,751) exceeds $1,200,
    so it is NOT flagged.  Lowering the sigma multiplier to 1.0 brings the
    cutoff down to ~$990, and the absolute floor lifts it to $1,000, so
    $1,200 is flagged.

    Demonstrates tuning the sensitivity via the thresholds parameter.
    """
    # With default sigma → NOT flagged
    default_findings = detect_risky_patterns(sample_df, "user_3")
    assert find_finding(default_findings, "unusual_large_expense") is None, (
        "Expected no flag with default sigma (cutoff > $1,200)"
    )

    # With sigma=1.0 → flagged
    tuned_findings = detect_risky_patterns(
        sample_df, "user_3", thresholds={"large_expense_sigma": 1.0}
    )
    finding = find_finding(tuned_findings, "unusual_large_expense")
    assert finding is not None
    assert finding["details"]["amount"]   == 1200.0
    assert finding["details"]["category"] == "luxury"


def test_raising_absolute_floor_suppresses_flag(sample_df):
    """
    When large_expense_amount is raised above the transaction amount ($1,500),
    the absolute floor dominates and the expense is NOT flagged, even though
    the sigma-based dynamic_cutoff (~$1,229) would flag it by default.

    cutoff = max(dynamic≈1,229,  floor=2,000) = 2,000
    $1,500 < 2,000  →  NOT flagged ✓
    """
    thresholds = {"large_expense_amount": 2000.0}
    findings = detect_risky_patterns(sample_df, "user_6", thresholds=thresholds)
    assert find_finding(findings, "unusual_large_expense") is None


def test_lowering_absolute_floor_flags_moderate_expenses(sample_df):
    """
    When large_expense_amount is lowered to $50 (below all user_6 expenses),
    AND large_expense_sigma is set near zero so dynamic_cutoff ≈ μ,
    every expense above the $50 floor is flagged.

    Verifies the absolute-floor path is the binding constraint when it is the
    larger of the two terms in max(dynamic_cutoff, large_expense_amount).
    """
    thresholds = {"large_expense_amount": 50.0, "large_expense_sigma": 0.0}
    findings = [f for f in detect_risky_patterns(sample_df, "user_6", thresholds=thresholds)
                if f["type"] == "unusual_large_expense"]
    # dynamic_cutoff = μ + 0.0 × σ = μ ≈ 209; cutoff = max(209, 50) = 209
    # All 11 expenses ≥ 209 qualify (they are exactly 80 or 1500)
    # 80 < 209 → NOT flagged; 1500 ≥ 209 → flagged
    # Only the $1,500 expense is >= cutoff (~209)
    assert len(findings) >= 1
    amounts = [f["details"]["amount"] for f in findings]
    assert 1500.0 in amounts
