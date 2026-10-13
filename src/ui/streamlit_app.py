"""
Virtual Financial Advisor – Streamlit UI
=========================================
Multi-tab dashboard providing:
  1. Financial Dashboard  – key metrics + spending breakdown chart.
  2. Risk Analysis        – detected risky patterns with severity badges.
  3. What-If Scenarios    – interactive savings / expense / investment simulations.
  4. AI Advisor Chat      – multi-turn conversation with the LangChain agent.

Run from the project root:
    streamlit run src/ui/streamlit_app.py
"""

import os
import sys

# Ensure project root is on PYTHONPATH when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_data
from src.genai_interface import generate_advice
from src.scenario_simulation import (
    simulate_expense_reduction,
    simulate_investment_growth,
    simulate_savings_increase,
)
from src.trend_detection import detect_risky_patterns, financial_health_metrics

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Virtual Financial Advisor",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data – cached so it is only loaded once per session
# ---------------------------------------------------------------------------
DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(PROJECT_ROOT, "data", "virtual_financial_advisor_data.csv"),
)


@st.cache_data(show_spinner="Loading transaction data…")
def get_data():
    """Load and cache the transaction CSV."""
    return load_data(DATA_PATH)


df = get_data()
all_user_ids = sorted(df["user_id"].unique().tolist())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💰 Virtual Financial Advisor")
st.sidebar.markdown("---")

selected_user: str = st.sidebar.selectbox(          # type: ignore[assignment]
    "Select User",
    options=all_user_ids,
    index=0,
    help="Choose the user whose data you want to analyse.",
) or (all_user_ids[0] if all_user_ids else "")

if not selected_user:
    st.error("No users found in the dataset.")
    st.stop()

st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigate",
    options=["📊 Dashboard", "⚠️ Risk Analysis", "🔮 What-If Scenarios", "🤖 AI Advisor Chat"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by LangChain · Groq Llama 3.3 · Streamlit")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
SEVERITY_COLOUR = {"high": "🔴", "medium": "🟡", "low": "🔵"}


def fmt_currency(value: float) -> str:
    return f"${value:,.2f}"


# ---------------------------------------------------------------------------
# TAB 1 – Financial Dashboard
# ---------------------------------------------------------------------------
if nav == "📊 Dashboard":
    st.title(f"📊 Financial Dashboard — {selected_user}")
    st.markdown("---")

    metrics = financial_health_metrics(df, selected_user)

    # Key metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric("💵 Total Income", fmt_currency(metrics["total_income"]))
    col2.metric("💳 Total Expenses", fmt_currency(metrics["total_expenses"]))
    savings_rate_pct = round(metrics["savings_rate"] * 100, 1)
    col3.metric(
        "🏦 Savings Rate",
        f"{savings_rate_pct} %",
        delta=f"{savings_rate_pct - 20:.1f} % vs 20 % target",
    )

    st.markdown("---")

    # Spending breakdown
    category_spend = metrics["category_spend"]
    if category_spend:
        col_pie, col_bar = st.columns(2)

        # Pie chart
        with col_pie:
            st.subheader("Spending by Category")
            fig_pie = px.pie(
                names=list(category_spend.keys()),
                values=list(category_spend.values()),
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Bar chart
        with col_bar:
            st.subheader("Expense Amounts by Category")
            fig_bar = px.bar(
                x=list(category_spend.keys()),
                y=list(category_spend.values()),
                labels={"x": "Category", "y": "Amount ($)"},
                color=list(category_spend.values()),
                color_continuous_scale="Reds",
            )
            fig_bar.update_layout(
                coloraxis_showscale=False,
                xaxis_tickangle=-30,
                margin=dict(t=10, b=60),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No expense data available for this user.")

    # Monthly trend
    st.markdown("---")
    st.subheader("Monthly Income vs Expenses")
    user_df = df[df["user_id"] == selected_user].copy()
    user_df["month"] = user_df["date"].dt.to_period("M").astype(str)
    monthly_income = (
        user_df[user_df["amount"] > 0].groupby("month")["amount"].sum().rename("Income")
    )
    monthly_expenses = (
        user_df[user_df["amount"] < 0].groupby("month")["amount"].sum().abs().rename("Expenses")
    )
    monthly = monthly_income.to_frame().join(monthly_expenses, how="outer").fillna(0).reset_index()

    if not monthly.empty:
        fig_line = px.line(
            monthly,
            x="month",
            y=["Income", "Expenses"],
            labels={"value": "Amount ($)", "month": "Month", "variable": "Type"},
            color_discrete_map={"Income": "#2ECC71", "Expenses": "#E74C3C"},
            markers=True,
        )
        fig_line.update_layout(xaxis_tickangle=-45, margin=dict(t=10, b=80))
        st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 – Risk Analysis
# ---------------------------------------------------------------------------
elif nav == "⚠️ Risk Analysis":
    st.title(f"⚠️ Risk Analysis — {selected_user}")
    st.markdown("---")

    findings = detect_risky_patterns(df, selected_user)

    if not findings:
        st.success("✅ No risky patterns detected.  This user's finances look healthy!")
    else:
        st.warning(f"**{len(findings)} risk finding(s) detected.**")
        for i, finding in enumerate(findings, start=1):
            icon = SEVERITY_COLOUR.get(finding["severity"], "⚪")
            severity_label = finding["severity"].upper()
            with st.expander(
                f"{icon} [{severity_label}] {finding['type'].replace('_', ' ').title()} — {finding['message']}",
                expanded=True,
            ):
                st.markdown(f"**Severity:** {severity_label}")
                st.markdown(f"**Message:** {finding['message']}")
                st.markdown("**Details:**")
                for k, v in finding["details"].items():
                    if v is not None:
                        st.markdown(f"- `{k}`: **{v}**")

    # Quick AI summary button
    st.markdown("---")
    if st.button("🤖 Get AI Summary of Risks"):
        if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
            st.error("Set GROQ_API_KEY (free) or OPENAI_API_KEY in your .env file to use AI features.")
        else:
            context = (
                f"User {selected_user} has the following risk findings: {findings}. "
                "Summarise them in plain English and give 3 actionable recommendations."
            )
            with st.spinner("Generating AI advice…"):
                advice = generate_advice(context)
            st.info(advice)

# ---------------------------------------------------------------------------
# TAB 3 – What-If Scenarios
# ---------------------------------------------------------------------------
elif nav == "🔮 What-If Scenarios":
    st.title(f"🔮 What-If Scenarios — {selected_user}")
    st.markdown("---")

    # ---- Scenario 1: Save more ----
    with st.expander("📈 Scenario 1 – Increase Monthly Savings", expanded=True):
        s1_col1, s1_col2 = st.columns(2)
        extra_pct = s1_col1.slider("Extra savings (% of monthly income)", 1, 50, 10, 1)
        sim_months = s1_col2.slider("Duration (months)", 1, 60, 12, 1)

        result = simulate_savings_increase(df, selected_user, extra_pct / 100, sim_months)
        st.markdown(f"**Current savings rate:** {round(result['current_savings_rate'] * 100, 1)} %")
        st.metric(
            f"Extra savings after {sim_months} months",
            fmt_currency(result["extra_savings"]),
        )

    # ---- Scenario 2: Cut expenses ----
    with st.expander("✂️ Scenario 2 – Reduce Monthly Expenses", expanded=True):
        s2_col1, s2_col2 = st.columns(2)
        cut_pct = s2_col1.slider("Expense reduction (%)", 1, 50, 15, 1)
        cut_months = s2_col2.slider("Duration (months) ", 1, 60, 12, 1)

        result2 = simulate_expense_reduction(df, selected_user, cut_pct / 100, cut_months)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Monthly Expenses", fmt_currency(result2["current_monthly_expenses"]))
        c2.metric("Extra Saved / Month", fmt_currency(result2["monthly_savings_gained"]))
        c3.metric(
            f"Total Extra Savings ({cut_months} mo.)",
            fmt_currency(result2["total_savings_gained"]),
        )
        st.markdown(f"**New estimated savings rate:** {round(result2['new_savings_rate'] * 100, 1)} %")

    # ---- Scenario 3: Invest savings ----
    with st.expander("📊 Scenario 3 – Invest Your Net Savings", expanded=True):
        s3_col1, s3_col2 = st.columns(2)
        ret_rate = s3_col1.slider("Annual return rate (%)", 1, 20, 7, 1)
        invest_years = s3_col2.slider("Investment horizon (years)", 1, 30, 10, 1)

        result3 = simulate_investment_growth(df, selected_user, ret_rate / 100, invest_years)
        i1, i2, i3 = st.columns(3)
        i1.metric("Annual Investable Savings", fmt_currency(result3["annual_investable_savings"]))
        i2.metric(
            f"Projected Portfolio ({invest_years} yr)",
            fmt_currency(result3["projected_portfolio_value"]),
        )
        i3.metric("Total Interest Earned", fmt_currency(result3["total_interest_earned"]))

        # Growth chart
        yearly = []
        for yr in range(1, invest_years + 1):
            r = simulate_investment_growth(df, selected_user, ret_rate / 100, yr)
            yearly.append({"Year": yr, "Portfolio Value ($)": r["projected_portfolio_value"]})

        fig_invest = px.area(
            yearly,
            x="Year",
            y="Portfolio Value ($)",
            color_discrete_sequence=["#3498DB"],
        )
        fig_invest.update_layout(margin=dict(t=10, b=40))
        st.plotly_chart(fig_invest, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 – AI Advisor Chat
# ---------------------------------------------------------------------------
elif nav == "🤖 AI Advisor Chat":
    st.title("🤖 AI Financial Advisor Chat")
    st.markdown(
        "The agent follows a **THINK → ACT → OBSERVE → REASON → RESPOND** loop, "
        "autonomously calling the right tools and maintaining full session memory."
    )
    st.markdown("---")

    # ── API key guard ──────────────────────────────────────────────────────
    api_key_set = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not api_key_set:
        st.error(
            "⚠️  No API key found.  "
            "Set **GROQ_API_KEY** (free) or **OPENAI_API_KEY** in your `.env` file and restart."
        )
        st.stop()

    # ── Session management ─────────────────────────────────────────────────
    # Recreate the session whenever the selected user changes
    if (
        "advisor_session" not in st.session_state
        or st.session_state.get("session_user") != selected_user
    ):
        with st.spinner(f"Initialising advisor session for {selected_user}…"):
            from src.agent.agent_core import create_session
            st.session_state.advisor_session = create_session(selected_user)
            st.session_state.session_user    = selected_user

    session = st.session_state.advisor_session

    # ── Autonomous analysis button ─────────────────────────────────────────
    col_analyze, col_clear = st.columns([3, 1])

    with col_analyze:
        if st.button("🔍 Auto-Analyze (full 5-step autonomous analysis)", type="primary"):
            with st.spinner(
                "Running autonomous analysis: health → risks → savings sim → "
                "expense sim → investment projection…"
            ):
                report = session.analyze()
            st.success("✅ Analysis complete")
            # The history is updated inside session.analyze() → will render below

    with col_clear:
        if st.button("🗑️ Clear session"):
            session.clear_history()
            st.rerun()

    st.markdown("---")

    # ── Render conversation history ────────────────────────────────────────
    for msg in session.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Interactive chat input ─────────────────────────────────────────────
    user_input = st.chat_input(
        f"Ask anything about {selected_user}'s finances…",
        key="chat_input",
    )

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = session.chat(user_input)
            st.markdown(answer)

    # ── Session info footer ────────────────────────────────────────────────
    if session.turn_count:
        st.caption(f"💬 {session.turn_count} conversation turn(s) in this session")
