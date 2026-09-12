from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import DEFAULT_PORTFOLIO, SchemeConfig, portfolio_statistics, simulate
from mortality import MortalityBasis, mortality_metadata


st.set_page_config(page_title="DB Pension ALM Engine", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
MORTALITY_FILE = ROOT / "2002_mortality_tables_single_lives.xlsx"


@st.cache_resource
def load_mortality() -> MortalityBasis:
    return MortalityBasis.from_workbook(MORTALITY_FILE)


@st.cache_data(show_spinner=False)
def run_model(config_dict: dict, portfolio_json: str):
    config = SchemeConfig(**config_dict)
    portfolio = pd.read_json(portfolio_json, orient="split")
    return simulate(config, load_mortality(), portfolio)


def money(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"£{value / 1_000_000_000:,.2f}bn"
    return f"£{value / 1_000_000:,.1f}m"


st.title("Defined Benefit Pension Asset Liability Model")
st.caption(
    "Model member inflows, contributions, retirements, pension payments, mortality, "
    "accrued liabilities and stochastic investment returns."
)

with st.sidebar:
    st.header("Run controls")
    projection_years = st.slider("Projection years", 10, 80, 40, 5)
    simulations = st.select_slider("Simulation paths", [100, 250, 500, 1_000, 2_000], 500)
    seed = st.number_input("Random seed", min_value=0, value=42)
    run = st.button("Run simulation", type="primary", use_container_width=True)

scheme_tab, portfolio_tab, results_tab, method_tab = st.tabs(
    ["Scheme", "Portfolio", "Results", "Method and limitations"]
)

with scheme_tab:
    st.subheader("Scheme and membership assumptions")
    c1, c2, c3 = st.columns(3)
    with c1:
        start_year = st.number_input("Starting year", 2000, 2100, 2026)
        initial_assets_m = st.number_input("Initial assets (£m)", 0.0, value=1_600.0, step=25.0)
        active_members = st.number_input("Current active members", 0, value=8_000, step=100)
        pensioners = st.number_input("Current pensioners", 0, value=4_000, step=100)
        new_entrants = st.number_input("New entrants per year", 0, value=300, step=25)
        entrant_growth = st.number_input("Annual entrant growth (%)", value=0.0, step=0.25)
    with c2:
        entry_age = st.number_input("Entry age", 18, 60, 25)
        retirement_age = st.number_input("Retirement age", 50, 75, 67)
        average_active_age = st.number_input("Average active age", 18, 74, 43)
        average_pensioner_age = st.number_input("Average pensioner age", 50, 100, 72)
        male_share = st.slider("Male share (%)", 0, 100, 55)
        average_service = st.number_input("Average past service (years)", 0.0, 50.0, 14.0)
    with c3:
        average_salary = st.number_input("Average salary (£)", 0.0, value=42_000.0, step=1_000.0)
        average_pension = st.number_input("Average pension in payment (£ pa)", 0.0, value=16_000.0, step=500.0)
        employee_contribution = st.number_input("Employee contribution (%)", 0.0, 50.0, 7.0)
        employer_contribution = st.number_input("Employer contribution (%)", 0.0, 100.0, 18.0)
        benefit_basis = st.selectbox("Benefit design", ["CARE", "Final salary approximation"])
        accrual_denominator = st.number_input("Accrual rate denominator", 20.0, 120.0, 60.0, help="60 means 1/60 of pensionable pay per year.")

    st.subheader("Economic and mortality assumptions")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        salary_growth = st.number_input("Salary growth (%)", value=3.5, step=0.25)
        care_revaluation = st.number_input("CARE revaluation (%)", value=2.5, step=0.25)
    with e2:
        pension_increase = st.number_input("Pension increases (%)", value=2.5, step=0.25)
        discount_rate = st.number_input("Liability discount rate (%)", value=4.5, step=0.25)
    with e3:
        mortality_improvement = st.number_input("Annual mortality improvement (%)", 0.0, 5.0, 1.25, step=0.05)
        mortality_level_volatility = st.number_input("Mortality level uncertainty (%)", 0.0, 30.0, 5.0, step=1.0)
    with e4:
        stochastic_deaths = st.toggle("Random member deaths", value=True)
        st.info("Liabilities are the present value of benefits accrued to date, not future service benefits.")

with portfolio_tab:
    st.subheader("Strategic asset allocation")
    st.write("Edit allocations and capital market assumptions. The portfolio is rebalanced annually.")
    portfolio = st.data_editor(
        DEFAULT_PORTFOLIO,
        hide_index=True,
        use_container_width=True,
        disabled=["Asset class"],
        column_config={
            "Allocation (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
            "Expected return (%)": st.column_config.NumberColumn(step=0.1),
            "Volatility (%)": st.column_config.NumberColumn(min_value=0.0, step=0.5),
        },
        key="portfolio_editor",
    )
    allocation_total = float(portfolio["Allocation (%)"].sum())
    if abs(allocation_total - 100.0) > 0.01:
        st.error(f"Allocations currently total {allocation_total:.1f}%. They must total 100%.")
    else:
        _, p_return, p_vol = portfolio_statistics(portfolio)
        p1, p2 = st.columns(2)
        p1.metric("Expected portfolio return", f"{p_return:.2%}")
        p2.metric("Portfolio volatility", f"{p_vol:.2%}")
        st.plotly_chart(
            px.pie(portfolio, values="Allocation (%)", names="Asset class", hole=0.45),
            use_container_width=True,
        )

config = SchemeConfig(
    projection_years=int(projection_years),
    simulations=int(simulations),
    seed=int(seed),
    start_year=int(start_year),
    initial_assets=float(initial_assets_m) * 1_000_000,
    active_members=int(active_members),
    pensioners=int(pensioners),
    new_entrants=int(new_entrants),
    entrant_growth=float(entrant_growth) / 100,
    male_share=float(male_share) / 100,
    average_active_age=int(average_active_age),
    average_pensioner_age=int(average_pensioner_age),
    entry_age=int(entry_age),
    retirement_age=int(retirement_age),
    average_salary=float(average_salary),
    average_service=float(average_service),
    average_pension=float(average_pension),
    employee_contribution=float(employee_contribution) / 100,
    employer_contribution=float(employer_contribution) / 100,
    accrual_denominator=float(accrual_denominator),
    salary_growth=float(salary_growth) / 100,
    pension_increase=float(pension_increase) / 100,
    care_revaluation=float(care_revaluation) / 100,
    benefit_basis="CARE" if benefit_basis == "CARE" else "Final salary",
    discount_rate=float(discount_rate) / 100,
    mortality_improvement=float(mortality_improvement) / 100,
    mortality_level_volatility=float(mortality_level_volatility) / 100,
    stochastic_deaths=bool(stochastic_deaths),
)

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if run:
    if abs(allocation_total - 100.0) > 0.01:
        st.error("Set portfolio allocations to 100% before running the model.")
    elif entry_age >= retirement_age:
        st.error("Entry age must be below retirement age.")
    else:
        with st.spinner("Simulating scheme membership, liabilities and assets..."):
            st.session_state.model_results = run_model(
                config.__dict__, portfolio.to_json(orient="split")
            )

with results_tab:
    results = st.session_state.model_results
    if results is None:
        st.info("Choose assumptions, then select Run simulation in the sidebar.")
    else:
        summary = results["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Initial funding ratio", f"{summary['initial_funding_ratio']:.1%}")
        m2.metric("Median end funding ratio", f"{summary['end_funding_ratio_median']:.1%}")
        m3.metric("Probability fully funded", f"{summary['fully_funded_probability']:.1%}")
        m4.metric("Probability assets exhausted", f"{summary['asset_exhaustion_probability']:.1%}")

        df = results["percentiles"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Year"], y=df["Assets p95"] / 1e6, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["Year"], y=df["Assets p05"] / 1e6, fill="tonexty", fillcolor="rgba(31,119,180,0.16)", line=dict(width=0), name="Assets 5th–95th percentile"))
        fig.add_trace(go.Scatter(x=df["Year"], y=df["Assets median"] / 1e6, name="Median assets", line=dict(color="#1f77b4", width=3)))
        fig.add_trace(go.Scatter(x=df["Year"], y=df["Liabilities median"] / 1e6, name="Median accrued liabilities", line=dict(color="#d62728", width=3)))
        fig.update_layout(title="Assets and accrued liabilities", yaxis_title="£ million", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            funding = go.Figure()
            funding.add_trace(go.Scatter(x=df["Year"], y=df["Funding ratio p95"], line=dict(width=0), showlegend=False))
            funding.add_trace(go.Scatter(x=df["Year"], y=df["Funding ratio p05"], fill="tonexty", fillcolor="rgba(44,160,44,0.16)", line=dict(width=0), name="5th–95th percentile"))
            funding.add_trace(go.Scatter(x=df["Year"], y=df["Funding ratio median"], name="Median", line=dict(color="#2ca02c", width=3)))
            funding.add_hline(y=1.0, line_dash="dash", annotation_text="100% funded")
            funding.update_layout(title="Funding ratio", yaxis_tickformat=".0%")
            st.plotly_chart(funding, use_container_width=True)
        with right:
            end_ratios = np.clip(results["end_funding_ratios"], 0, 4)
            hist = px.histogram(x=end_ratios, nbins=35, labels={"x": "End funding ratio"}, title="Distribution of end funding ratios")
            hist.update_xaxes(tickformat=".0%")
            st.plotly_chart(hist, use_container_width=True)

        cash = go.Figure()
        cash.add_trace(go.Scatter(x=df["Year"], y=df["Contributions median"] / 1e6, name="Contributions"))
        cash.add_trace(go.Scatter(x=df["Year"], y=df["Benefits median"] / 1e6, name="Pension payments"))
        cash.update_layout(title="Median annual scheme cash flows", yaxis_title="£ million")
        st.plotly_chart(cash, use_container_width=True)

        export = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download percentile results (CSV)", export, "db_pension_alm_results.csv", "text/csv")

with method_tab:
    meta = mortality_metadata()
    st.subheader("How the model works")
    st.markdown(
        """
        1. Active members contribute a percentage of payroll and accrue pension each year.
        2. Deaths are drawn from age and sex specific mortality rates. Surviving members age and retire at the selected retirement age.
        3. Pensioners receive indexed benefits while alive. New entrants replenish the active population.
        4. Accrued liabilities equal the discounted expected value of benefits earned to date, allowing for survival to retirement and pensioner mortality.
        5. Annual portfolio returns follow a correlated lognormal model based on the chosen strategic allocation. Contributions and benefits are assumed to occur evenly through the year.
        """
    )
    st.subheader("Mortality basis")
    st.write(meta)
    st.warning(meta["warning"])
    st.subheader("Material limitations")
    st.markdown(
        """
        - This is an aggregate cohort model, not a member by member actuarial valuation.
        - Spouse pensions, commutation, transfers, expenses, tax, salary dispersion and death benefits are excluded.
        - The final salary option is an approximation using uniform salary growth.
        - Capital market assumptions and correlations are illustrative and should be replaced with an approved basis.
        - Mortality improvement is a simplified constant annual reduction in qx; a production model should use a current CMI projection basis.
        - Results are educational scenario analysis, not financial or actuarial advice.
        """
    )
