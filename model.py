from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mortality import MortalityBasis


MAX_AGE = 120
SEXES = 2


@dataclass
class SchemeConfig:
    projection_years: int = 40
    simulations: int = 500
    seed: int = 42
    start_year: int = 2026
    initial_assets: float = 1_600_000_000
    active_members: int = 8_000
    pensioners: int = 4_000
    new_entrants: int = 300
    entrant_growth: float = 0.0
    male_share: float = 0.55
    average_active_age: int = 43
    active_age_spread: float = 9.0
    average_pensioner_age: int = 72
    pensioner_age_spread: float = 8.0
    entry_age: int = 25
    retirement_age: int = 67
    average_salary: float = 42_000
    average_service: float = 14.0
    average_pension: float = 16_000
    employee_contribution: float = 0.07
    employer_contribution: float = 0.18
    accrual_denominator: float = 60.0
    salary_growth: float = 0.035
    pension_increase: float = 0.025
    care_revaluation: float = 0.025
    benefit_basis: str = "CARE"
    discount_rate: float = 0.045
    mortality_improvement: float = 0.0125
    mortality_level_volatility: float = 0.05
    stochastic_deaths: bool = True


ASSET_CLASSES = [
    "UK equities",
    "Global equities",
    "Gilts",
    "Index linked gilts",
    "Corporate bonds",
    "Property",
    "Cash",
]

DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Asset class": ASSET_CLASSES,
        "Allocation (%)": [15.0, 25.0, 20.0, 10.0, 15.0, 10.0, 5.0],
        "Expected return (%)": [6.5, 7.0, 3.8, 3.5, 4.5, 5.5, 3.0],
        "Volatility (%)": [18.0, 17.0, 8.0, 7.0, 9.0, 14.0, 1.0],
    }
)

CORRELATION = np.array(
    [
        [1.00, 0.85, -0.10, -0.05, 0.20, 0.45, 0.00],
        [0.85, 1.00, -0.10, -0.05, 0.20, 0.40, 0.00],
        [-0.10, -0.10, 1.00, 0.75, 0.55, 0.05, 0.15],
        [-0.05, -0.05, 0.75, 1.00, 0.45, 0.10, 0.10],
        [0.20, 0.20, 0.55, 0.45, 1.00, 0.25, 0.15],
        [0.45, 0.40, 0.05, 0.10, 0.25, 1.00, 0.05],
        [0.00, 0.00, 0.15, 0.10, 0.15, 0.05, 1.00],
    ]
)


def portfolio_statistics(portfolio: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    weights = portfolio["Allocation (%)"].to_numpy(dtype=float) / 100.0
    means = portfolio["Expected return (%)"].to_numpy(dtype=float) / 100.0
    vols = portfolio["Volatility (%)"].to_numpy(dtype=float) / 100.0
    covariance = np.outer(vols, vols) * CORRELATION
    mean = float(weights @ means)
    volatility = float(np.sqrt(weights @ covariance @ weights))
    return weights, mean, volatility


def _allocate_population(total: int, ages: np.ndarray, centre: float, spread: float) -> np.ndarray:
    weights = np.exp(-0.5 * ((ages - centre) / max(spread, 1.0)) ** 2)
    weights /= weights.sum()
    raw = total * weights
    allocated = np.floor(raw).astype(int)
    remaining = total - allocated.sum()
    if remaining:
        order = np.argsort(raw - allocated)[::-1]
        allocated[order[:remaining]] += 1
    return allocated


def _initial_state(config: SchemeConfig) -> tuple[np.ndarray, ...]:
    active_n = np.zeros((config.simulations, SEXES, MAX_AGE + 1), dtype=float)
    active_salary = np.zeros_like(active_n)
    active_accrued = np.zeros_like(active_n)
    pensioner_n = np.zeros_like(active_n)
    pension_amount = np.zeros_like(active_n)

    active_ages = np.arange(18, config.retirement_age)
    pensioner_ages = np.arange(config.retirement_age, 106)
    sex_shares = [config.male_share, 1.0 - config.male_share]

    for sex, share in enumerate(sex_shares):
        n_active = round(config.active_members * share)
        counts = _allocate_population(
            n_active, active_ages, config.average_active_age, config.active_age_spread
        )
        active_n[:, sex, active_ages] = counts
        salary_by_age = config.average_salary * (
            (1.0 + config.salary_growth) ** (active_ages - config.average_active_age)
        )
        service_by_age = np.maximum(
            0.0, config.average_service + active_ages - config.average_active_age
        )
        active_salary[:, sex, active_ages] = counts * salary_by_age
        active_accrued[:, sex, active_ages] = (
            counts * salary_by_age * service_by_age / config.accrual_denominator
        )

        n_pensioners = round(config.pensioners * share)
        pension_counts = _allocate_population(
            n_pensioners,
            pensioner_ages,
            config.average_pensioner_age,
            config.pensioner_age_spread,
        )
        pensioner_n[:, sex, pensioner_ages] = pension_counts
        pension_amount[:, sex, pensioner_ages] = pension_counts * config.average_pension

    return active_n, active_salary, active_accrued, pensioner_n, pension_amount


def _mortality_vector(
    basis: MortalityBasis,
    sex: int,
    calendar_year: int,
    status: str,
    annual_improvement: float,
) -> np.ndarray:
    return np.array(
        [
            basis.qx(sex, age, calendar_year, status, annual_improvement)
            for age in range(MAX_AGE + 1)
        ]
    )


def _annuity_factor(
    basis: MortalityBasis,
    sex: int,
    age: int,
    calendar_year: int,
    config: SchemeConfig,
) -> float:
    survival = 1.0
    factor = 0.0
    for k in range(1, MAX_AGE - age + 1):
        q = basis.qx(
            sex,
            age + k - 1,
            calendar_year + k - 1,
            "pensioner",
            config.mortality_improvement,
        )
        survival *= 1.0 - q
        factor += (
            survival
            * ((1.0 + config.pension_increase) ** (k - 1))
            / ((1.0 + config.discount_rate) ** k)
        )
    return factor


def _liability_factors(
    basis: MortalityBasis, calendar_year: int, config: SchemeConfig
) -> tuple[np.ndarray, np.ndarray]:
    active_factors = np.zeros((SEXES, MAX_AGE + 1))
    pensioner_factors = np.zeros_like(active_factors)
    for sex in range(SEXES):
        for age in range(MAX_AGE + 1):
            pensioner_factors[sex, age] = _annuity_factor(
                basis, sex, age, calendar_year, config
            )
            if age >= config.retirement_age:
                active_factors[sex, age] = pensioner_factors[sex, age]
                continue
            survival = 1.0
            years_to_retirement = config.retirement_age - age
            for k in range(years_to_retirement):
                q = basis.qx(
                    sex,
                    age + k,
                    calendar_year + k,
                    "active",
                    config.mortality_improvement,
                )
                survival *= 1.0 - q
            active_factors[sex, age] = (
                survival
                * _annuity_factor(
                    basis,
                    sex,
                    config.retirement_age,
                    calendar_year + years_to_retirement,
                    config,
                )
                / ((1.0 + config.discount_rate) ** years_to_retirement)
            )
    return active_factors, pensioner_factors


def _apply_deaths(
    rng: np.random.Generator,
    counts: np.ndarray,
    quantities: list[np.ndarray],
    qx: np.ndarray,
    mortality_scale: np.ndarray,
    stochastic: bool,
) -> None:
    probabilities = np.clip(qx[None, :] * mortality_scale[:, None], 0.0, 1.0)
    old_counts = counts.copy()
    if stochastic:
        deaths = rng.binomial(np.rint(old_counts).astype(int), probabilities)
        survivors = old_counts - deaths
    else:
        survivors = old_counts * (1.0 - probabilities)
    ratio = np.divide(
        survivors,
        old_counts,
        out=np.zeros_like(survivors),
        where=old_counts > 0,
    )
    counts[:] = survivors
    for quantity in quantities:
        quantity *= ratio


def _age_one_year(array: np.ndarray) -> None:
    array[:, :, 1:] = array[:, :, :-1]
    array[:, :, 0] = 0.0


def simulate(
    config: SchemeConfig,
    mortality: MortalityBasis,
    portfolio: pd.DataFrame,
) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    _, portfolio_mean, portfolio_volatility = portfolio_statistics(portfolio)
    active_n, active_salary, active_accrued, pensioner_n, pension_amount = _initial_state(config)

    sims = config.simulations
    periods = config.projection_years + 1
    assets = np.full(sims, config.initial_assets, dtype=float)
    exhausted = np.zeros(sims, dtype=bool)
    mortality_scale = np.exp(
        rng.normal(
            -0.5 * config.mortality_level_volatility**2,
            config.mortality_level_volatility,
            size=sims,
        )
    )

    history = {
        "assets": np.zeros((periods, sims)),
        "liabilities": np.zeros((periods, sims)),
        "funding_ratio": np.zeros((periods, sims)),
        "contributions": np.zeros((periods, sims)),
        "benefits": np.zeros((periods, sims)),
        "active_members": np.zeros((periods, sims)),
        "pensioners": np.zeros((periods, sims)),
    }

    for t in range(periods):
        year = config.start_year + t
        active_factors, pensioner_factors = _liability_factors(mortality, year, config)
        liabilities = (
            (active_accrued * active_factors[None, :, :]).sum(axis=(1, 2))
            + (pension_amount * pensioner_factors[None, :, :]).sum(axis=(1, 2))
        )
        history["assets"][t] = assets
        history["liabilities"][t] = liabilities
        history["funding_ratio"][t] = np.divide(
            assets, liabilities, out=np.full_like(assets, np.nan), where=liabilities > 0
        )
        history["active_members"][t] = active_n.sum(axis=(1, 2))
        history["pensioners"][t] = pensioner_n.sum(axis=(1, 2))

        if t == config.projection_years:
            break

        contributions = active_salary.sum(axis=(1, 2)) * (
            config.employee_contribution + config.employer_contribution
        )
        benefits = pension_amount.sum(axis=(1, 2))
        history["contributions"][t + 1] = contributions
        history["benefits"][t + 1] = benefits

        z = rng.standard_normal(sims)
        log_drift = np.log1p(portfolio_mean) - 0.5 * portfolio_volatility**2
        annual_return = np.exp(log_drift + portfolio_volatility * z) - 1.0
        net_cashflow = contributions - benefits
        assets = assets * (1.0 + annual_return) + net_cashflow * (1.0 + 0.5 * annual_return)
        newly_exhausted = assets <= 0
        exhausted |= newly_exhausted
        assets = np.maximum(assets, 0.0)

        # Revalue accrued benefits and salaries before year-end demographic movements.
        revaluation = (
            config.care_revaluation
            if config.benefit_basis.upper() == "CARE"
            else config.salary_growth
        )
        active_accrued *= 1.0 + revaluation
        active_accrued += active_salary / config.accrual_denominator
        active_salary *= 1.0 + config.salary_growth
        pension_amount *= 1.0 + config.pension_increase

        for sex in range(SEXES):
            active_qx = _mortality_vector(
                mortality, sex, year, "active", config.mortality_improvement
            )
            pensioner_qx = _mortality_vector(
                mortality, sex, year, "pensioner", config.mortality_improvement
            )
            _apply_deaths(
                rng,
                active_n[:, sex, :],
                [active_salary[:, sex, :], active_accrued[:, sex, :]],
                active_qx,
                mortality_scale,
                config.stochastic_deaths,
            )
            _apply_deaths(
                rng,
                pensioner_n[:, sex, :],
                [pension_amount[:, sex, :]],
                pensioner_qx,
                mortality_scale,
                config.stochastic_deaths,
            )

        # Age all cohorts, then retire members who have reached retirement age.
        for state in (active_n, active_salary, active_accrued, pensioner_n, pension_amount):
            _age_one_year(state)

        retirement_slice = slice(config.retirement_age, MAX_AGE + 1)
        pensioner_n[:, :, retirement_slice] += active_n[:, :, retirement_slice]
        pension_amount[:, :, retirement_slice] += active_accrued[:, :, retirement_slice]
        active_n[:, :, retirement_slice] = 0.0
        active_salary[:, :, retirement_slice] = 0.0
        active_accrued[:, :, retirement_slice] = 0.0

        entrant_total = round(config.new_entrants * ((1.0 + config.entrant_growth) ** t))
        entrants_by_sex = [
            round(entrant_total * config.male_share),
            entrant_total - round(entrant_total * config.male_share),
        ]
        for sex, entrants in enumerate(entrants_by_sex):
            age = config.entry_age
            active_n[:, sex, age] += entrants
            active_salary[:, sex, age] += entrants * config.average_salary * (
                (1.0 + config.salary_growth) ** (age - config.average_active_age)
            )

    years = np.arange(config.start_year, config.start_year + periods)
    percentile_rows = []
    for t, year in enumerate(years):
        row = {"Year": int(year)}
        for metric, label in [
            ("assets", "Assets"),
            ("liabilities", "Liabilities"),
            ("funding_ratio", "Funding ratio"),
            ("contributions", "Contributions"),
            ("benefits", "Benefits"),
        ]:
            values = history[metric][t]
            row[f"{label} p05"] = np.nanpercentile(values, 5)
            row[f"{label} median"] = np.nanpercentile(values, 50)
            row[f"{label} p95"] = np.nanpercentile(values, 95)
        row["Active members median"] = np.nanmedian(history["active_members"][t])
        row["Pensioners median"] = np.nanmedian(history["pensioners"][t])
        percentile_rows.append(row)

    end_ratio = history["funding_ratio"][-1]
    summary = {
        "portfolio_expected_return": portfolio_mean,
        "portfolio_volatility": portfolio_volatility,
        "initial_funding_ratio": float(np.nanmedian(history["funding_ratio"][0])),
        "end_funding_ratio_median": float(np.nanmedian(end_ratio)),
        "fully_funded_probability": float(np.mean(end_ratio >= 1.0)),
        "asset_exhaustion_probability": float(np.mean(exhausted)),
        "end_assets_median": float(np.nanmedian(history["assets"][-1])),
        "end_liabilities_median": float(np.nanmedian(history["liabilities"][-1])),
    }
    return {
        "summary": summary,
        "percentiles": pd.DataFrame(percentile_rows),
        "end_funding_ratios": end_ratio,
        "history": history,
        "years": years,
    }
