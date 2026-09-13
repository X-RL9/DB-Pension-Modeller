from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = 252

DEFAULT_TICKER_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": ["SPY", "AGG", "TIP", "VNQ"],
        "Weight (%)": [40.0, 35.0, 15.0, 10.0],
    }
)


def clean_ticker_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    cleaned = portfolio.copy()
    cleaned["Ticker"] = cleaned["Ticker"].fillna("").astype(str).str.strip().str.upper()
    cleaned["Weight (%)"] = pd.to_numeric(cleaned["Weight (%)"], errors="coerce")
    cleaned = cleaned[(cleaned["Ticker"] != "") & cleaned["Weight (%)"].notna()]
    cleaned = cleaned[cleaned["Weight (%)"] > 0].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("Add at least one ticker with a positive weight.")
    if cleaned["Ticker"].duplicated().any():
        duplicates = ", ".join(cleaned.loc[cleaned["Ticker"].duplicated(), "Ticker"].unique())
        raise ValueError(f"Each ticker should appear once. Duplicate: {duplicates}.")
    if not np.isclose(cleaned["Weight (%)"].sum(), 100.0, atol=0.01):
        raise ValueError(f"Portfolio weights total {cleaned['Weight (%)'].sum():.2f}%, not 100%.")
    return cleaned


def download_adjusted_prices(tickers: list[str], period: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
            multi_level_index=True,
            timeout=20,
        )
    except Exception:
        raw = pd.DataFrame()

    if raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy() if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
    elif raw is not None and not raw.empty and "Close" in raw.columns:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
    else:
        prices = pd.DataFrame()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices.columns = [str(column).upper() for column in prices.columns]

    # If a bulk response is partial, retry the missing tickers individually.
    missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].dropna().empty]
    for ticker in missing:
        try:
            history = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=True,
                timeout=20,
                raise_errors=True,
            )
            if not history.empty and "Close" in history:
                prices[ticker] = history["Close"]
        except Exception:
            continue

    missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].dropna().empty]
    if missing:
        raise ValueError(
            "No usable price history for: "
            + ", ".join(missing)
            + ". Check the symbols or wait a few minutes if Yahoo Finance is rate limiting requests."
        )

    prices = prices[tickers].sort_index().ffill(limit=5).dropna(how="any")
    if len(prices) < TRADING_DAYS:
        raise ValueError("At least one year of overlapping daily prices is required.")
    return prices


def calculate_portfolio_history(
    prices: pd.DataFrame, portfolio: pd.DataFrame
) -> dict[str, object]:
    tickers = portfolio["Ticker"].tolist()
    weights = portfolio["Weight (%)"].to_numpy(dtype=float) / 100.0
    daily_returns = prices[tickers].pct_change(fill_method=None).dropna(how="any")
    portfolio_returns = daily_returns.mul(weights, axis=1).sum(axis=1)
    if len(portfolio_returns) < TRADING_DAYS - 1:
        raise ValueError("Insufficient overlapping return history for this portfolio.")

    log_returns = np.log1p(portfolio_returns.clip(lower=-0.999999))
    annual_return = float(np.expm1(log_returns.mean() * TRADING_DAYS))
    annual_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    asset_statistics = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight (%)": portfolio["Weight (%)"].to_numpy(dtype=float),
            "Annualised return (%)": [
                np.expm1(np.log1p(daily_returns[ticker].clip(lower=-0.999999)).mean() * TRADING_DAYS) * 100
                for ticker in tickers
            ],
            "Annualised volatility (%)": [
                daily_returns[ticker].std(ddof=1) * np.sqrt(TRADING_DAYS) * 100
                for ticker in tickers
            ],
        }
    )
    return {
        "prices": prices,
        "daily_returns": daily_returns,
        "portfolio_returns": portfolio_returns,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "asset_statistics": asset_statistics,
        "correlation": daily_returns.corr(),
    }


def historical_var_es(
    portfolio_returns: pd.Series, confidence: float, horizon_days: int
) -> dict[str, float | int]:
    if horizon_days < 1:
        raise ValueError("The risk horizon must be at least one trading day.")
    horizon_returns = (
        (1.0 + portfolio_returns)
        .rolling(horizon_days)
        .apply(np.prod, raw=True)
        .dropna()
        - 1.0
    )
    if horizon_returns.empty:
        raise ValueError("There is not enough history for the selected risk horizon.")
    losses = -horizon_returns.to_numpy(dtype=float)
    var = float(np.quantile(losses, confidence))
    tail_losses = losses[losses >= var]
    expected_shortfall = float(tail_losses.mean()) if len(tail_losses) else var
    return {
        "var": max(0.0, var),
        "expected_shortfall": max(0.0, expected_shortfall),
        "observations": int(len(horizon_returns)),
    }
