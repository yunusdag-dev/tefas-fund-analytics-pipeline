"""
Financial Transformation, Data Cleaning & Risk Analytics Engine for TEFAS Funds.

Transforms raw TEFAS daily series and investor transactions into:
1. Enriched daily fund time-series (daily return, cumulative return, 21d volatility, drawdown)
2. Fund Risk & Performance Scorecard (CAGR, Annualized Volatility, Sharpe, Sortino, Max Drawdown)
3. Investor Portfolio Valuation & PnL Summary
"""

import numpy as np
import pandas as pd

from config.settings import ANNUAL_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def clean_and_enrich_funds(funds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, deduplicate, and enrich daily TEFAS fund data with time-series financial metrics.

    Parameters
    ----------
    funds_df : pd.DataFrame
        Raw daily TEFAS fund data.

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched daily fund DataFrame including daily_return,
        cumulative_return, rolling_vol_21d, and drawdown.
    """
    if funds_df.empty:
        raise ValueError("funds_df cannot be empty.")

    df = funds_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Filter out invalid non-positive prices and deduplicate (fund_code, date)
    df = df[df["price"] > 0].drop_duplicates(subset=["fund_code", "date"], keep="last")
    df = df.sort_values(["fund_code", "date"]).reset_index(drop=True)

    # Compute time-series metrics per fund
    df["daily_return"] = df.groupby("fund_code")["price"].pct_change().fillna(0.0)

    df["cumulative_return"] = df.groupby("fund_code")["price"].transform(
        lambda s: (s / s.iloc[0]) - 1.0
    )

    df["rolling_vol_21d"] = df.groupby("fund_code")["daily_return"].transform(
        lambda s: s.rolling(window=21, min_periods=5).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).fillna(0.0)

    df["running_peak"] = df.groupby("fund_code")["price"].cummax()
    df["drawdown"] = (df["price"] - df["running_peak"]) / df["running_peak"]
    df = df.drop(columns=["running_peak"])

    return df


def calculate_fund_risk_metrics(
    enriched_funds_df: pd.DataFrame,
    risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Calculate institutional performance & risk ratios for each TEFAS fund:
    Total Return, Annualized Return, Annualized Volatility, Sharpe Ratio,
    Sortino Ratio, and Maximum Drawdown.

    Parameters
    ----------
    enriched_funds_df : pd.DataFrame
        Cleaned daily fund DataFrame produced by `clean_and_enrich_funds`.
    risk_free_rate : float
        Annual risk-free benchmark rate (e.g. 0.45 for 45% TL risk-free proxy).

    Returns
    -------
    pd.DataFrame
        Summary scorecard table with one row per fund_code.
    """
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    summaries = []

    for code, group in enriched_funds_df.groupby("fund_code"):
        group = group.sort_values("date")
        n_days = len(group)
        start_price = float(group["price"].iloc[0])
        end_price = float(group["price"].iloc[-1])

        total_return = (end_price / start_price) - 1.0
        years = max(n_days / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0

        # Exclude day 0 (where return is 0 by definition) if multiple days exist
        rets = group["daily_return"].iloc[1:] if n_days > 1 else group["daily_return"]
        daily_std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        annualized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe Ratio: (Annualized Return - Annual Risk-Free Rate) / Annualized Volatility
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / annualized_vol
            if annualized_vol > 1e-8
            else 0.0
        )

        # Sortino Ratio: uses downside deviation below daily risk-free return
        downside_diff = np.minimum(0.0, rets - daily_rf)
        downside_std = float(np.sqrt(np.mean(np.square(downside_diff)))) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino_ratio = (
            (annualized_return - risk_free_rate) / downside_std
            if downside_std > 1e-8
            else 0.0
        )

        max_drawdown = float(group["drawdown"].min())

        summaries.append(
            {
                "fund_code": code,
                "fund_name": str(group["fund_name"].iloc[-1]),
                "founder": str(group["founder"].iloc[-1]),
                "category": str(group["category"].iloc[-1]),
                "risk_level": int(group["risk_level"].iloc[-1]),
                "start_price": round(start_price, 6),
                "end_price": round(end_price, 6),
                "total_return_pct": round(total_return * 100.0, 2),
                "annualized_return_pct": round(annualized_return * 100.0, 2),
                "annualized_volatility_pct": round(annualized_vol * 100.0, 2),
                "sharpe_ratio": round(sharpe_ratio, 3),
                "sortino_ratio": round(sortino_ratio, 3),
                "max_drawdown_pct": round(max_drawdown * 100.0, 2),
                "latest_investor_count": int(group["investor_count"].iloc[-1]),
                "latest_portfolio_value_try": round(float(group["portfolio_value"].iloc[-1]), 2),
            }
        )

    return pd.DataFrame(summaries).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def calculate_investor_portfolios(
    investors_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    enriched_funds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate BUY and SELL transactions per investor & fund, mark holdings to the latest
    TEFAS fund price, and compute investor-level portfolio value, PnL, and ROI (%).

    Parameters
    ----------
    investors_df : pd.DataFrame
        Investor master table.
    transactions_df : pd.DataFrame
        Investor buy/sell transactions table.
    enriched_funds_df : pd.DataFrame
        Cleaned daily TEFAS fund data.

    Returns
    -------
    pd.DataFrame
        Investor portfolio summary with net invested capital, current market value,
        realized/unrealized PnL, and ROI percentage.
    """
    tx = transactions_df.copy()
    tx["signed_shares"] = np.where(tx["transaction_type"] == "BUY", tx["shares"], -tx["shares"])
    tx["buy_amount"] = np.where(tx["transaction_type"] == "BUY", tx["total_amount_try"], 0.0)
    tx["sell_amount"] = np.where(tx["transaction_type"] == "SELL", tx["total_amount_try"], 0.0)

    latest_prices = (
        enriched_funds_df.sort_values("date")
        .groupby("fund_code")["price"]
        .last()
        .to_dict()
    )

    pos = (
        tx.groupby(["investor_id", "fund_code"], as_index=False)
        .agg(
            raw_net_shares=("signed_shares", "sum"),
            total_buy_try=("buy_amount", "sum"),
            total_sell_try=("sell_amount", "sum"),
            tx_count=("transaction_id", "count"),
        )
    )

    # Floor net shares at 0 (no short-selling in TEFAS retail funds)
    pos["net_shares"] = np.maximum(0.0, pos["raw_net_shares"])
    pos["latest_price"] = pos["fund_code"].map(latest_prices).fillna(0.0)
    pos["current_position_value_try"] = np.round(pos["net_shares"] * pos["latest_price"], 2)

    inv_agg = (
        pos.groupby("investor_id", as_index=False)
        .agg(
            distinct_funds_count=("fund_code", "nunique"),
            total_transactions=("tx_count", "sum"),
            total_buy_try=("total_buy_try", "sum"),
            total_sell_try=("total_sell_try", "sum"),
            current_portfolio_value_try=("current_position_value_try", "sum"),
        )
    )

    portfolio_df = investors_df.merge(inv_agg, on="investor_id", how="left").fillna(
        {
            "distinct_funds_count": 0,
            "total_transactions": 0,
            "total_buy_try": 0.0,
            "total_sell_try": 0.0,
            "current_portfolio_value_try": 0.0,
        }
    )

    portfolio_df["distinct_funds_count"] = portfolio_df["distinct_funds_count"].astype(int)
    portfolio_df["total_transactions"] = portfolio_df["total_transactions"].astype(int)
    portfolio_df["net_pnl_try"] = np.round(
        portfolio_df["current_portfolio_value_try"]
        + portfolio_df["total_sell_try"]
        - portfolio_df["total_buy_try"],
        2,
    )
    portfolio_df["roi_pct"] = np.where(
        portfolio_df["total_buy_try"] > 0,
        np.round((portfolio_df["net_pnl_try"] / portfolio_df["total_buy_try"]) * 100.0, 2),
        0.0,
    )

    return portfolio_df.sort_values("current_portfolio_value_try", ascending=False).reset_index(drop=True)


def calculate_rfm_segments(
    investors_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) scores (1-5) and assign behavioral
    RFM segments for each investor based on transaction history.

    Parameters
    ----------
    investors_df : pd.DataFrame
        Investor master table.
    transactions_df : pd.DataFrame
        Transactions table containing transaction_date, total_amount_try, transaction_id.
    reference_date : pd.Timestamp | None
        Snapshot date for Recency calculation (defaults to max transaction_date + 1 day).

    Returns
    -------
    pd.DataFrame
        Investor RFM table with recency_days, frequency, monetary_try,
        r_score, f_score, m_score, rfm_score, and rfm_segment.
    """
    tx = transactions_df.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"]).dt.normalize()

    if reference_date is None:
        reference_date = tx["transaction_date"].max() + pd.Timedelta(days=1)

    rfm = (
        tx.groupby("investor_id", as_index=False)
        .agg(
            last_tx_date=("transaction_date", "max"),
            frequency=("transaction_id", "count"),
            monetary_try=("total_amount_try", "sum"),
        )
    )
    rfm["recency_days"] = (reference_date - rfm["last_tx_date"]).dt.days
    rfm["monetary_try"] = np.round(rfm["monetary_try"], 2)

    # Merge with investors so every investor with transactions has demographic metadata
    merged = investors_df[
        ["investor_id", "full_name", "city", "risk_profile"]
    ].merge(rfm, on="investor_id", how="inner")

    # Compute quintile scores (1 to 5) using rank to avoid duplicate bin edge errors
    # Lower recency_days -> higher r_score (5 is most recent)
    merged["r_score"] = pd.qcut(
        merged["recency_days"].rank(method="first", ascending=False),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    merged["f_score"] = pd.qcut(
        merged["frequency"].rank(method="first", ascending=True),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    merged["m_score"] = pd.qcut(
        merged["monetary_try"].rank(method="first", ascending=True),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    merged["rfm_score"] = (
        merged["r_score"].astype(str)
        + merged["f_score"].astype(str)
        + merged["m_score"].astype(str)
    )

    def _assign_segment(row: pd.Series) -> str:
        r, f, m = int(row["r_score"]), int(row["f_score"]), int(row["m_score"])
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions (Şampiyonlar)"
        if r >= 3 and f >= 4:
            return "Loyal Investors (Sadık Yatırımcılar)"
        if m >= 4 and r >= 3:
            return "Big Spenders (Yüksek Hacimliler)"
        if r >= 4 and f <= 3:
            return "Recent / Potential (Yeni & Potansiyel)"
        if r == 3 and f <= 3:
            return "Need Attention (İlgi Bekleyenler)"
        if r <= 2 and (f >= 3 or m >= 3):
            return "At Risk (Risk Altındakiler)"
        return "Hibernating (Uykuda Olanlar)"

    merged["rfm_segment"] = merged.apply(_assign_segment, axis=1)
    return merged.sort_values("monetary_try", ascending=False).reset_index(drop=True)

