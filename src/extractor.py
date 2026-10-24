"""
TEFAS (Türkiye Elektronik Fon Alım Satım Platformu) Data Extractor.

Fetches historical daily fund performance data (price, share count, investor count,
and total portfolio value) from TEFAS API, with an automatic, deterministic
Geometric Brownian Motion (GBM) fallback calibrated to real Turkish funds
so the pipeline never fails due to WAF blocks or offline environments.
"""

from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, Optional
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_HISTORY_DAYS,
    RANDOM_SEED,
    TARGET_FUNDS,
    TEFAS_API_URL,
    TEFAS_HEADERS,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


def fetch_tefas_fund_api(
    fund_code: str,
    start_date: str,
    end_date: str,
    timeout: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Attempt to fetch historical fund data directly from TEFAS public API.

    Parameters
    ----------
    fund_code : str
        3-letter TEFAS fund code (e.g., 'TI2', 'MAC', 'TCD').
    start_date : str
        Start date in 'DD.MM.YYYY' format required by TEFAS.
    end_date : str
        End date in 'DD.MM.YYYY' format required by TEFAS.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with raw TEFAS records if successful, otherwise None.
    """
    payload = urllib.parse.urlencode(
        {
            "fontip": "YAT",
            "sfontur": "",
            "fonkod": fund_code.upper(),
            "fongrup": "",
            "bastarih": start_date,
            "bittarih": end_date,
            "fonturkod": "",
            "fonunvantip": "",
        }
    ).encode("utf-8")

    req = urllib.request.Request(TEFAS_API_URL, data=payload, headers=TEFAS_HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_bytes = response.read()
            parsed = json.loads(raw_bytes.decode("utf-8"))
            records = parsed.get("data", [])
            if not records:
                return None

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["TARIH"].astype(float), unit="ms").dt.normalize()
            df["fund_code"] = df["FONKODU"].astype(str)
            df["fund_name"] = df["FONUNVAN"].astype(str)
            df["price"] = pd.to_numeric(df["FIYAT"], errors="coerce")
            df["share_count"] = pd.to_numeric(df["TEDAVULDEKI_PAY_SAYISI"], errors="coerce")
            df["investor_count"] = pd.to_numeric(df["KISI_SAYISI"], errors="coerce").astype("Int64")
            df["portfolio_value"] = pd.to_numeric(df["PORTFOY_BUYUKLUGU"], errors="coerce")

            meta = TARGET_FUNDS.get(fund_code.upper(), {})
            df["founder"] = meta.get("founder", "Bilinmeyen Portföy")
            df["category"] = meta.get("category", "Yatırım Fonu")
            df["risk_level"] = meta.get("risk_level", 5)
            df["data_source"] = "TEFAS_LIVE_API"

            cols = [
                "date",
                "fund_code",
                "fund_name",
                "founder",
                "category",
                "risk_level",
                "price",
                "share_count",
                "investor_count",
                "portfolio_value",
                "data_source",
            ]
            return df[cols].sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.debug("TEFAS live API unavailable for %s (%s). Using calibrated model.", fund_code, exc)
        return None


def simulate_tefas_fund_history(
    funds_config: Dict[str, Dict[str, Any]] = TARGET_FUNDS,
    n_days: int = DEFAULT_HISTORY_DAYS,
    end_date: Optional[datetime] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate realistic historical TEFAS daily fund data using Geometric Brownian Motion (GBM)
    calibrated to real Turkish investment funds.

    Parameters
    ----------
    funds_config : Dict[str, Dict[str, Any]]
        Dictionary of fund metadata and return/volatility parameters.
    n_days : int
        Number of trading (business) days to generate.
    end_date : Optional[datetime]
        End date for the historical series (defaults to today).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Daily historical TEFAS dataset across all configured funds.
    """
    rng = np.random.default_rng(seed)
    if end_date is None:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    business_dates = pd.bdate_range(end=end_date, periods=n_days)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # Shared market factor to induce realistic cross-fund correlation
    market_shocks = rng.standard_normal(size=n_days)
    frames = []

    for code, meta in funds_config.items():
        mu = float(meta["annual_return_mu"])
        sigma = float(meta["annual_volatility_sigma"])
        p0 = float(meta["initial_price"])
        base_inv = int(meta["base_investors"])

        # Combine idiosyncratic shock with market shock (higher correlation for equity funds)
        market_beta = 0.55 if meta["risk_level"] >= 6 else 0.10
        idio_shocks = rng.standard_normal(size=n_days)
        combined_shocks = market_beta * market_shocks + np.sqrt(1 - market_beta**2) * idio_shocks

        # Daily log returns via GBM
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * combined_shocks
        log_returns[0] = 0.0
        prices = np.round(p0 * np.exp(np.cumsum(log_returns)), 6)

        # Investor count evolves smoothly with fund performance and trend
        investor_drift = np.cumsum(rng.normal(loc=15.0, scale=85.0, size=n_days)).astype(int)
        investor_counts = np.maximum(1000, base_inv + investor_drift)

        # Share count and total AUM (Portfolio Value in TRY)
        avg_shares_per_investor = rng.uniform(18000, 45000)
        share_counts = np.round(investor_counts * avg_shares_per_investor * rng.uniform(0.98, 1.02, size=n_days), 2)
        portfolio_values = np.round(prices * share_counts, 2)

        fund_df = pd.DataFrame(
            {
                "date": business_dates,
                "fund_code": code,
                "fund_name": meta["fund_name"],
                "founder": meta["founder"],
                "category": meta["category"],
                "risk_level": int(meta["risk_level"]),
                "price": prices,
                "share_count": share_counts,
                "investor_count": investor_counts,
                "portfolio_value": portfolio_values,
                "data_source": "TEFAS_CALIBRATED_SIMULATION",
            }
        )
        frames.append(fund_df)

    return pd.concat(frames, ignore_index=True).sort_values(["fund_code", "date"]).reset_index(drop=True)


def extract_tefas_data(
    funds_config: Dict[str, Dict[str, Any]] = TARGET_FUNDS,
    n_days: int = DEFAULT_HISTORY_DAYS,
    try_live_api: bool = True,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Main extraction entry point: attempts live TEFAS API first (if enabled)
    and seamlessly falls back to calibrated historical data per fund.
    """
    end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(n_days * 1.45))

    start_str = start_dt.strftime("%d.%m.%Y")
    end_str = end_dt.strftime("%d.%m.%Y")

    fallback_all = simulate_tefas_fund_history(
        funds_config=funds_config,
        n_days=n_days,
        end_date=end_dt,
        seed=seed,
    )

    if not try_live_api:
        return fallback_all

    collected = []
    for code in funds_config:
        live_df = fetch_tefas_fund_api(code, start_str, end_str)
        if live_df is not None and not live_df.empty:
            collected.append(live_df)
        else:
            collected.append(fallback_all[fallback_all["fund_code"] == code].copy())

    return pd.concat(collected, ignore_index=True).sort_values(["fund_code", "date"]).reset_index(drop=True)
