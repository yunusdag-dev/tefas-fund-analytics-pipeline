"""
Synthetic Turkish Investor & TEFAS Portfolio Transaction Generator using Faker.

Generates realistic retail/institutional Turkish investor profiles and matches
their buy/sell fund transactions directly against the historical TEFAS fund prices.
"""

from typing import Tuple
from faker import Faker
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_NUM_INVESTORS,
    DEFAULT_NUM_TRANSACTIONS,
    RANDOM_SEED,
)


def generate_investors(
    n_investors: int = DEFAULT_NUM_INVESTORS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic Turkish investor demographic and risk-profile records.

    Parameters
    ----------
    n_investors : int
        Number of unique investors to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: investor_id, full_name, email, city, age,
        risk_profile, monthly_income_try, registration_date.
    """
    fake = Faker("tr_TR")
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    turkish_cities = [
        "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya",
        "Kocaeli", "Adana", "Konya", "Gaziantep", "Mersin",
        "Eskişehir", "Kayseri", "Samsun", "Denizli", "Muğla",
    ]
    city_weights = [
        0.34, 0.16, 0.12, 0.06, 0.05,
        0.04, 0.03, 0.03, 0.03, 0.03,
        0.03, 0.02, 0.02, 0.02, 0.02,
    ]
    risk_profiles = ["Muhafazakar", "Dengeli", "Agresif"]
    risk_weights = [0.30, 0.45, 0.25]

    records = []
    for idx in range(1, n_investors + 1):
        profile = str(rng.choice(risk_profiles, p=risk_weights))
        city = str(rng.choice(turkish_cities, p=city_weights))
        age = int(rng.integers(22, 68))

        base_income = 45000 if profile == "Muhafazakar" else (65000 if profile == "Dengeli" else 85000)
        income = float(np.round(rng.lognormal(mean=np.log(base_income), sigma=0.45), -2))

        records.append(
            {
                "investor_id": f"INV-{idx:04d}",
                "full_name": fake.name(),
                "email": fake.unique.email(),
                "city": city,
                "age": age,
                "risk_profile": profile,
                "monthly_income_try": income,
                "registration_date": pd.Timestamp(fake.date_between(start_date="-3y", end_date="-1y")),
            }
        )

    return pd.DataFrame(records)


def generate_transactions(
    investors_df: pd.DataFrame,
    funds_history_df: pd.DataFrame,
    n_transactions: int = DEFAULT_NUM_TRANSACTIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate realistic TEFAS fund BUY/SELL transactions matched to exact daily fund prices
    and investor risk profiles.

    Parameters
    ----------
    investors_df : pd.DataFrame
        Generated investors table.
    funds_history_df : pd.DataFrame
        Historical daily TEFAS fund prices table.
    n_transactions : int
        Total number of transactions to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Transactions table with transaction_id, investor_id, fund_code, transaction_date,
        transaction_type, unit_price, shares, total_amount_try, channel.
    """
    rng = np.random.default_rng(seed)

    # Group available price lookups by fund_code
    fund_codes = sorted(funds_history_df["fund_code"].unique())
    fund_risk_map = (
        funds_history_df.groupby("fund_code")["risk_level"].first().to_dict()
    )
    fund_lookup = {
        code: group[["date", "price"]].sort_values("date").reset_index(drop=True)
        for code, group in funds_history_df.groupby("fund_code")
    }

    channels = ["Mobil Şube", "İnternet Bankacılığı", "Yatırım Temsilcisi", "Otomatik Talimat"]
    channel_weights = [0.58, 0.27, 0.05, 0.10]

    investor_rows = investors_df.to_dict("records")
    tx_records = []

    for tx_idx in range(1, n_transactions + 1):
        inv = investor_rows[int(rng.integers(0, len(investor_rows)))]
        profile = inv["risk_profile"]

        # Weight fund choices according to investor's risk profile
        weights = []
        for code in fund_codes:
            r_level = fund_risk_map.get(code, 5)
            if profile == "Muhafazakar":
                w = 4.0 if r_level <= 5 else 1.0
            elif profile == "Agresif":
                w = 4.0 if r_level >= 6 else 0.8
            else:
                w = 2.5 if 4 <= r_level <= 6 else 1.5
            weights.append(w)
        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()

        chosen_fund = str(rng.choice(fund_codes, p=weights_arr))
        price_history = fund_lookup[chosen_fund]
        row_idx = int(rng.integers(0, len(price_history)))
        tx_date = price_history.loc[row_idx, "date"]
        unit_price = float(price_history.loc[row_idx, "price"])

        # 78% BUY, 22% SELL to reflect net portfolio accumulation
        tx_type = str(rng.choice(["BUY", "SELL"], p=[0.78, 0.22]))

        # Transaction amount scaled by investor's monthly income
        allocation_ratio = rng.uniform(0.08, 0.65)
        target_amount = max(1000.0, float(inv["monthly_income_try"]) * allocation_ratio)
        shares = np.round(target_amount / unit_price, 2)
        total_amount = np.round(shares * unit_price, 2)

        tx_records.append(
            {
                "transaction_id": f"TX-{tx_idx:06d}",
                "investor_id": inv["investor_id"],
                "fund_code": chosen_fund,
                "transaction_date": tx_date,
                "transaction_type": tx_type,
                "unit_price": unit_price,
                "shares": shares,
                "total_amount_try": total_amount,
                "channel": str(rng.choice(channels, p=channel_weights)),
            }
        )

    tx_df = pd.DataFrame(tx_records).sort_values("transaction_date").reset_index(drop=True)
    return tx_df


def build_raw_datasets(
    n_days: int = 252,
    n_investors: int = DEFAULT_NUM_INVESTORS,
    n_transactions: int = DEFAULT_NUM_TRANSACTIONS,
    try_live_api: bool = True,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience builder that extracts TEFAS fund history and generates
    matching investor and transaction tables.
    """
    from src.extractor import extract_tefas_data

    funds_df = extract_tefas_data(n_days=n_days, try_live_api=try_live_api, seed=seed)
    investors_df = generate_investors(n_investors=n_investors, seed=seed)
    transactions_df = generate_transactions(
        investors_df=investors_df,
        funds_history_df=funds_df,
        n_transactions=n_transactions,
        seed=seed,
    )
    return funds_df, investors_df, transactions_df
