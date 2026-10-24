"""
Central configuration settings for the TEFAS Fund Analytics & Portfolio Pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database configuration:
# Uses PostgreSQL if DATABASE_URL env var is set (e.g. postgresql+psycopg2://user:pass@localhost:5432/tefas_db)
# Otherwise defaults to a local SQLite database for out-of-the-box execution and testing.
DEFAULT_SQLITE_URL = f"sqlite:///{(DATA_DIR / 'tefas_analytics.db').as_posix()}"
DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

# TEFAS Public API Configuration
TEFAS_API_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryInfo"
TEFAS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.tefas.gov.tr/TarihselVeriler.aspx",
    "Origin": "https://www.tefas.gov.tr",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
}

# Target Turkish Investment Funds (TEFAS) with realistic baseline parameters
TARGET_FUNDS: Dict[str, Dict[str, Any]] = {
    "TI2": {
        "fund_name": "İş Portföy BIST 100 Dışı Şirketler Hisse Senedi Fonu",
        "founder": "İş Portföy",
        "category": "Hisse Senedi Şemsiye Fonu",
        "risk_level": 7,
        "initial_price": 12.40,
        "annual_return_mu": 0.58,
        "annual_volatility_sigma": 0.34,
        "base_investors": 48500,
    },
    "MAC": {
        "fund_name": "Marmara Capital Portföy Hisse Senedi Fonu",
        "founder": "Marmara Capital Portföy",
        "category": "Hisse Senedi Şemsiye Fonu",
        "risk_level": 7,
        "initial_price": 3.85,
        "annual_return_mu": 0.62,
        "annual_volatility_sigma": 0.32,
        "base_investors": 39200,
    },
    "TCD": {
        "fund_name": "Tacirler Portföy Değişken Fon",
        "founder": "Tacirler Portföy",
        "category": "Değişken Şemsiye Fonu",
        "risk_level": 6,
        "initial_price": 28.50,
        "annual_return_mu": 0.52,
        "annual_volatility_sigma": 0.28,
        "base_investors": 31000,
    },
    "IPB": {
        "fund_name": "İstanbul Portföy Birinci Değişken Fon",
        "founder": "İstanbul Portföy",
        "category": "Değişken Şemsiye Fonu",
        "risk_level": 6,
        "initial_price": 1.95,
        "annual_return_mu": 0.50,
        "annual_volatility_sigma": 0.25,
        "base_investors": 27400,
    },
    "AFT": {
        "fund_name": "Ak Portföy Yeni Teknolojiler Yabancı Hisse Senedi Fonu",
        "founder": "Ak Portföy",
        "category": "Hisse Senedi Şemsiye Fonu (Yabancı)",
        "risk_level": 6,
        "initial_price": 0.68,
        "annual_return_mu": 0.49,
        "annual_volatility_sigma": 0.27,
        "base_investors": 112000,
    },
    "YAY": {
        "fund_name": "Yapı Kredi Portföy Yabancı Teknoloji Sektörü Hisse Senedi Fonu",
        "founder": "Yapı Kredi Portföy",
        "category": "Hisse Senedi Şemsiye Fonu (Yabancı)",
        "risk_level": 6,
        "initial_price": 845.00,
        "annual_return_mu": 0.51,
        "annual_volatility_sigma": 0.26,
        "base_investors": 64000,
    },
    "TTA": {
        "fund_name": "İş Portföy Altın Fonu",
        "founder": "İş Portföy",
        "category": "Kıymetli Madenler Şemsiye Fonu",
        "risk_level": 5,
        "initial_price": 0.42,
        "annual_return_mu": 0.46,
        "annual_volatility_sigma": 0.18,
        "base_investors": 95000,
    },
    "TI1": {
        "fund_name": "İş Portföy Para Piyasası (TL) Fonu",
        "founder": "İş Portföy",
        "category": "Para Piyasası Şemsiye Fonu",
        "risk_level": 1,
        "initial_price": 4.10,
        "annual_return_mu": 0.47,
        "annual_volatility_sigma": 0.015,
        "base_investors": 145000,
    },
}

# Financial & Simulation Parameters
ANNUAL_RISK_FREE_RATE: float = 0.45  # 45% annual benchmark risk-free rate (TL Deposit / Policy Rate proxy)
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_HISTORY_DAYS: int = 252      # 1 year of business days
DEFAULT_NUM_INVESTORS: int = 300
DEFAULT_NUM_TRANSACTIONS: int = 3000
RANDOM_SEED: int = 42
