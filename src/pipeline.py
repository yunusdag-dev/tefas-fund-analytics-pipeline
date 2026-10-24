"""
End-to-End TEFAS Analytics & Portfolio Simulation ETL Orchestrator.

Runs:
1. Extract & Generate (TEFAS daily fund history + Turkish investors & transactions)
2. Transform (Data cleaning, time-series enrichment, Sharpe/Sortino/Drawdown, Investor PnL)
3. Load (SQLAlchemy relational persistence + CSV snapshots)
4. Verify (Analytical SQL queries summary)
"""

import logging
from typing import Dict, Any

from config.settings import (
    DATA_DIR,
    DATABASE_URL,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_NUM_INVESTORS,
    DEFAULT_NUM_TRANSACTIONS,
    RANDOM_SEED,
)
from src.generator import build_raw_datasets
from src.loader import get_db_engine, load_tables_to_db, run_sql_analytics
from src.transformer import (
    calculate_fund_risk_metrics,
    calculate_investor_portfolios,
    clean_and_enrich_funds,
)
from src.visualizer import build_tefas_dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("tefas_pipeline")


def run_etl_pipeline(
    db_url: str = DATABASE_URL,
    n_days: int = DEFAULT_HISTORY_DAYS,
    n_investors: int = DEFAULT_NUM_INVESTORS,
    n_transactions: int = DEFAULT_NUM_TRANSACTIONS,
    try_live_api: bool = True,
    save_csv: bool = True,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Execute the complete TEFAS ETL pipeline from extraction to SQL analytics.
    """
    logger.info("1/4 [EXTRACT] TEFAS fon verileri ve yatırımcı işlemleri hazırlanıyor...")
    raw_funds_df, investors_df, transactions_df = build_raw_datasets(
        n_days=n_days,
        n_investors=n_investors,
        n_transactions=n_transactions,
        try_live_api=try_live_api,
        seed=seed,
    )

    logger.info("2/4 [TRANSFORM] Finansal risk metrikleri (Sharpe, Sortino, Drawdown) ve PnL hesaplanıyor...")
    enriched_funds_df = clean_and_enrich_funds(raw_funds_df)
    risk_metrics_df = calculate_fund_risk_metrics(enriched_funds_df)
    portfolios_df = calculate_investor_portfolios(investors_df, transactions_df, enriched_funds_df)

    if save_csv:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        enriched_funds_df.to_csv(DATA_DIR / "tefas_daily_prices.csv", index=False)
        risk_metrics_df.to_csv(DATA_DIR / "tefas_risk_metrics.csv", index=False)
        investors_df.to_csv(DATA_DIR / "investors.csv", index=False)
        transactions_df.to_csv(DATA_DIR / "transactions.csv", index=False)
        portfolios_df.to_csv(DATA_DIR / "investor_portfolios.csv", index=False)
        logger.info("CSV veri setleri '%s' dizinine kaydedildi.", DATA_DIR)

    logger.info("3/4 [LOAD] Tablolar SQLAlchemy ile veritabanına (%s) yazılıyor...", db_url)
    engine = get_db_engine(db_url)
    loaded_counts = load_tables_to_db(
        enriched_funds_df=enriched_funds_df,
        risk_metrics_df=risk_metrics_df,
        investors_df=investors_df,
        transactions_df=transactions_df,
        portfolios_df=portfolios_df,
        engine=engine,
    )
    logger.info("Veritabanına yüklenen satır sayıları: %s", loaded_counts)

    logger.info("4/4 [SQL ANALYTICS & DASHBOARD] Analitik SQL sorguları ve Plotly Dashboard üretiliyor...")
    sql_reports = run_sql_analytics(engine)
    dashboard_path = build_tefas_dashboard(
        enriched_funds_df=enriched_funds_df,
        risk_metrics_df=risk_metrics_df,
        risk_profile_summary_df=sql_reports["risk_profile_summary"],
    )
    logger.info("İnteraktif Plotly Dashboard hazır: %s", dashboard_path)

    return {
        "enriched_funds": enriched_funds_df,
        "risk_metrics": risk_metrics_df,
        "investors": investors_df,
        "transactions": transactions_df,
        "portfolios": portfolios_df,
        "loaded_counts": loaded_counts,
        "sql_reports": sql_reports,
        "dashboard_path": dashboard_path,
    }


if __name__ == "__main__":
    results = run_etl_pipeline(try_live_api=True)
    print("\n=== TEFAS FON RİSK & PERFORMANS KARNESİ ===")
    print(
        results["risk_metrics"][
            [
                "fund_code",
                "total_return_pct",
                "annualized_volatility_pct",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown_pct",
            ]
        ].to_string(index=False)
    )
    print("\n=== RİSK PROFİLİNE GÖRE YATIRIMCI PERFORMANSI (SQL) ===")
    print(results["sql_reports"]["risk_profile_summary"].to_string(index=False))
    print("\n=== EN YÜKSEK PORTFÖY HACMİNE SAHİP İLK 5 ŞEHİR (SQL) ===")
    print(results["sql_reports"]["city_aum_summary"].head(5).to_string(index=False))
