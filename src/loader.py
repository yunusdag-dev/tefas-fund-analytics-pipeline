"""
Database Loader & SQL Analytics Layer using SQLAlchemy (PostgreSQL / SQLite compatible).

Persists transformed TEFAS fund time-series, risk scorecards, investors,
transactions, and portfolio valuations to a relational database and provides
analytical SQL query helpers.
"""

from typing import Dict
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config.settings import DATABASE_URL


def get_db_engine(db_url: str = DATABASE_URL) -> Engine:
    """
    Create and return a SQLAlchemy Engine instance.
    Supports both PostgreSQL (postgresql+psycopg2://...) and SQLite (sqlite:///...).
    """
    return create_engine(db_url, future=True)


def load_tables_to_db(
    enriched_funds_df: pd.DataFrame,
    risk_metrics_df: pd.DataFrame,
    investors_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    portfolios_df: pd.DataFrame,
    engine: Engine,
) -> Dict[str, int]:
    """
    Write all 5 core tables into the target relational database and create indexes.

    Returns
    -------
    Dict[str, int]
        Dictionary mapping table names to the number of rows loaded.
    """
    tables = {
        "tefas_daily_prices": enriched_funds_df,
        "tefas_risk_metrics": risk_metrics_df,
        "investors": investors_df,
        "transactions": transactions_df,
        "investor_portfolios": portfolios_df,
    }

    row_counts: Dict[str, int] = {}
    with engine.begin() as conn:
        for table_name, df in tables.items():
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists="replace",
                index=False,
                chunksize=500,
                method="multi",
            )
            row_counts[table_name] = len(df)

        # Create analytical indexes for fast SQL querying
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_daily_fund_date "
                "ON tefas_daily_prices (fund_code, date)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_tx_investor_fund "
                "ON transactions (investor_id, fund_code)"
            )
        )

    return row_counts


def run_sql_analytics(engine: Engine) -> Dict[str, pd.DataFrame]:
    """
    Execute analytical SQL queries against the loaded relational tables.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'risk_profile_summary': Investor performance grouped by risk_profile
        - 'fund_volume_summary': Transaction volume grouped by fund_code
        - 'city_aum_summary': Top cities by total portfolio value
    """
    risk_profile_sql = text(
        """
        SELECT
            risk_profile,
            COUNT(investor_id) AS investor_count,
            ROUND(AVG(current_portfolio_value_try), 2) AS avg_portfolio_try,
            ROUND(AVG(net_pnl_try), 2) AS avg_net_pnl_try,
            ROUND(AVG(roi_pct), 2) AS avg_roi_pct
        FROM investor_portfolios
        GROUP BY risk_profile
        ORDER BY avg_portfolio_try DESC
        """
    )

    fund_volume_sql = text(
        """
        SELECT
            t.fund_code,
            m.fund_name,
            m.sharpe_ratio,
            COUNT(t.transaction_id) AS tx_count,
            ROUND(SUM(CASE WHEN t.transaction_type = 'BUY' THEN t.total_amount_try ELSE 0 END), 2) AS total_buy_try,
            ROUND(SUM(CASE WHEN t.transaction_type = 'SELL' THEN t.total_amount_try ELSE 0 END), 2) AS total_sell_try
        FROM transactions t
        LEFT JOIN tefas_risk_metrics m ON t.fund_code = m.fund_code
        GROUP BY t.fund_code, m.fund_name, m.sharpe_ratio
        ORDER BY total_buy_try DESC
        """
    )

    city_aum_sql = text(
        """
        SELECT
            city,
            COUNT(investor_id) AS investor_count,
            ROUND(SUM(current_portfolio_value_try), 2) AS total_aum_try,
            ROUND(AVG(roi_pct), 2) AS avg_roi_pct
        FROM investor_portfolios
        GROUP BY city
        ORDER BY total_aum_try DESC
        LIMIT 10
        """
    )

    with engine.connect() as conn:
        risk_profile_df = pd.read_sql(risk_profile_sql, conn)
        fund_volume_df = pd.read_sql(fund_volume_sql, conn)
        city_aum_df = pd.read_sql(city_aum_sql, conn)

    return {
        "risk_profile_summary": risk_profile_df,
        "fund_volume_summary": fund_volume_df,
        "city_aum_summary": city_aum_df,
    }
