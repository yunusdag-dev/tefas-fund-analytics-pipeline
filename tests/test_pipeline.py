"""
Unit and Integration Test Suite for the TEFAS Fund Analytics & Portfolio Pipeline.
"""

import pandas as pd
import pytest

from config.settings import TARGET_FUNDS
from src.extractor import simulate_tefas_fund_history
from src.generator import generate_investors, generate_transactions
from src.loader import get_db_engine, load_tables_to_db, run_sql_analytics
from src.transformer import (
    calculate_fund_risk_metrics,
    calculate_investor_portfolios,
    calculate_rfm_segments,
    clean_and_enrich_funds,
)
from src.visualizer import build_tefas_dashboard, export_all_rfm_figures
from src.pipeline import run_etl_pipeline


def test_simulate_tefas_fund_history_schema_and_values() -> None:
    """Verify simulated TEFAS historical data has valid schema, positive prices, and all funds."""
    df = simulate_tefas_fund_history(n_days=60, seed=42)

    expected_cols = {
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
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == len(TARGET_FUNDS) * 60
    assert (df["price"] > 0).all()
    assert (df["portfolio_value"] > 0).all()
    assert set(df["fund_code"].unique()) == set(TARGET_FUNDS.keys())


def test_generate_investors_and_transactions() -> None:
    """Verify Turkish investor profiles and matched fund transactions."""
    funds_df = simulate_tefas_fund_history(n_days=30, seed=42)
    investors_df = generate_investors(n_investors=50, seed=42)
    tx_df = generate_transactions(
        investors_df=investors_df,
        funds_history_df=funds_df,
        n_transactions=200,
        seed=42,
    )

    assert len(investors_df) == 50
    assert investors_df["investor_id"].is_unique
    assert investors_df["email"].is_unique
    assert set(investors_df["risk_profile"].unique()).issubset({"Muhafazakar", "Dengeli", "Agresif"})
    assert (investors_df["monthly_income_try"] > 0).all()

    assert len(tx_df) == 200
    assert set(tx_df["transaction_type"].unique()).issubset({"BUY", "SELL"})
    assert (tx_df["shares"] > 0).all()
    assert (tx_df["total_amount_try"] > 0).all()


def test_clean_and_enrich_funds_filters_invalid_and_duplicates() -> None:
    """Verify dirty rows (negative prices, duplicates) are removed and drawdown is non-positive."""
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
            "fund_code": ["TI2", "TI2", "TI2", "TI2", "TI2"],
            "fund_name": ["Test Fon"] * 5,
            "founder": ["İş Portföy"] * 5,
            "category": ["Hisse"] * 5,
            "risk_level": [7] * 5,
            "price": [10.0, 10.5, -5.0, 12.0, 9.0],  # duplicate on Jan 1, negative on Jan 2
            "share_count": [1000.0] * 5,
            "investor_count": [100] * 5,
            "portfolio_value": [10000.0] * 5,
            "data_source": ["TEST"] * 5,
        }
    )

    cleaned = clean_and_enrich_funds(raw_df)

    # Duplicate Jan 1 kept last (10.5), negative Jan 2 removed -> 3 valid rows left
    assert len(cleaned) == 3
    assert list(cleaned["price"]) == [10.5, 12.0, 9.0]
    assert (cleaned["drawdown"] <= 0.0).all()
    # On 2026-01-04, peak was 12.0 and price is 9.0 -> drawdown = (9 - 12) / 12 = -0.25
    assert pytest.approx(cleaned["drawdown"].iloc[-1], rel=1e-4) == -0.25


def test_calculate_fund_risk_metrics_monotonic_growth() -> None:
    """Verify Sharpe, Sortino, and zero Max Drawdown on a monotonically growing fund."""
    dates = pd.bdate_range("2026-01-01", periods=50)
    # Alternating +0.25% and +0.45% daily returns -> strictly increasing with positive volatility
    prices = [10.0]
    for i in range(1, 50):
        step_ret = 0.0025 if i % 2 == 0 else 0.0045
        prices.append(prices[-1] * (1.0 + step_ret))
    raw_df = pd.DataFrame(
        {
            "date": dates,
            "fund_code": ["MAC"] * 50,
            "fund_name": ["Marmara Capital Hisse"] * 50,
            "founder": ["Marmara Capital"] * 50,
            "category": ["Hisse"] * 50,
            "risk_level": [7] * 50,
            "price": prices,
            "share_count": [10000.0] * 50,
            "investor_count": [500] * 50,
            "portfolio_value": [p * 10000.0 for p in prices],
            "data_source": ["TEST"] * 50,
        }
    )

    enriched = clean_and_enrich_funds(raw_df)
    metrics = calculate_fund_risk_metrics(enriched, risk_free_rate=0.45)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["max_drawdown_pct"] == 0.0
    assert row["total_return_pct"] > 0.0
    assert row["sharpe_ratio"] > 0.0


def test_calculate_investor_portfolios_pnl_accuracy() -> None:
    """Verify exact PnL and ROI calculation for controlled investor transactions."""
    investors_df = pd.DataFrame(
        [
            {
                "investor_id": "INV-0001",
                "full_name": "Ahmet Yılmaz",
                "email": "ahmet@example.com",
                "city": "İstanbul",
                "age": 35,
                "risk_profile": "Dengeli",
                "monthly_income_try": 50000.0,
                "registration_date": pd.Timestamp("2025-01-01"),
            }
        ]
    )

    # Fund price starts at 10 TL and ends at 15 TL
    raw_funds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "fund_code": ["TI2", "TI2"],
            "fund_name": ["İş Portföy Hisse", "İş Portföy Hisse"],
            "founder": ["İş Portföy", "İş Portföy"],
            "category": ["Hisse", "Hisse"],
            "risk_level": [7, 7],
            "price": [10.0, 15.0],
            "share_count": [10000.0, 10000.0],
            "investor_count": [100, 100],
            "portfolio_value": [100000.0, 150000.0],
            "data_source": ["TEST", "TEST"],
        }
    )
    enriched_funds = clean_and_enrich_funds(raw_funds)

    # Investor buys 100 shares at 10 TL (1000 TL) and sells 20 shares at 15 TL (300 TL)
    # Remaining: 80 shares worth 80 * 15 = 1200 TL.
    # Total PnL = 1200 (current) + 300 (sold) - 1000 (bought) = +500 TL (+50% ROI)
    tx_df = pd.DataFrame(
        [
            {
                "transaction_id": "TX-000001",
                "investor_id": "INV-0001",
                "fund_code": "TI2",
                "transaction_date": pd.Timestamp("2026-01-01"),
                "transaction_type": "BUY",
                "unit_price": 10.0,
                "shares": 100.0,
                "total_amount_try": 1000.0,
                "channel": "Mobil Şube",
            },
            {
                "transaction_id": "TX-000002",
                "investor_id": "INV-0001",
                "fund_code": "TI2",
                "transaction_date": pd.Timestamp("2026-01-02"),
                "transaction_type": "SELL",
                "unit_price": 15.0,
                "shares": 20.0,
                "total_amount_try": 300.0,
                "channel": "Mobil Şube",
            },
        ]
    )

    port_df = calculate_investor_portfolios(investors_df, tx_df, enriched_funds)
    row = port_df.iloc[0]

    assert row["current_portfolio_value_try"] == 1200.0
    assert row["net_pnl_try"] == 500.0
    assert row["roi_pct"] == 50.0


def test_database_loader_and_sql_queries() -> None:
    """Verify SQLAlchemy table persistence and analytical SQL queries on in-memory SQLite."""
    funds_df = simulate_tefas_fund_history(n_days=25, seed=42)
    investors_df = generate_investors(n_investors=30, seed=42)
    tx_df = generate_transactions(investors_df, funds_df, n_transactions=120, seed=42)

    enriched = clean_and_enrich_funds(funds_df)
    metrics = calculate_fund_risk_metrics(enriched)
    portfolios = calculate_investor_portfolios(investors_df, tx_df, enriched)

    engine = get_db_engine("sqlite:///:memory:")
    counts = load_tables_to_db(enriched, metrics, investors_df, tx_df, portfolios, engine)

    assert counts["tefas_daily_prices"] == len(enriched)
    assert counts["tefas_risk_metrics"] == len(metrics)
    assert counts["investors"] == 30
    assert counts["transactions"] == 120
    assert counts["investor_portfolios"] == 30

    reports = run_sql_analytics(engine)
    assert not reports["risk_profile_summary"].empty
    assert not reports["fund_volume_summary"].empty
    assert not reports["city_aum_summary"].empty


def test_plotly_dashboard_generation(tmp_path) -> None:
    """Verify Plotly HTML dashboard exports cleanly and contains chart markup."""
    funds_df = simulate_tefas_fund_history(n_days=20, seed=42)
    investors_df = generate_investors(n_investors=20, seed=42)
    tx_df = generate_transactions(investors_df, funds_df, n_transactions=80, seed=42)

    enriched = clean_and_enrich_funds(funds_df)
    metrics = calculate_fund_risk_metrics(enriched)
    portfolios = calculate_investor_portfolios(investors_df, tx_df, enriched)

    engine = get_db_engine("sqlite:///:memory:")
    load_tables_to_db(enriched, metrics, investors_df, tx_df, portfolios, engine)
    reports = run_sql_analytics(engine)

    out_file = tmp_path / "test_dashboard.html"
    generated_path = build_tefas_dashboard(
        enriched_funds_df=enriched,
        risk_metrics_df=metrics,
        risk_profile_summary_df=reports["risk_profile_summary"],
        output_path=out_file,
    )

    assert generated_path.exists()
    assert generated_path.stat().st_size > 1000


def test_full_etl_orchestrator_end_to_end() -> None:
    """Verify full run_etl_pipeline orchestrator produces DB tables, SQL reports, HTML dashboard, and RFM figures."""
    res = run_etl_pipeline(try_live_api=False)
    assert res["dashboard_path"].exists()
    assert res["rfm_figure_paths"]["treemap"].exists()
    assert res["rfm_figure_paths"]["donut"].exists()
    assert res["rfm_figure_paths"]["scatter"].exists()
    assert res["loaded_counts"]["tefas_daily_prices"] == 2016
    assert res["loaded_counts"]["investor_portfolios"] == 300


def test_rfm_segmentation_and_figures(tmp_path) -> None:
    """Verify RFM score/segment calculation and Plotly Treemap, Donut, and Scatter export."""
    funds_df = simulate_tefas_fund_history(n_days=30, seed=42)
    investors_df = generate_investors(n_investors=40, seed=42)
    tx_df = generate_transactions(investors_df, funds_df, n_transactions=200, seed=42)

    rfm_df = calculate_rfm_segments(investors_df, tx_df)
    assert not rfm_df.empty
    assert {"recency_days", "frequency", "monetary_try", "rfm_score", "rfm_segment"}.issubset(
        set(rfm_df.columns)
    )
    assert (rfm_df["recency_days"] >= 1).all()
    assert (rfm_df["monetary_try"] > 0).all()

    paths = export_all_rfm_figures(rfm_df, output_dir=tmp_path)
    for fig_path in paths.values():
        assert fig_path.exists()
        assert fig_path.stat().st_size > 1000




