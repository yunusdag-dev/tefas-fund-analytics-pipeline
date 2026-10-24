"""
Interactive Financial Visualization & Dashboard Builder using Plotly.

Generates an interactive 4-panel HTML dashboard (`reports/tefas_dashboard.html`)
containing:
1. Cumulative Return Curves (%) across TEFAS Funds
2. Risk-Return Scatter (Annualized Volatility vs. Annualized Return, sized by AUM, colored by Sharpe)
3. Historical Drawdown (Underwater) Curves (%)
4. Investor Portfolio Performance by Risk Profile
"""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import BASE_DIR


def build_tefas_dashboard(
    enriched_funds_df: pd.DataFrame,
    risk_metrics_df: pd.DataFrame,
    risk_profile_summary_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Create and export a 4-panel interactive Plotly HTML dashboard for TEFAS analytics.

    Returns
    -------
    Path
        Path to the generated HTML dashboard file.
    """
    if output_path is None:
        reports_dir = BASE_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / "tefas_dashboard.html"

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. TEFAS Fonları Kümülatif Getiri Eğrileri (%)",
            "2. Risk-Getiri Uzayı (Volatilite vs. Yıllık Getiri & Sharpe)",
            "3. Zirveden Düşüş / Drawdown Analizi (%)",
            "4. Yatırımcı Risk Profiline Göre Ort. Portföy & Net Kar (TL)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    palette = [
        "#2563eb", "#16a34a", "#dc2626", "#9333ea",
        "#ea580c", "#0891b2", "#ca8a04", "#4f46e5",
    ]

    # Panel 1 & Panel 3: Cumulative Return and Drawdown per fund
    for idx, (code, group) in enumerate(enriched_funds_df.groupby("fund_code")):
        color = palette[idx % len(palette)]
        group = group.sort_values("date")

        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["cumulative_return"] * 100.0,
                mode="lines",
                name=f"{code}",
                line=dict(color=color, width=2),
                legendgroup=code,
                hovertemplate=f"<b>{code}</b><br>Tarih: %{{x|%Y-%m-%d}}<br>Kümülatif Getiri: %{{y:.2f}}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["drawdown"] * 100.0,
                mode="lines",
                name=f"{code} DD",
                line=dict(color=color, width=1.5),
                legendgroup=code,
                showlegend=False,
                hovertemplate=f"<b>{code}</b><br>Tarih: %{{x|%Y-%m-%d}}<br>Drawdown: %{{y:.2f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Panel 2: Risk-Return Scatter
    fig.add_trace(
        go.Scatter(
            x=risk_metrics_df["annualized_volatility_pct"],
            y=risk_metrics_df["annualized_return_pct"],
            mode="markers+text",
            text=risk_metrics_df["fund_code"],
            textposition="top center",
            showlegend=False,
            marker=dict(
                size=22,
                color=risk_metrics_df["sharpe_ratio"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe", len=0.4, y=0.82),
                line=dict(width=1.5, color="#1e293b"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Yıllık Volatilite: %{x:.2f}%<br>"
                "Yıllık Getiri: %{y:.2f}%<br>"
                "Sharpe Oranı: %{marker.color:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # Panel 4: Investor Risk Profile Grouped Bar Chart
    fig.add_trace(
        go.Bar(
            x=risk_profile_summary_df["risk_profile"],
            y=risk_profile_summary_df["avg_portfolio_try"],
            name="Ort. Portföy Değeri (TL)",
            marker_color="#0ea5e9",
            hovertemplate="Profil: %{x}<br>Ort. Portföy: %{y:,.2f} TL<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=risk_profile_summary_df["risk_profile"],
            y=risk_profile_summary_df["avg_net_pnl_try"],
            name="Ort. Net Kar/Zarar (TL)",
            marker_color="#10b981",
            hovertemplate="Profil: %{x}<br>Ort. Net Kar: %{y:,.2f} TL<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Tarih", row=1, col=1)
    fig.update_yaxes(title_text="Kümülatif Getiri (%)", row=1, col=1)

    fig.update_xaxes(title_text="Yıllıklandırılmış Volatilite (%)", row=1, col=2)
    fig.update_yaxes(title_text="Yıllıklandırılmış Getiri (%)", row=1, col=2)

    fig.update_xaxes(title_text="Tarih", row=2, col=1)
    fig.update_yaxes(title_text="Maksimum Düşüş / Drawdown (%)", row=2, col=1)

    fig.update_xaxes(title_text="Yatırımcı Risk Profili", row=2, col=2)
    fig.update_yaxes(title_text="Tutar (TL)", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>TEFAS Yatırım Fonları Performans, Risk (Sharpe/Drawdown) ve Portföy Analitik Paneli</b>",
            x=0.5,
            font=dict(size=20),
        ),
        template="plotly_white",
        height=920,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path
