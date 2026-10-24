"""
Interactive Financial & RFM Visualization Suite using Plotly.

Generates:
1. RFM Segment Distribution Charts:
   - Interactive Treemap (`reports/figures/rfm_segments_treemap.html`)
   - Interactive Donut Chart (`reports/figures/rfm_segments_donut.html`)
2. Monetary (Harcama/Yatırım) vs. Recency (Yenilik) Scatter Plot colored by RFM Segment:
   - (`reports/figures/rfm_monetary_vs_recency_scatter.html`)
3. 4-Panel TEFAS Executive Dashboard (`reports/tefas_dashboard.html`)
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import BASE_DIR

FIGURES_DIR = BASE_DIR / "reports" / "figures"


def create_rfm_treemap(
    rfm_df: pd.DataFrame,
    output_dir: Path | None = FIGURES_DIR,
) -> go.Figure:
    """
    Build an interactive Plotly Treemap showing RFM Segment distribution
    (sized by investor count and colored by average Monetary value).
    Saves the chart to `reports/figures/rfm_segments_treemap.html` if output_dir is provided.
    """
    seg_summary = (
        rfm_df.groupby(["rfm_segment", "risk_profile"], as_index=False)
        .agg(
            investor_count=("investor_id", "count"),
            total_monetary_try=("monetary_try", "sum"),
            avg_monetary_try=("monetary_try", "mean"),
            avg_recency_days=("recency_days", "mean"),
        )
    )
    seg_summary["avg_monetary_try"] = seg_summary["avg_monetary_try"].round(2)
    seg_summary["avg_recency_days"] = seg_summary["avg_recency_days"].round(1)

    fig = px.treemap(
        seg_summary,
        path=[px.Constant("Tüm Yatırımcılar"), "rfm_segment", "risk_profile"],
        values="investor_count",
        color="avg_monetary_try",
        color_continuous_scale="Tealgrn",
        custom_data=["total_monetary_try", "avg_monetary_try", "avg_recency_days"],
        title="<b>RFM Segmentlerinin ve Risk Profillerinin Dağılımı (Treemap)</b>",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Yatırımcı Sayısı: %{value}<br>"
            "Toplam Hacim (Monetary): %{customdata[0]:,.2f} TL<br>"
            "Ortalama Hacim: %{customdata[1]:,.2f} TL<br>"
            "Ortalama Recency: %{customdata[2]:.1f} gün<extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=60, l=25, r=25, b=25),
        coloraxis_colorbar=dict(title="Ort. Monetary (TL)"),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_dir / "rfm_segments_treemap.html"), include_plotlyjs="cdn")

    return fig


def create_rfm_donut_chart(
    rfm_df: pd.DataFrame,
    output_dir: Path | None = FIGURES_DIR,
) -> go.Figure:
    """
    Build an interactive Plotly Donut Chart showing RFM Segment distribution.
    Saves the chart to `reports/figures/rfm_segments_donut.html` if output_dir is provided.
    """
    seg_counts = (
        rfm_df.groupby("rfm_segment", as_index=False)
        .agg(
            investor_count=("investor_id", "count"),
            total_monetary_try=("monetary_try", "sum"),
        )
        .sort_values("investor_count", ascending=False)
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=seg_counts["rfm_segment"],
                values=seg_counts["investor_count"],
                hole=0.48,
                textinfo="label+percent",
                customdata=seg_counts["total_monetary_try"],
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Yatırımcı Sayısı: %{value} (%{percent})<br>"
                    "Toplam Hacim (Monetary): %{customdata:,.2f} TL<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text="<b>RFM Segmentlerinin Yatırımcı Dağılımı (Donut Chart)</b>",
            x=0.5,
        ),
        template="plotly_white",
        height=550,
        annotations=[
            dict(
                text=f"<b>Toplam<br>{len(rfm_df)}<br>Yatırımcı</b>",
                x=0.5,
                y=0.5,
                font_size=15,
                showarrow=False,
            )
        ],
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_dir / "rfm_segments_donut.html"), include_plotlyjs="cdn")

    return fig


def create_rfm_scatter_plot(
    rfm_df: pd.DataFrame,
    output_dir: Path | None = FIGURES_DIR,
) -> go.Figure:
    """
    Build an interactive Plotly Scatter Plot of Monetary (Harcama/Yatırım Hacmi)
    vs. Recency (Yenilik - Gün), colored by RFM Segment and sized by Frequency.
    Saves the chart to `reports/figures/rfm_monetary_vs_recency_scatter.html` if output_dir is provided.
    """
    fig = px.scatter(
        rfm_df,
        x="recency_days",
        y="monetary_try",
        color="rfm_segment",
        size="frequency",
        hover_name="full_name",
        hover_data={
            "investor_id": True,
            "city": True,
            "risk_profile": True,
            "recency_days": True,
            "frequency": True,
            "monetary_try": ":,.2f",
            "rfm_score": True,
        },
        labels={
            "recency_days": "Recency (Son İşlemden Bu Yana Geçen Gün)",
            "monetary_try": "Monetary — Toplam Harcama / İşlem Hacmi (TL)",
            "rfm_segment": "RFM Segmenti",
            "frequency": "Frequency (İşlem Sayısı)",
        },
        title="<b>Monetary (Harcama) vs. Recency (Yenilik) Dağılımı — RFM Segmentlerine Göre</b>",
        size_max=22,
        opacity=0.85,
    )
    fig.update_layout(
        template="plotly_white",
        height=620,
        legend=dict(
            title="RFM Segmentleri",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            str(output_dir / "rfm_monetary_vs_recency_scatter.html"),
            include_plotlyjs="cdn",
        )

    return fig


def export_all_rfm_figures(
    rfm_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Dict[str, Path]:
    """
    Generate and save all RFM interactive figures under `reports/figures/`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    create_rfm_treemap(rfm_df, output_dir=output_dir)
    create_rfm_donut_chart(rfm_df, output_dir=output_dir)
    create_rfm_scatter_plot(rfm_df, output_dir=output_dir)

    return {
        "treemap": output_dir / "rfm_segments_treemap.html",
        "donut": output_dir / "rfm_segments_donut.html",
        "scatter": output_dir / "rfm_monetary_vs_recency_scatter.html",
    }


def build_tefas_dashboard(
    enriched_funds_df: pd.DataFrame,
    risk_metrics_df: pd.DataFrame,
    risk_profile_summary_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Create and export a 4-panel interactive Plotly HTML dashboard for TEFAS analytics.
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
