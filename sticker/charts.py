"""Plotly figures in the FairTag style: paper background inside white panels, recessive grid, euros on the axis."""

from __future__ import annotations

import plotly.graph_objects as go

from sticker import theme as t

CONFIG = {"displayModeBar": False}


def style(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=12, b=8), paper_bgcolor=t.PANEL, plot_bgcolor=t.PANEL,
        font=dict(family="Barlow, Segoe UI, sans-serif", size=13, color=t.INK), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=t.INK, font=dict(family="Barlow, sans-serif", color=t.INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=t.LINE, tickcolor=t.LINE, ticks="outside", zeroline=False,
                     title_font=dict(color=t.MUTED, size=12))
    fig.update_yaxes(gridcolor=t.GRID, linecolor=t.LINE, zeroline=False, title_font=dict(color=t.MUTED, size=12))
    return fig


def what_if_line(xs, ys, x_now, y_now, x_title: str, x_fmt: str = "", x_type: str = "linear") -> go.Figure:
    """The car's fair price as one feature changes, the rest held; the car itself marked."""
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=t.INK, width=2.5, shape="spline", smoothing=0.4),
                               hovertemplate=f"%{{x{x_fmt}}}<br><b>€%{{y:,.0f}}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=[x_now], y=[y_now], mode="markers", marker=dict(size=13, color=t.TAG, line=dict(color="#fff", width=2.5)),
                             hovertemplate=f"This car<br>%{{x{x_fmt}}}<br><b>€%{{y:,.0f}}</b><extra></extra>"))
    style(fig, 250)
    fig.update_xaxes(title_text=x_title, type=x_type)
    fig.update_yaxes(title_text="Fair price", tickprefix="€", tickformat=",.0f", rangemode="tozero")
    return fig
