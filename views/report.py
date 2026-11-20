"""Model report: accuracy on the held-out ads, by price band, predicted against asking, the price range, and the model
comparison (quality, speed, training time)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sticker import charts, theme as t, ui
from sticker.data import eur, listings, model


def _density(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram2d(
        x=df["fair"], y=df["Price"], xbins=dict(start=0, end=20000, size=500), ybins=dict(start=0, end=20000, size=500),
        colorscale=[[0, "#ffffff"], [0.02, "#ffe8d9"], [0.25, "#ffa94d"], [0.6, t.TAG], [1, "#5c1d04"]],
        colorbar=dict(title=dict(text="Ads", side="right"), thickness=12, outlinewidth=0),
        hovertemplate="Fair €%{x}<br>Asking €%{y}<br><b>%{z:,} ads</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 20000], y=[0, 20000], mode="lines", line=dict(color=t.INK, width=1.5, dash="dot"),
                             hoverinfo="skip", showlegend=False))
    charts.style(fig, 430)
    fig.update_xaxes(title_text="Fair price (model)", tickprefix="€", tickformat=",.0f", range=[0, 20000])
    fig.update_yaxes(title_text="Asking price (ad)", tickprefix="€", tickformat=",.0f", range=[0, 20000], gridcolor="rgba(0,0,0,0)")
    return fig


def _comparison_chart(comp: pd.DataFrame) -> go.Figure:
    comp = comp.sort_values("rmse", ascending=False)
    colors = [t.TAG if m == "LightGBM" else "#adb5bd" for m in comp["model"]]
    fig = go.Figure(go.Bar(y=comp["model"], x=comp["rmse"], orientation="h", marker=dict(color=colors, cornerradius=3),
                           text=[eur(v, 1) for v in comp["rmse"]], textposition="outside", cliponaxis=False,
                           textfont=dict(color=t.INK), hovertemplate="<b>%{y}</b><br>RMSE €%{x:,.0f}<extra></extra>"))
    charts.style(fig, 290)
    fig.update_xaxes(title_text="RMSE in cross-validation (lower is better)", tickprefix="€", tickformat=",.0f",
                     range=[0, float(comp["rmse"].max()) * 1.22])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(bargap=0.35)
    return fig


def render() -> None:
    meta = model().meta
    te, cv, rows = meta["test"], meta["cv"], meta["rows"]
    ui.page("Model report", "How accurate is the fair price?",
            f"Measured once on <b>{rows['test']:,} held-out ads</b> the model never saw in training. Asking prices for "
            "similar cars differ a lot, because ads do not record condition or equipment, so the report shows the typical "
            "error, the error by price band and how often the range holds the real price.")
    ui.stats([("RMSE", eur(te["rmse"], 1), "typical error (root mean squared)"),
              ("MAE", eur(te["mae"], 1), "average error in euros"),
              ("R²", f"{te['r2']:.3f}", "share of price variation explained"),
              ("Within 20%", f"{te['within_20']:.0%}", f"of ads; {te['within_10']:.0%} within 10%"),
              ("Range holds", f"{te['range_coverage']:.1%}", "of real asking prices (aim: 80%)")])

    with ui.panel("01", "Accuracy by price band",
                  "Cheap cars are hardest in percentage terms: a €300 error on a €600 car is half its price."):
        bb = pd.DataFrame(meta["by_band"])
        st.dataframe(pd.DataFrame({
            "Asking price (€)": bb["band"], "Ads": bb["cars"].map("{:,}".format),
            "Average error": bb["mae"].map(lambda v: eur(v, 10)), "Median error": bb["mdape"].map("{:.0%}".format),
            "Within 20%": bb["within_20"].map("{:.0%}".format), "Range holds": bb["coverage"].map("{:.0%}".format),
            "Average over (+) or under (−)": bb["bias"].map(lambda v: ("+" if v >= 0 else "−") + eur(abs(v), 10)),
        }), hide_index=True, width="stretch")
        lo, hi = meta["by_band"][0], meta["by_band"][-1]
        ui.note(f"Like any price model, FairTag pulls the extremes toward the middle: ads under €1,000 get a fair price "
                f"{eur(lo['bias'], 10)} higher on average, and ads over €10,000 one {eur(-hi['bias'], 10)} lower. Above "
                f"€6,000 the median error is {meta['by_band'][3]['mdape']:.0%} or less.")

    a, b = st.columns([1.15, 1], gap="medium")
    with a, ui.panel("02", "Fair price against asking price",
                     "Every held-out ad, counted in €500 squares. On the dotted line the two are equal."):
        st.plotly_chart(_density(listings()), config=charts.CONFIG, key="density")
    with b, ui.panel("03", "The price range",
                     "Built from cross-validated errors on the training ads: in each of 10 bands of predicted price, the "
                     "middle 80% of real asking prices around the prediction."):
        r = meta["range"]
        edges = r["edges"] + [None]
        band_rows = [(f"{eur(edges[i], 10)} to {eur(edges[i + 1], 10)}" if edges[i + 1] else f"{eur(edges[i], 10)} and up",
                      f"{r['low_ratio'][i]:.2f}x to {r['high_ratio'][i]:.2f}x") for i in range(len(r["low_ratio"]))]
        st.dataframe(pd.DataFrame(band_rows, columns=["Fair price", "Range (x fair price)"]), hide_index=True,
                     width="stretch", height=390)
        ui.note(f"On the held-out ads the range held <b>{te['range_coverage']:.1%}</b> of real asking prices. It is "
                f"wide (median width {te['range_width_median']:.0%} of the fair price) because asking prices for similar "
                "cars really are that spread out; it narrows as prices rise.")

    with ui.panel("04", "Why LightGBM",
                  "Six models, the same nine features, 5-fold cross-validation on the training ads. Fit time is for one "
                  f"fold (about {round(rows['train'] * 0.8, -3):,.0f} ads) on a desktop CPU; prediction time is for one car."):
        comp = pd.DataFrame(meta["comparison"])
        a, b = st.columns([1, 1.15], gap="medium")
        a.plotly_chart(_comparison_chart(comp), config=charts.CONFIG, key="comp_chart")
        b.dataframe(pd.DataFrame({"Model": comp["model"], "RMSE": comp["rmse"].map(lambda v: eur(v, 1)),
                                  "MAE": comp["mae"].map(lambda v: eur(v, 1)), "R²": comp["r2"].map("{:.3f}".format),
                                  "Fit (s)": comp["fit_s"].map("{:,.1f}".format),
                                  "Predict (ms)": comp["predict_ms"].map("{:.1f}".format)}), hide_index=True, width="stretch")
        tm = meta["timing"]
        lgb_row = comp.set_index("model").loc["LightGBM"]
        slow = comp.set_index("model").loc["CatBoost", "fit_s"] / lgb_row["fit_s"]
        ui.note(f"LightGBM had the lowest error with default settings and trains in seconds; CatBoost took about "
                f"{slow:.0f} times longer. The tuned LightGBM scores <b>{eur(cv['rmse'], 1)}</b> in cross-validation: "
                "settings move the error by only a few euros, because the data, not the model, limits accuracy. "
                f"Linear regression, the sanity check, misses by {eur(comp.set_index('model').loc['Linear regression (one-hot)', 'rmse'] - cv['rmse'], 10)} "
                f"more. The deployed model trains on all {rows['train']:,} training ads in <b>{tm['fit_s']:.0f} seconds</b> "
                f"and prices a car in <b>{tm['predict_ms']:.0f} ms</b>.")
    ui.footer()
