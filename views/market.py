"""Market guide: how asking prices fall with age by brand, what each detail is worth, which details matter most, and a
brand league table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fairtag.schema import FEATURE_LABELS, FEATURES
from sticker import charts, theme as t, ui
from sticker.data import brand_name, contributions, eur, km, market, model, model_name, reference, value_name

MIN_ADS = 40
DETAILS = {"Unrepaired damage": "NotRepaired", "Gearbox": "Gearbox", "Fuel": "FuelType", "Body type": "VehicleType",
           "Mileage": "Mileage", "Power": "Power", "Brand": "Brand"}
POWER_BINS = [20, 60, 80, 100, 125, 150, 200, 250, 601]
POWER_LABELS = ["20-59 hp", "60-79", "80-99", "100-124", "125-149", "150-199", "200-249", "250+ hp"]


@st.cache_data
def _age_curves(brands: tuple[str, ...]) -> pd.DataFrame:
    df = market()
    df = df[df["Brand"].isin(brands)]
    g = df.assign(year=df["Age"].round().astype(int)).query("year <= 25").groupby(["Brand", "year"], observed=True)["Price"]
    out = pd.DataFrame({"median": g.median(), "ads": g.size()}).reset_index()
    return out[out["ads"] >= MIN_ADS]


def _age_chart(curves: pd.DataFrame, brands: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, b in enumerate(brands):
        c = curves[curves["Brand"] == b]
        if c.empty:
            continue
        fig.add_trace(go.Scatter(x=c["year"], y=c["median"], mode="lines+markers", name=brand_name(b),
                                 line=dict(color=t.SERIES[i], width=2.5), marker=dict(size=8, color=t.SERIES[i], line=dict(color="#fff", width=2)),
                                 customdata=c["ads"], hovertemplate=f"<b>{brand_name(b)}</b>, %{{x}} years<br>Median asking €%{{y:,.0f}}"
                                                                    "<br>%{customdata:,} ads<extra></extra>"))
        last = c.iloc[-1]
        fig.add_annotation(x=last["year"], y=last["median"], text=brand_name(b), showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=t.INK, size=12))
    charts.style(fig, 380, legend=True)
    fig.update_xaxes(title_text="Age (years)", dtick=2, range=[-0.5, 27.5])
    fig.update_yaxes(title_text="Median asking price", tickprefix="€", tickformat=",.0f", rangemode="tozero")
    fig.update_layout(hovermode="closest")
    return fig


def _worth(detail: str) -> pd.DataFrame:
    c = contributions()
    f = DETAILS[detail]
    key = c[f]
    if f == "Power":
        key = pd.cut(c["Power"], POWER_BINS, labels=POWER_LABELS, right=False).astype(str).replace("nan", "Not given")
    elif f == "Mileage":
        key = c["Mileage"].map(km)
    elif f == "Brand":
        key = c["Brand"].map(brand_name)
    else:
        key = c[f].map(lambda v: value_name(f, v))
    g = c.groupby(key, observed=True)[f + "_eur"].agg(["mean", "size"]).rename(columns={"size": "ads"})
    g = g[g["ads"] >= MIN_ADS]
    if f in ("Mileage", "Power"):
        order = [km(m) for m in sorted(c["Mileage"].unique())] if f == "Mileage" else POWER_LABELS + ["Not given"]
        return g.reindex([o for o in order if o in g.index])
    return g.sort_values("mean")


def _worth_chart(g: pd.DataFrame, ordered: bool) -> go.Figure:
    colors = [t.UP if v >= 0 else t.DOWN for v in g["mean"]]
    fig = go.Figure(go.Bar(
        y=g.index, x=g["mean"], orientation="h", marker=dict(color=colors, cornerradius=3),
        text=[("+" if v >= 0 else "−") + eur(abs(v), 10) for v in g["mean"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=t.INK), customdata=g["ads"],
        hovertemplate="<b>%{y}</b><br>%{x:+,.0f} euros on average<br>%{customdata:,} ads in the sample<extra></extra>"))
    charts.style(fig, max(260, 34 * len(g) + 60))
    lo, hi = min(float(g["mean"].min()), 0.0), max(float(g["mean"].max()), 0.0)
    pad = (hi - lo) * 0.28
    fig.update_xaxes(title_text="Euros added or taken off, compared with the average ad", tickprefix="€", tickformat=",.0f",
                     range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)], zeroline=True, zerolinecolor=t.INK, zerolinewidth=1.5)
    fig.update_yaxes(autorange="reversed" if ordered else True, gridcolor="rgba(0,0,0,0)")
    fig.update_layout(bargap=0.35)
    return fig


def _importance_chart() -> go.Figure:
    imp = pd.Series(model().meta["importance"]).rename(index=FEATURE_LABELS).sort_values()
    fig = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h", marker=dict(color=t.INK, cornerradius=3),
                           text=[eur(v, 10) for v in imp.values], textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>moves a price by €%{x:,.0f} on average<extra></extra>"))
    charts.style(fig, 330)
    fig.update_xaxes(title_text="Average size of the detail's effect on a price", tickprefix="€", tickformat=",.0f",
                     range=[0, float(imp.max()) * 1.2])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(bargap=0.35)
    return fig


def _league() -> pd.DataFrame:
    df = market()
    like = df[df["Age"].between(5, 10)]
    g = df.groupby("Brand", observed=True).agg(ads=("Price", "size"), median=("Price", "median"), age=("Age", "median"))
    g["like"] = like.groupby("Brand", observed=True)["Price"].median()
    g["top_model"] = df.groupby("Brand", observed=True)["Model"].agg(lambda s: s[s != "unknown"].mode().iat[0]
                                                                    if (s != "unknown").any() else "unknown")
    g = g[g["ads"] >= 200].sort_values("ads", ascending=False)
    return pd.DataFrame({"Brand": g.index.map(brand_name), "Ads": g["ads"], "Most advertised model": g["top_model"].map(model_name),
                         "Median asking": g["median"].map(eur), "Median age": g["age"].map("{:.1f} years".format),
                         "Median asking at 5-10 years": g["like"].map(eur)})


def render() -> None:
    ui.page("Market guide", "How the used-car market prices a car",
            f"Read from <b>{len(market()):,} clean ads</b> and from the model: how prices fall with age, what each detail of a "
            "car is worth, and how the brands compare.")
    ref = reference()
    with ui.panel("01", "How prices fall with age",
                  f"Median asking price by age for up to four brands (ages with at least {MIN_ADS} ads)."):
        brands = st.multiselect("Brands to compare", list(ref["brands"]), default=["volkswagen", "bmw", "opel", "mercedes_benz"],
                                format_func=brand_name, max_selections=4, key="m_brands")
        if brands:
            st.plotly_chart(_age_chart(_age_curves(tuple(brands)), brands), config=charts.CONFIG, key="age_chart")
            ui.note("Raw medians mix everything else that differs between brands (models, power, mileage). For a single "
                    "car with the rest held fixed, use <b>Value a car</b>, panel 04.")
        else:
            st.info("Pick at least one brand.")

    a, b = st.columns([1.25, 1], gap="medium")
    with a, ui.panel("02", "What each detail is worth",
                     "Average euros the model adds or takes off for each value, compared with the average ad, across "
                     "8,000 held-out ads. Everything else about each car stays as it is."):
        detail = st.segmented_control("Detail", list(DETAILS), default="Unrepaired damage", key="m_detail",
                                      label_visibility="collapsed") or "Unrepaired damage"
        g = _worth(detail)
        st.plotly_chart(_worth_chart(g, DETAILS[detail] not in ("Mileage", "Power")), config=charts.CONFIG, key="worth_chart")
    with b, ui.panel("03", "Which details matter most",
                     "How far each detail moves a price on average, in either direction (mean absolute SHAP value)."):
        st.plotly_chart(_importance_chart(), config=charts.CONFIG, key="imp_chart")
        imp = pd.Series(model().meta["importance"]).sort_values(ascending=False)
        ui.note(f"<b>{FEATURE_LABELS[imp.index[0]]}</b> moves a price most (about {eur(imp.iloc[0], 10)} on average), "
                f"then <b>{FEATURE_LABELS[imp.index[1]].lower()}</b> ({eur(imp.iloc[1], 10)}) and "
                f"<b>{FEATURE_LABELS[imp.index[2]].lower()}</b> ({eur(imp.iloc[2], 10)}).")

    with ui.panel("04", "Brand league", "Brands with at least 200 ads. The last column compares brands at a similar age."):
        st.dataframe(_league(), hide_index=True, width="stretch", height=420, column_config={
            "Ads": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=int(market()["Brand"].value_counts().max())),
})
    ui.footer()
