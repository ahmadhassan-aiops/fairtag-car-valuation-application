"""Deal finder: filter the held-out ads and see which ones ask less than their fair range, which ask more, and why."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fairtag.model import VERDICTS, round_price
from fairtag.schema import MILEAGE_STEPS
from sticker import charts, theme as t, ui
from sticker.data import brand_name, eur, km, listings, model_name, reference, value_name

FAR = 0.5  # more than half below the fair price
SYMBOL = {"Below the range": "triangle-down", "Within the range": "circle", "Above the range": "triangle-up"}


def _filters() -> pd.DataFrame:
    df = listings()
    ref = reference()
    a, b, c = st.columns([1.2, 1.2, 1.4])
    brands = a.multiselect("Brands", list(ref["brands"]), format_func=brand_name, key="d_brands",
                           placeholder="All brands")
    models = sorted({m for br in brands for m in ref["brands"][br]}) if brands else []
    picked = b.multiselect("Models", models, format_func=model_name, key="d_models", disabled=not brands,
                           placeholder="All models" if brands else "Pick a brand first")
    lo, hi = c.slider("Asking price (€)", 100, 20000, (1000, 8000), step=100, key="d_budget")
    a, b, c, d = st.columns([1.1, 1.2, 1, 1.1], vertical_alignment="bottom")
    max_age = a.slider("At most this old (years)", 1, 30, 15, key="d_age")
    max_km = b.select_slider("At most this mileage", MILEAGE_STEPS, value=150000, format_func=km, key="d_km")
    gear = c.segmented_control("Gearbox", ["Any", "Manual", "Automatic"], default="Any", key="d_gear")
    no_damage = d.toggle("Hide unrepaired damage", value=True, key="d_damage")
    k = df["Price"].between(lo, hi) & (df["Age"] <= max_age) & (df["Mileage"] <= max_km)
    if brands:
        k &= df["Brand"].isin(brands)
    if picked:
        k &= df["Model"].isin(picked)
    if gear and gear != "Any":
        k &= df["Gearbox"] == {"Manual": "manual", "Automatic": "auto"}[gear]
    if no_damage:
        k &= df["NotRepaired"] != "yes"
    return df[k]


def _title(r) -> str:
    return f"{brand_name(r.Brand)} {model_name(r.Model)} · {int(r.RegistrationYear)}"


def _specs(r) -> str:
    p = "" if pd.isna(r.Power) else f" · {r.Power:.0f} hp"
    return f"{km(r.Mileage)}{p} · {value_name('Gearbox', r.Gearbox)} · {value_name('FuelType', r.FuelType)}"


def _open_in_valuer(r) -> None:
    from nav import VALUE
    ss = st.session_state
    ss.in_brand, ss.in_model = r.Brand, r.Model
    ss.in_type, ss.in_gear, ss.in_fuel, ss.in_damage = r.VehicleType, r.Gearbox, r.FuelType, r.NotRepaired
    ss.in_year, ss.in_month, ss.in_km = int(r.RegistrationYear), int(r.RegistrationMonth), int(r.Mileage)
    ss.in_power_known = not pd.isna(r.Power)
    ss.in_power = int(r.Power) if not pd.isna(r.Power) else 100
    ss.in_ask = int(r.Price)
    st.switch_page(VALUE)


def _scatter(df: pd.DataFrame) -> go.Figure:
    s = df.sample(min(len(df), 3000), random_state=0)
    fig = go.Figure()
    for v in VERDICTS:
        g = s[s["verdict"] == v]
        fig.add_trace(go.Scattergl(
            x=g["fair"], y=g["Price"], mode="markers", name=v,
            marker=dict(size=7, color=t.VERDICT_STYLE[v][0], symbol=SYMBOL[v], opacity=.55, line=dict(width=0)),
            customdata=np.c_[g["Brand"].map(brand_name), g["Model"].map(model_name), g["RegistrationYear"], g["low"], g["high"]],
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b> · %{customdata[2]}<br>Asking €%{y:,.0f}<br>"
                          "Fair €%{x:,.0f} (range €%{customdata[3]:,.0f} to €%{customdata[4]:,.0f})<extra></extra>"))
    top = float(s["Price"].max()) * 1.05 if len(s) else 20000
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=t.INK, width=1.5, dash="dot"),
                             hoverinfo="skip", name="Asking = fair", showlegend=True))
    charts.style(fig, 430, legend=True)
    fig.update_xaxes(title_text="Fair price (model)", tickprefix="€", tickformat=",.0f", rangemode="tozero")
    fig.update_yaxes(title_text="Asking price (ad)", tickprefix="€", tickformat=",.0f", rangemode="tozero")
    return fig


def render() -> None:
    ui.page("Deal finder", "Which ads ask less than they should?",
            f"These are <b>{len(listings()):,} real ads the model never saw in training</b>. Each one has its fair price and range. "
            "Filter for the car you want, and the ads asking less than their range come first.")
    with ui.panel("01", "What you are looking for"):
        df = _filters()
    below = df[df["verdict"] == VERDICTS[0]].sort_values("gap")
    above = df["verdict"] == VERDICTS[2]
    ui.stats([("Matching ads", f"{len(df):,}", "held-out ads that pass the filters"),
              ("Below the range", f"{len(below):,}", f"{len(below) / max(len(df), 1):.0%} of matching ads"),
              ("Within the range", f"{int((df['verdict'] == VERDICTS[1]).sum()):,}", "asking about what similar cars ask"),
              ("Above the range", f"{int(above.sum()):,}", f"{above.mean() if len(df) else 0:.0%} of matching ads")])
    if df.empty:
        st.info("No ads match these filters. Widen the price, age or mileage.")
        ui.footer()
        return

    with ui.panel("02", "Best-priced ads", "Ads below their range, the furthest below first."):
        far = below["gap"] < -FAR
        kind = st.segmented_control(
            "Show", ["Up to half off", "More than half off"], default="Up to half off", key="d_kind",
            label_visibility="collapsed",
            format_func=lambda v: f"{v} ({int((~far).sum() if v == 'Up to half off' else far.sum()):,})")
        pool = below[far] if kind == "More than half off" else below[~far]
        if kind == "More than half off":
            ui.caution("Asking less than half the fair price is rarely a plain bargain: such ads are often placeholder "
                       "prices, damaged or non-running cars, or parts sales the data does not label.")
        if pool.empty:
            st.info("No matching ad in this group. The closest prices are in the full list below.")
        else:
            top = pool.head(9)
            ui.deal_cards([dict(title=_title(r), specs=_specs(r), ask=r.Price, fair=r.fair, low=r.low, high=r.high,
                                verdict=r.verdict, gap=f"{abs(r.gap):.0%} below the fair price")
                           for r in top.itertuples()])
            a, b = st.columns([2, 1], vertical_alignment="bottom")
            labels = {i: f"{_title(r)} · {eur(r.Price, 1)}" for i, r in zip(top.index, top.itertuples())}
            pick = a.selectbox("Look at one in detail", list(labels), format_func=labels.get, key="d_pick")
            if b.button("Open in Value a car", icon=":material/sell:", width="stretch"):
                _open_in_valuer(top.loc[pick])
        ui.caution("A price below the range can be a real bargain, but the ad may hide what the data does not "
                   "record: general condition, accident history, service records, missing papers. Inspect the car.")

    with ui.panel("03", "Asking price against fair price",
                  "Each point is an ad (up to 3,000 shown). On the dotted line the seller asks exactly the fair price."):
        st.plotly_chart(_scatter(df), config=charts.CONFIG, key="deal_scatter")

    with ui.panel("04", "All matching ads", "Sort any column. The download holds every matching ad."):
        out = pd.DataFrame({
            "Car": [f"{brand_name(b)} {model_name(m)}" for b, m in zip(df["Brand"], df["Model"])],
            "Registered": df["RegistrationYear"].astype(int), "Mileage (km)": df["Mileage"].astype(int),
            "Power (hp)": df["Power"], "Gearbox": df["Gearbox"].map(lambda v: value_name("Gearbox", v)).astype(str),
            "Fuel": df["FuelType"].map(lambda v: value_name("FuelType", v)).astype(str),
            "Asking (€)": df["Price"].astype(int), "Fair (€)": df["fair"].map(round_price),
            "Low (€)": df["low"].map(round_price), "High (€)": df["high"].map(round_price),
            "Vs fair": df["gap"], "Verdict": df["verdict"],
        }).sort_values("Vs fair")
        st.dataframe(out, hide_index=True, width="stretch", height=380, column_config={
            "Vs fair": st.column_config.NumberColumn(format="percent"),
            "Power (hp)": st.column_config.NumberColumn(format="%d"),
            "Registered": st.column_config.NumberColumn(format="%d")})
        st.download_button("Download these ads (CSV)", out.to_csv(index=False).encode("utf-8"), "fairtag_ads.csv",
                           "text/csv", icon=":material/download:")
    ui.footer()
