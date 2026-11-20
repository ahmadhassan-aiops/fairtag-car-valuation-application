"""Value a car: describe it, get the fair asking price with its range, the itemised build-up, what-if curves and
similar ads from the data. An optional asking price is placed against the range."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from fairtag.schema import (FEATURE_LABELS, FEATURES, FUEL_TYPES, GEARBOXES, MARKET_YEAR, MILEAGE_STEPS, REPAIRED,
                            VEHICLE_TYPES, Car)
from sticker import charts, ui
from sticker.data import brand_name, eur, km, listings, market, model, model_name, reference, value_name

MONTHS = ["Not given", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
START = {"in_brand": "volkswagen", "in_model": "golf", "in_km": 150000, "in_month": 0, "in_damage": "no", "in_ask": None}


def _fill_defaults() -> None:
    """Typical body, gearbox, fuel, power and year for the chosen model (the seller can change each one)."""
    ref = reference()
    ss = st.session_state
    if ss.in_model not in ref["brands"][ss.in_brand]:
        ss.in_model = ref["brands"][ss.in_brand][0]
    d = ref["defaults"][f"{ss.in_brand}|{ss.in_model}"]
    ss.in_type, ss.in_gear, ss.in_fuel = d["vehicle_type"], d["gearbox"], d["fuel_type"]
    ss.in_power_known = d["power"] is not None
    ss.in_power = d["power"] or 100
    ss.in_year = d["year"]


def _init() -> None:
    if "in_brand" not in st.session_state:
        for k, v in START.items():
            st.session_state[k] = v
        _fill_defaults()


def _form() -> None:
    ref = reference()
    ss = st.session_state
    st.selectbox("Brand", list(ref["brands"]), format_func=brand_name, key="in_brand", on_change=_fill_defaults)
    st.selectbox("Model", ref["brands"][ss.in_brand], format_func=model_name, key="in_model", on_change=_fill_defaults,
                 help="Listed from most to least advertised for this brand.")
    a, b = st.columns([1.3, 1])
    a.number_input("First registered (year)", 1950, MARKET_YEAR, step=1, key="in_year")
    b.selectbox("Month", range(13), format_func=lambda m: MONTHS[m], key="in_month")
    st.select_slider("Mileage", MILEAGE_STEPS, format_func=km, key="in_km",
                     help="The ad site records mileage in steps; 150,000 means 150,000 km or more.")
    a, b = st.columns([1, 1.3], vertical_alignment="bottom")
    a.toggle("Power known", key="in_power_known")
    b.number_input("Power (hp)", 20, 600, step=5, key="in_power", disabled=not ss.in_power_known)
    a, b = st.columns(2)
    a.selectbox("Body type", VEHICLE_TYPES, format_func=lambda v: value_name("VehicleType", v), key="in_type")
    b.selectbox("Gearbox", GEARBOXES, format_func=lambda v: value_name("Gearbox", v), key="in_gear")
    a, b = st.columns(2)
    a.selectbox("Fuel", FUEL_TYPES, format_func=lambda v: value_name("FuelType", v), key="in_fuel")
    b.selectbox("Unrepaired damage", REPAIRED, format_func=lambda v: value_name("NotRepaired", v), key="in_damage")
    st.number_input("Asking price to check (optional, €)", min_value=100, max_value=100000, step=100, key="in_ask",
                    placeholder="For example 4500")


def _car() -> Car:
    ss = st.session_state
    return Car(brand=ss.in_brand, model=ss.in_model, vehicle_type=ss.in_type, gearbox=ss.in_gear, fuel_type=ss.in_fuel,
               not_repaired=ss.in_damage, registration_year=int(ss.in_year), registration_month=int(ss.in_month),
               power=int(ss.in_power) if ss.in_power_known else None, mileage=int(ss.in_km))


def _title(car: Car) -> str:
    return f"{brand_name(car.brand)} {model_name(car.model)} · {car.registration_year}"


def _specs(car: Car) -> str:
    parts = [km(car.mileage), f"{car.power} hp" if car.power else "power not given",
             value_name("VehicleType", car.vehicle_type), value_name("Gearbox", car.gearbox),
             value_name("FuelType", car.fuel_type), f"{car.age:.1f} years old in March 2016"]
    dmg = {"yes": "<b>unrepaired damage</b>", "no": "no unrepaired damage", "unknown": "damage not stated"}[car.not_repaired]
    return " · ".join(p for p in parts if p != "Not given") + f" · {dmg}"


def _value_text(f: str, row: dict) -> str:
    v = row[f]
    if f == "Brand":
        return brand_name(v)
    if f == "Model":
        return model_name(v)
    if f == "Age":
        return f"{v:.1f} years"
    if f == "Power":
        return "not given" if pd.isna(v) else f"{v:.0f} hp"
    if f == "Mileage":
        return km(v)
    return value_name(f, v)


def _verdict_text(ask: float, fair: float, low: float, high: float, verdict: str) -> str:
    gap = ask / fair - 1
    rel = f"{abs(gap):.0%} {'below' if gap < 0 else 'above'} the fair price"
    if verdict == "Below the range":
        return (f"Asking {eur(ask, 1)} is {rel}. Fewer than 1 in 10 ads for cars like this ask so little: a bargain, or "
                "something the ad does not show (condition, accident history, missing papers). Check before you buy.")
    if verdict == "Above the range":
        return (f"Asking {eur(ask, 1)} is {rel}. Fewer than 1 in 10 ads for cars like this ask so much: it needs "
                "something extra to justify it (low real mileage, full service history, rare equipment).")
    return f"Asking {eur(ask, 1)} is {rel}, inside the band that holds 8 in 10 asking prices for cars like this."


def _curve(car: Car, feature: str, values) -> np.ndarray:
    base = car.frame()
    rows = pd.concat([base] * len(values), ignore_index=True)
    rows[feature] = values
    return model().predict(rows)


def render() -> None:
    _init()
    ui.page("Value a car", "What should this car be listed for?",
            "Describe the car the way an ad would. FairTag gives the <b>fair asking price</b> (what similar cars were "
            "listed for), the range that holds 8 in 10 real asking prices, and how each detail moves the price. "
            "Prices are euros in the German market of early 2016, the market the model learned.")
    left, right = st.columns([1, 1.55], gap="medium")
    with left, ui.panel("01", "The car"):
        _form()
    try:
        car = _car()
    except ValidationError as e:
        with right:
            st.error(e.errors()[0]["msg"].removeprefix("Value error, ").capitalize())
        return
    m = model()
    x = car.frame()
    fair = float(m.predict(x)[0])
    low, high = (float(v[0]) for v in m.price_range(np.array([fair])))
    contrib = m.explain(x).iloc[0]
    ask = st.session_state.in_ask

    with right:
        with ui.panel("02", "The tag"):
            ui.price_tag(fair, low, high, _title(car), _specs(car))
            ui.range_bar(fair, low, high, ask)
            if ask:
                verdict = str(m.verdict(ask, low, high))
                st.html(ui.verdict_tag(verdict) + f'<p class="verdict-t">{_verdict_text(ask, fair, low, high, verdict)}</p>')
            else:
                st.html('<p class="verdict-t">Add an asking price in panel 01 to see where it sits against the range.</p>')
        with ui.panel("03", "How the price is built",
                      "Start from the average ad, then add or take off what each detail is worth for this car. The lines "
                      "add up exactly to the fair price."):
            row = x.iloc[0].to_dict()
            lines = sorted(((FEATURE_LABELS[f], _value_text(f, row), float(contrib[f])) for f in FEATURES),
                           key=lambda r: -abs(r[2]))
            ui.buildup(float(contrib["base"]), lines, max(float(contrib.sum()), fair))

    with ui.panel("04", "If it were older, or drove more",
                  "The fair price as one detail changes and everything else stays as described. The orange dot is this car."):
        a, b = st.columns(2)
        mk = market()
        same_km = mk.loc[mk["Mileage"] == car.mileage, "Age"]
        lo_age, hi_age = (float(v) for v in same_km.quantile([0.05, 0.95]))
        ages = np.round(np.arange(np.floor(min(lo_age, car.age)), np.ceil(max(hi_age, car.age)) + 0.01, 0.5), 1)
        a.plotly_chart(charts.what_if_line(ages, _curve(car, "Age", ages), car.age, fair, "Age (years)", ":.1f"),
                       config=charts.CONFIG, key="wi_age")
        same_age = mk.loc[(mk["Age"] - car.age).abs() <= 1.5, "Mileage"]
        lo_km, hi_km = (float(v) for v in same_age.quantile([0.05, 0.95])) if len(same_age) else (5000.0, 150000.0)
        steps = np.array([s for s in MILEAGE_STEPS if min(lo_km, car.mileage) <= s <= max(hi_km, car.mileage)], dtype=float)
        b.plotly_chart(charts.what_if_line(steps, _curve(car, "Mileage", steps), car.mileage, fair, "Mileage (km)", ":,.0f"),
                       config=charts.CONFIG, key="wi_km")
        per_year = _curve(car, "Age", np.array([car.age, car.age + 1]))
        change = per_year[0] - per_year[1]
        year_txt = (f"One more year takes about <b>{eur(change, 10)}</b> off this car's fair price."
                    if change > 5 else "At this age one more year barely changes this car's fair price.")
        ui.note(year_txt, "Each curve covers the ages (or mileages) that 9 in 10 real ads with this mileage (or age) "
                f"have: {lo_age:.0f} to {hi_age:.0f} years at {km(car.mileage)}, {km(lo_km)} to {km(hi_km)} at about "
                f"{car.age:.0f} years. Outside that the data holds too few ads to say.")

    with ui.panel("05", "Similar ads in the data",
                  "Held-out ads (never used in training) for the same model, within 1.5 years of age and one mileage step; "
                  "the closest in age are listed."):
        df = listings()
        k = MILEAGE_STEPS.index(car.mileage)
        near = MILEAGE_STEPS[max(0, k - 1): k + 2]
        sim = df[(df["Brand"] == car.brand) & (df["Model"] == car.model) & ((df["Age"] - car.age).abs() <= 1.5)
                 & (df["Mileage"].isin(near))]
        if sim.empty:
            st.info("No held-out ad matches this car closely. The price above still uses everything the model learned "
                    "from similar brands, ages and mileages.")
        else:
            ui.stats([("Similar ads", f"{len(sim):,}", "in the held-out data"),
                      ("Median asking", eur(sim["Price"].median()), "what these sellers asked"),
                      ("Middle half", f"{eur(sim['Price'].quantile(.25))} to {eur(sim['Price'].quantile(.75))}",
                       "half of them asked within this")])
            show = sim.assign(d=(sim["Age"] - car.age).abs()).sort_values("d").head(12)
            st.dataframe(pd.DataFrame({
                "Registered": show["RegistrationYear"].astype(int).astype(str),
                "Mileage": show["Mileage"].map(km), "Power": show["Power"].map(lambda v: "" if pd.isna(v) else f"{v:.0f} hp"),
                "Gearbox": show["Gearbox"].map(lambda v: value_name("Gearbox", v)),
                "Damage": show["NotRepaired"].map(lambda v: value_name("NotRepaired", v)),
                "Asking": show["Price"].map(lambda v: eur(v, 1)), "Fair price": show["fair"].map(eur),
            }), hide_index=True, width="stretch")
    ui.footer()
