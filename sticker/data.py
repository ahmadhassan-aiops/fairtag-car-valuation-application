"""Cached loaders for the model and the app's data (built by train.py), and display names."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fairtag.model import PriceModel

DATA = Path(__file__).resolve().parent.parent / "data"


@st.cache_resource(show_spinner="Loading the price model...")
def model() -> PriceModel:
    return PriceModel()


@st.cache_data
def listings() -> pd.DataFrame:
    """The held-out ads (20% of the clean ads) (never seen in training) with the model's fair price and range."""
    df = pd.read_parquet(DATA / "listings.parquet")
    df["verdict"] = model().verdict(df["Price"], df["low"], df["high"])
    df["gap"] = df["Price"] / df["fair"] - 1
    return df


@st.cache_data
def market() -> pd.DataFrame:
    """All clean ads (for the market guide)."""
    return pd.read_parquet(DATA / "market.parquet")


@st.cache_data
def reference() -> dict:
    return json.loads((DATA / "reference.json").read_text(encoding="utf-8"))


@st.cache_data
def contributions() -> pd.DataFrame:
    """Euros each feature added or took off (SHAP, columns '<feature>_eur'), for 8,000 random held-out ads, with their
    category values, power and mileage. Precomputed by train.py."""
    return pd.read_parquet(DATA / "contributions.parquet")


BRAND_NAMES = {"bmw": "BMW", "mercedes_benz": "Mercedes-Benz", "mini": "MINI", "sonstige_autos": "Other brands",
               "alfa_romeo": "Alfa Romeo", "land_rover": "Land Rover", "citroen": "Citroën", "seat": "SEAT",
               "kia": "Kia", "smart": "smart"}
MODEL_NAMES = {"unknown": "Model not given", "other": "Other model", "x_reihe": "X series", "z_reihe": "Z series",
               "m_reihe": "M series", "kaefer": "Käfer (Beetle)", "a_klasse": "A-Class", "b_klasse": "B-Class",
               "c_klasse": "C-Class", "e_klasse": "E-Class", "s_klasse": "S-Class", "m_klasse": "M-Class",
               "g_klasse": "G-Class", "cl": "CL", "sl": "SL", "slk": "SLK", "clk": "CLK", "glk": "GLK", "gl": "GL",
               "cc": "CC", "up": "up!", "500": "500", "x_type": "X-Type", "s_type": "S-Type"}
VALUE_NAMES = {
    "VehicleType": {"sedan": "Sedan", "small": "Small car", "wagon": "Estate", "bus": "Van / MPV", "convertible":
                    "Convertible", "coupe": "Coupé", "suv": "SUV", "other": "Other", "unknown": "Not given"},
    "Gearbox": {"manual": "Manual", "auto": "Automatic", "unknown": "Not given"},
    "FuelType": {"petrol": "Petrol", "diesel": "Diesel", "lpg": "LPG", "cng": "CNG", "hybrid": "Hybrid",
                 "electric": "Electric", "other": "Other", "unknown": "Not given"},
    "NotRepaired": {"no": "None", "yes": "Yes, unrepaired", "unknown": "Not given"},
}


def brand_name(b: str) -> str:
    return BRAND_NAMES.get(b, b.replace("_", " ").title())


def model_name(m: str) -> str:
    if m in MODEL_NAMES:
        return MODEL_NAMES[m]
    if m.endswith("er") and m[:-2].isdigit():
        return f"{m[:-2]} Series"
    if m.endswith("_reihe"):
        return m.replace("_reihe", " series").upper().replace(" SERIES", " series")
    return m.replace("_", " ").title() if not any(ch.isdigit() for ch in m) else m.replace("_", " ").upper()


def value_name(feature: str, v: str) -> str:
    return VALUE_NAMES.get(feature, {}).get(v, v)


def eur(x: float, step: int = 50) -> str:
    """Euro amount rounded to the step, with a thin thousands separator."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"€{int(round(float(x) / step) * step):,}"


def km(x: float) -> str:
    return "150,000+ km" if x >= 150000 else f"{int(x):,} km"
