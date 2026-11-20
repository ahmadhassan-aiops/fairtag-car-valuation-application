"""Cleaning of the raw car ads (car_data.csv), shared by train.py and the notebook's logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairtag.schema import CATEGORICAL, car_age

MIN_PRICE = 100          # asking prices under 100 euros are placeholders ("0", "1", "99"), not real prices
MIN_YEAR = 1950          # older registrations are typing errors (the data holds years 1000 and 9999)
MAX_KM_FIRST_YEAR = 50000  # under a year old with more km than this: the registration year is wrong
POWER_RANGE = (20, 600)  # horsepower outside this range is a typing error: set to missing, the model handles it


def clean(raw: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """Return one row per real ad with the model's features (Age added) and the asking Price. If a dict is given as
    report, the number of rows each step removed or changed is written into it."""
    r = report if report is not None else {}
    d = raw.copy()
    d["DateCreated"] = pd.to_datetime(d["DateCreated"], dayfirst=True)
    # The same ad crawled twice differs only in the crawl dates: keep it once.
    n = len(d)
    d = d.drop_duplicates(subset=[c for c in d.columns if c not in ("DateCrawled", "LastSeen")])
    r["duplicates"] = n - len(d)
    r["low_price"] = int((d["Price"] < MIN_PRICE).sum())
    d = d[d["Price"] >= MIN_PRICE]
    year = d["DateCreated"].dt.year
    ok = (d["RegistrationYear"] >= MIN_YEAR) & (d["RegistrationYear"] <= year)
    r["bad_year"] = int((~ok).sum())
    d = d[ok].copy()
    r["gasoline"] = int((d["FuelType"] == "gasoline").sum())
    # "gasoline" in the source is a mistranslation of the German "Diesel" category; "petrol" is Benzin.
    d["FuelType"] = d["FuelType"].replace({"gasoline": "diesel"})
    r["power_zero"] = int((d["Power"] == 0).sum())
    bad = (d["Power"] < POWER_RANGE[0]) | (d["Power"] > POWER_RANGE[1])
    r["bad_power"] = int(bad.sum())
    d.loc[bad, "Power"] = np.nan
    d["Age"] = car_age(d["RegistrationYear"].to_numpy(), d["RegistrationMonth"].to_numpy(),
                       d["DateCreated"].dt.year.to_numpy(), d["DateCreated"].dt.month.to_numpy())
    # Mostly ads giving the ad year as the registration year with 150,000 km: the year is wrong, so the age is unknown.
    wrong = (d["Age"] < 1) & (d["Mileage"] > MAX_KM_FIRST_YEAR)
    r["new_but_driven"] = int(wrong.sum())
    d = d[~wrong].copy()
    r["missing"] = {c: int(d[c].isna().sum()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        d[c] = d[c].fillna("unknown").astype(str)
    d["Power"] = d["Power"].astype(float)
    d["Mileage"] = d["Mileage"].astype(float)
    return d.reset_index(drop=True)
