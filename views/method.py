"""Data & method: the source data, cleaning, features, training, the price range, limits and how the app is built."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sticker import ui
from sticker.data import eur, model
from sticker.site import NOTEBOOK_URL


def render() -> None:
    meta = model().meta
    rows, cl, p = meta["rows"], meta["cleaning"], meta["params"]
    ui.page("Data & method", "How FairTag was built",
            f'The data, the cleaning, the model and its limits. The full Python work is in the '
            f'<a href="{NOTEBOOK_URL}" target="_blank">notebook</a>.')

    with ui.panel("01", "The data"):
        ui.sheet([
            ("Source", "Ads from a German used-car classifieds site, placed between March 2014 and April 2016 (almost "
                       "all in March 2016), prepared as the Rusty Bargain training case"),
            ("Size", f"{rows['raw']:,} ads, 16 columns; {rows['clean']:,} after cleaning"),
            ("Target", f"Asking price in euros (€100 to €20,000; median {eur(meta['price']['median'], 10)})"),
            ("Not used", "Ad and crawl dates (only to measure age), number of pictures (always 0), postal code"),
        ])

    with ui.panel("02", "Cleaning"):
        st.dataframe(pd.DataFrame([
            ("The same ad crawled more than once", f"{cl['duplicates']:,} rows", "Kept once (rows identical apart from the crawl dates)"),
            ("Asking price under €100", f"{cl['low_price']:,} ads", "Removed: 0, 1 or 99 euros are placeholders, not prices"),
            ("Registration year before 1950 or after the ad date", f"{cl['bad_year']:,} ads", "Removed: typing errors (years such as 1000 and 9999)"),
            ("Under a year old with more than 50,000 km", f"{cl['new_but_driven']:,} ads",
             "Removed: mostly the ad year given as the registration year (150,000 km, about €1,000), so the age is unknown"),
            ("Fuel labelled \"gasoline\"", f"{cl['gasoline']:,} ads", "Relabelled diesel: a translation error in the source (there is no diesel label otherwise)"),
            ("Power under 20 or over 600 hp", f"{cl['bad_power']:,} ads ({cl['power_zero']:,} of them 0 hp)", "Set to missing; the model handles missing power"),
            ("Missing body type, gearbox, fuel, model or damage", ", ".join(f"{k} {v:,}" for k, v in cl["missing"].items() if v),
             "Kept as their own value \"not given\": a seller leaving a field empty is information"),
        ], columns=["What was found", "How many", "How it was handled"]), hide_index=True, width="stretch")

    with ui.panel("03", "Features"):
        ui.sheet([
            ("Car", "Brand (40), model (250), body type, gearbox, fuel"),
            ("Condition", "Unrepaired damage (yes, no, not given), mileage (in the site's steps, capped at 150,000 km)"),
            ("Age", "Years from first registration to the ad, to the month (a missing month counts as mid-year)"),
            ("Power", "Horsepower, missing when not given or impossible"),
        ])

    with ui.panel("04", "Training"):
        ui.sheet([
            ("Split", f"{rows['train']:,} training ads and {rows['test']:,} held-out ads (80/20, random, seed 42). The "
                      "held-out ads were used once, for the final report"),
            ("Model choice", "Linear regression (sanity check), decision tree, random forest, LightGBM, XGBoost and "
                             "CatBoost compared with 5-fold cross-validation on the training ads"),
            ("Model", f"LightGBM regression on the raw price: {p['n_estimators']:,} trees, learning rate {p['learning_rate']}, "
                      f"up to {p['num_leaves']} leaves a tree, {p['colsample_bytree']:.0%} of features and {p['subsample']:.0%} of ads per tree. Brand, model and the other categories are used "
                      "directly as categories (no one-hot encoding)"),
            ("Tuning", "Random search over 24 settings of tree size, learning rate, sampling and regularisation, scored by cross-validation; "
                       "the tree count is the median best round with early stopping across the folds"),
            ("Price range", "Ratio of real to predicted price in cross-validation, measured in 10 bands of predicted price; "
                            "the range is the 10th to 90th percentile of that ratio"),
            ("Explanations", "SHAP values from LightGBM: euros each feature adds to or takes off the average ad, adding up "
                             "exactly to the prediction"),
        ])

    with ui.panel("05", "Limits"):
        st.markdown(
            "- **Asking prices, not sale prices.** The model learned what sellers asked; cars often sell for less.\n"
            "- **One market, one moment.** German ads from early 2016. Prices today and elsewhere differ.\n"
            "- **What ads do not record.** Condition, accident history, service records, equipment and colour are "
            "invisible to the model; they are why the range is wide.\n"
            "- **Mileage is capped.** 150,000 km means 150,000 or more, so very high mileage is not priced separately.\n"
            "- **5,000 km on an older car.** In this data such ads are mostly incomplete or unusual listings priced "
            "low, so FairTag prices a genuinely low-mileage older car like them.\n"
            "- **Rare cars.** Models with few ads borrow from similar brands, ages and power; their prices are less sure.")

    with ui.panel("06", "How this app is built"):
        st.markdown(
            f"Python, LightGBM, pandas, Plotly and Streamlit. `train.py` cleans the CSV, trains the model and writes "
            f"`model/` (the LightGBM model as a text file and `metadata.json`) and the app's data tables. The "
            f"`fairtag` package (pydantic input schema, model loading, price range, SHAP breakdown) has no Streamlit "
            f"code; tests cover the model and every page. Built on {meta['built']} with LightGBM "
            f"{meta['versions']['lightgbm']}.")
    ui.footer()
