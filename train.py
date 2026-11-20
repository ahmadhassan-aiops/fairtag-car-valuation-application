"""
Rebuild everything the app ships from car_data.csv: the LightGBM model and its metadata (model/), and the app's data
tables (data/). Same cleaning, split and settings as the notebook.

    python train.py --csv path/to/car_data.csv

The CSV (354,369 ads, about 37 MB) is not in this repo; it is in the portfolio project folder linked in the README.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from fairtag.model import FLOOR, MODEL_FILE, PriceModel
from fairtag.prep import clean
from fairtag.schema import CATEGORICAL, FEATURES

ROOT = Path(__file__).resolve().parent
SEED = 42

# Chosen by 5-fold cross-validation on the training split (random search in the notebook). 512 trees is the median
# best round count with early stopping across the folds; the final model uses it as a fixed number.
PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=255, min_child_samples=20, subsample=1.0,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, cat_smooth=30.0, min_data_per_group=20,
              max_cat_threshold=64, n_estimators=512, random_state=0, verbose=-1)
RANGE_COVER = (0.10, 0.90)   # the price range holds the middle 80% of asking prices for similar predictions
RANGE_BANDS = 10             # ranges are measured separately in 10 bands of predicted price
PRICE_BANDS = [100, 1000, 3000, 6000, 10000, 20001]
PRICE_LABELS = ["100-999", "1,000-2,999", "3,000-5,999", "6,000-9,999", "10,000-20,000"]


def as_model_frame(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c].where(x[c].isin(cats[c])), categories=cats[c])  # unseen value -> missing
    return x


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ape = np.abs(err) / y
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))), mdape=float(np.median(ape)),
                r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)), within_10=float(np.mean(ape <= 0.10)),
                within_20=float(np.mean(ape <= 0.20)), bias=float(np.mean(err)))


def main(csv: Path) -> None:
    raw = pd.read_csv(csv)
    steps: dict = {}
    d = clean(raw, steps)
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED)
    cats = {c: sorted(tr[c].unique().tolist()) for c in CATEGORICAL}
    X, y = as_model_frame(tr, cats), tr["Price"].to_numpy(float)
    Xt, yt = as_model_frame(te, cats), te["Price"].to_numpy(float)
    print(f"{len(d):,} clean ads: {len(tr):,} training, {len(te):,} held out")

    # 1. Out-of-fold predictions on the training split: honest errors to size the price range.
    oof = np.zeros(len(y))
    for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
        m = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[a], y[a])
        oof[b] = np.maximum(m.predict(X.iloc[b]), FLOOR)
        print(f"  fold {i + 1}: RMSE {np.sqrt(np.mean((y[b] - oof[b]) ** 2)):,.0f}")
    cv = metrics(y, oof)
    q = np.quantile(oof, np.linspace(0, 1, RANGE_BANDS + 1))
    edges = [0.0] + q[1:-1].tolist()
    band = np.clip(np.searchsorted(edges, oof, side="right") - 1, 0, RANGE_BANDS - 1)
    ratio = y / oof
    low = [float(np.quantile(ratio[band == k], RANGE_COVER[0])) for k in range(RANGE_BANDS)]
    high = [float(np.quantile(ratio[band == k], RANGE_COVER[1])) for k in range(RANGE_BANDS)]

    # 2. Final model on the whole training split.
    t = time.perf_counter()
    model = lgb.LGBMRegressor(**PARAMS).fit(X, y)
    fit_s = time.perf_counter() - t
    (ROOT / "model").mkdir(exist_ok=True)
    model.booster_.save_model(str(ROOT / "model" / MODEL_FILE))

    # 3. One evaluation on the held-out 20%.
    pred = np.maximum(model.predict(Xt), FLOOR)
    test = metrics(yt, pred)
    tb = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, RANGE_BANDS - 1)
    lo, hi = np.maximum(pred * np.asarray(low)[tb], FLOOR), pred * np.asarray(high)[tb]
    test["range_coverage"] = float(np.mean((yt >= lo) & (yt <= hi)))
    test["range_width_median"] = float(np.median((hi - lo) / pred))
    by_band = []
    for label, a, b in zip(PRICE_LABELS, PRICE_BANDS[:-1], PRICE_BANDS[1:]):
        k = (yt >= a) & (yt < b)
        mm = metrics(yt[k], pred[k])
        by_band.append(dict(band=label, cars=int(k.sum()), mae=mm["mae"], mdape=mm["mdape"], within_20=mm["within_20"],
                            bias=mm["bias"], coverage=float(np.mean((yt[k] >= lo[k]) & (yt[k] <= hi[k])))))
    one = Xt.iloc[[0]]
    t = time.perf_counter()
    for _ in range(50):
        model.predict(one)
    predict_ms = (time.perf_counter() - t) / 50 * 1000
    sample = Xt.sample(5000, random_state=0)
    contrib = model.booster_.predict(sample, pred_contrib=True)[:, :-1]
    importance = dict(zip(FEATURES, np.abs(contrib).mean(axis=0).round(1).tolist()))
    print(f"held out: RMSE {test['rmse']:,.0f}  MAE {test['mae']:,.0f}  R2 {test['r2']:.3f}  "
          f"range coverage {test['range_coverage']:.1%}")

    comparison = json.loads((ROOT / "model" / "comparison.json").read_text(encoding="utf-8"))
    meta = dict(
        model="LightGBM gradient-boosted trees (regression, squared error)", target="Price (euros)",
        features=FEATURES, categorical=CATEGORICAL, categories=cats, params=PARAMS,
        range=dict(cover=RANGE_COVER, edges=edges, low_ratio=low, high_ratio=high),
        rows=dict(raw=len(raw), clean=len(d), train=len(tr), test=len(te)), cleaning=steps,
        price=dict(median=float(d["Price"].median()), mean=float(d["Price"].mean())),
        cv=cv, test=test, by_band=by_band, importance=importance, comparison=comparison,
        timing=dict(fit_s=round(fit_s, 1), predict_ms=round(predict_ms, 2)),
        versions=dict(lightgbm=lgb.__version__, pandas=pd.__version__, numpy=np.__version__), built=str(date.today()),
    )
    (ROOT / "model" / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # 4. App data: the held-out ads with the model's price, all clean ads for the market guide, form reference.
    pm = PriceModel()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    keep = CATEGORICAL + ["RegistrationYear", "RegistrationMonth", "Age", "Power", "Mileage", "Price"]
    held = te[keep].copy()
    held["fair"] = pm.predict(held)
    held["low"], held["high"] = pm.price_range(held["fair"].to_numpy())
    held = held.reset_index(drop=True)
    _slim(held).to_parquet(data / "listings.parquet", index=False)
    # SHAP values for a sample of held-out ads (TreeSHAP takes about 13 ms an ad, too slow to run in the app)
    sample = held.sample(8000, random_state=0)
    contrib = pm.explain(sample).add_suffix("_eur").rename(columns={"base_eur": "base"}).astype("float32")
    _slim(sample[CATEGORICAL + ["Power", "Mileage"]].copy()).join(contrib).reset_index(drop=True).to_parquet(
        data / "contributions.parquet", index=False)
    _slim(d[keep].copy()).to_parquet(data / "market.parquet", index=False)

    ref = {"brands": {}, "defaults": {}}
    for brand, g in d.groupby("Brand"):
        ref["brands"][brand] = g["Model"].value_counts().index.tolist()
    ref["brands"] = dict(sorted(ref["brands"].items(), key=lambda kv: -int((d["Brand"] == kv[0]).sum())))
    for (brand, mdl), g in d.groupby(["Brand", "Model"]):
        def top(col):
            v = g.loc[g[col] != "unknown", col]
            return v.mode().iat[0] if len(v) else "unknown"
        ref["defaults"][f"{brand}|{mdl}"] = dict(
            ads=len(g), vehicle_type=top("VehicleType"), gearbox=top("Gearbox"), fuel_type=top("FuelType"),
            power=int(g["Power"].median()) if g["Power"].notna().any() else None,
            year=int(g["RegistrationYear"].median()))
    (data / "reference.json").write_text(json.dumps(ref, indent=1), encoding="utf-8")
    print("saved model/ and data/")


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in CATEGORICAL:
            df[c] = df[c].astype("category")
        elif c in ("RegistrationYear", "RegistrationMonth"):
            df[c] = df[c].astype("int16")
        else:
            df[c] = df[c].astype("float32")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "car_data.csv")
    main(ap.parse_args().csv)
