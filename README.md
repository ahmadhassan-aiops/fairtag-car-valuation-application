# FairTag: used-car valuation application

**Live app:** [fairtag-car-valuation-application.streamlit.app](https://fairtag-car-valuation-application.streamlit.app/) · **Notebook and project folder:** [AIOPS-Portfolio-Projects](https://github.com/ahmadhassan-aiops/AIOPS-Portfolio-Projects/tree/main/2.%20Machine%20Learning%20Projects/Rusty%20Bargain%20Company%20ML%20Model%20To%20Attract%20New%20Buyers%20For%20The%20Used%20Cars)

What should a used car be listed for? FairTag, by **Ahmad Hassan** (AI Engineer), answers in seconds for the
Rusty Bargain brief: a buyer or seller describes a car and gets its **fair asking price**, the **range** that holds
8 in 10 real asking prices for cars like it, and an **itemised breakdown** of what each detail adds or takes off.
A LightGBM model, trained on 247,442 German ads and tested once on 61,861 it never saw, has a typical error (RMSE)
of **€1,502**, explains **89%** of price variation (R² 0.891), and its range holds **80.3%** of real asking prices.
It trains in about 16 seconds and prices a car in about 6 ms.

- **Value a car**: describe the car (brand, model, registration, mileage, power, body, gearbox, fuel, damage), get
  the price on an orange tag with its range, place an asking price against it, read the itemised build-up (SHAP
  values in euros that add up exactly to the price), see how the price changes with age and mileage, and compare
  with similar held-out ads.
- **Deal finder**: filter 61,861 held-out ads by brand, model, budget, age, mileage and gearbox; see the ads asking
  less than their range (realistic bargains apart from ads more than half off), an asking-against-fair chart, the full
  sortable list and a CSV download. Any ad opens in Value a car.
- **Market guide**: how prices fall with age by brand, what each detail is worth on average, which details matter
  most, and a brand league table.
- **Model report**: held-out accuracy overall and by price band, fair against asking price, how the range is built
  and how often it holds, and the six-model comparison on quality, speed and training time.
- **Data & method**: the data, every cleaning step with its count, the features, training, limits.

## The model

The project notebook was audited and rebuilt before the app. The original notebook reported an RMSE of about 980,
which came from test data leaking into training; the honest figure for this data is about €1,500. The rebuild also
fixed the fuel labels (the source calls diesel cars "gasoline"), removed reposted ads, placeholder prices and
impossible registration years, and removed ads under a year old with more than 50,000 km (the ad year given as the
registration year). Details are in the notebook.

| Model (5-fold cross-validation, default settings) | RMSE | R² | Fit time per fold | Predict one car |
|---|---|---|---|---|
| Linear regression (sanity check) | €2,631 | 0.668 | 2 s | 9 ms |
| Decision tree | €1,832 | 0.839 | 2 s | 7 ms |
| Random forest | €1,590 | 0.879 | 18 s | 58 ms |
| **LightGBM** | **€1,518** | **0.889** | **17 s** | **7 ms** |
| XGBoost | €1,540 | 0.886 | 40 s | 14 ms |
| CatBoost | €1,548 | 0.885 | 344 s | 2 ms |

LightGBM was then tuned by random search with cross-validation. The deployed model (512 trees, 255 leaves, learning
rate 0.08) uses brand, model and the other categories natively, leaves missing power as missing, and predicts the raw
price in euros (log price was tested and scored worse on RMSE). The range comes from cross-validated errors: in 10
bands of predicted price, the 10th to 90th percentile of real/predicted price.

## Why this stack

| Need | Choice |
|---|---|
| Fast, accurate prices from 40 brands and 250 models | **LightGBM** with native categorical splits: best RMSE of six models, trains in seconds |
| Explanations a buyer understands | **TreeSHAP** from LightGBM (`pred_contrib`), shown as euro line items that sum to the price |
| An honest "how sure" | A **range** from cross-validated errors, checked on held-out ads (80.3% coverage for an 80% range) |
| Reject impossible inputs | **pydantic** schema (mileage steps, registration not after March 2016, power 20 to 600 hp) |
| Trust that the shipped file is the evaluated model | **pytest** + GitHub Actions: saved predictions, SHAP additivity, range, schema, cleaning, every page |

`fairtag/` has no Streamlit dependency, so the same model can sit behind an API for a listings site.

## How it is built

```
app.py              entry point (custom top bar, hidden native navigation)
nav.py              page registry
views/              value, deals, market, report, method
fairtag/            schema.py (pydantic), prep.py (cleaning), model.py (price, range, SHAP, verdict)
sticker/            window-sticker UI kit (numbered panels, price tag, range bar, itemised build-up), charts, theme
model/              fairtag_lightgbm.txt + metadata.json + comparison.json
data/               held-out ads with fair price and range, all clean ads, SHAP sample, form reference
train.py            rebuilds model/ and data/ from the CSV
tests/              model and page tests
```

## Run locally

```bash
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
python train.py --csv "path/to/car_data.csv"   # rebuild the model and data
```

`packages.txt` installs `libgomp1`, which LightGBM needs on Linux hosts such as Streamlit Community Cloud.

## Data and limits

354,369 ads from a German used-car classifieds site (March 2014 to April 2016, nearly all March 2016), prepared as the
Rusty Bargain training case from a public Kaggle dataset; 309,303 remain after cleaning. Prices are **asking** prices,
not sale prices, in one market at one moment. Ads do not record condition, accident history or equipment, which is why
the range is wide. FairTag is a portfolio demonstration, not a valuation service.
