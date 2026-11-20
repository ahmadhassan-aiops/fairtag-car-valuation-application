"""The saved model, its range and explanations, the input schema and the cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fairtag.model import VERDICTS, PriceModel
from fairtag.prep import clean
from fairtag.schema import FEATURES, Car

DATA = pd.read_parquet("data/listings.parquet")


@pytest.fixture(scope="module")
def m() -> PriceModel:
    return PriceModel()


def test_predictions_match_saved_listings(m):
    s = DATA.sample(500, random_state=1)
    assert np.allclose(m.predict(s), s["fair"], rtol=1e-4)


def test_explanation_adds_up(m):
    s = DATA.sample(200, random_state=2)
    c = m.explain(s)
    raw = m.booster.predict(m.frame(s))
    assert np.allclose(c.sum(axis=1), raw, atol=1e-3)
    assert list(c.columns) == FEATURES + ["base"]


def test_range_brackets_the_price(m):
    fair = np.array([150.0, 800, 3000, 9000, 19000])
    lo, hi = m.price_range(fair)
    assert np.all(lo <= fair) and np.all(fair <= hi) and np.all(lo >= 100)


def test_verdicts(m):
    assert list(m.verdict([50, 500, 5000], [100, 100, 100], [1000, 1000, 1000])) == VERDICTS


def test_reported_quality(m):
    t = m.meta["test"]
    assert t["rmse"] < 1600 and t["r2"] > 0.87 and 0.75 < t["range_coverage"] < 0.85


def test_car_schema_and_prediction(m):
    car = Car(brand="volkswagen", model="golf", registration_year=2008, registration_month=5, mileage=125000, power=105,
              gearbox="manual", fuel_type="petrol", vehicle_type="sedan", not_repaired="no")
    assert 6 < car.age < 9
    price = m.predict(car.frame())[0]
    assert 2000 < price < 8000
    older = Car(**{**car.model_dump(), "registration_year": 1999})
    assert m.predict(older.frame())[0] < price


def test_unknown_values_still_price(m):
    car = Car(brand="volkswagen", model="not_a_model", registration_year=2010, power=None)
    assert m.predict(car.frame())[0] >= 100


@pytest.mark.parametrize("bad", [dict(mileage=12345), dict(registration_year=2016, registration_month=7),
                                 dict(registration_year=1900), dict(power=5)])
def test_schema_rejects(bad):
    with pytest.raises(ValidationError):
        Car(**{"brand": "bmw", "model": "3er", "registration_year": 2005, **bad})


def test_clean():
    raw = pd.DataFrame({
        "DateCrawled": ["24/03/2016 11:52", "25/03/2016 11:52", "24/03/2016 11:52", "24/03/2016 11:52"],
        "Price": [5000, 5000, 50, 3000], "VehicleType": ["sedan", "sedan", None, None],
        "RegistrationYear": [2010, 2010, 2005, 2005], "Gearbox": ["manual"] * 4, "Power": [0, 0, 100, 900],
        "Model": ["golf", "golf", "polo", None], "Mileage": [150000] * 4, "RegistrationMonth": [0, 0, 3, 3],
        "FuelType": ["gasoline", "gasoline", "petrol", "petrol"], "Brand": ["volkswagen"] * 4, "NotRepaired": [None] * 4,
        "DateCreated": ["24/03/2016 00:00"] * 4, "NumberOfPictures": [0] * 4, "PostalCode": [1] * 4,
        "LastSeen": ["01/04/2016 00:00"] * 4})
    rep = {}
    d = clean(raw, rep)
    assert len(d) == 2 and rep["duplicates"] == 1 and rep["low_price"] == 1
    assert d.loc[0, "FuelType"] == "diesel" and d["Power"].isna().all()
    assert d.loc[1, "Model"] == "unknown" and abs(d.loc[0, "Age"] - (6 - 3 / 12)) < 1e-9
