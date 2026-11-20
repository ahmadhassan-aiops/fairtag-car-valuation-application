"""The car description the model needs, validated with pydantic, and the conversion to a model row."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

CATEGORICAL = ["Brand", "Model", "VehicleType", "Gearbox", "FuelType", "NotRepaired"]
NUMERIC = ["Age", "Power", "Mileage"]
FEATURES = CATEGORICAL + NUMERIC

# The ads were placed between March 2014 and April 2016, almost all in March 2016. A car's age is measured to the
# month the ad went online; new valuations use March 2016, the market the model learned.
MARKET_YEAR, MARKET_MONTH = 2016, 3

# The site records mileage in fixed steps; 150,000 means "150,000 km or more".
MILEAGE_STEPS = [5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 125000, 150000]

VEHICLE_TYPES = ["sedan", "small", "wagon", "bus", "convertible", "coupe", "suv", "other", "unknown"]
GEARBOXES = ["manual", "auto", "unknown"]
FUEL_TYPES = ["petrol", "diesel", "lpg", "cng", "hybrid", "electric", "other", "unknown"]
REPAIRED = ["no", "yes", "unknown"]

FEATURE_LABELS = {
    "Brand": "Brand", "Model": "Model", "VehicleType": "Body type", "Gearbox": "Gearbox", "FuelType": "Fuel",
    "NotRepaired": "Unrepaired damage", "Age": "Age", "Power": "Power", "Mileage": "Mileage",
}


def car_age(reg_year, reg_month, ad_year, ad_month):
    """Age in years at the time of the ad. Month 0 means 'not given' and counts as mid-year. Never below 0."""
    reg_month = np.where(np.asarray(reg_month) == 0, 6, reg_month)
    age = np.asarray(ad_year - np.asarray(reg_year), dtype=float) + (np.asarray(ad_month) - reg_month) / 12
    return np.clip(age, 0, None)


class Car(BaseModel):
    """One used car as a seller would describe it in an ad."""

    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)
    vehicle_type: Literal[tuple(VEHICLE_TYPES)] = "unknown"
    gearbox: Literal[tuple(GEARBOXES)] = "unknown"
    fuel_type: Literal[tuple(FUEL_TYPES)] = "unknown"
    not_repaired: Literal[tuple(REPAIRED)] = "unknown"
    registration_year: int = Field(ge=1950, le=MARKET_YEAR)
    registration_month: int = Field(default=0, ge=0, le=12, description="0 = not given")
    power: int | None = Field(default=None, ge=20, le=600, description="Horsepower; None if not known")
    mileage: int = 150000

    @field_validator("mileage")
    @classmethod
    def _mileage_step(cls, v: int) -> int:
        if v not in MILEAGE_STEPS:
            raise ValueError(f"mileage must be one of {MILEAGE_STEPS}")
        return v

    @field_validator("registration_month")
    @classmethod
    def _not_future(cls, v: int, info) -> int:
        if info.data.get("registration_year") == MARKET_YEAR and v > MARKET_MONTH:
            raise ValueError("a car registered after March 2016 is newer than the market data")
        return v

    @property
    def age(self) -> float:
        return float(car_age(self.registration_year, self.registration_month, MARKET_YEAR, MARKET_MONTH))

    def row(self) -> dict:
        return {"Brand": self.brand, "Model": self.model, "VehicleType": self.vehicle_type, "Gearbox": self.gearbox,
                "FuelType": self.fuel_type, "NotRepaired": self.not_repaired, "Age": self.age,
                "Power": float("nan") if self.power is None else float(self.power), "Mileage": float(self.mileage)}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.row()])[FEATURES]
