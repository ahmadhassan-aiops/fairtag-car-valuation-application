"""Every page renders without an exception, including the main interactions."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/value.py", "views/deals.py", "views/market.py", "views/report.py", "views/method.py"]


def run(module: str) -> AppTest:
    script = f"""
import importlib
from sticker.ui import inject_css
inject_css()
importlib.import_module("{module.replace('/', '.').removesuffix('.py')}").render()
"""
    return AppTest.from_string(script, default_timeout=90).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, at.exception


def test_value_with_asking_price_and_brand_change():
    at = run("views/value.py")
    at.number_input(key="in_ask").set_value(1500).run()
    assert not at.exception
    assert any("Below the range" in h.proto.body or "range" in h.proto.body for h in at.get("html"))
    at.selectbox(key="in_brand").set_value("bmw").run()
    assert not at.exception
    assert at.selectbox(key="in_model").value == "3er"
    at.toggle(key="in_power_known").set_value(False).run()
    assert not at.exception


def test_value_rejects_future_registration():
    at = run("views/value.py")
    at.number_input(key="in_year").set_value(2016).run()
    at.selectbox(key="in_month").set_value(9).run()
    assert at.error and "newer than the market data" in at.error[0].value


def test_deals_filters():
    at = run("views/deals.py")
    at.multiselect(key="d_brands").set_value(["bmw"]).run()
    assert not at.exception
    at.multiselect(key="d_models").set_value(["3er"]).run()
    at.segmented_control(key="d_gear").set_value("Automatic").run()
    assert not at.exception
    at.slider(key="d_budget").set_value((100, 300)).run()
    assert not at.exception


@pytest.mark.parametrize("detail", ["Gearbox", "Fuel", "Body type", "Mileage", "Power", "Brand"])
def test_market_details(detail):
    at = run("views/market.py")
    at.segmented_control(key="m_detail").set_value(detail).run()
    assert not at.exception
