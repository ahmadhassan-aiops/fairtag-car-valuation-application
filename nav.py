"""Page registry (custom top bar, native navigation hidden), shared by app.py and pages that link to other pages."""

import streamlit as st

from views import deals, market, method, report, value

VALUE = st.Page(value.render, title="Value a car", icon=":material/sell:", url_path="value", default=True)
DEALS = st.Page(deals.render, title="Deal finder", icon=":material/manage_search:", url_path="deals")
MARKET = st.Page(market.render, title="Market guide", icon=":material/insights:", url_path="market")
REPORT = st.Page(report.render, title="Model report", icon=":material/speed:", url_path="report")
METHOD = st.Page(method.render, title="Data & method", icon=":material/description:", url_path="method")

PAGES = [VALUE, DEALS, MARKET, REPORT, METHOD]
