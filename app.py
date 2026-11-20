"""
FairTag: what a used car should be listed for. Price a car with a range and an itemised breakdown, find ads priced
below their range, read the market, and check how accurate the model is. Model in model/ (built by train.py), code in
fairtag/.
"""

import base64
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FairTag | Used-car valuation | Ahmad Hassan", page_icon=":material/sell:",
                   layout="wide", initial_sidebar_state="collapsed")

from nav import PAGES  # noqa: E402
from sticker.ui import inject_css  # noqa: E402

LOGO = base64.b64encode((Path(__file__).parent / "assets" / "logo.svg").read_bytes()).decode()

inject_css()
current = st.navigation(PAGES, position="hidden")
cols = st.columns([1.35, 1, 1, 1.05, 1.05, 1.1], vertical_alignment="center")
cols[0].html(f'<div class="brandbar"><img class="mark" src="data:image/svg+xml;base64,{LOGO}" alt=""><div><p class="name">Fair<b>Tag</b></p>'
             '<p class="tagline">Used-car valuation</p></div></div>')
for col, p in zip(cols[1:], PAGES):
    col.page_link(p, label=p.title, icon=p.icon, width="stretch")
k = [p.title for p in PAGES].index(current.title) + 2
st.html(f'<style>[data-testid="stHorizontalBlock"]:has(.brandbar) > [data-testid="stColumn"]:nth-child({k}) a '
        '{ background: var(--ink); border-color: var(--ink); } [data-testid="stHorizontalBlock"]:has(.brandbar) > '
        f'[data-testid="stColumn"]:nth-child({k}) a * {{ color: #fff !important; }}</style><div class="rule"></div>')
current.run()
