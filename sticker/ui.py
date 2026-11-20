"""FairTag components: a dealer's window-sticker look. Every block sits in its own white panel with a black header bar
(condensed capitals and an orange number tag), figures in Barlow, the fair price on an orange tag, a range bar with the
asking price marked, and the price build-up as itemised sticker lines."""

from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

from sticker import theme as t
from sticker.data import eur
from sticker.site import CODE_URL, DATA_CREDIT, LEARN_URL, NOTEBOOK_URL

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600;700&display=swap');
:root {{ --bg:{t.BG}; --panel:{t.PANEL}; --line:{t.LINE}; --ink:{t.INK}; --muted:{t.MUTED}; --faint:{t.FAINT};
         --bar:{t.BAR}; --tag:{t.TAG}; --tag-dark:{t.TAG_DARK}; --tag-soft:{t.TAG_SOFT}; --head:{t.HEAD}; --body:{t.BODY}; }}
html, body, [data-testid="stAppViewContainer"] {{ font-family: var(--body); color: var(--ink); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: var(--bg); }}
.block-container {{ padding-top: 1.4rem; max-width: 1240px; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.msr {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; display: inline-block;
        -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }}
.num {{ font-variant-numeric: tabular-nums; }}

/* Bordered containers are the sticker panels: square-ish corners, a firm outline */
[data-testid="stVerticalBlock"][data-test-scroll-behavior] {{ border: 1.5px solid var(--ink) !important; border-radius: 4px !important;
    background: var(--panel); box-shadow: 3px 3px 0 rgba(22, 24, 29, .10); }}

/* Top bar: logo block and page tabs */
.brandbar {{ display: flex; align-items: center; gap: .7rem; padding: .2rem 0; }}
.brandbar .mark {{ width: 2.6rem; height: 2.6rem; }}
.brandbar .name {{ font-family: var(--head); font-size: 1.9rem; font-weight: 700; letter-spacing: .02em; line-height: 1; margin: 0; }}
.brandbar .name b {{ color: var(--tag); font-weight: 700; }}
.brandbar .tagline {{ font-size: .76rem; color: var(--muted); margin: .1rem 0 0; letter-spacing: .04em; text-transform: uppercase; font-weight: 600; }}
[data-testid="stPageLink"] a {{ border: 1.5px solid var(--line); border-radius: 4px; background: var(--panel); padding: .35rem .4rem;
    justify-content: center; }}
[data-testid="stPageLink"] a p {{ font-family: var(--head); font-size: .98rem; white-space: nowrap; font-weight: 600; letter-spacing: .03em; text-transform: uppercase; }}
[data-testid="stPageLink"] a:hover {{ border-color: var(--ink); background: var(--panel); }}
[data-testid="stPageLink"] a[aria-current="page"], [data-testid="stPageLink-NavLink"][aria-current="page"] {{ background: var(--ink); border-color: var(--ink); }}
[data-testid="stPageLink"] a[aria-current="page"] p, [data-testid="stPageLink"] a[aria-current="page"] span {{ color: #fff !important; }}
.rule {{ height: 3px; background: var(--ink); margin: .6rem 0 1.3rem; border-radius: 2px; }}

/* Page head */
.kicker {{ display: inline-block; font-family: var(--head); font-size: .95rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
           color: #fff; background: var(--tag); border-radius: 3px; padding: .1rem .55rem; margin: 0 0 .5rem; }}
.title {{ font-family: var(--head); font-size: 2.6rem; font-weight: 700; margin: 0; line-height: 1.05; color: var(--ink); }}
.lead {{ color: var(--muted); font-size: 1.02rem; line-height: 1.6; max-width: 820px; margin: .55rem 0 1.2rem; }}
.lead b {{ color: var(--ink); font-weight: 600; }}

/* Panel header bar, stretched to the panel edges */
.ph {{ display: flex; align-items: center; gap: .7rem; background: var(--bar); color: #fff; margin: -1rem -1rem .9rem;
       padding: .5rem 1rem; border-radius: 2px 2px 0 0; }}
.ph-n {{ flex: 0 0 auto; background: var(--tag); color: #fff; font-family: var(--head); font-weight: 700; font-size: .95rem;
         padding: .05rem .5rem; border-radius: 3px; clip-path: polygon(0 0, 85% 0, 100% 50%, 85% 100%, 0 100%); padding-right: .8rem; }}
.ph-t {{ font-family: var(--head); font-size: 1.2rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; margin: 0; line-height: 1.2; }}
.ph-s {{ font-size: .88rem; color: var(--muted); margin: -.4rem 0 .8rem; line-height: 1.5; }}

/* Stat tiles */
.stats {{ display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: .7rem; margin: 0 0 1rem; }}
@media (max-width: 820px) {{ .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.stat {{ background: var(--panel); border: 1.5px solid var(--ink); border-radius: 4px; padding: .7rem .9rem; box-shadow: 3px 3px 0 rgba(22,24,29,.10); }}
.stat-l {{ font-family: var(--head); font-size: .9rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); margin: 0; }}
.stat-v {{ font-size: 1.8rem; font-weight: 700; margin: .1rem 0; color: var(--ink); line-height: 1.1; font-variant-numeric: tabular-nums; }}
.stat-n {{ font-size: .8rem; color: var(--faint); margin: 0; line-height: 1.4; }}

/* The price tag */
.tagrow {{ display: grid; grid-template-columns: auto 1fr; gap: 1.4rem; align-items: center; }}
@media (max-width: 760px) {{ .tagrow {{ grid-template-columns: 1fr; }} }}
.ptag {{ position: relative; background: var(--tag); color: #fff; padding: .8rem 1.4rem .8rem 2.3rem; border-radius: 6px;
         clip-path: polygon(1.1rem 0, 100% 0, 100% 100%, 1.1rem 100%, 0 50%); min-width: 13rem; }}
.ptag::before {{ content: ""; position: absolute; left: .5rem; top: 50%; width: .55rem; height: .55rem; margin-top: -.275rem;
                 border-radius: 50%; background: var(--panel); }}
.ptag-l {{ font-family: var(--head); font-size: .9rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin: 0; opacity: .92; }}
.ptag-v {{ font-size: 2.7rem; font-weight: 700; margin: .05rem 0 0; line-height: 1; font-variant-numeric: tabular-nums; }}
.ptag-r {{ font-size: .88rem; margin: .3rem 0 0; font-weight: 500; }}
.car-t {{ font-family: var(--head); font-size: 1.55rem; font-weight: 700; margin: 0; line-height: 1.15; }}
.car-s {{ font-size: .9rem; color: var(--muted); margin: .3rem 0 0; line-height: 1.55; }}

/* Range bar */
.rbar {{ position: relative; margin: 2.4rem .4rem 2.6rem; height: 12px; border-radius: 6px;
         background: linear-gradient(90deg, {t.BELOW_SOFT} 0 var(--l), #dee2e6 var(--l) var(--h), {t.ABOVE_SOFT} var(--h) 100%);
         border: 1px solid var(--line); }}
.rbar .band {{ position: absolute; top: -1px; bottom: -1px; left: var(--l); width: calc(var(--h) - var(--l)); background: #adb5bd; border-radius: 3px; }}
.rbar .pin {{ position: absolute; top: 50%; transform: translate(-50%, -50%); }}
.rbar .pin i {{ display: block; width: 4px; height: 30px; border-radius: 2px; background: var(--c); }}
.rbar .pin span {{ position: absolute; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: .8rem; font-weight: 600; color: var(--ink); }}
.rbar .pin.up span {{ bottom: 34px; }} .rbar .pin.down span {{ top: 34px; color: var(--muted); font-weight: 500; }}
.rbar .fair i {{ width: 18px; height: 18px; border-radius: 50%; background: var(--tag); border: 3px solid #fff; box-shadow: 0 0 0 2px var(--tag); }}
.rbar-ends {{ display: flex; justify-content: space-between; font-size: .76rem; color: var(--faint); margin: -1.9rem .4rem 0; }}

/* Verdict tag */
.verdict {{ display: inline-flex; align-items: center; gap: .4rem; border-radius: 4px; padding: .3rem .7rem; font-weight: 700;
            font-size: .92rem; color: var(--c); background: var(--s); border: 1.5px solid var(--c); }}
.verdict .msr {{ font-size: 1.15rem; }}
.verdict-t {{ margin: .6rem 0 0; color: var(--ink); line-height: 1.55; font-size: .95rem; }}

/* Itemised build-up */
.items {{ font-size: .95rem; margin: 0; }}
.it {{ display: grid; grid-template-columns: 1fr auto; gap: .6rem; align-items: baseline; padding: .32rem 0; border-bottom: 1px dotted var(--line); }}
.it-l {{ display: flex; gap: .4rem; align-items: baseline; min-width: 0; }}
.it-l b {{ font-weight: 600; white-space: nowrap; }}
.it-l span {{ color: var(--muted); font-size: .86rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.it-v {{ font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.it-v.up {{ color: {t.UP}; }} .it-v.down {{ color: {t.DOWN}; }}
.it-bar {{ grid-column: 1 / -1; height: 5px; border-radius: 3px; background: var(--c); width: var(--w); margin-top: .15rem; opacity: .75; }}
.it.base, .it.total {{ border-bottom: none; }}
.it.base {{ border-bottom: 1.5px solid var(--ink); }}
.it.total {{ border-top: 3px double var(--ink); margin-top: .2rem; padding-top: .5rem; font-size: 1.08rem; }}
.it.total .it-v {{ color: var(--tag-dark); font-size: 1.2rem; }}

/* Deal cards */
.deals {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .7rem; }}
.deal {{ border: 1.5px solid var(--line); border-left: 6px solid var(--c); border-radius: 4px; padding: .65rem .8rem; background: var(--panel); }}
.deal-t {{ font-family: var(--head); font-size: 1.15rem; font-weight: 700; margin: 0; line-height: 1.2; }}
.deal-s {{ font-size: .82rem; color: var(--muted); margin: .15rem 0 .45rem; }}
.deal-p {{ display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; font-variant-numeric: tabular-nums; }}
.deal-p b {{ font-size: 1.35rem; }} .deal-p span {{ font-size: .82rem; color: var(--muted); }}
.deal-g {{ font-size: .85rem; font-weight: 700; color: var(--c); margin: .25rem 0 0; }}

/* Notes, cautions, spec sheet */
.note {{ background: var(--tag-soft); border-left: 4px solid var(--tag); border-radius: 0 4px 4px 0; padding: .6rem .85rem; margin: .5rem 0 .2rem;
         font-size: .92rem; line-height: 1.6; color: var(--ink); }}
.note p {{ margin: 0 0 .4rem; }} .note p:last-child {{ margin: 0; }}
.caution {{ display: flex; gap: .55rem; align-items: flex-start; background: #fff9db; border: 1px solid #f0d36b; border-radius: 4px;
            padding: .55rem .8rem; font-size: .9rem; line-height: 1.55; margin: .5rem 0; }}
.caution .msr {{ color: #946200; font-size: 1.2rem; margin-top: .05rem; }}
.sheet {{ display: grid; grid-template-columns: max-content 1fr; gap: .1rem 1.2rem; font-size: .92rem; margin: 0; }}
.sheet dt {{ font-family: var(--head); font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); font-size: .9rem; }}
.sheet dd {{ margin: 0; font-weight: 500; }}
.foot {{ margin: 2.2rem 0 .5rem; padding-top: 1rem; border-top: 3px solid var(--ink); font-size: .82rem; color: var(--muted); line-height: 1.7; }}
.foot a {{ color: var(--tag-dark); font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def icon(name: str) -> str:
    return f'<span class="msr" aria-hidden="true">{name}</span>'


def page(kicker: str, title: str, lead: str) -> None:
    st.html(f'<p class="kicker">{html.escape(kicker)}</p><h1 class="title">{html.escape(title)}</h1><p class="lead">{lead}</p>')


@contextmanager
def panel(num: str, title: str, sub: str = ""):
    """A sticker panel: black header bar with an orange number tag, then the content."""
    with st.container(border=True):
        st.html(f'<div class="ph"><span class="ph-n">{html.escape(num)}</span><p class="ph-t">{html.escape(title)}</p></div>'
                + (f'<p class="ph-s">{sub}</p>' if sub else ""))
        yield


def stats(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div class="stat"><p class="stat-l">{html.escape(a)}</p><p class="stat-v">{b}</p><p class="stat-n">{c}</p></div>'
                    for a, b, c in items)
    st.html(f'<div class="stats" style="--n:{len(items)}">{cells}</div>')


def verdict_tag(verdict: str) -> str:
    c, s, ic = t.VERDICT_STYLE[verdict]
    return f'<span class="verdict" style="--c:{c};--s:{s}">{icon(ic)}{html.escape(verdict)}</span>'


def price_tag(fair: float, low: float, high: float, car: str, specs: str) -> None:
    st.html(f'<div class="tagrow"><div class="ptag"><p class="ptag-l">Fair asking price</p><p class="ptag-v">{eur(fair)}</p>'
            f'<p class="ptag-r">Range {eur(low)} to {eur(high)}</p></div>'
            f'<div><p class="car-t">{html.escape(car)}</p><p class="car-s">{specs}</p></div></div>')


def range_bar(fair: float, low: float, high: float, ask: float | None = None) -> None:
    """The range on a scale from half the low end to 1.4x the high end; the fair price and the asking price pinned."""
    lo_s, hi_s = low * 0.5, high * 1.4
    if ask:
        lo_s, hi_s = min(lo_s, ask * 0.9), max(hi_s, ask * 1.08)

    def x(v):
        return max(0.0, min(100.0, (v - lo_s) / (hi_s - lo_s) * 100))

    pins = f'<div class="pin fair up" style="left:{x(fair):.2f}%"><i></i><span>Fair {eur(fair)}</span></div>'
    pins += f'<div class="pin down" style="left:{x(low):.2f}%;--c:#868e96"><i></i><span>{eur(low)}</span></div>'
    pins += f'<div class="pin down" style="left:{x(high):.2f}%;--c:#868e96"><i></i><span>{eur(high)}</span></div>'
    if ask:
        c = t.VERDICT_STYLE[_verdict(ask, low, high)][0]
        side = "up" if abs(x(ask) - x(fair)) > 14 else "down"
        if side == "down":
            pins = pins.replace(f'<span>{eur(low)}</span>', '<span></span>').replace(f'<span>{eur(high)}</span>', '<span></span>')
        pins += f'<div class="pin {side}" style="left:{x(ask):.2f}%;--c:{c}"><i></i><span style="color:{c}">Asking {eur(ask)}</span></div>'
    st.html(f'<div class="rbar" style="--l:{x(low):.2f}%;--h:{x(high):.2f}%"><div class="band"></div>{pins}</div>')


def _verdict(ask: float, low: float, high: float) -> str:
    return "Below the range" if ask < low else "Above the range" if ask > high else "Within the range"


def buildup(base: float, lines: list[tuple[str, str, float]], total: float) -> None:
    """Itemised sticker: the average ad, one line per feature (label, the car's value, euros), the fair price."""
    biggest = max([abs(v) for _, _, v in lines] + [1.0])
    rows = [f'<div class="it base"><div class="it-l"><b>Average ad in the data</b><span>starting point</span></div>'
            f'<div class="it-v num">{eur(base)}</div></div>']
    for label, value, euros in lines:
        cls, sign, c = ("up", "+", t.UP) if euros >= 5 else ("down", "−", t.DOWN) if euros <= -5 else ("", "", t.FAINT)
        rows.append(f'<div class="it"><div class="it-l"><b>{html.escape(label)}</b><span>{html.escape(value)}</span></div>'
                    f'<div class="it-v {cls} num">{sign}{eur(abs(euros), 10)}</div>'
                    f'<div class="it-bar" style="--c:{c};--w:{abs(euros) / biggest * 100:.1f}%"></div></div>')
    rows.append(f'<div class="it total"><div class="it-l"><b>Fair asking price</b></div><div class="it-v num">{eur(total)}</div></div>')
    st.html(f'<div class="items">{"".join(rows)}</div>')


def deal_cards(cards: list[dict]) -> None:
    out = []
    for c in cards:
        col = t.VERDICT_STYLE[c["verdict"]][0]
        out.append(f'<div class="deal" style="--c:{col}"><p class="deal-t">{html.escape(c["title"])}</p>'
                   f'<p class="deal-s">{html.escape(c["specs"])}</p><div class="deal-p"><b>{eur(c["ask"], 1)}</b>'
                   f'<span>fair {eur(c["fair"])} · range {eur(c["low"])} to {eur(c["high"])}</span></div>'
                   f'<p class="deal-g">{icon(t.VERDICT_STYLE[c["verdict"]][2])} {html.escape(c["gap"])}</p></div>')
    st.html(f'<div class="deals">{"".join(out)}</div>')


def sheet(rows: list[tuple[str, str]]) -> None:
    st.html('<dl class="sheet">' + "".join(f"<dt>{html.escape(a)}</dt><dd>{b}</dd>" for a, b in rows) + "</dl>")


def note(*paras: str) -> None:
    st.html('<div class="note">' + "".join(f"<p>{p}</p>" for p in paras) + "</div>")


def caution(text: str) -> None:
    st.html(f'<div class="caution">{icon("info")}<div>{text}</div></div>')


def footer() -> None:
    st.html(f'<div class="foot">Model and app by <b>Ahmad Hassan</b>, AI Engineer · <a href="{NOTEBOOK_URL}" target="_blank">'
            f'Notebook</a> · <a href="{CODE_URL}" target="_blank">App code</a> · <a href="{LEARN_URL}" target="_blank">'
            f"Learn with Ahmad</a><br>{DATA_CREDIT}</div>")
