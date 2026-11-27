"""Shared dark console theme and small HTML components."""

from __future__ import annotations

from html import escape
from typing import Dict, Optional, Sequence, Tuple, Union

import streamlit as st

INK, PAPER, LINE = "#EEF3FC", "#070A11", "#202A3D"
MUTED, GREEN, BLUE, CORAL, AMBER = "#7187A8", "#00D6C2", "#55C8FF", "#FF63C7", "#FFC400"

_CSS = """
<style>
:root {
  color-scheme: dark;
  --ms-bg:#070A11; --ms-panel:#0E1524; --ms-panel-deep:#090F1D;
  --ms-border:#202A3D; --ms-text:#EEF3FC; --ms-muted:#7187A8;
  --ms-teal:#00D6C2; --ms-blue:#55C8FF; --ms-pink:#D879FF; --ms-amber:#FFC400;
}
.stApp { background:var(--ms-bg); color:var(--ms-text); }
.block-container { max-width:1540px; padding:0 2.1rem 3rem; }
footer { visibility:hidden; }
header[data-testid="stHeader"] { height:0; background:transparent; }
[data-testid="stToolbar"] { right:1rem; }
[data-testid="stSidebar"] { display:none; }
.stApp, .stApp p, .stApp label, .stApp li { color:var(--ms-text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  color:var(--ms-text); font-family:Inter, "Segoe UI", Arial, sans-serif;
  letter-spacing:-.035em; font-weight:650;
}
.stApp h1 { font-size:2.6rem; line-height:1.12; padding:0; margin:.2rem 0 .55rem; }
.stApp h2 { font-size:1.65rem; }
.stApp h3 { font-size:1.2rem; }
.stApp [data-testid="stCaptionContainer"], .stApp small { color:var(--ms-muted); }
a { color:var(--ms-teal) !important; }

.st-key-ms-topbar {
  position:sticky; top:0; z-index:999; margin:0 -2.1rem 1.25rem;
  padding:.65rem 2.1rem .6rem; background:rgba(7,10,17,.97);
  border-bottom:1px solid #1A2231; backdrop-filter:blur(16px);
}
.ms-brand { display:flex; align-items:center; gap:.7rem; min-height:2.5rem; white-space:nowrap; }
.ms-brand-mark { display:grid; place-items:center; width:2.3rem; height:2.3rem; border:1px solid #075D5A;
  border-radius:50%; background:#092125; color:var(--ms-teal); font-size:1.2rem; }
.ms-brand-name { font-size:.93rem; line-height:1.15; font-weight:750; color:var(--ms-text); }
.ms-brand-sub { display:block; margin-top:.15rem; font-size:.54rem; font-weight:700;
  letter-spacing:.18em; color:#536783; }
.ms-nav-current { display:inline-flex; align-items:center; min-height:2.1rem; padding:0 .8rem;
  border-radius:999px; background:#1C293E; color:#F2F6FD; font-size:.77rem; font-weight:650; white-space:nowrap; }
.st-key-ms-topbar [data-testid="stPageLink"] { min-height:2.1rem; }
.st-key-ms-topbar [data-testid="stPageLink"] a { display:inline-flex; align-items:center; min-height:2.1rem;
  padding:0 .55rem; border-radius:999px; color:#7187A8 !important; font-size:.77rem;
  font-weight:550; white-space:nowrap; text-decoration:none; }
.st-key-ms-topbar [data-testid="stPageLink"] a:hover { color:#EEF3FC !important; background:#121B2B; }
.ms-engine { display:flex; align-items:center; justify-content:flex-end; gap:.45rem; min-height:2.1rem;
  color:#7187A8; font-size:.61rem; font-weight:750; letter-spacing:.15em; white-space:nowrap; }
.ms-engine-dot { color:var(--ms-teal); font-size:.9rem; }
.st-key-ms-topbar [data-testid="stPopover"] button {
  min-width:2.2rem; max-width:9rem; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; border:0; background:transparent;
}

.ms-eyebrow { margin:0 0 .6rem; color:var(--ms-teal); font-size:.68rem; font-weight:800;
  letter-spacing:.19em; text-transform:uppercase; }
.ms-lede { max-width:72ch; margin:.15rem 0 1.65rem; color:#91A4C0 !important; font-size:1rem; line-height:1.65; }
.ms-group { margin:.85rem 0 .35rem; padding-top:.9rem; border-top:1px solid var(--ms-border);
  color:var(--ms-text); font-size:.96rem; font-weight:650; }
.ms-group.first { margin-top:.15rem; padding-top:0; border-top:0; }
.ms-hint { margin:.2rem 0 .6rem; color:var(--ms-muted); font-size:.84rem; line-height:1.5; }

.stApp [data-testid="stVerticalBlockBorderWrapper"] {
  border-color:var(--ms-border) !important; border-radius:18px !important; background:var(--ms-panel);
}
.stApp [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--ms-border); }
.stApp [data-testid="stTabs"] button[data-baseweb="tab"] { color:var(--ms-muted); }
.stApp [data-testid="stTabs"] button[aria-selected="true"] { color:var(--ms-teal); }
.stApp [data-testid="stExpander"] { border-color:var(--ms-border); border-radius:12px; background:#0A1020; }
.stApp [data-testid="stTextInput"] input, .stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background:#080E1B; border-color:#293750; border-radius:11px; color:var(--ms-text);
}
.stApp [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--ms-teal); }
.stApp [data-testid="stFileUploader"] section { border:1px dashed #2B3A54; border-radius:15px; background:#090F1D; }
.stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"] { border:1px solid var(--ms-border); border-radius:12px; overflow:hidden; }
.stApp [data-testid="stAlert"] { border-radius:12px; }
.stApp [data-testid="stButton"] button, .stApp [data-testid="stDownloadButton"] button {
  border:1px solid #2B3A54; border-radius:11px; background:#101A2C; color:#EAF2FC;
}
.stApp [data-testid="stButton"] button:hover, .stApp [data-testid="stDownloadButton"] button:hover {
  border-color:var(--ms-teal); color:var(--ms-teal);
}
.stApp :is(button, a, input, [role="slider"], [role="combobox"]):focus-visible {
  outline:3px solid #FFC400 !important; outline-offset:3px !important;
}
.stApp [data-testid="stPlotlyChart"] { border:1px solid #1E293C; border-radius:14px; background:#0B1220; }

.ms-readout { display:flex; flex-wrap:wrap; gap:0; margin:.45rem 0 1.2rem; padding:.5rem .35rem;
  border:1px solid var(--ms-border); border-radius:16px; background:var(--ms-panel); }
.ms-item { flex:1; min-width:100px; margin:.25rem 0; padding:.55rem .7rem;
  border-right:1px solid #293750; }
.ms-item:last-child { border-right:0; }
.ms-val { color:var(--ms-teal); font-size:1.55rem; font-weight:700; letter-spacing:-.03em; line-height:1.2; }
.ms-val small { margin-left:.28rem; color:var(--ms-muted); font-size:.75rem; font-weight:500; }
.ms-lab { margin-top:.3rem; color:#7187A8; font-size:.65rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
.ms-item.ms-bad .ms-val { color:var(--ms-amber); }
.ms-item.ms-good .ms-val { color:var(--ms-teal); }
.ms-verdict { margin:.4rem 0 1rem; padding:1rem 1.2rem; border:1px solid; border-radius:14px; }
.ms-verdict.ms-bad { border-color:#754816; background:#21170B; color:#FFDDA2; }
.ms-verdict.ms-good { border-color:#075D52; background:#071B1A; color:#B2FFF1; }
.ms-verdict.ms-warn { border-color:#7C6015; background:#1D190C; color:#FFE7A5; }
.ms-vh { font-size:1.3rem; font-weight:700; }
.ms-vd { margin-top:.35rem; color:#A6B5C9; font-size:.9rem; line-height:1.5; }
.ms-prov { display:flex; align-items:baseline; flex-wrap:wrap; gap:.6rem; margin:.15rem 0 .8rem;
  color:var(--ms-muted); font-size:.85rem; }
.ms-badge { display:inline-block; padding:.2rem .65rem; border-radius:999px; font-size:.72rem; font-weight:700; }
.ms-badge.real { background:#0A302E; color:#5FF7E2; }
.ms-badge.synth { background:#33270C; color:#FFD55A; }
.ms-badge.user { background:#102C42; color:#72CFFF; }

.st-key-ms-hero { position:relative; overflow:hidden; padding:2.8rem 3rem; border:1px solid #1C3440;
  border-radius:26px; background:radial-gradient(ellipse at 78% 10%, #092C31 0, transparent 40%),
  linear-gradient(115deg,#0E1728 0%,#0A111D 70%,#071014 100%); }
.st-key-ms-hero h2 { max-width:850px; margin:.6rem 0 1rem; font-size:clamp(2.4rem,4vw,3.65rem);
  line-height:1.08; letter-spacing:-.05em; }
.st-key-ms-hero p { max-width:75ch; color:#9AAEC9 !important; font-size:1.02rem; line-height:1.7; }
.ms-hero-metrics { display:flex; align-items:center; justify-content:flex-end; gap:1.8rem;
  min-height:100%; border-left:1px solid #1C3440; padding-left:1.6rem; }
.st-key-ms-hero [data-testid="stPageLink"] a { display:inline-flex; padding:.7rem 1rem; border-radius:12px;
  background:var(--ms-teal); color:#031411 !important; font-weight:750; text-decoration:none; }
.st-key-ms-hero [data-testid="stPageLink"] a p { color:#031411 !important; }
.ms-hero-stat strong { display:block; color:var(--ms-teal); font-size:1.65rem; font-weight:750; }
.ms-hero-stat span { color:#7187A8; font-size:.58rem; letter-spacing:.15em; }
.ms-section-kicker { margin:0 0 .45rem; color:var(--ms-teal); font-size:.65rem; font-weight:800; letter-spacing:.2em; }
.ms-workspace-card { min-height:180px; }
.ms-card-icon { color:var(--ms-teal); font-size:1.35rem; }
.ms-organism-chip { display:inline-block; margin:.35rem .25rem .1rem 0; padding:.42rem .78rem;
  border:1px solid #2A3952; border-radius:999px; color:#DCE7F5; font-size:.84rem; }
.ms-data-contract { margin:.1rem 0 1rem; }
.ms-data-contract strong { display:block; color:var(--ms-teal); font-family:Consolas,monospace; font-size:.8rem; }
.ms-data-contract span { color:var(--ms-muted); font-size:.76rem; line-height:1.5; }
.ms-run-row { display:flex; justify-content:space-between; gap:1rem; padding:1rem .2rem;
  border-bottom:1px solid #202A3D; }
.ms-run-row:last-child { border-bottom:0; }
.ms-run-title { color:#EAF2FC; font-weight:650; }
.ms-run-sub { margin-top:.25rem; color:#7187A8; font-size:.78rem; }
.ms-run-metric { align-self:center; color:var(--ms-teal); font-family:Consolas,monospace;
  font-size:.8rem; font-weight:700; white-space:nowrap; }
.ms-footer { margin-top:2rem; padding:1rem .2rem; border-top:1px solid #1A2231;
  color:#526783; font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; }

@media(max-width:1200px) {
  .ms-brand-sub, .ms-engine { display:none; }
}
@media(max-width:900px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .st-key-ms-topbar { margin-left:-1rem; margin-right:-1rem; padding-left:1rem; padding-right:1rem; }
  .st-key-ms-hero { padding:1.5rem; }
  .ms-hero-metrics { justify-content:flex-start; border-left:0; padding:.8rem 0 0; }
}
@media(max-width:640px) {
  .stApp h1 { font-size:2rem; }
  .block-container { padding-bottom:2rem; }
  .st-key-ms-topbar { overflow-x:auto; }
  .ms-readout { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }
  .ms-item { min-width:0; border-right:0; border-bottom:1px solid #293750; }
  .ms-run-row { align-items:flex-start; flex-direction:column; gap:.35rem; }
  .ms-run-metric { align-self:flex-start; }
}
</style>
"""


def inject_css() -> None:
    """Call once per script run, before any page content."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_topbar(pages: Dict[str, st.Page], active_key: str, username: str, on_logout) -> None:
    """Render app navigation above the active page."""
    labels = (
        ("overview", "Overview"),
        ("assays", "Assays"),
        ("hurdles", "Hurdle model"),
        ("simulator", "Cold chain"),
        ("runs", "Saved runs"),
    )
    with st.container(key="ms-topbar"):
        cols = st.columns(
            [1.5, 0.78, 0.78, 1.05, 0.83, 0.92, 0.9, 0.5],
            gap="small",
            vertical_alignment="center",
        )
        with cols[0]:
            st.markdown(
                '<div class="ms-brand"><span class="ms-brand-mark">⚗</span><span>'
                '<span class="ms-brand-name">MicrobeSim</span>'
                '<span class="ms-brand-sub">BIOANALYTICS CONSOLE</span></span></div>',
                unsafe_allow_html=True,
            )
        for column, (key, label) in zip(cols[1:6], labels):
            with column:
                if key == active_key:
                    st.markdown(
                        f'<div class="ms-nav-current">{escape(label)}</div>', unsafe_allow_html=True
                    )
                else:
                    st.page_link(pages[key], label=label, width="stretch")
        with cols[6]:
            st.markdown(
                '<div class="ms-engine"><span class="ms-engine-dot">●</span> NUMERICAL ENGINE ONLINE</div>',
                unsafe_allow_html=True,
            )
        with cols[7]:
            with st.popover(username, help="Account and safety information"):
                st.markdown("**MicrobeSim workspace**")
                st.caption(f"Signed in as {username}")
                st.caption("Screening and comparative analysis only; not regulatory evidence.")
                st.page_link(pages["overview"], label="Model notes")
                st.page_link(pages["runs"], label="Analysis ledger")
                if st.button("Sign out", key="ms-sign-out", width="stretch"):
                    on_logout()
                    st.rerun()


def page_header(title: str, lede: str, eyebrow: str = "MICROBESIM / ANALYTICAL WORKSPACE") -> None:
    st.markdown(f'<div class="ms-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="ms-lede">{escape(lede)}</p>', unsafe_allow_html=True)


def group(title: str, first: bool = False) -> None:
    """A titled group inside an inputs panel."""
    cls = "ms-group first" if first else "ms-group"
    st.markdown(f'<div class="{cls}">{escape(title)}</div>', unsafe_allow_html=True)


def hint(text: str) -> None:
    st.markdown(f'<div class="ms-hint">{escape(text)}</div>', unsafe_allow_html=True)


Item = Tuple[str, Union[str, Tuple[str, str]], Optional[str]]


def readout(items: Sequence[Item]) -> None:
    """Render key values in a row; tuple values carry a unit."""
    cells = []
    for label, value, tone in items:
        if isinstance(value, tuple):
            val_html = f"{escape(str(value[0]))}<small>{escape(str(value[1]))}</small>"
        else:
            val_html = escape(str(value))
        cls = f"ms-item ms-{tone}" if tone else "ms-item"
        cells.append(
            f'<div class="{cls}"><div class="ms-val">{val_html}</div>'
            f'<div class="ms-lab">{escape(label)}</div></div>'
        )
    st.markdown(
        f'<div class="ms-readout" role="group" aria-label="Key analysis results">'
        f'{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def verdict(tone: str, headline: str, detail: str = "") -> None:
    """Render a full-width result banner."""
    detail_html = f'<div class="ms-vd">{escape(detail)}</div>' if detail else ""
    st.markdown(
        f'<div class="ms-verdict ms-{escape(tone)}" role="status" '
        f'aria-live="polite" aria-atomic="true">'
        f'<div class="ms-vh">{escape(headline)}</div>{detail_html}</div>',
        unsafe_allow_html=True,
    )


def provenance(
    kind: str, n_rows: int, t_range: Optional[Tuple[float, float]] = None, n_temps: int = 0
) -> None:
    """Say where the training data came from, and how much to trust it."""
    if kind == "real":
        span = f" between {t_range[0]:g} and {t_range[1]:g} °C" if t_range else ""
        text = f"{n_rows} growth records at {n_temps} temperatures{span}."
        if n_rows < 60:
            text += " Small sample, so treat predictions as approximate."
        badge = '<span class="ms-badge real">Installed real data</span>'
    elif kind == "synthetic":
        text = "Generated from illustrative parameters. Good for exploring the app, not for safety decisions."
        badge = '<span class="ms-badge synth">Synthetic demo data</span>'
    else:
        text = f"{n_rows} rows from your file."
        badge = '<span class="ms-badge user">Your dataset</span>'
    st.markdown(
        f'<div class="ms-prov">{badge}<span>{escape(text)}</span></div>', unsafe_allow_html=True
    )


def step_block(title: str, body: str) -> None:
    """A simple titled paragraph used on workspace cards."""
    st.markdown(
        f'<div class="ms-card-icon">↗</div><h3>{escape(title)}</h3>'
        f'<p class="ms-hint">{escape(body)}</p>',
        unsafe_allow_html=True,
    )
