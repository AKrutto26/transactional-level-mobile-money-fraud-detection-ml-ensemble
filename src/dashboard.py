"""
dashboard.py
------------
Mlinzi: fraud analyst dashboard (Streamlit) for the explainable mobile money fraud
detection prototype. "Mlinzi" is Swahili for guard or protector. Decisions always come from the running API; the dashboard
adds the analyst workspace around them.

Monitor
  Command centre        headline KPIs, decisions over time, open alerts and cases, calibration, audit chain
  Velocity & patterns   alert bursts per simulation hour, hour-by-type heatmap, fast-filling recipients
  Network               fan-in graph of senders to shared recipients (mule-account signal)
Investigate
  Alert queue           filterable queue with alert age, open detail or start a case
  Cases                 assignment, priority, notes, escalation, closure (posted to the API), report export
  Transaction detail    verdict, reasons, rules, hash-chained events, analyst review
  Explore & SQL         drill-down filters, CSV export, read-only SQL console over local activity
Operate
  Send a transaction    customer simulator with step-up verification
  Verification          simulated SMS outbox and code entry
  Live replay           streams unseen test-period transactions into the API
  Rule tuning           threshold what-if on labelled replay results; candidate rule validation
  System health         API, pipeline latency, audit chain, data stores, integrations, controls
Research
  Model evidence        tables and figures from results/

Local stores written by this dashboard (data/state/):
  activity.db   every transaction this dashboard sends and the decision returned. Replay rows also
                carry the PaySim label, used only for evaluation views; the API never sees it.
  cases.db      case management records and notes. The final outcome of a case is posted to the
                API review endpoint, so the decision itself lands in the hash-chained audit log.

Run (with the API already running in another terminal):
    python -m streamlit run src/dashboard.py

Optional files: assets/mlinzi_mark.svg (logo), assets/mlinzi_icon.png (browser tab icon),
.streamlit/config.toml (dark base theme).

Changelog:
  v2.3 — light-only presentation design: white background and black accents, with colour reserved
         for data. Risk bands use green, yellow, orange and red; charts use a distinct categorical palette.
  v2.2 — monochrome design system: grey frosted frame, white cards, black hero panel and black
         primary pills, greyscale risk ramp with red reserved for Critical. Command Centre gains a
         summary hero card, a detection-rate ring and header actions.
  v2.1 — Mlinzi identity (logo, browser tab title and icon). Visual system aligned with Apple's HIG:
         solid elevated surfaces for content, blur only on navigation, systemBlue for actions,
         backdrop photo removed for readability. Model evidence rebuilt around the model
         selection argument and tolerant of results/ column names. Explanations for low-risk
         decisions ("why it was cleared") alongside flagged ones.
  v2.0 — analyst workspace: command centre KPIs, velocity and pattern views, network graph,
         case management with escalation and report export, drill-down explorer with SQL,
         threshold tuning, system health and integrations status. Charts use Altair and
         Graphviz, both bundled with Streamlit (no new dependencies).
  v1.2 — dark glass interface; coverage-aware NOT ASSESSED state for non-model types.
  v1.1 — review confirmation survives the page refresh; width="stretch".
"""

import base64
import html
import json
import re
import secrets
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import FIGURES_DIR, METRICS_DIR, RAW_DATA_PATH, SHAP_DIR, VAL_MAX_STEP  # noqa: E402

API = "http://127.0.0.1:8000"
STATE_DIR = ROOT / "data" / "state"
STORE_PATH = STATE_DIR / "feature_store.db"
REPLAY_PATH = STATE_DIR / "replay_test_period.parquet"
CURSOR_PATH = STATE_DIR / "replay_cursor.json"
ACTIVITY_PATH = STATE_DIR / "activity.db"
CASES_PATH = STATE_DIR / "cases.db"
ICON_PATH = ROOT / "assets" / "mlinzi_icon.png"

# Transaction types the model was trained on. PaySim contains labelled fraud only
# for these two, so every other type is outside model coverage.
MODEL_TYPES = {"TRANSFER", "CASH_OUT"}
TX_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]

def detect_appearance() -> str:
    """Follow the viewer's appearance (dark-mode.md: people expect apps to respect their preference).
    Streamlit reports the active theme from 1.46; older versions fall back to dark."""
    try:
        kind = st.context.theme.type
        if kind in ("light", "dark"):
            return kind
    except AttributeError:
        pass
    return "dark"


APPEARANCE = "light"  # presentation design: white background, black accents, colour reserved for data

# Apple system colours for each appearance (color.md › Specifications). Light-mode text colours use
# deeper values where the standard ones fall below 4.5:1 on white (accessibility.md › Vision).
TOKENS = {
    # Monochrome system: greys and white/black, with red kept as the single accent for critical risk.
    "light": {
        "page": "#FFFFFF",
        "frame": "transparent", "frame_border": "transparent",
        "card": "#FFFFFF", "card_solid": "#FFFFFF", "card_border": "#E8E8ED",
        "card_hl": "inset 0 1px 0 rgba(255,255,255,1)",
        "shadow": "0 1px 2px rgba(0,0,0,.03), 0 8px 24px rgba(0,0,0,.04)",
        "hero": "#1D1D1F", "hero_ink": "#FFFFFF", "hero_ink2": "rgba(255,255,255,.62)",
        "tile": "#FFFFFF", "tile_ink": "#1D1D1F", "tile_alt": "#E8E8ED",
        "chip": "#FFFFFF", "chip_strong": "#1D1D1F",
        "chip_on": "#1D1D1F", "chip_on_ink": "#FFFFFF",
        "bg": "#FFFFFF", "surface": "#FFFFFF", "surface2": "#F5F5F7", "surface3": "#E8E8ED",
        "sep": "#E8E8ED", "ink": "#1D1D1F", "ink2": "#48484A", "ink3": "#6E6E73",
        # data colours (charts only); the interface itself stays black and white
        "blue": "#0A84FF", "blue_hover": "#0071E3", "blue_tint": "#1D1D1F",
        "hover": "rgba(0,0,0,.04)", "sidebar": "#FFFFFF", "glow": "transparent",
        "tab_on": "#1D1D1F", "img_filter": "none",
        "axis": "#6E6E73", "grid": "#EDEDF0", "domain": "#C7C7CC",
        "neutral": "#C7C7CC", "neutral2": "#8E8E93", "neutral3": "#D1D1D6",
        "c1": "#0A84FF", "c2": "#AF52DE", "orange": "#FF9500", "red": "#FF3B30", "green": "#34C759",
        "bands": {"LOW": "#34C759", "MEDIUM": "#FFCC00", "HIGH": "#FF9500", "CRITICAL": "#FF3B30",
                  "NOT_ASSESSED": "#8E8E93"},
        "contrast": "--sep: rgba(0,0,0,.3); --ink2: #1C1C1E; --ink3: #3A3A3C;",
    },
    "dark": {
        "page": ("radial-gradient(900px 640px at 18% 8%, rgba(255,255,255,.07), transparent 62%),"
                 "radial-gradient(820px 700px at 88% 92%, rgba(255,255,255,.05), transparent 60%), #0B0B0C"),
        "frame": "rgba(255,255,255,.025)", "frame_border": "rgba(255,255,255,.07)",
        "card": "rgba(28,28,30,.94)", "card_solid": "#1C1C1E", "card_border": "rgba(255,255,255,.06)",
        "card_hl": "inset 0 1px 0 rgba(255,255,255,.05)",
        "shadow": "0 1px 2px rgba(0,0,0,.5), 0 12px 32px rgba(0,0,0,.35)",
        "hero": "#F2F2F4", "hero_ink": "#1D1D1F", "hero_ink2": "rgba(29,29,31,.62)",
        "tile": "#1D1D1F", "tile_ink": "#FFFFFF", "tile_alt": "#636366",
        "chip": "rgba(255,255,255,.07)", "chip_strong": "#F2F2F4",
        "chip_on": "#F2F2F4", "chip_on_ink": "#000000",
        "bg": "#0B0B0C", "surface": "#1C1C1E", "surface2": "#2C2C2E", "surface3": "#3A3A3C",
        "sep": "rgba(255,255,255,.09)", "ink": "#F5F5F7", "ink2": "#C7C7CC", "ink3": "#98989D",
        "blue": "#F5F5F7", "blue_hover": "#FFFFFF", "blue_tint": "#F2F2F4",
        "hover": "rgba(255,255,255,.06)", "sidebar": "#1C1C1E", "glow": "transparent",
        "tab_on": "#F2F2F4", "img_filter": "brightness(.9)",
        "axis": "#AEAEB2", "grid": "#2C2C2E", "domain": "#48484A",
        "neutral": "#48484A", "neutral2": "#8E8E93", "neutral3": "#3A3A3C",
        "c1": "#AEAEB2", "c2": "#636366", "orange": "#D1D1D6", "red": "#FF453A", "green": "#F5F5F7",
        "bands": {"LOW": "#48484A", "MEDIUM": "#8E8E93", "HIGH": "#D1D1D6", "CRITICAL": "#FF453A",
                  "NOT_ASSESSED": "#636366"},
        "contrast": "--sep: rgba(255,255,255,.4); --ink2: #F2F2F7; --ink3: #C7C7CC;",
    },
}
T = TOKENS[APPEARANCE]
BAND_COLOURS = T["bands"]
BAND_LABELS = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High", "CRITICAL": "Critical",
               "NOT_ASSESSED": "Not assessed"}
RISK_BANDS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ALL_BANDS = RISK_BANDS + ["NOT_ASSESSED"]
FLAGGED = ("MEDIUM", "HIGH", "CRITICAL")
UP, DOWN = (T["red"], T["green"]) if APPEARANCE == "light" else ("#FF6961", "#4CD787")

CASE_STATUSES = ["Open", "Investigating", "Escalated", "Closed: fraud", "Closed: legitimate"]
CASE_PRIORITIES = ["P1 urgent", "P2 high", "P3 normal"]
ESCALATION_TARGETS = ["Senior investigator", "Compliance and AML", "Mobile network fraud desk",
                      "Law enforcement (DCI cybercrime)"]

NAV = {
    "Monitor": ["Command Centre", "Velocity & Patterns", "Network"],
    "Investigate": ["Alert Queue", "Cases", "Transaction Detail", "Explore & SQL"],
    "Operate": ["Send a Transaction", "Verification", "Live Replay", "Rule Tuning", "System Health"],
    "Research": ["Model Evidence"],
}
PAGES = [p for group in NAV.values() for p in group]
NAV_ICONS = {
    "Command Centre": ":material/space_dashboard:", "Velocity & Patterns": ":material/monitoring:",
    "Network": ":material/hub:", "Alert Queue": ":material/notifications:", "Cases": ":material/folder_open:",
    "Transaction Detail": ":material/receipt_long:", "Explore & SQL": ":material/manage_search:",
    "Send a Transaction": ":material/send:", "Verification": ":material/sms:", "Live Replay": ":material/play_circle:",
    "Rule Tuning": ":material/tune:", "System Health": ":material/health_and_safety:",
    "Model Evidence": ":material/science:",
}
OFFLINE_OK = ("Model Evidence", "Explore & SQL", "Velocity & Patterns", "Network", "Rule Tuning")

if "page" not in st.session_state or st.session_state["page"] not in PAGES:
    st.session_state["page"] = "Command Centre"
st.set_page_config(page_title=f"Mlinzi | {st.session_state['page']}",
                   page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🛡️", layout="wide")


# ── Visual system ──────────────────────────────────────────────────────────
MARK_SVG = (ROOT / "assets" / "mlinzi_mark.svg")


@st.cache_resource
def mark_svg() -> str:
    """Inline mark whose colours come from the appearance tokens (black on light, white on dark)."""
    return ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' role='img' aria-label='Mlinzi'>"
            "<rect class='m-bg' width='64' height='64' rx='16'/>"
            "<path class='m-fg' d='M11 32 C19 20 45 20 53 32 C45 44 19 44 11 32 Z' fill='none' stroke-width='4' "
            "stroke-linejoin='round'/><circle class='m-dot' cx='32' cy='32' r='7.5'/>"
            "<circle class='m-pupil' cx='32' cy='32' r='3'/></svg>")

def token_css() -> str:
    t = T
    return (":root {"
            f"--bg:{t['bg']};--surface:{t['surface']};--surface-2:{t['surface2']};--surface-3:{t['surface3']};"
            f"--sep:{t['sep']};--ink:{t['ink']};--ink-2:{t['ink2']};--ink-3:{t['ink3']};"
            f"--blue:{t['ink']};--blue-hover:{t['ink2']};--blue-tint:{t['blue_tint']};--ice:{t['ink']};"
            f"--hover:{t['hover']};--sidebar:{t['sidebar']};--glow:{t['glow']};--tab-on:{t['tab_on']};"
            f"--img-filter:{t['img_filter']};--glass:{t['surface']};--glass-strong:{t['surface2']};"
            f"--rim:{t['sep']};--rim-top:{t['sep']};--radius:24px;"
            f"--page:{t['page']};--card:{t['card']};--card-solid:{t['card_solid']};--card-border:{t['card_border']};"
            f"--card-hl:{t['card_hl']};--shadow:{t['shadow']};--chip:{t['chip']};--chip-strong:{t['chip_strong']};"
            f"--chip-on:{t['chip_on']};--chip-on-ink:{t['chip_on_ink']};"
            f"--frame:{t['frame']};--frame-border:{t['frame_border']};--hero:{t['hero']};--hero-ink:{t['hero_ink']};"
            f"--hero-ink2:{t['hero_ink2']};--tile:{t['tile']};--tile-ink:{t['tile_ink']};--tile-alt:{t['tile_alt']};"
            "--grad:var(--chip-on);"
            "--font:-apple-system,BlinkMacSystemFont,'SF Pro Text','SF Pro Display','Inter','Segoe UI',Roboto,sans-serif;}"
            "@media (prefers-contrast: more) {:root{" + t["contrast"].replace("--ink2", "--ink-2").replace("--ink3", "--ink-3") + "}}")


def inject_css() -> None:
    # Apple dark-mode system values: solid base and elevated surfaces for content, blur reserved for
    # the floating navigation layer, systemBlue for interaction only, band colours for risk only.
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..700&display=swap');

__TOKENS__

.stApp {
  background:
    radial-gradient(900px 500px at 100% -10%, var(--glow), transparent 70%),
    var(--bg);
  color: var(--ink);
}
html, body, .stApp, p, li, label, input, textarea, h1, h2, h3, h4,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], .stButton button {
  font-family: var(--font) !important;
  -webkit-font-smoothing: antialiased;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1320px; }
.stApp p, .stApp li, .stApp label { font-size: 14px; line-height: 1.45; }
[data-testid="stCaptionContainer"] p { font-size: 12.5px; }

h2, h3 { color: var(--ink); letter-spacing: -0.015em; font-weight: 600; }
h3 { font-size: 17px !important; line-height: 22px; margin-top: .1rem; }
[data-testid="stCaptionContainer"] { color: var(--ink-3); }

/* Navigation: the one translucent layer */
[data-testid="stSidebar"] > div:first-child {
  background: var(--sidebar);
  backdrop-filter: blur(30px) saturate(180%);
  -webkit-backdrop-filter: blur(30px) saturate(180%);
  border-right: 1px solid var(--sep);
}
@media (prefers-reduced-transparency: reduce) {
  [data-testid="stSidebar"] > div:first-child { background: var(--surface); backdrop-filter: none; }
}
[data-testid="stSidebar"] .stButton button {
  justify-content: flex-start; border-radius: 8px; padding: .34rem .65rem; min-height: 32px;
  background: transparent; border: 0; color: var(--ink-2); font-weight: 500; font-size: .92rem;
}
[data-testid="stSidebar"] .stButton button > div { justify-content: flex-start; }
[data-testid="stSidebar"] .stButton button:hover { background: var(--hover); color: var(--ink); border: 0; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--blue-tint); color: var(--ink); }
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: var(--blue-tint); }
[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] { color: var(--blue); font-size: 1.15rem; }

.fa-brand { display: flex; align-items: center; gap: .7rem; margin: .1rem 0 1.1rem; }
.fa-brand svg { width: 36px; height: 36px; flex: 0 0 36px; filter: drop-shadow(0 3px 10px rgba(0,113,227,.3)); }
.fa-brand b { display: block; font: 700 1.22rem var(--font); letter-spacing: -.02em; color: var(--ink); line-height: 1.1; }
.fa-brand span { display: block; font-size: .78rem; color: var(--ink-3); margin-top: .1rem; }
.fa-group { color: var(--ink-3); font-size: .74rem; font-weight: 600; margin: 1rem 0 .2rem .65rem; }

/* Content surfaces: solid, elevated, no blur */
[data-testid="stMetric"],
[data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  background: var(--surface);
  border: 1px solid var(--sep) !important;
  border-radius: var(--radius);
}
[data-testid="stMetric"] { padding: .9rem 1.05rem; }
[data-testid="stForm"] { padding: 1.25rem 1.3rem .5rem; }
[data-testid="stMetricLabel"] { color: var(--ink-2); }
[data-testid="stMetricLabel"] p { font-size: .84rem; }
[data-testid="stMetricValue"] {
  font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--ink);
  font-size: 1.75rem;
}

[data-testid="stDataFrame"] { border: 1px solid var(--sep); border-radius: 12px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 10px; background: #fff; filter: var(--img-filter); }

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stNumberInput > div > div {
  background: var(--surface-2) !important; border-color: transparent !important; border-radius: 9px !important;
}

/* Buttons: systemBlue for the prominent action, filled grey for the rest */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
  border-radius: 9px; padding: .42rem 1rem; min-height: 34px; font-weight: 500;
  background: var(--surface-2); border: 0; color: var(--ink);
}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  background: var(--surface-3); color: var(--ink); border: 0;
}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button:is([kind="primary"],[kind="primaryFormSubmit"]) {
  background: var(--blue); color: #fff;
}
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:is([kind="primary"],[kind="primaryFormSubmit"]):hover {
  background: var(--blue-hover); color: #fff;
}
button:focus-visible { outline: 3px solid var(--blue) !important; outline-offset: 2px; }

/* Tabs as a segmented control */
[data-baseweb="tab-list"] { gap: 2px; background: var(--surface-2); border-radius: 9px; padding: 2px; width: fit-content; }
[data-baseweb="tab"] { border-radius: 7px; padding: .3rem .9rem; height: auto; }
[data-baseweb="tab"][aria-selected="true"] { background: var(--tab-on); box-shadow: 0 1px 3px rgba(0,0,0,.12); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stProgress"] > div > div { border-radius: 999px; }
hr { border-color: var(--sep) !important; }

/* Page title */
.fa-title { margin: 0 0 1.4rem; }
.fa-title h1 {
  font-family: var(--font); font-size: 28px; font-weight: 700;
  letter-spacing: -0.02em; line-height: 34px; margin: 0; padding: 0; color: var(--ink);
}
.fa-title p { color: var(--ink-2); font-size: 15px; max-width: 70ch; margin: .45rem 0 0; line-height: 1.5; }

.fa-pill {
  display: inline-flex; align-items: center; gap: .45rem; padding: .22rem .7rem .22rem .55rem;
  border-radius: 999px; font-weight: 600; font-size: .86rem; color: var(--ink);
  background: color-mix(in srgb, var(--band) 20%, var(--surface));
}
.fa-pill i { width: .5rem; height: .5rem; border-radius: 50%; background: var(--band); display: inline-block; }

/* Signature element: the verdict */
.fa-verdict {
  position: relative; overflow: hidden; border-radius: 18px; padding: 1.6rem 1.8rem 1.4rem;
  background: var(--surface); border: 1px solid var(--sep); margin: .3rem 0 1.2rem;
  animation: fa-rise .35s cubic-bezier(.2,.8,.2,1);
}
.fa-verdict::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--band);
}
.fa-verdict::after {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(600px 220px at 0% 0%, color-mix(in srgb, var(--band) 16%, transparent), transparent 70%);
  pointer-events: none;
}
.fa-verdict .row { position: relative; z-index: 1; display: flex; flex-wrap: wrap; align-items: flex-end; gap: 2.4rem; }
.fa-verdict .band { font-size: clamp(2.2rem, 4.4vw, 3.3rem); font-weight: 700; letter-spacing: -0.035em; line-height: 1; color: var(--ink); }
.fa-verdict .band small { display: flex; align-items: center; gap: .45rem; font-size: 13px; font-weight: 600; letter-spacing: 0; color: var(--ink-2); margin-bottom: .5rem; }
.fa-verdict .band small::before { content: ""; width: .6rem; height: .6rem; border-radius: 50%; background: var(--band); }
.fa-verdict .stat { font-variant-numeric: tabular-nums; }
.fa-verdict .stat b { display: block; font-size: 1.55rem; font-weight: 600; letter-spacing: -0.02em; color: var(--ink); }
.fa-verdict .stat span { color: var(--ink-2); font-size: .84rem; }
.fa-verdict .meta { position: relative; z-index: 1; color: var(--ink-3); font-size: .82rem; margin-top: 1.1rem; }
@keyframes fa-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .fa-verdict { animation: none; } }

.fa-note {
  border-radius: 12px; padding: .9rem 1.1rem; margin: 0 0 1.1rem; color: var(--ink);
  background: var(--surface); border: 1px solid var(--sep); border-left: 4px solid #98A2B3;
  line-height: 1.5; font-size: .93rem;
}
.fa-note strong { color: var(--ink); }

.fa-reason { display: flex; gap: .85rem; padding: .62rem 0; border-bottom: 1px solid var(--sep); }
.fa-reason:last-child { border-bottom: 0; }
.fa-reason .pts { flex: 0 0 4.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.fa-reason .txt { color: var(--ink-2); line-height: 1.45; }
.fa-reason .txt b { color: var(--ink); font-weight: 600; }

.fa-status { display: flex; align-items: center; gap: .5rem; font-weight: 500; font-size: .9rem; margin: .1rem 0 .2rem .1rem; }
.fa-status i { width: .5rem; height: .5rem; border-radius: 50%; display: inline-block; }

.fa-card { border-radius: var(--radius); padding: 1rem 1.1rem; height: 100%; background: var(--surface); border: 1px solid var(--sep); }
.fa-card h4 { margin: 0 0 .3rem; font: 600 .98rem var(--font); color: var(--ink); }
.fa-card p { margin: 0; color: var(--ink-2); font-size: .88rem; line-height: 1.45; }
.fa-card .state { display: inline-flex; align-items: center; gap: .45rem; font-size: .8rem; color: var(--ink-2); margin-bottom: .55rem; }
.fa-card .state i { width: .45rem; height: .45rem; border-radius: 50%; display: inline-block; }

.fa-timeline { border-left: 2px solid var(--surface-3); margin: .3rem 0 1rem .35rem; padding-left: 1.1rem; }
.fa-timeline div { position: relative; padding: .4rem 0 .65rem; }
.fa-timeline div::before { content: ""; position: absolute; left: -1.5rem; top: .8rem; width: .6rem; height: .6rem;
  border-radius: 50%; background: var(--blue); border: 2px solid var(--bg); }
.fa-timeline b { color: var(--ink); font-weight: 600; }
.fa-timeline span { color: var(--ink-3); font-size: .8rem; }
.fa-timeline p { margin: .2rem 0 0; color: var(--ink-2); line-height: 1.45; }
.fa-empty { color: var(--ink-2); padding: 1.4rem 0; }

.fa-finding { padding: .8rem 0; border-bottom: 1px solid var(--sep); display: flex; gap: 1rem; }
.fa-finding:last-child { border-bottom: 0; }
.fa-finding b { flex: 0 0 9.5rem; color: var(--ink); font-weight: 600; }
.fa-finding span { color: var(--ink-2); line-height: 1.5; }
</style>
""".replace("__TOKENS__", token_css()), unsafe_allow_html=True)


inject_css()

st.markdown("""
<style>
/* Widget layer: soft squircle cards over an ambient colour field */
.stApp { background: var(--page) !important; background-attachment: fixed !important; }
[data-testid="stMetric"], [data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.fa-card, .fa-verdict, .w {
  background: var(--card) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 24px;
  box-shadow: var(--card-hl), var(--shadow);
  backdrop-filter: blur(40px) saturate(160%);
  -webkit-backdrop-filter: blur(40px) saturate(160%);
}
.fa-verdict { border-radius: 30px; }
.fa-note { border-radius: 18px; background: var(--card) !important; box-shadow: var(--card-hl), var(--shadow);
  border: 1px solid var(--card-border); border-left: 4px solid var(--ink-3); }
@media (prefers-reduced-transparency: reduce) {
  [data-testid="stMetric"], [data-testid="stForm"],
  [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
  .fa-card, .fa-verdict, .w, .fa-note { background: var(--card-solid) !important; backdrop-filter: none; }
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { padding: .35rem .25rem; }

.w { padding: 1rem 1.15rem .95rem; min-height: 172px; display: flex; flex-direction: column; overflow: hidden; }
.w-label { display: flex; align-items: center; gap: .45rem; font-size: 13px; font-weight: 600; color: var(--ink-2); }
.w-label i { width: .55rem; height: .55rem; border-radius: 50%; background: var(--c); flex: 0 0 auto; }
.w-value { margin-top: .4rem; font-size: 40px; font-weight: 600; letter-spacing: -.035em; line-height: 1;
  font-variant-numeric: tabular-nums; color: var(--ink); white-space: nowrap; }
.w-value.long { font-size: 26px; line-height: 1.15; }
.w-value span { font-size: 17px; font-weight: 500; color: var(--ink-3); margin-left: .3rem; letter-spacing: 0; }
.w-viz { margin-top: auto; padding-top: .75rem; }
.w-viz svg { width: 100%; height: 44px; display: block; overflow: visible; }
.w-note { font-size: 12px; color: var(--ink-3); margin-top: .4rem; }

[data-testid="stMetric"] { padding: 1rem 1.15rem; }
[data-testid="stMetricValue"] { font-size: 32px; font-weight: 600; letter-spacing: -.03em; }

/* Large title */
.fa-title h1 { font-size: 34px; line-height: 41px; letter-spacing: -.025em; }

/* Floating glass sidebar */
section[data-testid="stSidebar"] { background: transparent !important; border: 0 !important; }
section[data-testid="stSidebar"] > div:first-child {
  margin: 12px 0 12px 12px; border-radius: 26px; height: calc(100vh - 24px);
  background: var(--sidebar) !important; border: 1px solid var(--card-border) !important;
  box-shadow: var(--card-hl), var(--shadow);
}
[data-testid="stSidebar"] .stButton button { border-radius: 14px; min-height: 38px; padding: .4rem .75rem; box-shadow: none; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--chip-strong); color: var(--ink);
  box-shadow: var(--card-hl); filter: none; }
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: var(--chip-strong); filter: none; }

/* Buttons: one gradient for the main action, frosted chips for the rest */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
  border-radius: 14px; min-height: 38px; padding: .45rem 1.1rem; background: var(--chip);
  border: 1px solid var(--card-border); box-shadow: var(--card-hl);
}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button:is([kind="primary"],[kind="primaryFormSubmit"]) {
  background: var(--grad); color: #fff; border: 0; font-weight: 600;
  box-shadow: 0 8px 22px rgba(91,108,255,.30), inset 0 1px 0 rgba(255,255,255,.35);
}
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:is([kind="primary"],[kind="primaryFormSubmit"]):hover {
  background: var(--grad); color: #fff; filter: brightness(1.06) saturate(1.05);
}

/* Tabs and segmented controls as pills */
[data-baseweb="tab-list"] { background: transparent; gap: .4rem; padding: 0; }
[data-baseweb="tab"] { border-radius: 999px; padding: .42rem 1rem; background: var(--chip); border: 1px solid var(--card-border); }
[data-baseweb="tab"][aria-selected="true"] { background: var(--chip-on); box-shadow: none; }
[data-baseweb="tab"][aria-selected="true"] p { color: var(--chip-on-ink) !important; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; background: var(--chip); border: 1px solid var(--card-border); }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[kind="pillsActive"] { background: var(--chip-on); color: var(--chip-on-ink); }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p { color: var(--chip-on-ink); }

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stNumberInput > div > div {
  background: var(--chip) !important; border: 1px solid var(--card-border) !important; border-radius: 12px !important;
}
[data-testid="stDataFrame"] { border-radius: 18px; border: 1px solid var(--card-border); }
.fa-pill { border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Monochrome layer: white cards and black accents in a frosted frame */
.block-container {
  background: var(--frame); border: 1px solid var(--frame-border); border-radius: 36px;
  backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
  padding: 2rem 2.3rem 2.5rem !important; margin-top: 1.4rem; margin-bottom: 1.4rem;
}
[data-testid="stMetric"], [data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
.fa-card, .w {
  background: var(--card) !important; border: 1px solid var(--card-border) !important;
  border-radius: 28px; box-shadow: var(--shadow); backdrop-filter: none; -webkit-backdrop-filter: none;
}
.fa-note { background: var(--card) !important; border-radius: 22px; border-left: 4px solid var(--ink-3); }

/* Sidebar: solid white card, black pill for the active page */
section[data-testid="stSidebar"] > div:first-child {
  background: var(--sidebar) !important; border-radius: 32px; border: 1px solid var(--card-border) !important;
  backdrop-filter: none; box-shadow: var(--shadow);
}
.fa-group { text-transform: uppercase; letter-spacing: .06em; font-size: 11px; font-weight: 600; color: var(--ink-3); margin: 1.2rem 0 .3rem .7rem; }
[data-testid="stSidebar"] .stButton button { border-radius: 14px; color: var(--ink-2); background: transparent; border: 0 !important; }
[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] { color: var(--ink-2); }
[data-testid="stSidebar"] .stButton button:hover { background: var(--hover); color: var(--ink); }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
  background: var(--chip-on) !important; color: var(--chip-on-ink) !important; box-shadow: none;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] p,
[data-testid="stSidebar"] .stButton button[kind="primary"] [data-testid="stIconMaterial"] { color: var(--chip-on-ink) !important; }

/* Buttons: black pill for the main action, white outlined pill for the rest */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
  border-radius: 999px; background: var(--card-solid); border: 1px solid var(--ink) !important;
  color: var(--ink); box-shadow: none; min-height: 40px; padding: .45rem 1.2rem;
}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  background: var(--hover); color: var(--ink); border-color: var(--ink) !important;
}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button:is([kind="primary"],[kind="primaryFormSubmit"]),
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:is([kind="primary"],[kind="primaryFormSubmit"]):hover {
  background: var(--chip-on); color: var(--chip-on-ink); border: 1px solid var(--chip-on) !important;
  box-shadow: none; filter: none;
}
.stButton button[kind="primary"]:hover { opacity: .88; }

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stNumberInput > div > div {
  background: var(--surface-2) !important; border: 1px solid transparent !important; border-radius: 14px !important;
}
[data-baseweb="tab"] { background: var(--card-solid); border: 1px solid var(--card-border); }
[data-testid="stDataFrame"] { border-radius: 22px; background: var(--card-solid); }

/* Widgets */
.w { padding: 1.15rem 1.25rem 1rem; min-height: 190px; }
.w-label { color: var(--ink); font-size: 15px; font-weight: 600; }
.w-label i { display: none; }
.w-value { font-size: 42px; font-weight: 500; letter-spacing: -.04em; margin-top: .7rem; }
.w-value span { font-size: 13px; color: var(--ink-3); font-weight: 400; }
.w-note { color: var(--ink-3); }

/* Hero card: the dark "overall" panel */
.hero { background: var(--hero); color: var(--hero-ink); border-radius: 28px; padding: 1.25rem 1.35rem 1.2rem;
  min-height: 190px; box-shadow: var(--shadow); }
.hero-head { display: flex; justify-content: space-between; align-items: center; }
.hero-head b { font-size: 17px; font-weight: 600; }
.hero-head span { font-size: 12px; color: var(--hero-ink2); }
.hero-stats { display: flex; align-items: center; gap: 1.1rem; margin: .9rem 0 .8rem; }
.hero-stats div { display: flex; align-items: center; gap: .55rem; }
.hero-stats b { font-size: 40px; font-weight: 500; letter-spacing: -.04em; line-height: 1; font-variant-numeric: tabular-nums; }
.hero-stats span { font-size: 12px; line-height: 1.25; color: var(--hero-ink2); }
.hero-stats em { width: 1px; align-self: stretch; background: var(--hero-ink2); opacity: .4; }
.hero-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: .65rem; }
.tile { background: var(--tile); color: var(--tile-ink); border-radius: 18px; padding: .7rem .6rem .6rem; text-align: center; }
.tile.alt { background: var(--tile-alt); }
.tile i { display: block; width: .6rem; height: .6rem; border-radius: 50%; margin: 0 auto .35rem; background: var(--d); }
.tile b { display: block; font-size: 26px; font-weight: 500; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.tile small { font-size: 11.5px; opacity: .75; }

/* The verdict uses the hero treatment */
.fa-verdict { background: var(--hero) !important; border: 0 !important; border-radius: 30px; box-shadow: var(--shadow); }
.fa-verdict::after { background: none; }
.fa-verdict .band, .fa-verdict .stat b { color: var(--hero-ink); }
.fa-verdict .band small, .fa-verdict .stat span, .fa-verdict .meta { color: var(--hero-ink2); }

/* Brand mark follows the appearance */
.fa-brand svg { filter: none; }
.fa-brand .m-bg { fill: var(--ink); }
.fa-brand .m-fg { stroke: var(--card-solid); }
.fa-brand .m-dot { fill: var(--card-solid); }
.fa-brand .m-pupil { fill: var(--ink); }
.fa-pill { background: var(--surface-2); }
</style>
""", unsafe_allow_html=True)



def show_chart(c) -> None:
    c = (c.configure(background="transparent", font="-apple-system, BlinkMacSystemFont, Inter, Segoe UI, sans-serif")
          .configure_view(stroke=None)
          .configure_axis(labelColor=T["axis"], titleColor=T["axis"], gridColor=T["grid"],
                          domainColor=T["domain"], tickColor=T["domain"], labelFontSize=11, titleFontWeight=500)
          .configure_axisX(grid=False)
          .configure_legend(labelColor=T["axis"], titleColor=T["axis"], orient="bottom")
          .configure_title(color=T["ink"], fontWeight=600, anchor="start", fontSize=14))
    try:
        st.altair_chart(c, width="stretch", theme=None)
    except TypeError:
        st.altair_chart(c, use_container_width=True, theme=None)

def stretch(fn, *args, **kwargs):
    """Call a Streamlit element with width="stretch", falling back on older versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


# ── Widgets (big number, small unit, a glanceable trace) ───────────────────
def _spark_points(values) -> list[tuple[float, float]]:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(v) < 2:
        return []
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    n = len(v) - 1
    return [(100 * i / n, 36 - 30 * (x - lo) / span) for i, x in enumerate(v)]


def sparkline(values, colour: str) -> str:
    pts = _spark_points(values)
    if not pts:
        return ""
    gid = f"g{abs(hash((tuple(round(y, 2) for _, y in pts), colour))) % 10**8}"
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    area = f"0,40 {line} 100,40"
    return (f"<svg viewBox='0 0 100 40' preserveAspectRatio='none' aria-hidden='true'>"
            f"<defs><linearGradient id='{gid}' x1='0' y1='0' x2='0' y2='1'>"
            f"<stop offset='0' stop-color='{colour}' stop-opacity='.35'/>"
            f"<stop offset='1' stop-color='{colour}' stop-opacity='0'/></linearGradient></defs>"
            f"<polygon points='{area}' fill='url(#{gid})'/>"
            f"<polyline points='{line}' fill='none' stroke='{colour}' stroke-width='2' "
            f"stroke-linejoin='round' stroke-linecap='round' vector-effect='non-scaling-stroke'/></svg>")


def bars(values, colours) -> str:
    """Thin rounded bars, one per category, like an activity strip."""
    v = [max(float(x), 0.0) for x in values]
    if not v or max(v) == 0:
        return ""
    top, n = max(v), len(v)
    gap = 100 / n
    out = []
    for i, (x, c) in enumerate(zip(v, colours)):
        h = max(4.0, 36 * x / top)
        out.append(f"<rect x='{i * gap + gap * .3:.2f}' y='{40 - h:.2f}' width='{gap * .4:.2f}' height='{h:.2f}' "
                   f"rx='{min(gap * .2, 3):.2f}' fill='{c}'/>")
    return f"<svg viewBox='0 0 100 40' preserveAspectRatio='none' aria-hidden='true'>{''.join(out)}</svg>"


def widget(label: str, value: str, unit: str = "", colour: str | None = None, viz: str = "",
           note: str = "") -> str:
    colour = colour or T["blue"]
    unit_html = f"<span>{html.escape(unit)}</span>" if unit else ""
    note_html = f"<div class='w-note'>{html.escape(note)}</div>" if note else ""
    viz_html = f"<div class='w-viz'>{viz}</div>" if viz else ""
    return (f"<div class='w' style='--c:{colour}' role='group' aria-label='{html.escape(label)}: "
            f"{html.escape(value)} {html.escape(unit)}'>"
            f"<div class='w-label'><i></i>{html.escape(label)}</div>"
            f"<div class='w-value{' long' if len(value) > 7 else ''}'>{html.escape(value)}{unit_html}</div>"
            f"{viz_html}{note_html}</div>")


def ring(fraction: float | None, colour: str | None = None) -> str:
    """Progress ring for a share between 0 and 1."""
    if fraction is None or pd.isna(fraction):
        return ""
    f = max(0.0, min(1.0, float(fraction)))
    c = 2 * np.pi * 26
    return (f"<svg viewBox='0 0 64 64' style='width:64px;height:64px' aria-hidden='true'>"
            f"<circle cx='32' cy='32' r='26' fill='none' stroke='{T['neutral3']}' stroke-width='7'/>"
            f"<circle cx='32' cy='32' r='26' fill='none' stroke='{colour or T['ink']}' stroke-width='7' stroke-linecap='round' "
            f"stroke-dasharray='{c * f:.1f} {c:.1f}' transform='rotate(-90 32 32)'/></svg>")


def hero_card(title: str, meta: str, stats: list[tuple[str, str]], tiles: list[tuple[str, str, str, bool]]) -> str:
    """Dark summary panel: two headline numbers and three tiles (value, label, dot colour, alternate fill)."""
    stat_html = "<em></em>".join(
        f"<div><b>{html.escape(v)}</b><span>{html.escape(l).replace(' ', '<br>', 1)}</span></div>" for v, l in stats)
    tile_html = "".join(
        f"<div class='tile{' alt' if alt else ''}' style='--d:{c}'><i></i><b>{html.escape(v)}</b>"
        f"<small>{html.escape(l)}</small></div>" for v, l, c, alt in tiles)
    return (f"<div class='hero'><div class='hero-head'><b>{html.escape(title)}</b><span>{html.escape(meta)}</span></div>"
            f"<div class='hero-stats'>{stat_html}</div><div class='hero-tiles'>{tile_html}</div></div>")


def widget_row(items: list[str]) -> None:
    cols = st.columns(len(items), gap="medium")
    for col, item in zip(cols, items):
        col.markdown(item, unsafe_allow_html=True)


def split_value(text: str) -> tuple[str, str]:
    """'38 ms' -> ('38', 'ms'); '87.2%' -> ('87.2', '%')."""
    m = re.match(r"^\s*([-+]?[\d,]*\.?\d+)\s*(.*)$", str(text))
    return (m.group(1), m.group(2)) if m else (str(text), "")


def band_scale(bands=ALL_BANDS):
    return alt.Scale(domain=[BAND_LABELS[b] for b in bands], range=[BAND_COLOURS[b] for b in bands])


# ── API client ─────────────────────────────────────────────────────────────
@st.cache_resource
def api_key() -> str:
    raw = (ROOT / ".env").read_bytes()
    text = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
    for line in text.splitlines():
        if line.strip().startswith("FRAUD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("No FRAUD_API_KEY line in .env")


def api(method: str, path: str, body: dict | None = None):
    """Returns (status_code, json). status_code is None when the API is unreachable."""
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-API-Key", api_key())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except json.JSONDecodeError:
            return e.code, None
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None, None


def new_account(prefix: str = "C") -> str:
    return prefix + str(secrets.randbelow(10**12) + 10**12)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ── Local stores (activity and cases) ──────────────────────────────────────
def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource
def init_local_stores() -> bool:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with connect(ACTIVITY_PATH) as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""CREATE TABLE IF NOT EXISTS activity (
            tx_id TEXT PRIMARY KEY, source TEXT, step INTEGER, tx_type TEXT, amount REAL,
            name_orig TEXT, name_dest TEXT, old_balance REAL, api_band TEXT, band TEXT,
            risk_score REAL, status TEXT, action TEXT, latency_ms REAL, is_fraud INTEGER, sent_at TEXT)""")
    with connect(CASES_PATH) as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""CREATE TABLE IF NOT EXISTS cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT, tx_id TEXT UNIQUE, status TEXT, priority TEXT,
            assignee TEXT, escalated_to TEXT, opened_at TEXT, updated_at TEXT, closed_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS case_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, author TEXT, kind TEXT,
            note TEXT, created_at TEXT)""")
    return True


init_local_stores()


def log_activity(rows: list[dict]) -> None:
    if not rows:
        return
    cols = ["tx_id", "source", "step", "tx_type", "amount", "name_orig", "name_dest", "old_balance",
            "api_band", "band", "risk_score", "status", "action", "latency_ms", "is_fraud", "sent_at"]
    with connect(ACTIVITY_PATH) as c:
        c.executemany(f"INSERT OR REPLACE INTO activity ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [tuple(r.get(k) for k in cols) for r in rows])
    load_activity.clear()


def activity_row(body: dict, res: dict, source: str, latency_ms: float, is_fraud=None) -> dict:
    return {"tx_id": res["tx_id"], "source": source, "step": body["step"], "tx_type": body["type"],
            "amount": body["amount"], "name_orig": body["nameOrig"], "name_dest": body["nameDest"],
            "old_balance": body["oldbalanceOrg"], "api_band": res["risk_band"],
            "band": effective_band(res["risk_band"], body["type"]), "risk_score": res.get("risk_score"),
            "status": res.get("status"), "action": res.get("recommended_action"),
            "latency_ms": latency_ms, "is_fraud": is_fraud, "sent_at": now_utc()}


@st.cache_data(ttl=5, show_spinner=False)
def load_activity() -> pd.DataFrame:
    with connect(ACTIVITY_PATH) as c:
        df = pd.read_sql_query("SELECT * FROM activity", c)
    if not df.empty:
        df["hour"] = df["step"] % 24
        df["band_label"] = df["band"].map(BAND_LABELS).fillna(df["band"])
        df["flagged"] = df["band"].isin(FLAGGED)
    return df


def cases_df() -> pd.DataFrame:
    with connect(CASES_PATH) as c:
        return pd.read_sql_query("SELECT * FROM cases ORDER BY case_id DESC", c)


def case_for(tx_id: str) -> dict | None:
    with connect(CASES_PATH) as c:
        row = c.execute("SELECT * FROM cases WHERE tx_id = ?", (tx_id,)).fetchone()
    return dict(row) if row else None


def case_ref(case_id: int) -> str:
    return f"CASE-{case_id:04d}"


def open_case(tx_id: str, priority: str, assignee: str, note: str = "") -> int:
    existing = case_for(tx_id)
    if existing:
        return existing["case_id"]
    ts = now_utc()
    with connect(CASES_PATH) as c:
        cur = c.execute("INSERT INTO cases (tx_id, status, priority, assignee, opened_at, updated_at) "
                        "VALUES (?, 'Open', ?, ?, ?, ?)", (tx_id, priority, assignee, ts, ts))
        case_id = cur.lastrowid
        c.execute("INSERT INTO case_notes (case_id, author, kind, note, created_at) VALUES (?, ?, ?, ?, ?)",
                  (case_id, assignee, "Opened", note or "Case opened from alert.", ts))
    return case_id


def update_case(case_id: int, author: str, kind: str, note: str, **fields) -> None:
    ts = now_utc()
    fields["updated_at"] = ts
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect(CASES_PATH) as c:
        c.execute(f"UPDATE cases SET {sets} WHERE case_id = ?", (*fields.values(), case_id))
        c.execute("INSERT INTO case_notes (case_id, author, kind, note, created_at) VALUES (?, ?, ?, ?, ?)",
                  (case_id, author, kind, note, ts))


def case_notes(case_id: int) -> list[dict]:
    with connect(CASES_PATH) as c:
        rows = c.execute("SELECT * FROM case_notes WHERE case_id = ? ORDER BY note_id", (case_id,)).fetchall()
    return [dict(r) for r in rows]


# ── Replay position (kept consistent with the feature store) ───────────────
def store_seed_stamp() -> str | None:
    if not STORE_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(STORE_PATH.as_uri() + "?mode=ro", uri=True)
        row = conn.execute("SELECT value FROM store_meta WHERE key = 'seeded_at'").fetchone()
        conn.close()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def load_cursor() -> dict:
    stamp = store_seed_stamp()
    if CURSOR_PATH.exists():
        cur = json.loads(CURSOR_PATH.read_text())
        if cur.get("seed") == stamp:
            return cur
    return {"seed": stamp, "position": 0, "last_step": VAL_MAX_STEP + 1}


def save_cursor(cur: dict) -> None:
    CURSOR_PATH.write_text(json.dumps(cur))


@st.cache_data(show_spinner="Preparing test-period transactions (first time only, about 20 seconds) ...")
def replay_frame() -> pd.DataFrame:
    if not REPLAY_PATH.exists():
        raw = pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "amount", "nameOrig",
                                                 "oldbalanceOrg", "nameDest", "isFraud"])
        raw[raw["step"] > VAL_MAX_STEP].reset_index(drop=True).to_parquet(REPLAY_PATH, index=False)
    return pd.read_parquet(REPLAY_PATH)


# ── Coverage ───────────────────────────────────────────────────────────────
def effective_band(band: str | None, tx_type: str | None) -> str | None:
    """The band to display. Types outside model coverage are never shown as LOW."""
    if band == "NOT_ASSESSED" or (tx_type and tx_type not in MODEL_TYPES):
        return "NOT_ASSESSED"
    return band


def coverage_note(api_band: str | None, tx_type: str | None) -> None:
    routed = (f" The API routed it as <strong>{BAND_LABELS.get(api_band, api_band)}</strong> by rule; "
              "that is a default, not a risk assessment.") if api_band and api_band != "NOT_ASSESSED" else ""
    st.markdown(
        f"<div class='fa-note'><strong>Outside model coverage.</strong> This is a "
        f"{html.escape(str(tx_type))} transaction. The model was trained on TRANSFER and CASH_OUT only, "
        f"because the training data contains no labelled fraud for any other type, so it cannot judge "
        f"merchant, till or paybill fraud.{routed}</div>", unsafe_allow_html=True)


# ── Shared display helpers ─────────────────────────────────────────────────
def page_title(title: str, lead: str | None = None) -> None:
    lead_html = f"<p>{html.escape(lead)}</p>" if lead else ""
    st.markdown(f"<div class='fa-title'><h1>{html.escape(title)}</h1>{lead_html}</div>",
                unsafe_allow_html=True)


def pill(band: str | None) -> str:
    if not band:
        return ""
    colour = BAND_COLOURS.get(band, "#8E8E93")
    return (f"<span class='fa-pill' style='--band:{colour}'><i></i>"
            f"{html.escape(BAND_LABELS.get(band, band))}</span>")


def verdict(band: str | None, score: float | None, status: str, extra_label: str, extra_value: str,
            meta: str) -> None:
    colour = BAND_COLOURS.get(band, "#8E8E93")
    assessed = band != "NOT_ASSESSED"
    score_txt = f"{score:.1%}" if assessed and score is not None else "Not scored"
    head = "Risk band" if assessed else "Model coverage"
    st.markdown(f"""
<div class='fa-verdict' style='--band:{colour}'>
  <div class='row'>
    <div class='band'><small>{head}</small>{html.escape(BAND_LABELS.get(band, str(band)))}</div>
    <div class='stat'><b>{score_txt}</b><span>Fraud probability</span></div>
    <div class='stat'><b>{html.escape(extra_value)}</b><span>{html.escape(extra_label)}</span></div>
    <div class='stat'><b>{html.escape(status.replace('_', ' ').title())}</b><span>Status</span></div>
  </div>
  <div class='meta'>{html.escape(meta)}</div>
</div>""", unsafe_allow_html=True)


REASON_HEADINGS = {"CRITICAL": "Why it was flagged", "HIGH": "Why it was flagged", "MEDIUM": "Why it was flagged",
                   "LOW": "Why it was cleared", "NOT_ASSESSED": "Why it wasn't scored"}


def explain(reasons: list, band: str | None) -> None:
    """Explanation panel for any decision. Low-risk decisions are explained too: the factors that
    pulled the probability down come first, because that is what an analyst checks on a missed fraud."""
    st.subheader(REASON_HEADINGS.get(band, "Why"))
    reasons = reasons or []
    model_reasons = [r for r in reasons if r.get("direction") != "routing rule"]
    if band == "LOW":
        reasons = sorted(reasons, key=lambda r: r.get("impact_pct_points", 0.0))
        if model_reasons:
            down = [r for r in model_reasons if r.get("impact_pct_points", 0.0) < 0]
            st.caption(f"{len(down)} of {len(model_reasons)} listed factors lowered the fraud probability. "
                       "Factors that raised it are still shown, so a cleared decision can be challenged.")
        else:
            st.markdown("<div class='fa-note'><strong>No explanation returned.</strong> The model scored this "
                        "transaction as low risk, but the API did not attach the reasons. Cleared decisions need "
                        "an explanation as much as flagged ones: it is how a missed fraud gets diagnosed.</div>",
                        unsafe_allow_html=True)
            return
    if band == "NOT_ASSESSED" and not reasons:
        st.caption("No model ran on this transaction, so there are no feature contributions to show.")
        return
    show_reasons(reasons)


def show_reasons(reasons: list) -> None:
    if not reasons:
        st.caption("No reasons recorded.")
        return
    rows = []
    for r in reasons:
        name, desc = html.escape(str(r.get("reason", ""))), html.escape(str(r.get("description", "")))
        if r.get("direction") == "routing rule":
            pts_html = "<span class='pts' style='color:#98A2B3'>Rule</span>"
        else:
            pts = r.get("impact_pct_points", 0.0)
            colour, arrow = (UP, "▲") if pts > 0 else (DOWN, "▼")
            pts_html = f"<span class='pts' style='color:{colour}'>{arrow} {pts:+.1f}</span>"
        rows.append(f"<div class='fa-reason'>{pts_html}<span class='txt'><b>{name}</b>. {desc}</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    if any(r.get("direction") != "routing rule" for r in reasons):
        st.caption("Values are percentage-point contributions to the fraud probability.")


def show_rules(rules_fired: list) -> None:
    if not rules_fired:
        st.caption("No rule conditions met.")
        return
    rows = []
    for r in rules_fired:
        acting = r["role"] != "UNUSED"
        colour = T["orange"] if acting else T["ink3"]
        note = "" if acting else " Monitored only: it failed validation precision."
        rows.append(f"<div class='fa-reason'><span class='pts' style='color:{colour}'>"
                    f"{html.escape(r['role'].title())}</span><span class='txt'><b>{html.escape(r['rule'])}</b>. "
                    f"{html.escape(r['description'])}{note}</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def card(title: str, body: str, state: str | None = None, colour: str | None = None) -> str:
    colour = colour or T["ink3"]
    st_html = (f"<div class='state'><i style='background:{colour}'></i>{html.escape(state)}</div>"
               if state else "")
    return f"<div class='fa-card'>{st_html}<h4>{html.escape(title)}</h4><p>{body}</p></div>"


def empty(message: str) -> None:
    st.markdown(f"<div class='fa-empty'>{html.escape(message)}</div>", unsafe_allow_html=True)


def wait_for_explanation(tx_id: str, seconds: float = 8.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        code, rec = api("GET", f"/transactions/{tx_id}")
        if code == 200 and rec.get("explanation_status") == "COMPLETE":
            return rec
        time.sleep(0.4)
    return None


def go(page: str, **state) -> None:
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["page"] = page


def open_detail(tx_id: str) -> None:
    go("Transaction Detail", detail_tx=tx_id)


def panel():
    return st.container(border=True)


def selected_row(event) -> int | None:
    try:
        rows = event.selection.rows
        return rows[0] if rows else None
    except AttributeError:
        return None


def age_text(ts) -> str:
    if pd.isna(ts):
        return ""
    mins = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    if mins < 60:
        return f"{mins:.0f} min"
    if mins < 60 * 48:
        return f"{mins / 60:.1f} h"
    return f"{mins / 1440:.0f} d"


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_fraud"].notna()] if not df.empty else df


def detection_stats(df: pd.DataFrame) -> dict | None:
    lab = labelled(df)
    lab = lab[lab["tx_type"].isin(MODEL_TYPES)] if not lab.empty else lab
    if lab.empty:
        return None
    fraud, legit = lab[lab["is_fraud"] == 1], lab[lab["is_fraud"] == 0]
    tp = int(fraud["flagged"].sum())
    fp = int(legit["flagged"].sum())
    return {"fraud": len(fraud), "legit": len(legit), "tp": tp, "fp": fp,
            "recall": tp / len(fraud) if len(fraud) else None,
            "fpr": fp / len(legit) if len(legit) else None,
            "precision": tp / (tp + fp) if (tp + fp) else None}


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


# ── Sidebar ────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
code, health = api("GET", "/health")
health_ms = (time.perf_counter() - t0) * 1000
with st.sidebar:
    st.markdown(f"<div class='fa-brand'>{mark_svg()}<div><b>Mlinzi</b>"
                "<span>Mobile money fraud intelligence</span></div></div>", unsafe_allow_html=True)
    if code == 200:
        st.markdown("<div class='fa-status'><i style='background:#30D158;box-shadow:0 0 8px #30D158'></i>"
                    "API online</div>", unsafe_allow_html=True)
        st.caption(f"Model {health.get('model_version')}")
    else:
        st.markdown("<div class='fa-status'><i style='background:#FF453A'></i>API offline</div>",
                    unsafe_allow_html=True)
        st.code("python -m uvicorn src.api:app --port 8000", language="bash")
    for group, pages in NAV.items():
        st.markdown(f"<div class='fa-group'>{group}</div>", unsafe_allow_html=True)
        for p in pages:
            active = st.session_state["page"] == p
            nav_kwargs = dict(key=f"nav_{p}", on_click=go, args=(p,), type="primary" if active else "secondary")
            try:
                stretch(st.button, p, icon=NAV_ICONS.get(p), **nav_kwargs)
            except TypeError:  # Streamlit without button icons
                stretch(st.button, p, **nav_kwargs)
    st.divider()
    st.caption("Research prototype on simulated PaySim data. Not connected to any live payment system. "
               "The model covers TRANSFER and CASH_OUT only.")

page = st.session_state["page"]
API_UP = code == 200
if not API_UP and page not in OFFLINE_OK:
    page_title(page)
    st.warning("This page needs the API. Start it, then refresh.")
    st.stop()


# ── Monitor ────────────────────────────────────────────────────────────────
def page_command():
    _, m = api("GET", "/metrics")
    m = m or {}
    _, alerts = api("GET", "/alerts?limit=500")
    alerts = alerts or []
    head, actions = st.columns([3, 1.4], vertical_alignment="center")
    with head:
        page_title("Command Centre", "What the system is deciding right now, how well it is catching fraud, "
                                     "and what is waiting for an analyst.")
    with actions:
        a1, a2 = st.columns(2)
        a1.button("New Transaction", type="primary", icon=":material/add:", on_click=go, args=("Send a Transaction",))
        a2.button(f"Alerts ({len(alerts)})", icon=":material/notifications:", on_click=go, args=("Alert Queue",))
    act = load_activity()
    cases = cases_df()
    stats = detection_stats(act)
    open_cases = cases[~cases["status"].str.startswith("Closed")] if not cases.empty else cases

    bands = m.get("transactions_by_band", {})
    per_step = act.groupby("step").agg(n=("tx_id", "size"), flagged=("flagged", "sum"),
                                       latency=("latency_ms", "median")).tail(48) if not act.empty else None
    lab = labelled(act)
    lab = lab[lab["tx_type"].isin(MODEL_TYPES)] if not lab.empty else lab
    recall_trace = fpr_trace = []
    if not lab.empty:
        g = lab.groupby("step").apply(lambda d: pd.Series({
            "tp": int((d["flagged"] & (d["is_fraud"] == 1)).sum()), "pos": int((d["is_fraud"] == 1).sum()),
            "fp": int((d["flagged"] & (d["is_fraud"] == 0)).sum()), "neg": int((d["is_fraud"] == 0).sum())}))
        cum = g.cumsum()
        recall_trace = (cum["tp"] / cum["pos"].replace(0, np.nan)).tail(48).tolist()
        fpr_trace = (cum["fp"] / cum["neg"].replace(0, np.nan)).tail(48).tolist()
    band_counts = {b: 0 for b in ALL_BANDS}
    for a in alerts:
        band_counts[effective_band(a["risk_band"], a.get("tx_type")) or "LOW"] += 1
    status_counts = [int((open_cases["status"] == s_).sum()) for s_ in CASE_STATUSES[:3]] if not open_cases.empty else []
    rec_v, rec_u = split_value(pct(stats and stats["recall"]))
    fpr_v, fpr_u = split_value(pct(stats and stats["fpr"]))
    med = act["latency_ms"].median() if not act.empty else None
    top = st.columns([2.2, 1, 1, 1, 1], gap="medium")
    top[0].markdown(hero_card(
        "Overall", "Live decisions",
        [(f"{sum(bands.values()):,}", "Transactions decided"), (f"{len(alerts):,}", "Open alerts")],
        [(f"{band_counts['MEDIUM']:,}", "Medium", BAND_COLOURS["MEDIUM"], True),
         (f"{band_counts['HIGH']:,}", "High", BAND_COLOURS["HIGH"], False),
         (f"{band_counts['CRITICAL']:,}", "Critical", BAND_COLOURS["CRITICAL"], False)]), unsafe_allow_html=True)
    top[1].markdown(widget("Detection Rate", rec_v, rec_u if stats else "", T["green"],
                           ring(stats["recall"], T["green"]) if stats else "",
                           "Labelled frauds flagged" if stats else "Run a replay"), unsafe_allow_html=True)
    top[2].markdown(widget("False Positives", fpr_v, fpr_u if stats else "", T["ink"],
                           sparkline(fpr_trace, T["orange"]), "Legitimate customers flagged" if stats else "Run a replay"),
                    unsafe_allow_html=True)
    top[3].markdown(widget("Decision Time", f"{med:.0f}" if med is not None else "n/a", "ms" if med is not None else "",
                           T["blue"], sparkline(per_step["latency"], T["blue"]) if per_step is not None else "",
                           "Median round trip"), unsafe_allow_html=True)
    top[4].markdown(widget("Open Cases", f"{len(open_cases):,}", "", T["ink"],
                           bars(status_counts, [T["blue"], T["c2"], T["orange"]]) if status_counts else "",
                           "Open, investigating, escalated"), unsafe_allow_html=True)
    if stats is None:
        st.caption("Detection and false positive rates appear after a live replay, which carries dataset labels.")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with panel():
            st.subheader("Decisions by simulation hour")
            if act.empty:
                empty("No activity yet. Send a transaction or start a live replay.")
            else:
                recent = act[act["step"] >= act["step"].max() - 71]
                g = recent.groupby(["step", "band_label"]).size().reset_index(name="n")
                show_chart(alt.Chart(g).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                    x=alt.X("step:O", title="Simulation step (hour)", axis=alt.Axis(labelAngle=0, labelOverlap=True)),
                    y=alt.Y("n:Q", title="Transactions", stack="zero"),
                    color=alt.Color("band_label:N", scale=band_scale(), title=None),
                    tooltip=["step", "band_label", "n"]).properties(height=260))
    with right:
        with panel():
            st.subheader("Where decisions landed")
            shown = [b for b in ALL_BANDS if bands.get(b) or b in RISK_BANDS]
            g = pd.DataFrame({"band": [BAND_LABELS[b] for b in shown], "n": [bands.get(b, 0) for b in shown]})
            if g["n"].sum():
                show_chart(alt.Chart(g).mark_arc(innerRadius=62, cornerRadius=4, padAngle=0.02).encode(
                    theta="n:Q", color=alt.Color("band:N", scale=band_scale(shown), title=None),
                    tooltip=["band", "n"]).properties(height=260))
            else:
                empty("No decisions yet.")
            if "NOT_ASSESSED" not in bands:
                st.caption("Low includes PAYMENT, CASH_IN and DEBIT routed by rule until the API reports them separately.")

    left, right = st.columns(2, gap="large")
    with left:
        with panel():
            st.subheader("Highest-risk open alerts")
            if not alerts:
                empty("No open alerts.")
            else:
                top = sorted(alerts, key=lambda a: a.get("risk_score", 0), reverse=True)[:6]
                for a in top:
                    cols = st.columns([1.3, 1.2, 1.3, 1])
                    cols[0].markdown(pill(effective_band(a["risk_band"], a.get("tx_type"))), unsafe_allow_html=True)
                    cols[1].markdown(f"**{a.get('amount', 0):,.0f}**  \n{a.get('tx_type', '')}")
                    cols[2].caption(f"{a['tx_id'][:14]}…  \n{a.get('status', '').replace('_', ' ').lower()}")
                    cols[3].button("Open", key=f"cc_{a['tx_id']}", on_click=open_detail, args=(a["tx_id"],))
    with right:
        with panel():
            st.subheader("Ongoing cases")
            if open_cases.empty:
                empty("No open cases. Start one from the alert queue.")
            else:
                view = open_cases.head(8).copy()
                view["Case"] = view["case_id"].map(case_ref)
                st.dataframe(view[["Case", "status", "priority", "assignee", "updated_at"]].rename(columns={
                    "status": "Status", "priority": "Priority", "assignee": "Assigned To",
                    "updated_at": "Updated (UTC)"}), hide_index=True, width="stretch")
                st.button("Go to Cases", on_click=go, args=("Cases",))

    left, right = st.columns(2, gap="large")
    with left:
        with panel():
            st.subheader("Calibration against analyst outcomes")
            mon = m.get("monitoring", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Alerts reviewed", mon.get("reviewed_alerts", 0))
            if mon.get("reviewed_alerts"):
                c2.metric("Confirmed fraud rate", f"{mon['confirmed_fraud_rate_among_reviewed']:.1%}")
                c3.metric("Mean predicted risk", f"{mon['mean_predicted_risk_among_reviewed']:.1%}")
                if mon.get("recalibration_suggested"):
                    st.warning("Predicted risk and confirmed outcomes have drifted apart. Recalibration suggested.")
                else:
                    st.caption("Calibration gap within tolerance, or fewer than 50 reviews so far.")
    with right:
        with panel():
            st.subheader("Audit log integrity")
            st.caption("Every decision, verification and review is hash-chained. Any later edit breaks the chain.")
            if st.button("Verify Hash Chain"):
                _, chain = api("GET", "/audit/verify")
                if chain and chain.get("valid"):
                    st.success(f"Chain intact: {chain['events_checked']:,} events verified.")
                elif chain:
                    st.error(f"Chain broken at event {chain['first_broken_event_id']} "
                             f"({chain['event_type']} for {chain['tx_id']}).")


def page_patterns():
    page_title("Velocity and Patterns", "Bursts of risky activity, when risk concentrates, and recipient "
                                        "accounts filling up faster than normal.")
    act = load_activity()
    if act.empty:
        empty("No activity yet. Start a live replay to populate these views.")
        return

    with panel():
        st.subheader("Alert bursts")
        st.caption("Flagged transactions per simulation hour against a rolling baseline. A burst is an hour "
                   "above the 24-hour rolling median plus three median absolute deviations.")
        g = act.groupby("step").agg(total=("tx_id", "size"), flagged=("flagged", "sum")).reset_index()
        g["baseline"] = g["flagged"].rolling(24, min_periods=6).median()
        mad = (g["flagged"] - g["baseline"]).abs().rolling(24, min_periods=6).median()
        g["threshold"] = g["baseline"] + 3 * mad.clip(lower=1)
        g["burst"] = g["flagged"] > g["threshold"]
        base = alt.Chart(g).encode(x=alt.X("step:Q", title="Simulation step (hour)"))
        layers = (base.mark_area(opacity=.35, color=T["neutral"]).encode(y=alt.Y("total:Q", title="Transactions"))
                  + base.mark_line(color=T["orange"], strokeWidth=2).encode(y="flagged:Q")
                  + base.mark_line(color=T["neutral2"], strokeDash=[4, 4]).encode(y="threshold:Q")
                  + base.transform_filter("datum.burst").mark_point(color=T["red"], size=70, filled=True)
                  .encode(y="flagged:Q", tooltip=["step", "flagged", "total"]))
        show_chart(layers.properties(height=280))
        st.caption(f"{int(g['burst'].sum())} burst hours detected. Grey area is all traffic, orange is flagged, "
                   "dashed is the burst threshold.")

    left, right = st.columns(2, gap="large")
    with left:
        with panel():
            st.subheader("Flag rate by hour of day")
            h = act[act["tx_type"].isin(MODEL_TYPES)].groupby(["hour", "tx_type"]).agg(
                n=("tx_id", "size"), rate=("flagged", "mean")).reset_index()
            if h.empty:
                empty("No TRANSFER or CASH_OUT activity yet.")
            else:
                show_chart(alt.Chart(h).mark_rect(cornerRadius=3).encode(
                    x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("tx_type:N", title=None),
                    color=alt.Color("rate:Q", title="Flag rate", scale=alt.Scale(scheme="inferno"),
                                    legend=alt.Legend(format="%")),
                    tooltip=["hour", "tx_type", "n", alt.Tooltip("rate:Q", format=".1%")]).properties(height=160))
                st.caption("PaySim has no location data, so time is the available contextual axis.")
    with right:
        with panel():
            st.subheader("Amount by band")
            a = act[act["amount"] > 0]
            show_chart(alt.Chart(a).mark_tick(opacity=.5, thickness=1.5).encode(
                x=alt.X("amount:Q", scale=alt.Scale(type="log"), title="Amount (log scale)"),
                y=alt.Y("band_label:N", title=None, sort=[BAND_LABELS[b] for b in ALL_BANDS]),
                color=alt.Color("band_label:N", scale=band_scale(), legend=None)).properties(height=160))

    with panel():
        st.subheader("Fast-filling recipients")
        window = st.slider("Window (simulation hours)", 6, 72, 24, step=6)
        latest = act["step"].max()
        w = act[act["step"] > latest - window]
        r = w.groupby("name_dest").agg(inbound=("tx_id", "size"), senders=("name_orig", "nunique"),
                                       value=("amount", "sum"), flagged=("flagged", "sum"),
                                       max_risk=("risk_score", "max")).reset_index()
        r = r[r["inbound"] >= 2].sort_values(["flagged", "inbound"], ascending=False).head(25)
        if r.empty:
            empty("No recipient received more than one transaction in this window.")
        else:
            st.dataframe(r.rename(columns={"name_dest": "Recipient", "inbound": "Inbound", "senders": "Distinct Senders",
                                           "value": "Value Received", "flagged": "Flagged",
                                           "max_risk": "Highest Risk"}),
                         hide_index=True, width="stretch",
                         column_config={"Value Received": st.column_config.NumberColumn(format="%.0f"),
                                        "Highest Risk": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                        format="%.2f")})
            st.caption("Many distinct senders paying one account in a short window is a common mule-account signal.")


def page_network():
    page_title("Network", "Senders linked through the recipient accounts they paid. Shared recipients with "
                          "several flagged inbound payments are the shape a mule account or fraud ring leaves.")
    act = load_activity()
    if act.empty:
        empty("No activity yet. Start a live replay to populate the graph.")
        return
    c1, c2, c3 = st.columns(3)
    hubs_n = c1.slider("Recipients to show", 3, 25, 10)
    min_in = c2.slider("Minimum inbound payments", 2, 20, 2)
    only_flagged = c3.toggle("Only recipients with a flagged payment", value=True)

    stats = act.groupby("name_dest").agg(inbound=("tx_id", "size"), flagged=("flagged", "sum"),
                                         value=("amount", "sum")).reset_index()
    stats = stats[stats["inbound"] >= min_in]
    if only_flagged:
        stats = stats[stats["flagged"] > 0]
    hubs = stats.sort_values(["flagged", "inbound"], ascending=False).head(hubs_n)
    if hubs.empty:
        empty("No recipients match these filters yet. Lower the minimum or replay more transactions.")
        return
    edges = act[act["name_dest"].isin(hubs["name_dest"])].sort_values("risk_score", ascending=False).head(220)

    def q(s):
        return '"' + str(s).replace('"', "") + '"'
    lines = ['digraph G {', 'bgcolor="transparent"; rankdir=LR; splines=true; overlap=false; nodesep=0.12;',
             'node [fontname="Helvetica", fontsize=9, fontcolor="#FFFFFF", style="filled", penwidth=0];',
             'edge [arrowsize=0.45, penwidth=1.1];']
    for h in hubs.itertuples():
        frac = h.flagged / h.inbound
        colour = T["red"] if frac >= .5 else T["orange"] if h.flagged else T["neutral3"]
        lines.append(f'{q(h.name_dest)} [shape=circle, width={0.55 + min(h.inbound, 30) / 30:.2f}, '
                     f'fillcolor="{colour}", label="{h.name_dest[-6:]}\\n{h.inbound} in"];')
    for s in edges["name_orig"].unique():
        lines.append(f'{q(s)} [shape=point, width=0.09, fillcolor="{T["neutral2"]}"];')
    for e in edges.itertuples():
        lines.append(f'{q(e.name_orig)} -> {q(e.name_dest)} [color="{BAND_COLOURS.get(e.band, "#666")}"];')
    lines.append("}")
    with panel():
        stretch(st.graphviz_chart, "\n".join(lines))
        st.caption("Large circles are recipients (last six digits shown); red means at least half their inbound "
                   "payments were flagged. Small dots are senders. Edge colour is the risk band of that payment.")
    st.dataframe(hubs.rename(columns={"name_dest": "Recipient", "inbound": "Inbound", "flagged": "Flagged",
                                      "value": "Value Received"}), hide_index=True, width="stretch")
    st.caption("PaySim origin accounts rarely transact twice, so links form through shared recipients rather than "
               "shared devices or IPs. Device and IP linkage needs a device-intelligence feed (see System health).")


# ── Investigate ────────────────────────────────────────────────────────────
def page_alerts():
    page_title("Alert Queue", "Held, declined and pending transactions awaiting action, highest risk first.")
    _, alerts = api("GET", "/alerts?limit=500")
    if not alerts:
        st.success("No open alerts.")
        return
    df = pd.DataFrame(alerts)
    df["band"] = [effective_band(b, t) for b, t in zip(df["risk_band"], df["tx_type"])]
    df["created"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["Age"] = df["created"].map(age_text)
    cases = cases_df()
    case_map = dict(zip(cases["tx_id"], cases["case_id"].map(case_ref))) if not cases.empty else {}
    df["Case"] = df["tx_id"].map(case_map).fillna("")

    c1, c2, c3 = st.columns(3)
    bands = c1.multiselect("Band", [b for b in ALL_BANDS if b in set(df["band"])],
                           format_func=BAND_LABELS.get)
    types = c2.multiselect("Type", sorted(df["tx_type"].unique()))
    min_amt = c3.number_input("Minimum amount", min_value=0.0, value=0.0, step=10_000.0)
    f = df.copy()
    if bands:
        f = f[f["band"].isin(bands)]
    if types:
        f = f[f["tx_type"].isin(types)]
    f = f[f["amount"] >= min_amt].sort_values("risk_score", ascending=False).reset_index(drop=True)
    f["Band"] = f["band"].map(BAND_LABELS)

    st.caption(f"{len(f):,} of {len(df):,} alerts. Select a row to act on it.")
    event = st.dataframe(
        f[["tx_id", "Band", "risk_score", "status", "tx_type", "amount", "name_dest", "Age", "Case"]].rename(
            columns={"tx_id": "Transaction", "risk_score": "Risk", "status": "Status", "tx_type": "Type",
                     "amount": "Amount", "name_dest": "Recipient"}),
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="alert_table",
        column_config={"Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                       "Amount": st.column_config.NumberColumn(format="%.0f")})
    i = selected_row(event)
    if i is None:
        return
    row = f.iloc[i]
    with panel():
        st.markdown(f"{pill(row['band'])}&nbsp;&nbsp;**{row['tx_id']}**, {row['tx_type']} of "
                    f"{row['amount']:,.0f} to {row['name_dest']}", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1, 1, 3])
        b1.button("Open Detail", type="primary", on_click=open_detail, args=(row["tx_id"],))
        existing = case_for(row["tx_id"])
        if existing:
            b2.button(f"Go to {case_ref(existing['case_id'])}", on_click=go, args=("Cases",),
                      kwargs={"case_sel": existing["case_id"]})
        elif b2.button("Start a Case"):
            prio = "P1 urgent" if row["band"] == "CRITICAL" else "P2 high" if row["band"] == "HIGH" else "P3 normal"
            cid = open_case(row["tx_id"], prio, st.session_state.get("analyst", "analyst"))
            go("Cases", case_sel=cid)
            st.rerun()


def case_report(case: dict, rec: dict | None, notes: list[dict]) -> str:
    lines = [f"# Fraud case report: {case_ref(case['case_id'])}", "",
             f"- Transaction: {case['tx_id']}", f"- Status: {case['status']}", f"- Priority: {case['priority']}",
             f"- Assigned to: {case['assignee']}", f"- Escalated to: {case.get('escalated_to') or 'Not escalated'}",
             f"- Opened (UTC): {case['opened_at']}", f"- Closed (UTC): {case.get('closed_at') or 'Open'}", ""]
    if rec:
        band = effective_band(rec["risk_band"], rec["tx_type"])
        lines += ["## Transaction", "",
                  f"{rec['tx_type']} of {rec['amount']:,.0f} from {rec['name_orig']} to {rec['name_dest']} "
                  f"at simulation step {rec['step']}.", "",
                  f"- Risk band: {BAND_LABELS.get(band, band)}",
                  f"- Fraud probability: {rec['risk_score']:.1%}" if band != "NOT_ASSESSED" else
                  "- Fraud probability: not scored (outside model coverage)",
                  f"- System status: {rec['status']}", "", "## Why the system flagged it", ""]
        for r in rec.get("reasons", []):
            pts = r.get("impact_pct_points")
            lines.append(f"- {r.get('reason')}: {r.get('description')}"
                         + (f" ({pts:+.1f} pts)" if isinstance(pts, (int, float)) else ""))
        lines += ["", "## Audit events", ""]
        lines += [f"- {e['created_at']} {e['event_type']} (hash {e['hash'][:16]}…)" for e in rec.get("events", [])]
    lines += ["", "## Investigation log", ""]
    lines += [f"- {n['created_at']} · {n['kind']} · {n['author']}: {n['note']}" for n in notes]
    lines += ["", "_Generated by the fraud analyst dashboard. Research prototype on simulated PaySim data._"]
    return "\n".join(lines)


def page_cases():
    page_title("Cases", "Track each investigation from detection to resolution: who owns it, what was found, "
                        "where it was escalated and how it closed.")
    st.session_state.setdefault("analyst", "analyst")
    cases = cases_df()
    if cases.empty:
        empty("No cases yet. Select an alert in the alert queue and start a case.")
        return
    case_colours = [T["blue"], T["c1"], T["orange"], T["red"], T["green"]]
    widget_row([widget(s_, f"{int((cases['status'] == s_).sum())}", "", c_)
                for s_, c_ in zip(CASE_STATUSES, case_colours)])
    closed = cases.dropna(subset=["closed_at"])
    if not closed.empty:
        dur = (pd.to_datetime(closed["closed_at"]) - pd.to_datetime(closed["opened_at"])).dt.total_seconds() / 3600
        st.caption(f"Median time to resolution: {dur.median():.1f} hours across {len(closed)} closed cases.")

    flt = st.segmented_control("Show", ["Active", "All", "Closed"], default="Active", key="case_filter")
    view = cases if flt == "All" else (cases[cases["status"].str.startswith("Closed")] if flt == "Closed"
                                       else cases[~cases["status"].str.startswith("Closed")])
    view = view.reset_index(drop=True)
    labels = {int(r.case_id): f"{case_ref(r.case_id)}  ·  {r.status}  ·  {r.priority}  ·  {r.assignee}"
              for r in view.itertuples()}
    if not labels:
        empty("No cases in this view.")
        return
    ids = list(labels)
    pre = st.session_state.get("case_sel")
    case_id = st.selectbox("Case", ids, index=ids.index(pre) if pre in ids else 0, format_func=labels.get)
    st.session_state["case_sel"] = case_id
    case = next(r for r in cases.to_dict("records") if r["case_id"] == case_id)
    code, rec = api("GET", f"/transactions/{case['tx_id']}")
    rec = rec if code == 200 else None

    left, right = st.columns([3, 2], gap="large")
    with left:
        if rec:
            band = effective_band(rec["risk_band"], rec["tx_type"])
            verdict(band, rec.get("risk_score"), rec["status"], "Amount", f"{rec['amount']:,.0f}",
                    f"{case_ref(case_id)}, {rec['tx_type']} from {rec['name_orig']} to {rec['name_dest']}")
            with panel():
                explain(rec.get("reasons", []), band)
        st.subheader("Investigation log")
        notes = case_notes(case_id)
        st.markdown("<div class='fa-timeline'>" + "".join(
            f"<div><b>{html.escape(n['kind'])}</b> <span>{html.escape(n['created_at'])} UTC, "
            f"{html.escape(n['author'])}</span><p>{html.escape(n['note'])}</p></div>" for n in notes)
            + "</div>", unsafe_allow_html=True)
        with st.form(f"note_{case_id}", clear_on_submit=True):
            note = st.text_area("Add a finding", placeholder="e.g. Recipient received 14 payments from new senders in 6 hours")
            if st.form_submit_button("Add to Log") and note.strip():
                update_case(case_id, st.session_state["analyst"], "Note", note.strip(), status=(
                    "Investigating" if case["status"] == "Open" else case["status"]))
                st.rerun()
    with right:
        closed_now = case["status"].startswith("Closed")
        with panel():
            st.subheader("Ownership")
            analyst = st.text_input("You are", key="analyst")
            a1, a2 = st.columns(2)
            assignee = a1.text_input("Assigned To", value=case["assignee"], disabled=closed_now)
            priority = a2.selectbox("Priority", CASE_PRIORITIES, index=CASE_PRIORITIES.index(case["priority"])
                                    if case["priority"] in CASE_PRIORITIES else 2, disabled=closed_now)
            if st.button("Save Ownership", disabled=closed_now):
                update_case(case_id, analyst, "Updated", f"Assigned to {assignee}, priority {priority}.",
                            assignee=assignee, priority=priority)
                st.rerun()
        with panel():
            st.subheader("Escalate")
            target = st.selectbox("Escalate to", ESCALATION_TARGETS, disabled=closed_now)
            reason = st.text_input("Reason", placeholder="e.g. Suspected mule network, more than five victims",
                                   disabled=closed_now)
            if st.button("Escalate Case", disabled=closed_now or not reason.strip()):
                update_case(case_id, analyst, "Escalated", f"To {target}: {reason.strip()}",
                            status="Escalated", escalated_to=target)
                st.rerun()
            st.caption("Escalation is recorded in the case log. No message leaves this prototype.")
        with panel():
            st.subheader("Close")
            if closed_now:
                st.caption(f"Closed {case['closed_at']} UTC as {case['status'].split(': ')[1]}.")
            else:
                outcome = st.radio("Outcome", ["FRAUD", "LEGITIMATE"], horizontal=True,
                                   format_func=lambda o: "Confirmed fraud" if o == "FRAUD" else "Legitimate")
                summary = st.text_area("Closing summary", placeholder="What was found and why it was decided")
                if st.button("Close Case", type="primary", disabled=not summary.strip()):
                    c_, r_ = api("POST", f"/alerts/{case['tx_id']}/review",
                                 {"outcome": outcome, "analyst": analyst, "note": summary.strip()})
                    already = rec and rec["status"].startswith("REVIEWED")
                    if c_ == 200 or already:
                        update_case(case_id, analyst, "Closed",
                                    summary.strip() + ("" if c_ == 200 else " (review already on the audit log)"),
                                    status="Closed: fraud" if outcome == "FRAUD" else "Closed: legitimate",
                                    closed_at=now_utc())
                        st.rerun()
                    else:
                        st.error(f"The API rejected the review ({c_}): {r_}")
                st.caption("Closing posts the outcome to the API, so it lands in the hash-chained audit log.")
        st.download_button("Download Case Report", case_report(case, rec, case_notes(case_id)),
                           file_name=f"{case_ref(case_id)}.md", mime="text/markdown")


def page_detail():
    page_title("Transaction Detail")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    tx_id = st.text_input("Transaction ID", value=st.session_state.get("detail_tx", ""))
    if not tx_id:
        st.caption("Open a transaction from the alert queue, or paste an ID.")
        return
    tx_id = tx_id.strip()
    code, rec = api("GET", f"/transactions/{tx_id}")
    if code != 200:
        st.error("Transaction not found." if code in (404, 422) else f"Error {code}")
        return
    decision_event = next((e for e in rec["events"] if e["event_type"] == "DECISION"), None)
    decision = decision_event["payload"]["decision"] if decision_event else {}
    tx = decision_event["payload"]["transaction"] if decision_event else {}

    band = effective_band(rec["risk_band"], rec["tx_type"])
    verdict(band, rec.get("risk_score"), rec["status"], "Amount", f"{rec['amount']:,.0f}",
            f"{rec['tx_type']} from {rec['name_orig']} to {rec['name_dest']}, simulation step {rec['step']} "
            f"(hour {rec['step'] % 24:02d}:00), sender balance before {tx.get('oldbalanceOrg', 0) or 0:,.0f}, "
            f"model {decision.get('model_version', '')}")
    if band == "NOT_ASSESSED":
        coverage_note(rec["risk_band"], rec["tx_type"])

    existing = case_for(tx_id)
    if existing:
        st.button(f"Go to {case_ref(existing['case_id'])} ({existing['status'].lower()})", on_click=go,
                  args=("Cases",), kwargs={"case_sel": existing["case_id"]})
    elif st.button("Start a Case"):
        prio = "P1 urgent" if band == "CRITICAL" else "P2 high" if band == "HIGH" else "P3 normal"
        cid = open_case(tx_id, prio, st.session_state.get("analyst", "analyst"))
        go("Cases", case_sel=cid)
        st.rerun()

    left, right = st.columns(2, gap="large")
    with left:
        with panel():
            explain(rec["reasons"], band)
    with right:
        with panel():
            st.subheader("Rules")
            show_rules(decision.get("rules_fired", []))

    act = load_activity()
    if not act.empty:
        hist = act[(act["name_dest"] == rec["name_dest"]) & (act["tx_id"] != tx_id)]
        with panel():
            st.subheader("Other payments to this recipient")
            if hist.empty:
                st.caption("None seen by this dashboard.")
            else:
                st.caption(f"{len(hist)} payments from {hist['name_orig'].nunique()} senders, "
                           f"{int(hist['flagged'].sum())} flagged.")
                st.dataframe(hist.sort_values("step", ascending=False).head(15)[
                    ["tx_id", "step", "tx_type", "amount", "name_orig", "band_label", "risk_score"]].rename(columns={
                        "tx_id": "Transaction", "step": "Step", "tx_type": "Type", "amount": "Amount",
                        "name_orig": "Sender", "band_label": "Band", "risk_score": "Risk"}),
                    hide_index=True, width="stretch")

    st.subheader("Event history")
    st.caption("Each event's hash includes the previous one, so any later edit breaks the chain.")
    st.dataframe(pd.DataFrame([{"#": e["event_id"], "Event": e["event_type"], "Time (UTC)": e["created_at"],
                                "Hash": e["hash"][:20] + "..."} for e in rec["events"]]),
                 hide_index=True, width="stretch")

    if not rec["status"].startswith("REVIEWED"):
        st.subheader("Quick review")
        st.caption("For a straightforward alert. Use a case when it needs investigation or escalation.")
        with st.form(f"review_{tx_id}"):
            outcome = st.radio("Outcome", ["FRAUD", "LEGITIMATE"], horizontal=True,
                               format_func=lambda o: "Confirm fraud" if o == "FRAUD" else "Mark legitimate")
            analyst = st.text_input("Analyst", value=st.session_state.get("analyst", "analyst"))
            note = st.text_area("Note", placeholder="e.g. customer confirmed by call-back")
            if st.form_submit_button("Submit Review", type="primary"):
                c, r = api("POST", f"/alerts/{tx_id}/review", {"outcome": outcome, "analyst": analyst, "note": note})
                if c == 200:
                    st.session_state["flash"] = f"Review recorded: {r['status'].replace('_', ' ').title()}"
                    st.rerun()
                else:
                    st.error(f"Review rejected ({c}): {r}")


SQL_EXAMPLES = {
    "Recipients paid by the most distinct senders": (
        "SELECT name_dest, COUNT(*) AS payments, COUNT(DISTINCT name_orig) AS senders,\n"
        "       SUM(band IN ('MEDIUM','HIGH','CRITICAL')) AS flagged, ROUND(SUM(amount)) AS value\n"
        "FROM activity\nGROUP BY name_dest HAVING payments > 1\nORDER BY senders DESC, flagged DESC\nLIMIT 50;"),
    "Missed frauds (labelled, not flagged)": (
        "SELECT tx_id, step, tx_type, amount, old_balance, band, ROUND(risk_score, 3) AS risk\n"
        "FROM activity\nWHERE is_fraud = 1 AND band NOT IN ('MEDIUM','HIGH','CRITICAL')\nORDER BY amount DESC;"),
    "False alarms by type and band": (
        "SELECT tx_type, band, COUNT(*) AS n\nFROM activity\nWHERE is_fraud = 0 AND band IN ('MEDIUM','HIGH','CRITICAL')\n"
        "GROUP BY tx_type, band ORDER BY n DESC;"),
    "Accounts that were fully drained": (
        "SELECT tx_id, tx_type, amount, old_balance, band, ROUND(risk_score, 3) AS risk\n"
        "FROM activity\nWHERE old_balance > 0 AND amount >= old_balance\nORDER BY risk_score DESC LIMIT 100;"),
    "Case outcomes against model risk": (
        "SELECT c.status, COUNT(*) AS cases, ROUND(AVG(a.risk_score), 3) AS mean_risk\n"
        "FROM cases.cases c LEFT JOIN activity a ON a.tx_id = c.tx_id\nGROUP BY c.status;"),
}


def run_sql(query: str) -> pd.DataFrame:
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise ValueError("Run one statement at a time.")
    if not re.match(r"(?is)^(select|with)\b", q):
        raise ValueError("Only SELECT queries are allowed here.")
    conn = sqlite3.connect(ACTIVITY_PATH.as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute(f"ATTACH DATABASE '{CASES_PATH.as_uri()}?mode=ro' AS cases")
        return pd.read_sql_query(f"SELECT * FROM ({q}) LIMIT 5000", conn)
    finally:
        conn.close()


def page_explore():
    page_title("Explore and SQL", "Drill into everything this dashboard has sent, export what you find, or "
                                  "query it directly.")
    act = load_activity()
    tabs = st.tabs(["Drill-down", "SQL console"])
    with tabs[0]:
        if act.empty:
            empty("No activity yet. Send a transaction or start a live replay.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            types = c1.multiselect("Type", TX_TYPES)
            bands = c2.multiselect("Band", ALL_BANDS, format_func=BAND_LABELS.get)
            source = c3.multiselect("Source", sorted(act["source"].dropna().unique()))
            label = c4.segmented_control("Dataset label", ["Any", "Fraud", "Legitimate", "Unlabelled"], default="Any") or "Any"
            lo, hi = int(act["step"].min()), int(act["step"].max())
            c5, c6 = st.columns(2)
            steps = c5.slider("Simulation steps", lo, max(hi, lo + 1), (lo, max(hi, lo + 1)))
            risk = c6.slider("Fraud probability", 0.0, 1.0, (0.0, 1.0), step=0.01)
            f = act[act["step"].between(*steps)]
            f = f[f["risk_score"].fillna(0).between(*risk)]
            if types:
                f = f[f["tx_type"].isin(types)]
            if bands:
                f = f[f["band"].isin(bands)]
            if source:
                f = f[f["source"].isin(source)]
            if label != "Any":
                f = f[f["is_fraud"].isna()] if label == "Unlabelled" else f[f["is_fraud"] == (label == "Fraud")]
            f = f.sort_values("risk_score", ascending=False).reset_index(drop=True)
            m = st.columns(4)
            m[0].metric("Transactions", f"{len(f):,}")
            m[1].metric("Value", f"{f['amount'].sum():,.0f}")
            m[2].metric("Flagged", f"{int(f['flagged'].sum()):,}")
            m[3].metric("Labelled fraud", f"{int((f['is_fraud'] == 1).sum()):,}")
            event = st.dataframe(
                f[["tx_id", "step", "tx_type", "amount", "name_orig", "name_dest", "band_label", "risk_score",
                   "status", "is_fraud", "source"]].rename(columns={
                    "tx_id": "Transaction", "step": "Step", "tx_type": "Type", "amount": "Amount",
                    "name_orig": "Sender", "name_dest": "Recipient", "band_label": "Band", "risk_score": "Risk",
                    "status": "Status", "is_fraud": "Label", "source": "Source"}),
                hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="explore_table",
                column_config={"Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                               "Amount": st.column_config.NumberColumn(format="%.0f")})
            b1, b2 = st.columns([1, 4])
            b1.download_button("Export CSV", f.drop(columns=["flagged"]).to_csv(index=False),
                               file_name="fraud_explore_export.csv", mime="text/csv")
            i = selected_row(event)
            if i is not None:
                b2.button(f"Open {f.iloc[i]['tx_id']}", type="primary", on_click=open_detail,
                          args=(f.iloc[i]["tx_id"],))
    with tabs[1]:
        st.caption("Read-only SQLite. Tables: activity, and cases.cases and cases.case_notes. "
                   "Results are capped at 5,000 rows.")
        ex = st.selectbox("Start from an example", list(SQL_EXAMPLES))
        query = st.text_area("Query", value=SQL_EXAMPLES[ex], height=170, key=f"sql_{ex}")
        if st.button("Run Query", type="primary"):
            try:
                res = run_sql(query)
                st.caption(f"{len(res):,} rows")
                st.dataframe(res, hide_index=True, width="stretch")
                st.download_button("Export Result", res.to_csv(index=False), file_name="query_result.csv",
                                   mime="text/csv")
            except (ValueError, sqlite3.Error, pd.errors.DatabaseError) as e:
                st.error(str(e))


# ── Operate ────────────────────────────────────────────────────────────────
PRESETS = {
    "Everyday payment": {"type": "PAYMENT", "amount": 1_500.0, "hour": 14, "balance": 20_000.0},
    "Risky night transfer": {"type": "TRANSFER", "amount": 181_000.0, "hour": 4, "balance": 181_000.0},
    "Daytime cash-out to new account": {"type": "CASH_OUT", "amount": 60_000.0, "hour": 10, "balance": 60_000.0},
    "Large payment to a new till": {"type": "PAYMENT", "amount": 1_500_000.0, "hour": 22, "balance": 2_000_000.0},
    "Custom": {"type": "TRANSFER", "amount": 10_000.0, "hour": 12, "balance": 50_000.0},
}


def page_send():
    page_title("Send a Transaction", "Acts as the mobile money app: the transaction goes to the API "
                                     "and the decision comes back.")
    cur = load_cursor()
    day = cur["last_step"] // 24
    preset = st.selectbox("Start from an example", list(PRESETS), key="preset")
    p = PRESETS[preset]
    with st.form(f"send_{preset}"):
        c1, c2, c3 = st.columns(3)
        tx_type = c1.selectbox("Type", TX_TYPES, index=TX_TYPES.index(p["type"]))
        amount = c2.number_input("Amount", min_value=1.0, max_value=1e10, value=p["amount"], step=1_000.0)
        hour = c3.slider("Hour of day", 0, 23, p["hour"])
        c4, c5, c6 = st.columns(3)
        balance = c4.number_input("Sender's balance before", min_value=0.0, value=p["balance"], step=1_000.0)
        name_orig = c5.text_input("Sender account", value=new_account())
        name_dest = c6.text_input("Recipient account (new by default)",
                                  value=new_account("M" if tx_type == "PAYMENT" else "C"))
        st.caption(f"Simulation day {day}. It follows the replay position so history stays in time order.")
        submitted = st.form_submit_button("Send Transaction", type="primary")

    if submitted:
        step = day * 24 + hour
        body = {"step": int(step), "type": tx_type, "amount": float(amount),
                "nameOrig": name_orig.strip(), "nameDest": name_dest.strip(), "oldbalanceOrg": float(balance)}
        t = time.perf_counter()
        code, res = api("POST", "/transactions", body)
        latency = (time.perf_counter() - t) * 1000
        if code != 201:
            st.error(f"Rejected ({code}): {res}")
            return
        log_activity([activity_row(body, res, "manual", latency)])
        st.session_state["last_decision"] = res
        st.session_state["last_body"] = body

    res = st.session_state.get("last_decision")
    if not res:
        return
    sent_type = st.session_state.get("last_body", {}).get("type")
    if res.get("explanation_status") == "PENDING":
        with st.spinner("Decision returned. Computing the explanation ..."):
            rec = wait_for_explanation(res["tx_id"])
        if rec:
            res["top_reasons"], res["explanation_status"] = rec["reasons"], "COMPLETE"
            st.session_state["last_decision"] = res

    band = effective_band(res["risk_band"], sent_type)
    verdict(band, res.get("risk_score"), res["status"], "Decision time",
            f"{res['timings_ms'].get('decision_total_ms', 0):.0f} ms",
            f"Transaction {res['tx_id']}, recommended action {res['recommended_action']}")
    if band == "NOT_ASSESSED":
        coverage_note(res["risk_band"], sent_type)

    left, right = st.columns(2, gap="large")
    with left:
        with panel():
            explain(res.get("top_reasons", []), band)
    with right:
        with panel():
            st.subheader("Rules")
            show_rules(res.get("rules_fired", []))

    if res["status"] == "PENDING_VERIFICATION":
        st.write("")
        with panel():
            st.subheader("Step-up verification")
            _, msgs = api("GET", f"/outbox?tx_id={res['tx_id']}")
            if msgs:
                st.info(f"Simulated SMS to {msgs[0]['recipient']}:\n\n{msgs[0]['message']}")
            code_in = st.text_input("Enter the 6-digit code", max_chars=6, key=f"code_{res['tx_id']}")
            if st.button("Verify Code", type="primary"):
                c, r = api("POST", f"/transactions/{res['tx_id']}/verify", {"code": code_in})
                if c == 200 and r["result"] == "VERIFIED":
                    st.success("Verified. The transaction can proceed.")
                    res["status"] = "VERIFIED"
                elif c == 200:
                    st.error(f"{r['result'].replace('_', ' ').title()}"
                             + (f" ({r['attempts_left']} attempts left)" if "attempts_left" in r else ""))
                else:
                    st.error("Enter exactly 6 digits.")
    if res["status"] in ("HELD", "DECLINED"):
        st.button("Open in Transaction Detail", on_click=open_detail, args=(res["tx_id"],))


def page_verification():
    page_title("Verification", "The simulated SMS channel. Codes appear here instead of on a phone.")
    _, msgs = api("GET", "/outbox?limit=20")
    if msgs:
        st.dataframe(pd.DataFrame(msgs)[["tx_id", "recipient", "message", "created_at"]]
                     .rename(columns={"tx_id": "Transaction", "recipient": "To", "message": "Message",
                                      "created_at": "Sent (UTC)"}), hide_index=True, width="stretch")
    else:
        st.caption("No messages yet. A code is sent when a transaction needs step-up verification.")
    st.write("")
    with st.form("verify_form"):
        c1, c2 = st.columns(2)
        tx_id = c1.text_input("Transaction ID")
        code_in = c2.text_input("6-digit code", max_chars=6)
        if st.form_submit_button("Verify", type="primary"):
            c, r = api("POST", f"/transactions/{tx_id.strip()}/verify", {"code": code_in})
            if c == 200:
                (st.success if r["result"] == "VERIFIED" else st.error)(
                    f"{r['result'].replace('_', ' ').title()}. Status {r['status'].replace('_', ' ').lower()}.")
            else:
                st.error("Check the transaction ID and that the code is 6 digits.")


def page_replay():
    page_title("Live Replay", "Streams real test-period transactions, never seen in training, into the API "
                              "in their original order. It resumes where it stopped; reseeding the feature "
                              "store restarts it.")
    if store_seed_stamp() is None:
        st.error("Feature store not seeded. Run: python -m src.feature_store --seed")
        return
    frame = replay_frame()
    cur = load_cursor()
    st.progress(cur["position"] / len(frame),
                text=f"Position {cur['position']:,} of {len(frame):,} test-period transactions "
                     f"(simulation step {cur['last_step']})")
    c1, c2 = st.columns(2)
    n = c1.number_input("Transactions to send", min_value=10, max_value=10_000, value=300, step=50)
    delay = c2.slider("Pause between transactions (ms)", 0, 500, 0, step=25)
    if cur["position"] >= len(frame):
        st.info("All test-period transactions have been replayed. Reseed the feature store to start again.")
        return
    if not st.button("Start Replay", type="primary"):
        return

    batch = frame.iloc[cur["position"]: cur["position"] + int(n)]
    bar = st.progress(0.0)
    live = st.empty()
    counts = {b: 0 for b in ALL_BANDS}
    fraud_sent = fraud_flagged = fraud_held = legit_flagged = 0
    latencies, pending = [], []
    for i, r in enumerate(batch.itertuples(index=False), start=1):
        body = {"step": int(r.step), "type": r.type, "amount": float(r.amount), "nameOrig": r.nameOrig,
                "nameDest": r.nameDest, "oldbalanceOrg": float(r.oldbalanceOrg)}
        t = time.perf_counter()
        code, res = api("POST", "/transactions", body)
        latency = (time.perf_counter() - t) * 1000
        latencies.append(latency)
        cur["position"] += 1
        cur["last_step"] = int(r.step)
        save_cursor(cur)
        if code != 201:
            st.error(f"Stopped: API returned {code} for row {cur['position']}: {res}")
            break
        pending.append(activity_row(body, res, "replay", latency, int(r.isFraud)))
        band = effective_band(res["risk_band"], r.type)
        counts[band] = counts.get(band, 0) + 1
        flagged = band in FLAGGED
        if r.isFraud:
            fraud_sent += 1
            fraud_flagged += flagged
            fraud_held += band in ("HIGH", "CRITICAL")
        else:
            legit_flagged += flagged
        if delay:
            time.sleep(delay / 1000)
        if i % 10 == 0 or i == len(batch):
            log_activity(pending)
            pending = []
            bar.progress(i / len(batch), text=f"{i:,} / {len(batch):,} sent")
            with live.container():
                cols = st.columns(5)
                for col, b in zip(cols, ALL_BANDS):
                    col.metric(BAND_LABELS[b], f"{counts.get(b, 0):,}")
                cols = st.columns(4)
                cols[0].metric("Real frauds sent", fraud_sent)
                cols[1].metric("Frauds flagged (Medium+)", fraud_flagged)
                cols[2].metric("Frauds held or declined", fraud_held)
                cols[3].metric("Legitimate customers flagged", legit_flagged)
    log_activity(pending)
    if latencies:
        st.success(f"Replay batch finished. Median round-trip {np.median(latencies):.0f} ms per transaction "
                   f"(95th percentile {np.percentile(latencies, 95):.0f} ms), including HTTP.")
        st.caption("Fraud labels come from the dataset and are stored only for evaluation views; the API never "
                   "sees them. Not assessed counts PAYMENT, CASH_IN and DEBIT, which carry no fraud labels.")


def page_tuning():
    page_title("Rule Tuning", "See what a different alert threshold would do before anyone changes it. "
                              "Uses labelled replay results; the live bands are set in the API.")
    act = load_activity()
    lab = labelled(act)
    lab = lab[lab["tx_type"].isin(MODEL_TYPES) & lab["risk_score"].notna()] if not lab.empty else lab
    tabs = st.tabs(["Threshold what-if", "Candidate rules"])
    with tabs[0]:
        if lab.empty or lab["is_fraud"].sum() == 0:
            empty("Needs labelled replay results that include at least one fraud. Run a longer live replay.")
        else:
            y, s = lab["is_fraud"].astype(int).to_numpy(), lab["risk_score"].to_numpy()
            thr = st.slider("Alert when fraud probability is at least", 0.01, 0.99, 0.50, step=0.01)
            pred = s >= thr
            tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
            fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
            c = st.columns(5)
            c[0].metric("Frauds caught", f"{tp} of {tp + fn}")
            c[1].metric("Recall", pct(tp / (tp + fn) if tp + fn else None))
            c[2].metric("Precision", pct(tp / (tp + fp) if tp + fp else None))
            c[3].metric("False positive rate", pct(fp / (fp + tn) if fp + tn else None))
            c[4].metric("Alerts per 1,000", f"{1000 * (tp + fp) / len(y):.1f}")
            grid = np.round(np.linspace(0.01, 0.99, 99), 2)
            rows = []
            for g in grid:
                p_ = s >= g
                tp_, fp_ = (p_ & (y == 1)).sum(), (p_ & (y == 0)).sum()
                rows.append({"threshold": g, "Recall": tp_ / y.sum(),
                             "Precision": tp_ / (tp_ + fp_) if tp_ + fp_ else np.nan})
            curve = pd.DataFrame(rows).melt("threshold", var_name="metric", value_name="value")
            rule = alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(color=T["ink"], strokeDash=[3, 3]).encode(
                x="threshold:Q")
            show_chart((alt.Chart(curve).mark_line(strokeWidth=2.2).encode(
                x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("metric:N", title=None, scale=alt.Scale(range=[T["c1"], T["c2"]])),
                tooltip=["threshold", "metric", alt.Tooltip("value:Q", format=".1%")]) + rule).properties(height=280))
            st.caption(f"Based on {len(y):,} labelled TRANSFER and CASH_OUT replay transactions. A small sample "
                       "gives an unstable curve; the thesis results use the full test set.")
    with tabs[1]:
        p = find_result("rules_validation.csv")
        if p:
            st.dataframe(pd.read_csv(p), hide_index=True, width="stretch")
            st.caption("Candidate rules scored on the validation period. None met the precision bar, so they run "
                       "as monitored-only and never change a decision.")
        else:
            st.caption("Not found yet: rules_validation.csv")


INTEGRATIONS = [
    ("Device intelligence", "SEON, ThreatMetrix or iovation", "Device fingerprints, emulator and bot signals, "
     "shared-device links between accounts. PaySim has no device fields."),
    ("SIM-swap check", "Mobile network operator API", "Recent SIM replacement on the sender's line, one of the "
     "strongest signals for social-engineering takeover in mobile money."),
    ("Geolocation", "Cell-tower or IP location", "Impossible travel and regional risk views. PaySim has no "
     "location fields, so no map is shown rather than an invented one."),
    ("CRM and KYC", "Customer records", "Account age, KYC tier and contact history to enrich each alert."),
    ("Case escalation", "Email or ticketing", "Send escalated cases to compliance or law enforcement. Escalation "
     "is recorded locally only."),
]


def page_system():
    page_title("System Health", "Whether each part of the pipeline is running, how fast, and which external "
                                "signals are connected.")
    _, m = api("GET", "/metrics")
    m = m or {}
    act = load_activity()
    lat = act.groupby("step")["latency_ms"].median().tail(48) if not act.empty else []
    widget_row([
        widget("API", "Online" if API_UP else "Offline", "", T["green"] if API_UP else T["red"]),
        widget("Health Check", f"{health_ms:.0f}", "ms", T["blue"], sparkline(lat, T["blue"]),
               "Decision latency trace below"),
        widget("Model", str((health or {}).get("model_version", "n/a")), "", T["c2"]),
        widget("Feature Store", "Seeded" if store_seed_stamp() else "Not seeded", "",
               T["green"] if store_seed_stamp() else T["orange"]),
    ])

    left, right = st.columns(2, gap="large")
    with left:
        with panel():
            st.subheader("Decision latency")
            if act.empty:
                empty("No activity yet.")
            else:
                l1, l2, l3 = st.columns(3)
                l1.metric("Median", f"{act['latency_ms'].median():.0f} ms")
                l2.metric("95th percentile", f"{act['latency_ms'].quantile(.95):.0f} ms")
                l3.metric("Slowest", f"{act['latency_ms'].max():.0f} ms")
                show_chart(alt.Chart(act[act["latency_ms"] < act["latency_ms"].quantile(.99)]).mark_bar(
                    color=T["neutral2"], opacity=.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                    x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=40), title="Round trip (ms)"),
                    y=alt.Y("count():Q", title="Transactions")).properties(height=200))
    with right:
        with panel():
            st.subheader("Pipeline status")
            statuses = m.get("transactions_by_status", {})
            if statuses:
                st.dataframe(pd.DataFrame(statuses.items(), columns=["Status", "Count"])
                             .sort_values("Count", ascending=False), hide_index=True, width="stretch")
            else:
                st.caption("No transactions yet.")
            if st.button("Verify Audit Hash Chain"):
                _, chain = api("GET", "/audit/verify")
                if chain and chain.get("valid"):
                    st.success(f"Chain intact: {chain['events_checked']:,} events verified.")
                elif chain:
                    st.error(f"Chain broken at event {chain['first_broken_event_id']}.")

    st.subheader("Controls in place")
    cols = st.columns(3)
    controls = [
        ("API key authentication", "Every request carries an API key; requests without it are refused."),
        ("Hash-chained audit log", "Decisions, verifications and reviews are chained, so tampering is detectable."),
        ("Simulated data only", "Accounts are PaySim identifiers. No real customer data enters the system."),
    ]
    for col, (t, b) in zip(cols, controls):
        col.markdown(card(t, html.escape(b), "Active", T["green"]), unsafe_allow_html=True)

    st.subheader("External signals")
    st.caption("Production fraud platforms combine the model with these feeds. None is connected in this prototype.")
    cols = st.columns(3)
    for i, (t, vendor, b) in enumerate(INTEGRATIONS):
        cols[i % 3].markdown(card(t, f"{html.escape(vendor)}. {html.escape(b)}", "Not connected", T["ink3"]),
                             unsafe_allow_html=True)
        if i % 3 == 2:
            cols = st.columns(3)


# ── Research ───────────────────────────────────────────────────────────────
RESULT_DIRS = [METRICS_DIR, FIGURES_DIR, SHAP_DIR, ROOT / "results"]
SELECTED_HINTS = ("soft", "vot", "ensemble", "rf+xgb", "rf + xgb")


def find_result(name: str) -> Path | None:
    """Locate a results file in the configured folders, then anywhere under results/."""
    for d in RESULT_DIRS:
        if (Path(d) / name).exists():
            return Path(d) / name
    base = ROOT / "results"
    hits = sorted(base.rglob(name)) if base.exists() else []
    return hits[0] if hits else None


def read_result(name: str) -> pd.DataFrame | None:
    p = find_result(name)
    if p is None:
        return None
    try:
        df = pd.read_csv(p)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return None
    df.columns = df.columns.str.strip()  # some logs were written with padded column names
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def pick(df: pd.DataFrame, *candidates: str) -> str | None:
    """First column whose normalised name matches a candidate exactly, then by containment."""
    cols = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in cols:
            return cols[norm(c)]
    for c in candidates:
        for k, v in cols.items():
            if norm(c) in k:
                return v
    return None


def is_selected(label: str) -> bool:
    low = str(label).lower()
    return any(h in low for h in SELECTED_HINTS)


def show_csv(name: str, caption: str, sort: str | None = None, cols: list | None = None):
    df = read_result(name)
    if df is None:
        st.caption(f"Not found: {name}. Looked in {', '.join(sorted({str(Path(d).relative_to(ROOT)) for d in RESULT_DIRS if Path(d).is_relative_to(ROOT)}))} and below results/.")
        return
    if sort and sort in df:
        df = df.sort_values(sort, ascending=False)
    if cols:
        keep = [c for c in cols if c in df]
        df = df[keep] if keep else df
    st.markdown(f"**{caption}**")
    st.dataframe(df, hide_index=True, width="stretch")


def show_figure(name: str, caption: str):
    p = find_result(name)
    if p:
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"Not found: {name}")


def finding(label: str, text: str) -> str:
    return f"<div class='fa-finding'><b>{html.escape(label)}</b><span>{text}</span></div>"


def model_ranking(log: pd.DataFrame) -> pd.DataFrame | None:
    metric = pick(log, "pr_auc", "prauc", "average_precision", "auprc", "pr_auc_test")
    model = pick(log, "model", "model_name", "estimator", "name")
    if metric is None or model is None:
        return None
    fs = pick(log, "feature_set", "features")
    if fs:  # the balance ablation is discussed in the findings, not ranked against the candidates
        log = log[~log[fs].astype(str).str.contains("ablation", case=False)].reset_index(drop=True)
    lo, hi = pick(log, "pr_auc_ci_low", "ci_low", "lower"), pick(log, "pr_auc_ci_high", "ci_high", "upper")
    out = pd.DataFrame({"label": log[model].astype(str),
                        "pr_auc": pd.to_numeric(log[metric], errors="coerce")})
    out["lo"] = pd.to_numeric(log[lo], errors="coerce") if lo else np.nan
    out["hi"] = pd.to_numeric(log[hi], errors="coerce") if hi else np.nan
    for name, cands in {"recall": ("recall",), "precision": ("precision",), "fpr": ("fpr", "false_positive_rate"),
                        "train_s": ("train_time_s", "train_time", "fit_time"),
                        "infer_ms": ("inference_ms_per_txn", "inference_ms", "latency_ms")}.items():
        c = pick(log, *cands)
        out[name] = pd.to_numeric(log[c], errors="coerce") if c else np.nan
    out = out.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False).drop_duplicates("label")
    out["selected"] = out["label"].map(is_selected)
    out["role"] = np.where(out["selected"], "Deployed ensemble", "Candidate")
    return out.reset_index(drop=True)


def seed_long(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalise a seed-robustness table to long form: seed, model, value."""
    seed = pick(df, "seed", "random_state", "run")
    if seed is None:
        return None
    model = pick(df, "model", "model_name", "variant", "name")
    metric = pick(df, "pr_auc", "prauc", "average_precision", "auprc")
    if model and metric:
        out = df[[seed, model, metric]].copy()
        out.columns = ["seed", "model", "value"]
    else:
        # Wide form: one <model>_pr_auc column per model. Training times and win flags are ignored,
        # and only the XGBoost reference and the challenger are compared.
        cols = [c for c in df.columns if c != seed and norm(c).endswith("prauc")]
        ref = next((c for c in cols if "xgb" in c.lower()), None)
        chal = next((c for c in cols if any(k in c.lower() for k in ("soft", "vot", "hybrid", "iso"))), None)
        if not ref or not chal:
            return None
        nice = {ref: "XGBoost", chal: "Soft voting" if any(k in chal.lower() for k in ("soft", "vot"))
                else "IF hybrid"}
        out = df.melt(id_vars=[seed], value_vars=[ref, chal], var_name="model", value_name="value")
        out = out.rename(columns={seed: "seed"})
        out["model"] = out["model"].map(nice)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["model"] = out["model"].astype(str)
    return out.dropna(subset=["value"])


def seed_summary(long: pd.DataFrame) -> tuple[str, str, int, int, float] | None:
    """(challenger, reference, wins, seeds, mean difference) when exactly two models are compared."""
    models = list(long["model"].unique())
    if len(models) != 2:
        return None
    wide = long.pivot_table(index="seed", columns="model", values="value")
    a = next((m for m in models if is_selected(m) or "iso" in m.lower() or "hybrid" in m.lower()), models[0])
    b = next(m for m in models if m != a)
    diff = wide[a] - wide[b]
    return a, b, int((diff > 0).sum()), int(diff.notna().sum()), float(diff.mean())


def seed_chart(long: pd.DataFrame, title: str):
    """Slope chart: XGBoost on the left, the challenger on the right, one line per seed,
    green where the challenger won and red where it lost."""
    models = list(long["model"].unique())
    ref = "XGBoost" if "XGBoost" in models else models[0]
    chal = next(m for m in models if m != ref)
    wide = long.pivot_table(index="seed", columns="model", values="value")
    won = (wide[chal] > wide[ref]).rename("won").reset_index()
    d = long.merge(won, on="seed")
    d["Outcome"] = np.where(d["won"], f"{chal} wins", f"{chal} loses")
    order = [ref, chal]
    lines = alt.Chart(d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
        x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title="Test PR-AUC", scale=alt.Scale(zero=False)),
        detail="seed:N",
        color=alt.Color("Outcome:N", title=None, scale=alt.Scale(
            domain=[f"{chal} wins", f"{chal} loses"], range=[T["green"], T["red"]])),
        tooltip=["seed", "model", alt.Tooltip("value:Q", format=".4f")])
    means = long.groupby("model", as_index=False)["value"].mean()
    mean_line = alt.Chart(means).mark_line(color=T["ink"], strokeWidth=3.5,
                                           point=alt.OverlayMarkDef(size=110, shape="diamond", filled=True,
                                                                    color=T["ink"])).encode(
        x=alt.X("model:N", sort=order), y="value:Q",
        tooltip=["model", alt.Tooltip("value:Q", format=".4f", title="Mean")])
    show_chart((lines + mean_line).properties(height=260, title=title))
    wins = int(won["won"].sum())
    st.caption(f"{chal} better on {wins} of {len(won)} seeds. The thick line is the mean of all seeds.")


def page_evidence():
    page_title("Model Evidence", "Why the deployed model is a soft-voting ensemble of Random Forest and XGBoost, "
                                 "what it relies on, and how the full system was tested.")
    tabs = st.tabs(["Why this model", "Robustness", "Risk bands", "Explainability", "Rules", "System tests",
                    "Data sources"])
    log = read_result("experiment_log.csv")
    rank = model_ranking(log) if log is not None else None
    ens_seeds = read_result("ensemble_seed_robustness.csv")
    iso_seeds = read_result("seed_robustness.csv")
    ens_long = seed_long(ens_seeds) if ens_seeds is not None else None
    iso_long = seed_long(iso_seeds) if iso_seeds is not None else None

    with tabs[0]:
        if log is None:
            st.markdown("<div class='fa-note'><strong>experiment_log.csv was not found.</strong> This tab is built "
                        "from it. Run the experiment script that writes it, or check the Data sources tab to see "
                        "which results files the dashboard can see.</div>", unsafe_allow_html=True)
        elif rank is None:
            st.markdown("<div class='fa-note'><strong>experiment_log.csv was found but has no recognisable model "
                        "and PR-AUC columns.</strong> The Data sources tab lists its columns.</div>",
                        unsafe_allow_html=True)
        else:
            findings = []
            sel = rank[rank["selected"]]
            if not sel.empty:
                s0 = sel.iloc[0]
                pos = int(rank.index[rank["label"] == s0["label"]][0]) + 1
                ci = (f", 95% CI {s0['lo']:.3f} to {s0['hi']:.3f}" if pd.notna(s0["lo"]) and pd.notna(s0["hi"]) else "")
                findings.append(finding("Ranking", f"The ensemble ranks {pos} of {len(rank)} configurations on test "
                                        f"PR-AUC ({s0['pr_auc']:.3f}{ci}). PR-AUC is the headline metric because "
                                        "fraud is a small minority class, where ROC-AUC flatters every model; a "
                                        "random classifier scores 0.033 here."))
                if pos > 1:
                    top = rank.iloc[0]
                    findings.append(finding("Not the top score", f"{html.escape(top['label'])} scores higher "
                                            f"({top['pr_auc']:.3f}). The selection rests on stability across seeds, "
                                            "not on a single run, so check the robustness evidence below."))
            if ens_long is not None and (summ := seed_summary(ens_long)):
                a, b, wins, n, d = summ
                findings.append(finding("Stability", f"{html.escape(a)} beat {html.escape(b)} on {wins} of {n} seeds, "
                                        f"by {d:+.4f} PR-AUC on average."))
            if iso_long is not None and (summ := seed_summary(iso_long)):
                a, b, wins, n, d = summ
                verdict_txt = ("A gain that does not hold on every seed is noise, and the hybrid adds a second "
                               "model to train and score." if wins < n else
                               "It won on every seed, so the rejection must rest on cost or size of gain; "
                               "state that reason explicitly in the thesis.")
                findings.append(finding("Rejected option", f"{html.escape(a)} against {html.escape(b)}: better on "
                                        f"{wins} of {n} seeds, {d:+.4f} on average. {verdict_txt}"))
            if not sel.empty and pd.notna(sel.iloc[0]["infer_ms"]):
                findings.append(finding("Cost", f"Scoring takes {sel.iloc[0]['infer_ms']:.3f} ms per transaction, "
                                        "well inside a real-time payment budget."))
            fs_col = pick(log, "feature_set", "features")
            m_col = pick(log, "pr_auc", "prauc", "average_precision")
            if fs_col and m_col:
                abl = log[log[fs_col].astype(str).str.contains("ablation", case=False)]
                if not abl.empty:
                    top_abl = pd.to_numeric(abl[m_col], errors="coerce").max()
                    findings.append(finding("Balance ablation", f"Adding the sender-balance features lifts PR-AUC "
                                            f"to {top_abl:.3f}. That is not a better detector: PaySim creates fraud by "
                                            "emptying the victim's account, so these features nearly reveal the "
                                            "label. They are excluded from the deployed model and from this chart."))
            findings.append(finding("Selection rule", "Adopt a more complex model only if its gain over XGBoost "
                                    "holds across all five random seeds and stays inside the latency budget."))
            with panel():
                st.subheader("The case for the ensemble")
                st.markdown("".join(findings), unsafe_allow_html=True)

            with panel():
                st.subheader("Test PR-AUC by configuration")
                ch = rank.copy()
                order = ch["label"].tolist()
                base = alt.Chart(ch).encode(y=alt.Y("label:N", sort=order, title=None,
                                                    axis=alt.Axis(labelLimit=320)))
                bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
                    x=alt.X("pr_auc:Q", title="PR-AUC (test)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("role:N", title=None,
                                    scale=alt.Scale(domain=["Deployed ensemble", "Candidate"],
                                                    range=[T["blue"], T["neutral3"]])),
                    tooltip=["label", alt.Tooltip("pr_auc:Q", format=".4f"), alt.Tooltip("lo:Q", format=".4f"),
                             alt.Tooltip("hi:Q", format=".4f")])
                layers = bars
                if ch["lo"].notna().any():
                    layers += base.mark_rule(color=T["ink"], strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q")
                layers += alt.Chart(pd.DataFrame({"x": [0.033]})).mark_rule(color=T["orange"], strokeDash=[4, 3]).encode(x="x:Q")
                show_chart(layers.properties(height=max(160, 30 * len(ch))))
                st.caption("Black or white lines are 95% bootstrap confidence intervals. The dashed orange line is a random "
                           "classifier (0.033, the test-set fraud rate).")

            c1, c2 = st.columns(2, gap="large")
            with c1:
                with panel():
                    st.subheader("Catch rate against false alarms")
                    pr = rank.dropna(subset=["recall", "fpr"])
                    if pr.empty:
                        empty("The log has no recall and false positive rate columns.")
                    else:
                        show_chart(alt.Chart(pr).mark_circle(size=140, opacity=.95).encode(
                            x=alt.X("fpr:Q", title="False positive rate", axis=alt.Axis(format="%")),
                            y=alt.Y("recall:Q", title="Recall", axis=alt.Axis(format="%")),
                            color=alt.Color("role:N", legend=None, scale=alt.Scale(
                                domain=["Deployed ensemble", "Candidate"], range=[T["blue"], T["neutral2"]])),
                            tooltip=["label", alt.Tooltip("recall:Q", format=".1%"),
                                     alt.Tooltip("fpr:Q", format=".2%")]).properties(height=240))
                        st.caption("Up and to the left is better: more fraud caught, fewer customers disturbed.")
            with c2:
                with panel():
                    st.subheader("Cost to run")
                    cost = rank.dropna(subset=["infer_ms"])
                    if cost.empty:
                        empty("The log has no inference time column.")
                    else:
                        show_chart(alt.Chart(cost).mark_bar(cornerRadiusEnd=4, height=12).encode(
                            y=alt.Y("label:N", sort=cost["label"].tolist(), title=None, axis=alt.Axis(labelLimit=220)),
                            x=alt.X("infer_ms:Q", title="Inference (ms per transaction)"),
                            color=alt.Color("role:N", legend=None, scale=alt.Scale(
                                domain=["Deployed ensemble", "Candidate"], range=[T["blue"], T["neutral3"]])),
                            tooltip=["label", alt.Tooltip("infer_ms:Q", format=".4f"),
                                     alt.Tooltip("train_s:Q", format=".1f")]).properties(height=max(160, 26 * len(cost))))
            with st.expander("Full experiment log"):
                show_csv("experiment_log.csv", "Experiment log, ranked by test PR-AUC", sort=pick(log, "pr_auc"))

    with tabs[1]:
        for long, title in ((ens_long, "Soft voting against XGBoost across seeds (adopted)"),
                            (iso_long, "Isolation Forest hybrid against XGBoost across seeds (rejected)")):
            with panel():
                if long is None:
                    st.subheader(title)
                    empty("Not found, or the file has no seed column.")
                else:
                    seed_chart(long, title)
                    st.caption("A model worth adopting wins on every seed, not just on average.")
        show_csv("paired_comparisons.csv", "Paired bootstrap comparisons")
        with st.expander("Raw seed tables"):
            show_csv("ensemble_seed_robustness.csv", "Soft voting against XGBoost")
            show_csv("seed_robustness.csv", "Isolation Forest hybrid against XGBoost")
    with tabs[2]:
        show_csv("risk_bands_test.csv", "Risk bands on the test set")
        show_csv("calibration_metrics.csv", "Calibration")
        show_figure("calibration_curve.png", "Calibration curve")
    with tabs[3]:
        st.caption("Global explanations from SHAP on the test set, then decision-level explanations, "
                   "including a cleared transaction.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            show_figure("shap_grouped_importance.png", "What the model relies on overall")
            show_figure("shap_waterfall_best_catch.png", "A fraud it caught")
            show_figure("shap_waterfall_missed_fraud.png", "A fraud it missed: why it looked low risk")
        with c2:
            show_figure("shap_beeswarm.png", "Direction of each feature's effect")
            show_figure("shap_waterfall_false_alarm.png", "A legitimate transaction it flagged")
            show_figure("shap_dependence_transfer.png", "How the TRANSFER effect depends on context")
        show_csv("transfer_effect.csv", "TRANSFER effect by context")
        with panel():
            st.subheader("Why a transaction was cleared")
            act = load_activity()
            low = act[(act["band"] == "LOW")] if not act.empty else act
            if low.empty or not API_UP:
                empty("Needs the API and at least one low-risk TRANSFER or CASH_OUT decision. Run a live replay.")
            else:
                pick_from = low.sort_values("amount", ascending=False).head(50)
                tx = st.selectbox("Low-risk decision (largest amounts first)", pick_from["tx_id"].tolist(),
                                  format_func=lambda t: f"{t}, {pick_from.set_index('tx_id').loc[t, 'tx_type']}, "
                                                        f"{pick_from.set_index('tx_id').loc[t, 'amount']:,.0f}")
                code_, rec_ = api("GET", f"/transactions/{tx}")
                if code_ == 200:
                    explain(rec_.get("reasons", []), "LOW")
    with tabs[4]:
        show_csv("rules_validation.csv", "Candidate rules on validation (none qualified)")
    with tabs[5]:
        show_csv("api_functional_tests.csv", "API functional tests")
        show_csv("feature_parity.csv", "Feature parity: live store against training")
        show_csv("latency_benchmark.csv", "Latency benchmark")
    with tabs[6]:
        st.caption("Every results file the dashboard can see, with its columns. If a tab above is empty, "
                   "compare the expected file name with this list.")
        rows = []
        seen = set()
        for d in RESULT_DIRS:
            d = Path(d)
            if not d.exists():
                continue
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() not in (".csv", ".png") or f in seen:
                    continue
                seen.add(f)
                cols = ""
                if f.suffix.lower() == ".csv":
                    try:
                        cols = ", ".join(pd.read_csv(f, nrows=0).columns)
                    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
                        cols = "(unreadable)"
                rows.append({"File": f.name, "Folder": str(f.parent.relative_to(ROOT)) if f.is_relative_to(ROOT)
                             else str(f.parent), "Columns": cols})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            empty("No results files found under the configured folders.")


{"Command Centre": page_command, "Velocity & Patterns": page_patterns, "Network": page_network,
 "Alert Queue": page_alerts, "Cases": page_cases, "Transaction Detail": page_detail,
 "Explore & SQL": page_explore, "Send a Transaction": page_send, "Verification": page_verification,
 "Live Replay": page_replay, "Rule Tuning": page_tuning, "System Health": page_system,
 "Model Evidence": page_evidence}[page]()
