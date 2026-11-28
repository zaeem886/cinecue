"""CineCue website. Run with: python -m streamlit run app.py"""
from datetime import datetime
import html
import json
from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from cinecue.artifacts import load_bundle, save_bundle  # noqa: E402
from cinecue.context import DAYPARTS, WEEKDAYS, ContextRanker, daypart  # noqa: E402
from cinecue.data import table_path  # noqa: E402
from cinecue.demo import context_tables, fit_demo, fit_tables, recommend_in_context  # noqa: E402
from cinecue.first_run import official_tables  # noqa: E402
from cinecue.likes import load_likes, make_like, save_likes  # noqa: E402
from cinecue.search import TitleSearch, display_title, label  # noqa: E402

LIKES_FILE = ROOT / "data/likes.json"  # local only; see .gitignore
PAGES = ["Discover", "Movie library", "Watchlist", "Project results"]
PAGE_ICONS = {"Discover": ":material/explore:", "Movie library": ":material/video_library:",
              "Watchlist": ":material/bookmarks:", "Project results": ":material/insights:"}
DAYPART_ICONS = {0: "☾", 1: "☀", 2: "◐", 3: "☾"}
# Context weights you choose in Discover. MovieLens showed no measurable gain from any of them
# (results/context_summary.json), so "Off" equals the validated final ranker.
LEAN = {"Off": [1.0, 0, 0, 0, 0], "Light": [1.0, 0.5, 0.2, 0.25, 0], "Strong": [1.0, 1.5, 0.5, 0.75, 0]}

st.set_page_config(page_title="CineCue | Movie discovery", page_icon="🎬", layout="wide")
if toast := st.session_state.pop("toast", None):
    st.toast(toast[0], icon=toast[1])


def phase(name, signature):
    """Return 'a' or 'b', flipping whenever `signature` changes.

    Streamlit reuses DOM nodes across reruns, so a CSS entrance animation plays only once.
    Switching between two identical keyframe names makes it replay when the content changes.
    """
    store = st.session_state.setdefault("_phases", {})
    last = store.get(name)
    if last is None or last[0] != signature:
        store[name] = (signature, "b" if last and last[1] == "a" else "a")
    return store[name][1]


# ---------- Likes and the current moment ----------
st.session_state.setdefault("watchlist", {})
if "likes" not in st.session_state:
    st.session_state.likes = load_likes(LIKES_FILE)
    st.session_state.likes_rev = 0


def set_likes(likes, message=None):
    """Replace the likes, persist them, and tell widgets that mirror them to refresh."""
    st.session_state.likes = likes
    st.session_state.likes_rev += 1
    save_likes(LIKES_FILE, likes)
    if message:
        st.session_state.toast = (message, ":material/favorite:")


def moment():
    """The context recommendations use: the clock, or the day and time picked in the sidebar."""
    if st.session_state.get("ctx_live", True):
        now = datetime.now().astimezone()
        return now.weekday(), daypart(now.hour)
    return (WEEKDAYS.index(st.session_state.get("ctx_day", WEEKDAYS[0])),
            DAYPARTS.index(st.session_state.get("ctx_part", DAYPARTS[3])))


def toggle_like(movie_id, title):
    likes = st.session_state.likes
    if any(like["movie_id"] == movie_id for like in likes):
        set_likes([like for like in likes if like["movie_id"] != movie_id], f"Removed {label(title)} from your likes")
    else:
        weekday, part = moment()
        set_likes(likes + [make_like(movie_id, weekday=weekday, part=part, title=title)],
                  f"Liked {label(title)} · {WEEKDAYS[weekday]} {DAYPARTS[part]}")


def nav_label(name):
    count = len(st.session_state.watchlist) if name == "Watchlist" else 0
    return f"{PAGE_ICONS[name]} {name}" + (f" :primary-badge[{count}]" if count else "")


# ---------- Sidebar ----------
weekday, part = moment()
with st.sidebar:
    st.markdown('<div class="brand"><span class="logo" aria-hidden="true">C</span><div>Cine<span>Cue.</span>'
                '<small>Movie picks, with a reason</small></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Browse</div>', unsafe_allow_html=True)
    page = st.radio("Explore", PAGES, format_func=nav_label, label_visibility="collapsed", key="page")

    st.markdown('<div class="side-label">Right now</div>', unsafe_allow_html=True)
    with st.container(key="ctxcard"):
        on_day = sum(like["weekday"] == weekday for like in st.session_state.likes)
        hint = (f"{on_day} of your likes are from {WEEKDAYS[weekday]}s" if on_day else
                f"Like a movie and it remembers it was a {WEEKDAYS[weekday]}")
        st.markdown(f'<div class="ctx-now" title="Likes you make now are stamped with this day and time.">'
                    f'<span class="ctx-icon">{DAYPART_ICONS[part]}</span><div><b>{WEEKDAYS[weekday]}</b>'
                    f'<span>{DAYPARTS[part]}</span></div></div><div class="ctx-hint">{hint}</div>',
                    unsafe_allow_html=True)
        live = st.toggle("Use my clock", value=True, key="ctx_live",
                         help="Off: pick a day and time to see how your picks change, e.g. a Monday evening.")
        if not live:
            day_col, part_col = st.columns(2)
            day_col.selectbox("Day", WEEKDAYS, key="ctx_day", label_visibility="collapsed")
            part_col.selectbox("Time", DAYPARTS, index=3, key="ctx_part", label_visibility="collapsed")

    likes_now = st.session_state.likes
    st.markdown(f'<div class="side-label">Your likes <b>{len(likes_now)}</b></div>', unsafe_allow_html=True)
    if likes_now:
        recent = sorted(likes_now, key=lambda like: like["liked_at"], reverse=True)[:5]
        st.markdown('<ul class="like-list">' + "".join(
            f'<li class="{"same" if like["weekday"] == weekday else ""}" '
            f'title="{html.escape(label(like["title"]) if like["title"] else str(like["movie_id"]))} · liked on a '
            f'{WEEKDAYS[like["weekday"]]} {DAYPARTS[like["part"]]}"><span class="t">'
            f'{html.escape(display_title(like["title"])[0] if like["title"] else "Movie " + str(like["movie_id"]))}'
            f'</span><span class="d">{WEEKDAYS[like["weekday"]][:3]} {DAYPARTS[like["part"]]}</span></li>'
            for like in recent) + "</ul>"
            + (f'<div class="ctx-hint">+ {len(likes_now) - 5} more in Discover</div>' if len(likes_now) > 5 else ""),
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="ctx-hint">Tap ♥ on any movie. Each like remembers the day and time.</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="side-gap"></div>', unsafe_allow_html=True)
    dark_mode = st.toggle(":material/dark_mode: Dark mode", key="dark_mode")
page_phase = phase("page", page)

# ---------- Theme ----------
palette = (
    dict(bg="#0f1519", surface="#172127", control="#1f2c34", text="#e6eef2", muted="#9fb2bf",
         border="#2c3b45", accent="#5fd0c2", accent_soft="rgba(95,208,194,.14)", bar="#2c3b45")
    if dark_mode else
    dict(bg="#f4f6f7", surface="#ffffff", control="#eef3f4", text="#16262b", muted="#566770",
         border="#dbe3e6", accent="#007e77", accent_soft="rgba(0,126,119,.09)", bar="#e3eaec")
)
tokens = ";".join(f"--cue-{name.replace('_', '-')}:{value}" for name, value in palette.items())
st.markdown(f"<style>:root{{{tokens};color-scheme:{'dark' if dark_mode else 'light'}}}</style>",
            unsafe_allow_html=True)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');
.stApp {background:var(--cue-bg);color:var(--cue-text)}
html,body,[data-testid="stApp"] {font-family:'DM Sans',sans-serif}
.block-container {max-width:1180px;padding-top:1.6rem;padding-bottom:3rem}
[data-testid="stHeader"] {background:var(--cue-bg)}
[data-testid="stHeader"] button {color:var(--cue-text)}
[data-testid="stSidebar"] {background:var(--cue-surface);border-right:1px solid var(--cue-border);min-width:240px;max-width:270px}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.35rem}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.45rem .6rem;border-radius:8px;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:var(--cue-accent-soft)}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp summary,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:var(--cue-text)}
[data-testid="stWidgetLabel"] p {font-weight:500}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {color:var(--cue-muted)!important}
/* Bordered containers are keyed (cuecard-*, controls) so they can be styled; Streamlit 1.67 has no wrapper test id. */
[class*="st-key-cuecard"],.st-key-controls {background:var(--cue-surface);border-color:var(--cue-border)!important;border-radius:14px}
.stButton button,.stDownloadButton button,.stLinkButton a {border-radius:8px;background:var(--cue-surface);
  color:var(--cue-text);border-color:var(--cue-border);font-weight:500;
  transition:transform .15s ease,background-color .2s ease,border-color .2s ease,color .2s ease}
.stButton button:hover,.stDownloadButton button:hover,.stLinkButton a:hover {border-color:var(--cue-accent);color:var(--cue-accent)}
.stButton button:active,.stDownloadButton button:active,.stLinkButton a:active {transform:scale(.97)}
[class*="-btn-on-"] button {background:var(--cue-accent-soft);border-color:var(--cue-accent);color:var(--cue-accent)}
[data-testid="stToast"] {background:var(--cue-surface);color:var(--cue-text);border:1px solid var(--cue-border)}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],.stApp input,
[data-baseweb="popover"],[role="listbox"],[role="option"] {background:var(--cue-control)!important;color:var(--cue-text)!important}
[data-baseweb="select"] span,[data-baseweb="select"] input {color:var(--cue-text)!important}
[data-baseweb="select"] div {background-color:var(--cue-control)!important;border-color:var(--cue-border)!important}
[data-baseweb="select"] svg,[data-baseweb="input"] svg {fill:var(--cue-text)}
[data-baseweb="popover"] *,[role="option"] * {color:var(--cue-text)}
.stApp input::placeholder {color:var(--cue-muted)}
[data-baseweb="tag"] {background:var(--cue-accent-soft)!important;color:var(--cue-text)}
[data-testid="stExpander"] details {background:var(--cue-surface);border-color:var(--cue-border)}
[data-testid="stAlert"] {background:var(--cue-control);color:var(--cue-text)}

/* Streamlit widgets drawn with the light base theme: re-colour them from the tokens. */
[data-testid="stCheckbox"]:not([data-selected="true"]) label > div:not([data-testid]) {background:var(--cue-border)!important}
[data-testid="stSelectbox"] [role="group"],[data-testid="stMultiSelect"] [role="group"] {background:var(--cue-control)!important;
  border-color:var(--cue-border)!important}
[data-testid="stSelectbox"] button,[data-testid="stMultiSelect"] button {background:transparent!important}
[data-testid="stSelectbox"] svg,[data-testid="stMultiSelect"] svg {fill:var(--cue-text);color:var(--cue-text)}
[data-testid="stNumberInput"] input,[data-testid="stNumberInput"] button {background:var(--cue-control)!important;color:var(--cue-text)!important}
/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {min-width:300px;max-width:300px}
[data-testid="stSidebarUserContent"] {padding-top:.2rem}
[data-testid="stSidebarHeader"] {height:2.6rem;margin-bottom:0}
/* Streamlit pulls each markdown block up by 1rem; sidebar blocks here carry their own spacing. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {margin-bottom:0}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.55rem}
.brand {display:flex;align-items:center;gap:.7rem;font-size:1.45rem;font-weight:700;letter-spacing:-.05em;
  padding:.1rem 0 .9rem;color:var(--cue-text);line-height:1.1}
.brand > div > span {color:var(--cue-accent)}
.brand .logo {display:grid;place-items:center;width:2.3rem;height:2.3rem;border-radius:11px;flex-shrink:0;
  background:linear-gradient(140deg,var(--cue-accent),#0a4d5c);color:#fff;font-size:1.2rem;letter-spacing:0;
  box-shadow:0 6px 16px -8px var(--cue-accent)}
.brand small {display:block;font-size:.74rem;font-weight:500;letter-spacing:0;color:var(--cue-muted);margin-top:.2rem}
.side-label {display:flex;align-items:center;gap:.45rem;font-size:.7rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--cue-muted);margin:.9rem 0 .1rem .1rem}
.side-label b {font-size:.7rem;padding:0 .4rem;border-radius:99px;background:var(--cue-accent-soft);color:var(--cue-accent)}
/* Navigation: the radio becomes a list of full-width rows with an accent bar on the active page. */
[data-testid="stSidebar"] [role="radiogroup"] {gap:.15rem}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.5rem .7rem;border-radius:9px;width:100%;margin:0;
  position:relative;cursor:pointer}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:not([data-testid]) {display:none}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {font-size:.93rem;font-weight:500;display:flex;align-items:center;gap:.6rem}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p span[role="img"] {color:var(--cue-muted);font-size:1.2rem}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background:var(--cue-accent-soft)}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p span[role="img"] {color:var(--cue-accent);font-weight:600}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before {content:"";position:absolute;left:0;
  top:22%;bottom:22%;width:3px;border-radius:3px;background:var(--cue-accent)}
[data-testid="stTooltipIcon"] svg {fill:none!important}
.st-key-ctxcard {background:var(--cue-control);border:1px solid var(--cue-border);border-radius:12px;padding:.75rem .8rem .55rem;
  gap:.35rem!important}
.st-key-ctxcard [data-testid="stMarkdownContainer"] p {margin:0}
.ctx-now {display:flex;align-items:center;gap:.65rem}
.ctx-now b {display:block;font-size:1.02rem;line-height:1.2;color:var(--cue-text)}
.ctx-now span:not(.ctx-icon) {display:block;font-size:.82rem;line-height:1.2;color:var(--cue-muted)}
.ctx-icon {display:grid;place-items:center;width:2.1rem;height:2.1rem;border-radius:50%;background:var(--cue-accent-soft);
  color:var(--cue-accent);font-size:1.05rem}
.ctx-hint {font-size:.78rem;color:var(--cue-muted);margin-top:.45rem;line-height:1.4}
.like-list {list-style:none;margin:0;padding:0;display:grid;grid-template-columns:minmax(0,1fr);gap:.2rem;width:100%}
.like-list li {display:flex;justify-content:space-between;align-items:center;gap:.5rem;padding:.32rem .5rem;
  border-radius:8px;font-size:.83rem;margin:0;min-width:0;max-width:100%;box-sizing:border-box}
.like-list li:hover {background:var(--cue-control)}
.like-list .t {flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--cue-text)}
.like-list .d {flex-shrink:0;font-size:.72rem;color:var(--cue-muted)}
.like-list li.same .d {color:var(--cue-accent);font-weight:600}
.like-list li.same .t::before {content:"● ";color:var(--cue-accent);font-size:.6rem;vertical-align:.15em}
.side-gap {height:.2rem;border-top:1px solid var(--cue-border);margin-top:.8rem}

/* ---------- Discover controls and context banner ---------- */
.ctl-head {font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--cue-muted);
  margin:.1rem 0 -.2rem}
.ctx-banner {display:flex;align-items:center;gap:.8rem;padding:.65rem .85rem;border-radius:11px;
  background:var(--cue-accent-soft);font-size:.9rem;margin:.1rem 0 .2rem}
.ctx-banner .ctx-icon {background:var(--cue-surface)}
.ctx-banner b {color:var(--cue-text)}
.ctx-banner span.note {color:var(--cue-muted);font-size:.8rem;display:block}
/* Pills and segmented controls: drawn from the tokens so both themes stay readable. */
[data-testid="stButtonGroup"] button[data-variant] {background:var(--cue-surface)!important;color:var(--cue-text)!important;
  border-color:var(--cue-border)!important}
[data-testid="stButtonGroup"] button[data-variant] * {color:inherit!important}
[data-testid="stButtonGroup"] button[data-variant][data-selected="true"] {background:var(--cue-accent-soft)!important;
  color:var(--cue-accent)!important;border-color:var(--cue-accent)!important}
[data-testid="stButtonGroup"] button[data-variant]:hover:not(:disabled) {border-color:var(--cue-accent)!important;color:var(--cue-accent)!important}
[data-testid="stButtonGroup"] button[data-variant]:disabled {opacity:.5}

/* ---------- Hover tooltip on movie cards ---------- */
.poster {outline:none}
.poster .tip {position:absolute;inset:0;z-index:3;display:flex;flex-direction:column;gap:.4rem;overflow-y:auto;
  scrollbar-width:none;padding:.75rem .85rem;background:linear-gradient(180deg,rgba(7,12,15,.55) 0%,rgba(7,12,15,.93) 45%);
  color:#eef4f6;opacity:0;transform:translateY(8px);transition:opacity .22s ease,transform .25s var(--ease-out);
  pointer-events:none;text-align:left}
[class*="st-key-cuecard"]:hover .poster .tip {pointer-events:auto}
[class*="st-key-cuecard"]:hover .poster .tip,.poster:focus-within .tip,.poster:focus .tip {opacity:1;transform:none}
.tip .tip-title {font-weight:700;font-size:.98rem;line-height:1.25;margin-top:auto} /* bottom-aligned, scrolls if long */
.tip .tip-title small {font-weight:500;opacity:.7;margin-left:.3rem}
.tip ul {list-style:none;margin:0;padding:0;display:grid;gap:.3rem}
.tip li {display:flex;gap:.5rem;align-items:flex-start;font-size:.8rem;line-height:1.35;margin:0}
.tip li i {font-style:normal;flex-shrink:0;width:1.15rem;height:1.15rem;display:grid;place-items:center;border-radius:5px;
  background:rgba(255,255,255,.14);font-size:.72rem}
.tip li.hot i {background:var(--cue-accent);color:#fff}
.tip li b {color:#fff}
.poster .lift {position:absolute;left:.7rem;bottom:.6rem;z-index:1;background:var(--cue-accent);color:#fff;border-radius:6px;
  padding:.1rem .45rem;font-size:.72rem;font-weight:700}
.poster .liked {position:absolute;right:.7rem;top:.6rem;z-index:1;background:rgba(0,0,0,.42);color:#ff8fa3;border-radius:6px;
  padding:.05rem .4rem;font-size:.8rem}
[class*="-like-on-"] button {color:#d6456a;border-color:#d6456a55}
.hero {padding:.4rem 0 1.1rem}
.eyebrow {color:var(--cue-accent);font-weight:600;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase}
.hero h1 {font-size:2.35rem;line-height:1.1;font-weight:700;letter-spacing:-.04em;margin:.35rem 0 .5rem;padding:0}
.hero p {color:var(--cue-muted)!important;font-size:1.05rem;max-width:640px;margin:0}
.stats {display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0 .2rem}
.stat {background:var(--cue-surface);border:1px solid var(--cue-border);border-radius:999px;padding:.35rem .85rem;
  font-size:.85rem;color:var(--cue-muted)}
.stat b {color:var(--cue-text);font-weight:600}
.section {display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:.4rem;margin:1.4rem 0 .6rem}
.section h3 {font-size:1.25rem;font-weight:700;margin:0;padding:0}
.section span {color:var(--cue-muted);font-size:.88rem}

.poster {height:230px;border-radius:10px;position:relative;overflow:hidden;display:flex;align-items:flex-end;
  padding:.7rem .8rem;color:#fff}
.poster .mark {position:absolute;right:.6rem;top:.2rem;font-size:5.2rem;font-weight:700;opacity:.18;letter-spacing:-.05em}
.poster .genre {font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.92}
.poster .year {position:absolute;left:.7rem;top:.65rem;background:rgba(0,0,0,.32);border-radius:6px;padding:.1rem .45rem;
  font-size:.75rem;font-weight:600}
.poster.has-img {align-items:center;justify-content:center;padding:.8rem;background:#0b1114}
.poster.has-img .bg {position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
  filter:blur(16px) brightness(.5) saturate(1.2);transform:scale(1.15)}
.poster.has-img .fg {position:relative;height:100%;aspect-ratio:2/3;object-fit:cover;border-radius:6px;
  box-shadow:0 8px 22px rgba(0,0,0,.5)}
.poster.has-img .year {z-index:1}
.movie-title {font-size:1.04rem;font-weight:600;line-height:1.3;margin:.75rem 0 .35rem;height:2.6em;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.chips {display:flex;flex-wrap:nowrap;gap:.3rem;margin-bottom:.55rem;height:1.45rem;overflow:hidden}
.chip {font-size:.72rem;border:1px solid var(--cue-border);border-radius:999px;padding:.08rem .5rem;color:var(--cue-muted);
  white-space:nowrap;flex-shrink:0}
.score {display:flex;align-items:center;gap:.55rem;font-size:.85rem;color:var(--cue-muted);margin-bottom:.55rem;height:1.35rem}
.score b {color:var(--cue-text)}
.meter {flex:1;height:6px;border-radius:99px;background:var(--cue-bar);overflow:hidden}
.meter i {display:block;height:100%;background:var(--cue-accent);border-radius:99px}
.reason {background:var(--cue-accent-soft);border-radius:9px;padding:.55rem .65rem;font-size:.84rem;line-height:1.45;
  margin-bottom:.6rem;height:5.9rem;box-sizing:border-box;overflow:hidden}
.reason .why {display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.reason .label {display:block;font-size:.7rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  color:var(--cue-accent);margin-bottom:.15rem}
.kpi {background:var(--cue-surface);border:1px solid var(--cue-border);border-radius:14px;padding:1rem 1.1rem;height:100%}
.kpi .name {font-size:.78rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--cue-muted)}
.kpi .value {font-size:2rem;font-weight:700;letter-spacing:-.03em;margin:.25rem 0 .1rem}
.kpi .note {font-size:.84rem;color:var(--cue-muted)}
.kpi .note b {color:var(--cue-accent)}
.tbl-wrap {overflow-x:auto;border:1px solid var(--cue-border);border-radius:12px;background:var(--cue-surface);margin-bottom:1rem}
.tbl {width:100%;border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums;margin:0!important}
.tbl th,.tbl td {border-left:0!important;border-right:0!important;border-top:0!important;background:transparent}
.tbl th,.tbl td {padding:.6rem .85rem;border-bottom:1px solid var(--cue-border);text-align:right;white-space:nowrap}
.tbl thead th {color:var(--cue-muted);font-weight:600;font-size:.78rem;letter-spacing:.04em;text-transform:uppercase}
.tbl th:first-child {text-align:left;font-weight:500;color:var(--cue-text)}
.tbl thead th:first-child {color:var(--cue-muted)}
.tbl tbody tr:last-child th,.tbl tbody tr:last-child td {border-bottom:0}
.tbl tr.final {background:var(--cue-accent-soft)}
.tbl tr.final th {font-weight:700}
.bars {display:grid;gap:.55rem;margin:.4rem 0 .6rem}
.bar-row {display:grid;grid-template-columns:minmax(120px,220px) 1fr 3.2rem;align-items:center;gap:.75rem;font-size:.88rem}
.bar-row .name {color:var(--cue-muted)}
.bar-row .name.best {color:var(--cue-text);font-weight:600}
.bar-track {height:12px;background:var(--cue-bar);border-radius:99px;overflow:hidden}
.bar-track i {display:block;height:100%;background:var(--cue-accent);border-radius:99px}
.bar-row .num {text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.footer {border-top:1px solid var(--cue-border);margin-top:3rem;padding-top:1rem;color:var(--cue-muted);font-size:.85rem}
.badge {display:inline-flex;align-items:center;gap:.5rem;font-size:.85rem;color:var(--cue-muted)}
.badge b {min-width:1.6rem;padding:.05rem .45rem;border-radius:999px;background:var(--cue-accent-soft);
  color:var(--cue-accent);text-align:center;font-variant-numeric:tabular-nums}
.poster .rank {position:absolute;right:.7rem;bottom:.6rem;z-index:1;background:rgba(0,0,0,.4);border-radius:6px;
  padding:.1rem .45rem;font-size:.75rem;font-weight:700;font-variant-numeric:tabular-nums}
.poster.has-img .rank {bottom:.65rem}
@media(max-width:640px){.hero h1{font-size:1.8rem}.kpi .value{font-size:1.6rem}}

/* ---------- Motion ----------
   Each entrance keyframe exists twice (-a/-b). phase() in Python flips the class between them when
   the content changes, which restarts the animation on DOM nodes Streamlit reuses. Fill mode is
   `backwards` so a finished animation stops holding `transform`, and hover transforms still work. */
@keyframes cue-rise-a {from {opacity:0;transform:translateY(14px)} to {opacity:1;transform:none}}
@keyframes cue-rise-b {from {opacity:0;transform:translateY(14px)} to {opacity:1;transform:none}}
@keyframes cue-grow-a {from {transform:scaleX(0)} to {transform:scaleX(1)}}
@keyframes cue-grow-b {from {transform:scaleX(0)} to {transform:scaleX(1)}}
@keyframes cue-pop-a {0% {transform:scale(1)} 40% {transform:scale(1.3)} 100% {transform:scale(1)}}
@keyframes cue-pop-b {0% {transform:scale(1)} 40% {transform:scale(1.3)} 100% {transform:scale(1)}}
@keyframes cue-pulse {0%,100% {box-shadow:0 0 0 0 var(--cue-accent)} 60% {box-shadow:0 0 0 .4rem transparent}}
@keyframes cue-sheen {from {transform:translateX(-130%) skewX(-18deg)} to {transform:translateX(330%) skewX(-18deg)}}
.ph-a,[class*="st-key-cuecard-a-"] {--rise:cue-rise-a;--grow:cue-grow-a;--pop:cue-pop-a}
.ph-b,[class*="st-key-cuecard-b-"] {--rise:cue-rise-b;--grow:cue-grow-b;--pop:cue-pop-b}
:root {--ease-out:cubic-bezier(.2,.7,.2,1)}

/* Smooth light/dark switch */
.stApp,[data-testid="stSidebar"],[data-testid="stHeader"],.stat,.kpi,.chip,.tbl-wrap {
  transition:background-color .35s ease,border-color .35s ease,color .35s ease}

/* Page header, stat pills, KPI tiles */
.hero > * {animation:var(--rise) .6s var(--ease-out) backwards}
.hero > :nth-child(2) {animation-delay:.07s}
.hero > :nth-child(3) {animation-delay:.14s}
.eyebrow::before {content:"";display:inline-block;width:.42rem;height:.42rem;margin-right:.5rem;border-radius:50%;
  background:var(--cue-accent);vertical-align:.12em;animation:cue-pulse 2.6s ease-out infinite}
.stats .stat {animation:var(--rise) .5s var(--ease-out) backwards;animation-delay:calc(.2s + var(--i,0) * 60ms)}
.stats .stat:nth-child(2) {--i:1} .stats .stat:nth-child(3) {--i:2}
.stats .stat:nth-child(4) {--i:3} .stats .stat:nth-child(5) {--i:4}
.stColumn:nth-child(2) {--col:1} .stColumn:nth-child(3) {--col:2} .stColumn:nth-child(4) {--col:3}
.kpi {animation:var(--rise) .55s var(--ease-out) backwards;animation-delay:calc(.15s + var(--col,0) * 80ms);
  transition:transform .25s var(--ease-out),border-color .25s ease,background-color .35s ease}
.kpi:hover {transform:translateY(-3px);border-color:var(--cue-accent)}
.badge b {display:inline-block;animation:var(--pop) .45s var(--ease-out)}

/* Movie cards: staggered by row and column, lift on hover */
[class*="st-key-cuecard"] {animation:var(--rise) .55s var(--ease-out) backwards;
  animation-delay:calc(var(--row,0) * 90ms + var(--col,0) * 60ms);
  transition:transform .28s var(--ease-out),box-shadow .28s ease,border-color .28s ease,background-color .35s ease}
[class*="st-key-cuecard"][class*="-row1-"] {--row:1} [class*="st-key-cuecard"][class*="-row2-"] {--row:2}
[class*="st-key-cuecard"][class*="-row3-"] {--row:3} [class*="st-key-cuecard"][class*="-row4-"] {--row:4}
[class*="st-key-cuecard"][class*="-row5-"] {--row:5}
[class*="st-key-cuecard"]:hover {transform:translateY(-4px);border-color:var(--cue-accent)!important;
  box-shadow:0 18px 34px -20px rgba(0,0,0,.45)}
.poster::after {content:"";position:absolute;top:0;bottom:0;left:0;width:35%;pointer-events:none;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.22),transparent);transform:translateX(-130%) skewX(-18deg)}
[class*="st-key-cuecard"]:hover .poster::after {animation:cue-sheen .85s ease-out}
.poster .mark,.poster.has-img .fg {transition:transform .45s var(--ease-out),opacity .45s ease}
[class*="st-key-cuecard"]:hover .poster .mark {transform:scale(1.12) rotate(-4deg);opacity:.28}
[class*="st-key-cuecard"]:hover .poster.has-img .fg {transform:scale(1.04)}
.meter i {transform-origin:left;animation:var(--grow) .9s var(--ease-out) backwards;
  animation-delay:calc(.3s + var(--row,0) * 90ms + var(--col,0) * 60ms)}
.reason {animation:var(--rise) .5s var(--ease-out) backwards;
  animation-delay:calc(.25s + var(--row,0) * 90ms + var(--col,0) * 60ms)}

/* Results table and bar chart */
.tbl tbody tr {animation:var(--rise) .45s var(--ease-out) backwards;animation-delay:calc(.2s + var(--i,0) * 60ms)}
.tbl tbody tr {transition:background-color .2s ease}
.tbl tbody tr:not(.final):hover {background:var(--cue-control)}
.bar-track i {transform-origin:left;animation:var(--grow) 1s var(--ease-out) backwards;
  animation-delay:calc(.2s + var(--i,0) * 90ms)}
[data-testid="stSidebar"] [role="radiogroup"] label {transition:background-color .2s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:var(--cue-control)}

@media (prefers-reduced-motion: reduce) {
  *,*::before,*::after {animation:none!important;transition:none!important}
}
</style>""", unsafe_allow_html=True)

# Poster colours per primary genre; all dark enough for white text.
GENRE_COLORS = {
    "Action": "#9a3c2a", "Adventure": "#8a5a17", "Animation": "#22735d", "Children's": "#2f6c94",
    "Comedy": "#8f6a12", "Crime": "#3d3f66", "Documentary": "#4c5f32", "Drama": "#5d3b78",
    "Fantasy": "#3e4f86", "Film-Noir": "#2f3438", "Horror": "#6b1d2b", "Musical": "#82386a",
    "Mystery": "#304a63", "Romance": "#963d5b", "Sci-Fi": "#17616f", "Thriller": "#55392a",
    "War": "#4f5034", "Western": "#80491f", "Children": "#2f6c94", "IMAX": "#2b4a5e",
}
def hero(eyebrow, title, subtitle):
    st.markdown(f'<div class="hero ph-{page_phase}"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title, note=""):
    st.markdown(f'<div class="section"><h3>{html.escape(title)}</h3><span>{html.escape(note)}</span></div>',
                unsafe_allow_html=True)


def because_text(movie):
    """'Because you liked <b>Inside Man</b> on a Monday evening, like now · shares Crime' (html, plain)."""
    shared = movie.get("shared") or []
    when = f" {movie['because_when']}" if movie.get("because_when") else ""
    tail = f" · shares {', '.join(shared)}" if shared else ""
    plain = f"Because you liked {label(movie['because'])}{when}{tail}"
    rich = f"Because you liked <b>{html.escape(label(movie['because']))}</b>{html.escape(when + tail)}"
    return rich, plain


def tooltip_html(movie, name, year, genres):
    """Hover/focus panel over the poster: every signal behind the card, as short text lines."""
    lines = []
    if movie.get("prediction") is not None:
        lines.append(("★", f"<b>{movie['prediction']:.1f}</b> stars predicted for you", False))
    if movie.get("moved", 0) > 0 and movie.get("context"):
        lines.append(("↑", f"Up <b>{movie['moved']}</b> place{'s' if movie['moved'] != 1 else ''} for "
                            f"{html.escape(movie['context'])}", True))
    if movie.get("because"):
        lines.append(("♥", because_text(movie)[0], bool(movie.get("because_when", "")
                                                          and "like now" in movie["because_when"])))
    if movie.get("context_tags"):
        lines.append(("◷", "Lifted by your <b>" + html.escape(", ".join(movie["context_tags"])) + "</b>", True))
    if movie.get("mood_match"):
        lines.append(("◎", f"Matches your mood: <b>{html.escape(', '.join(movie['mood_match']))}</b>", True))
    if movie.get("ratings"):
        lines.append(("☰", f"{movie['ratings']:,} MovieLens ratings", False))
    if movie.get("movie_id") is not None and int(movie["movie_id"]) in liked_ids():
        lines.append(("♥", "In your likes", False))
    if not lines:
        lines.append(("☰", html.escape(" · ".join(genres) or "Movie"), False))
    items = "".join(f'<li class="{"hot" if hot else ""}"><i>{icon}</i><span>{text}</span></li>'
                    for icon, text, hot in lines)
    return (f'<div class="tip" role="tooltip"><div class="tip-title">{html.escape(name)}'
            f'{f"<small>{year}</small>" if year else ""}</div><ul>{items}</ul></div>')


def liked_ids():
    return {like["movie_id"] for like in st.session_state.likes}


def card_html(movie, explanations, rank=None):
    name, year = display_title(movie["title"])
    genres = [g for g in str(movie["genres"]).split("|") if g]
    color = GENRE_COLORS.get(genres[0] if genres else "", "#34444f")
    second = GENRE_COLORS.get(genres[1], "#0b1114") if len(genres) > 1 else "#0b1114"
    initials = "".join(word[0] for word in re.findall(r"[A-Za-z0-9]+", name)[:2]).upper() or "C"
    badge = (f'<span class="year">{year}</span>' if year else "") + \
        (f'<span class="rank" title="Rank in this list">#{rank}</span>' if rank else "")
    if movie.get("moved", 0) > 0 and movie.get("context"):
        badge += f'<span class="lift">↑{movie["moved"]}</span>'
    if movie.get("movie_id") is not None and int(movie["movie_id"]) in liked_ids():
        badge += '<span class="liked">♥</span>'
    tip = tooltip_html(movie, name, year, genres)
    url = poster_url(movie.get("movie_id"))
    if url:
        poster = (f'<div class="poster has-img" tabindex="0">{badge}<img class="bg" src="{url}" alt="" aria-hidden="true" '
                  f'loading="lazy"><img class="fg" src="{url}" alt="Poster: {html.escape(name)}" loading="lazy">'
                  f'{tip}</div>')
    else:
        poster = (f'<div class="poster" tabindex="0" style="background:linear-gradient(140deg,{color} 10%,'
                  f'{second} 115%)">{badge}<span class="mark" aria-hidden="true">{html.escape(initials)}</span>'
                  f'<span class="genre">{html.escape(genres[0] if genres else "Movie")}</span>{tip}</div>')
    parts = [
        poster,
        f'<div class="movie-title" title="{html.escape(name)}">{html.escape(name)}</div>',
        '<div class="chips">' + "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genres[:4]) + "</div>",
    ]
    if movie.get("prediction") is not None:
        width = max(0.0, min(1.0, (movie["prediction"] - 1) / 4)) * 100
        parts.append(f'<div class="score"><span>Predicted <b>{movie["prediction"]:.1f}★</b></span>'
                     f'<span class="meter"><i style="width:{width:.0f}%"></i></span></div>')
    elif movie.get("ratings"):
        parts.append(f'<div class="score"><span><b>{movie["ratings"]:,}</b> ratings</span></div>')
    else:
        parts.append('<div class="score"></div>')  # keeps every card the same height
    if explanations:
        if movie.get("because"):
            text, plain = because_text(movie)
        elif movie.get("mood_match"):
            plain = f"Matches your mood: {', '.join(movie['mood_match'])}"
            text = html.escape(plain)
        else:
            plain = (movie.get("reason") or "Suggested from your ratings").rstrip(".")
            text = html.escape(plain)
        parts.append(f'<div class="reason"><span class="label">Why this pick</span>'
                     f'<span class="why" title="{html.escape(plain)}.">{text}.</span></div>')
    return "".join(parts)


def cards(rows, prefix, explanations=True, ranked=False,
          empty="No movies here yet. Try another filter or save a movie from Discover."):
    if not rows:
        st.info(empty)
        return
    keys = [str(movie.get("movie_id", movie["title"])) for movie in rows]
    # New list -> new phase -> the cards animate in again. Saving a movie keeps the list, so no replay.
    card_phase = phase(f"cards-{prefix}", tuple(keys))
    for start in range(0, len(rows), 3):
        row = min(start // 3, 5)  # the CSS stagger stops growing after the sixth row
        for offset, (column, movie) in enumerate(zip(st.columns(3, gap="medium"), rows[start:start + 3])):
            key = keys[start + offset]
            with column, st.container(border=True, key=f"cuecard-{card_phase}-{prefix}-row{row}-{key}"):
                rank = start + offset + 1 if ranked else None
                st.markdown(card_html(movie, explanations, rank), unsafe_allow_html=True)
                saved = key in st.session_state.watchlist
                movie_id = movie.get("movie_id")
                liked = movie_id is not None and int(movie_id) in liked_ids()
                like_col, save_col = st.columns([1, 2], gap="small")
                if movie_id is not None and like_col.button(
                        "♥ Liked" if liked else "♡ Like", key=f"{prefix}-like-{'on' if liked else 'off'}-{key}",
                        width="stretch", help="Unlike" if liked else
                        f"Like it now. CineCue remembers it was a {WEEKDAYS[weekday]} {DAYPARTS[part]}."):
                    toggle_like(int(movie_id), movie["title"])
                    st.rerun()
                if save_col.button("✓ Saved" if saved else "＋ Watchlist",
                                   key=f"{prefix}-btn-{'on' if saved else 'off'}-{key}", width="stretch",
                                   help="Remove from your watchlist" if saved else "Save to your watchlist"):
                    name = label(movie["title"])
                    if saved:
                        del st.session_state.watchlist[key]
                        st.session_state.toast = (f"Removed {name} from your watchlist", ":material/bookmark_remove:")
                    else:
                        st.session_state.watchlist[key] = movie
                        st.session_state.toast = (f"Saved {name} to your watchlist", ":material/bookmark_added:")
                    st.rerun()


# ---------- Data and models ----------
required = ["train.csv", "validation.csv", "movies_clean.csv", "movie_features.csv",
            "rating_summary.json", "ranking_summary.json"]
processed = next((p for p in [ROOT / "results", ROOT / "data/processed"]
                  if all(table_path(p, n).is_file() for n in required)), None)
large_artifact = ROOT / "artifacts/recommender-32m.pkl"  # built by scripts/build_app_catalog.py
artifact = ROOT / "artifacts/recommender.pkl"

active_artifact = large_artifact if large_artifact.is_file() else artifact
posters_file = ROOT / "results/posters.csv"  # built by scripts/fetch_posters.py


@st.cache_data
def load_posters(signature):
    if not posters_file.is_file():
        return {}
    frame = pd.read_csv(posters_file, usecols=["movie_id", "poster_url"])
    # Only Wikimedia image URLs are rendered; anything else falls back to the genre tile.
    frame = frame.loc[frame.poster_url.str.match(r"^https://(upload|thumb)\.wikimedia\.org/")]
    return {int(m): html.escape(u, quote=True) for m, u in zip(frame.movie_id, frame.poster_url)}


POSTERS = load_posters(posters_file.stat().st_mtime_ns if posters_file.is_file() else 0)


def poster_url(movie_id):
    return POSTERS.get(int(movie_id)) if movie_id is not None else None


@st.cache_resource(show_spinner=False)
def title_search(signature, _catalog):
    """Search index for the loaded catalog; rebuilt only when the model changes."""
    return TitleSearch(_catalog.title.to_dict())


@st.cache_resource(show_spinner="Preparing your library. The first launch can take a few minutes.")
def models(signature):
    if active_artifact.is_file():
        return load_bundle(active_artifact)
    if processed is None:
        history, catalog = official_tables(ROOT / "data/raw")
        bundle = fit_tables(history, catalog, dict(factors=64, reg=0.08, n_iter=10, seed=0),
                            dict(ease_reg=30000, alpha=0.75, rerank_weight=1.5))
    else:
        bundle = fit_demo(processed)
    save_bundle(artifact, bundle)
    return bundle


def summary(name):
    path = next((p / name for p in [ROOT / "results", ROOT / "data/processed"] if (p / name).is_file()), None)
    return json.loads(path.read_text(encoding="utf-8")) if path else None


@st.cache_resource(show_spinner=False)
def context_model(signature, _catalog):
    """Genre matrix and weekday genre table for the loaded catalog (built by scripts/evaluate_context.py)."""
    found = summary("context_summary.json")
    return context_tables(_catalog, found), found


# ---------- Pages ----------
if page == "Watchlist":
    hero("Watchlist", "Your next movie nights.", "Everything you saved, in one place. Download the list to keep it.")
    saved = list(st.session_state.watchlist.values())
    if saved:
        st.download_button("Download watchlist (CSV)",
                           pd.DataFrame(saved).drop(columns=["shared"], errors="ignore").to_csv(index=False),
                           "cinecue-watchlist.csv", "text/csv")
        st.caption("Saved for this browser session only.")
    cards(saved, "saved", empty="Nothing saved yet. Open Discover and save a few picks.")

elif page == "Project results":
    hero("Project results", "Behind the recommendations.",
         "How CineCue ranks movies, and how well it does on two years of held-out MovieLens ratings.")
    ranking, rating, explained = summary("ranking_summary.json"), summary("rating_summary.json"), \
        summary("explanation_summary.json")
    if ranking and rating:
        test = ranking["test_known_users"]
        rmse = rating["test"]
        best_rmse_name = "MF + session (online)" if "MF + session (online)" in rmse else "MF (online)"
        tiles = [
            ("Ranking quality", f"{test['blend']['ndcg']:.3f}",
             f"NDCG@10, <b>+{test['blend']['ndcg'] / test['popularity']['ndcg'] - 1:.0%}</b> vs popularity"),
            ("Rating accuracy", f"{rmse[best_rmse_name]['RMSE']:.3f}",
             f"RMSE in stars, <b>{rmse[best_rmse_name]['RMSE'] / rmse['global mean']['RMSE'] - 1:.0%}</b> vs global mean"),
            ("Catalog reach", f"{test['blend + re-rank']['coverage']:.1%}",
             f"with discovery on, vs <b>{test['popularity']['coverage']:.1%}</b> for popularity"),
        ]
        if explained:
            named = next(r for r in explained["deletion"] if r["condition"] == "named movie")
            control = next(r for r in explained["deletion"] if r["condition"] == "random")
            tiles.append(("Honest explanations", f"{named['share_left_top10']:.0%}",
                          f"of picks drop out when the named movie is removed, vs <b>{control['share_left_top10']:.0%}</b> at random"))
        for column, (name, value, note) in zip(st.columns(len(tiles), gap="small"), tiles):
            column.markdown(f'<div class="kpi ph-{page_phase}"><div class="name">{name}</div><div class="value">{value}</div>'
                            f'<div class="note">{note}</div></div>', unsafe_allow_html=True)
        section("Top-10 ranking on the test period", f"{int(test['blend']['users']):,} users known at the cutoff")
        final = {"popularity": "Most popular", "MF": "Predicted stars (MF)", "EASE": "EASE",
                 "blend": "CineCue blend", "blend + re-rank": "CineCue blend + discovery"}
        table = pd.DataFrame({final[k]: test[k] for k in final if k in test}).T
        shown = table[["precision", "recall", "ndcg", "coverage", "long_tail_share"]].astype(float).rename(columns={
            "precision": "Precision@10", "recall": "Recall@10", "ndcg": "NDCG@10",
            "coverage": "Catalog coverage", "long_tail_share": "Long-tail share"})
        formats = {"Precision@10": "{:.3f}", "Recall@10": "{:.3f}", "NDCG@10": "{:.3f}",
                   "Catalog coverage": "{:.1%}", "Long-tail share": "{:.1%}"}
        header = "".join(f"<th>{html.escape(c)}</th>" for c in shown.columns)
        body = "".join(
            f'<tr class="{"final" if name == "CineCue blend" else ""}" style="--i:{i}"><th scope="row">{html.escape(name)}</th>'
            + "".join(f"<td>{formats[c].format(row[c])}</td>" for c in shown.columns) + "</tr>"
            for i, (name, row) in enumerate(shown.iterrows()))
        st.markdown(f'<div class="tbl-wrap ph-{page_phase}"><table class="tbl"><thead><tr><th>Model</th>{header}</tr></thead>'
                    f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)
        ndcg = shown["NDCG@10"].sort_values(ascending=False)
        st.markdown(f'<div class="bars ph-{page_phase}" role="img" aria-label="NDCG@10 by model">' + "".join(
            f'<div class="bar-row" style="--i:{i}"><span class="name{" best" if i == 0 else ""}">{html.escape(name)}</span>'
            f'<span class="bar-track"><i style="width:{value / ndcg.max() * 100:.1f}%"></i></span>'
            f'<span class="num">{value:.3f}</span></div>' for i, (name, value) in enumerate(ndcg.items())) + "</div>",
            unsafe_allow_html=True)
        st.caption("A hit is a movie the user rated 4 or 5 stars later. Variants with recency weighting were "
                   "evaluated in notebook 6 and are not part of CineCue.")
    else:
        st.info("Evaluation summaries appear here when notebook results are available.")
    app_summary = summary("app_catalog_summary.json")
    if app_summary and large_artifact.is_file():
        facts, held = app_summary["catalog"], app_summary["holdout"]
        section("The website's catalog", f"{facts['dataset']}, a laptop-sized slice")
        st.write(f"The research results above use MovieLens 1M (films up to 2000). The website serves a newer, larger "
                 f"catalog: the {facts['movies']:,} most-rated films of MovieLens 32M ({facts['first_year']}–"
                 f"{facts['last_year']}), learned from {facts['ratings']:,} ratings by {facts['users']:,} sampled users. "
                 f"The same pipeline was re-tuned on each user's last {held['ratings_per_user']} ratings: NDCG@10 "
                 f"**{held['selected']['ndcg']:.3f}** against {held['popularity_ndcg']:.3f} for popularity "
                 f"({held['tuning_users']:,} users), with EASE penalty {held['selected']['ease_reg']:,} and blend "
                 f"weight {held['selected']['alpha']}. This is a sanity check, not a second study.")
    context = summary("context_summary.json")
    if context:
        section("Does the day of the week matter?", f"{context['events']['test']:,} test user-days, "
                                                     f"{context['events']['test_users']:,} users")
        st.write("A context re-ranker was trained on the validation period to use when a like happened: likes "
                 "from the same weekday or time of day, and how a user's genres shift on that weekday. It "
                 "was then checked once on the test period. Every gain is a paired bootstrap 95% interval "
                 "against the final ranker.")
        names = {"+ same weekday": "Same-weekday likes", "+ same daypart": "Same time-of-day likes",
                 "+ weekday genres (user)": "Your genres on this weekday",
                 "+ weekday genres (everyone)": "Everyone's genres on this weekday", "all context": "All context signals"}
        body = "".join(
            f'<tr style="--i:{i}"><th scope="row">{html.escape(names[v["variant"]])}</th>'
            f'<td>{v["validation_cv_ndcg"]:.4f}</td><td>{v["test_ndcg"]:.4f}</td>'
            f'<td>{v["vs_final_difference"]:+.4f}</td>'
            f'<td>[{v["vs_final_ci_low"]:+.4f}, {v["vs_final_ci_high"]:+.4f}]</td></tr>'
            for i, v in enumerate(r for r in context["variants"] if r["variant"] in names))
        base = next(r for r in context["variants"] if r["variant"] == "final ranker")
        st.markdown(f'<div class="tbl-wrap ph-{page_phase}"><table class="tbl"><thead><tr><th>Signal added</th>'
                    f'<th>Validation NDCG@10</th><th>Test NDCG@10</th><th>Gain</th><th>95% interval</th></tr></thead>'
                    f'<tbody><tr class="final"><th scope="row">Final ranker (no context)</th>'
                    f'<td>{base["validation_cv_ndcg"]:.4f}</td><td>{base["test_ndcg"]:.4f}</td><td>—</td><td>—</td></tr>'
                    f"{body}</tbody></table></div>", unsafe_allow_html=True)
        st.write("**No context signal helped.** Every interval includes zero or sits below it, so the validated "
                 "ranker uses none. MovieLens timestamps record when people *rated*, in UTC, often many movies "
                 "in one sitting, not when they watched. In Discover, *Lean on this day's likes* is therefore your "
                 "preference, and the likes this website records with their real local time are the data a "
                 "future test would need.")
    section("How your list is made")
    st.write("**EASE** learns which movies appear together in rating histories. **Matrix factorization** "
             "learns patterns in star ratings. CineCue blends both scores, skips movies you already rated "
             "and movies with fewer than 20 ratings, and names the liked movie that pushed each pick up the most. "
             "**Discover less popular films** reorders the top candidates with a popularity penalty.")
    st.write("Predicted stars come from matrix factorization on the whole profile. The rating-session "
             "adjustment from notebook 5 needs a live rating session, so these lists do not use it. "
             "UTC time and recency adjustments were tested and left out.")
    st.link_button("View source and notebooks", "https://github.com/zaeem886/cinecue")

else:
    if active_artifact.is_file():
        signature = ("saved-model", str(active_artifact), active_artifact.stat().st_mtime_ns)
    else:
        signature = ((str(processed), tuple(table_path(processed, n).stat().st_mtime_ns for n in required))
                     if processed else "ml-1m")
    try:
        bundle = models(signature)
    except Exception as error:  # surface a readable message instead of a stack trace
        st.error(f"The movie library could not load. {error}")
        if st.button("Try again"):
            st.rerun()
        st.stop()
    history, catalog, counts, settings = bundle[0], bundle[1], bundle[4], bundle[5]
    years = pd.to_numeric(catalog.title.str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce")
    dataset = settings.get("dataset", "MovieLens 1M")
    st.sidebar.caption(f"Catalog: {dataset} · {len(catalog):,} movies · {int(years.min())}–{int(years.max())}")
    titles = catalog.title.to_dict()
    genres = sorted({g for value in catalog.genres for g in str(value).split("|") if g})

    if page == "Movie library":
        hero("Movie library", "Find an old favorite.",
             f"Browse {len(catalog):,} movies. Search a title, filter by genre, and save anything for later.")
        query = st.text_input("Search movies", placeholder="Title, words or year, e.g. inside man 2006",
                              icon=":material/search:",
                              help="Every word must appear in the title, in any order. Accents and punctuation "
                                   "are ignored. Press Enter to search.")
        selected = st.pills("Genres", genres, selection_mode="multi", key="library_genres") or []
        st.caption("Hover a poster for details. ♡ Like stamps the movie with "
                   f"{WEEKDAYS[weekday]} {DAYPARTS[part]}.")
        records = catalog.assign(ratings=counts).rename_axis("movie_id").reset_index()[["movie_id", "title", "genres", "ratings"]]
        by_id = {int(r["movie_id"]): r for r in records.to_dict("records")}
        ranked, fuzzy = title_search(signature, catalog)(query, {m: r["ratings"] for m, r in by_id.items()})
        filtered = [by_id[m] for m in ranked
                    if not selected or set(selected).intersection(str(by_id[m]["genres"]).split("|"))]
        if fuzzy and filtered:
            st.caption(f"No exact match for “{query.strip()}”. Showing close titles.")
        pages = max(1, (len(filtered) + 11) // 12)
        info, pager = st.columns([3, 1])
        number = pager.number_input("Page", min_value=1, max_value=pages, value=1)
        order = "best matches first" if query.strip() else "most rated first"
        info.caption(f"{len(filtered):,} movies · {order} · page {number} of {pages}")
        cards(filtered[(number - 1) * 12:number * 12], "catalog", False,
              empty="No movies match. Try a shorter search or fewer genres.")

    else:
        hero("Discover", "Find your next movie.",
             "Pick a few movies you love, or start from a MovieLens profile. Every pick comes with the reason it was chosen.")
        st.markdown(f'<div class="stats ph-{page_phase}"><span class="stat"><b>{len(catalog):,}</b> movies</span>'
                    f'<span class="stat"><b>{history.user_id.nunique():,}</b> MovieLens profiles</span>'
                    f'<span class="stat"><b>{len(history):,}</b> ratings learned from</span>'
                    f'<span class="stat">Films from <b>{int(years.min())}</b> to <b>{int(years.max())}</b></span>'
                    f'<span class="stat"><b>{sum(poster_url(m) is not None for m in catalog.index):,}</b> posters</span></div>',
                    unsafe_allow_html=True)
        tables, context_summary = context_model(signature, catalog)
        likes = [like for like in st.session_state.likes if like["movie_id"] in titles]
        when = f"{WEEKDAYS[weekday]} {DAYPARTS[part]}"
        with st.container(border=True, key="controls"):
            st.markdown('<div class="ctl-head">Your taste</div>', unsafe_allow_html=True)
            mode = st.segmented_control("Start from", [":material/favorite: My likes", ":material/person: MovieLens profile"],
                                        default=":material/favorite: My likes", key="start_from",
                                        label_visibility="collapsed") or ":material/favorite: My likes"
            user = None
            if mode.endswith("My likes"):
                # The picker mirrors the likes, which card hearts and the editor below also change.
                # Streamlit drops a widget's state on pages that do not draw it, so the picker is
                # re-seeded from the likes whenever its state is missing or stale, and only a real
                # edit (the on_change callback) ever changes the likes.
                if "fav_select" not in st.session_state or \
                        st.session_state.get("fav_rev") != st.session_state.likes_rev:
                    st.session_state.fav_select = [like["movie_id"] for like in likes]
                    st.session_state.fav_rev = st.session_state.likes_rev

                def picked_changed(titles=titles, weekday=weekday, part=part):
                    picked = st.session_state.fav_select
                    current = {like["movie_id"] for like in st.session_state.likes}
                    kept = [like for like in st.session_state.likes if like["movie_id"] in picked
                            or like["movie_id"] not in titles]
                    added = [make_like(m, weekday=weekday, part=part, title=titles[m])
                             for m in picked if m not in current]
                    set_likes(kept + added)
                    st.session_state.fav_rev = st.session_state.likes_rev

                st.multiselect(
                    "Movies you love", sorted(titles, key=lambda m: label(titles[m])),
                    format_func=lambda m: label(titles[m]), key="fav_select", on_change=picked_changed,
                    # "contains" + no select-all: Enter adds the one highlighted title, never every fuzzy match.
                    filter_mode="contains", select_all=False,
                    placeholder="Start typing a title, e.g. Inside Man",
                    help=f"Each movie you add is stamped with the moment: right now that's {when}.")
                if likes:
                    with st.expander(f"When you liked them · {len(likes)}", icon=":material/schedule:"):
                        st.caption("Each like keeps the day and time it happened. Change them here to try, "
                                   "for example, \"I liked Inside Man on a Monday\".")
                        frame = pd.DataFrame({"movie_id": [like["movie_id"] for like in likes],
                                              "Movie": [label(titles[like["movie_id"]]) for like in likes],
                                              "Day": [WEEKDAYS[like["weekday"]] for like in likes],
                                              "Time of day": [DAYPARTS[like["part"]] for like in likes]})
                        edited = st.data_editor(
                            frame, hide_index=True, width="stretch", key=f"likes_editor_{st.session_state.likes_rev}",
                            column_order=["Movie", "Day", "Time of day"], disabled=["Movie"],
                            column_config={"Day": st.column_config.SelectboxColumn("Day", options=list(WEEKDAYS),
                                                                                   required=True),
                                           "Time of day": st.column_config.SelectboxColumn(
                                               "Time of day", options=list(DAYPARTS), required=True)})
                        changes = {int(m): (WEEKDAYS.index(day), DAYPARTS.index(tod))
                                   for m, day, tod, day0, tod0 in zip(edited.movie_id, edited["Day"],
                                                                      edited["Time of day"], frame["Day"],
                                                                      frame["Time of day"])
                                   if (day, tod) != (day0, tod0)}
                        if changes:
                            set_likes([like | ({"weekday": changes[like["movie_id"]][0],
                                                "part": changes[like["movie_id"]][1]}
                                               if like["movie_id"] in changes else {})
                                       for like in st.session_state.likes])
                            st.rerun()
            else:
                user = st.selectbox("MovieLens user", sorted(history.user_id.unique().tolist()),
                                    help="MovieLens profiles carry no like times here, so only the mood applies.")

            st.markdown('<div class="ctl-head">Tonight</div>', unsafe_allow_html=True)
            mood = st.pills("In the mood for", genres, selection_mode="multi", key="mood",
                            help="Genres you are in the mood for. They are boosted on top of your taste.")
            mood_col, lean_col, count_col = st.columns([1.1, 1.1, 0.8])
            strength = mood_col.segmented_control(
                "Mood strength", ["Nudge", "Strong", "Only these"], default="Strong", key="mood_strength",
                disabled=not mood, help="Nudge: a light boost. Strong: mood genres lead. Only these: nothing else.")
            lean = lean_col.segmented_control(
                f"Lean on {WEEKDAYS[weekday]} likes", list(LEAN), default="Strong", key="ctx_lean",
                disabled=user is not None,
                help=f"How much likes made on a {WEEKDAYS[weekday]} (and in the {DAYPARTS[part]}) count right now. "
                     "Your preference: on MovieLens data this signal showed no measurable gain (see Project results).")
            count = count_col.segmented_control("Picks", [6, 12, 18], default=12, key="pick_count") or 12
            left, right = st.columns(2)
            discover = left.toggle("Discover less popular films", disabled=user is None and not likes)
            explanations = right.toggle("Show why", value=True)

        lean = lean or "Off"
        ranker = ContextRanker(LEAN[lean] if user is None else LEAN["Off"])
        strength_key = {"Nudge": "nudge", "Strong": "strong", "Only these": "only"}.get(strength or "Strong")
        rows = recommend_in_context(bundle, tables, ranker, likes=None if user is not None else likes,
                                    user_id=user, k=count, discover=discover, weekday=weekday, part=part,
                                    mood=set(mood or []), mood_strength=strength_key)
        on_day = sum(like["weekday"] == weekday for like in likes)
        if user is None and likes:
            if lean != "Off" and on_day:
                text = (f"<b>{on_day} of your {len(likes)} likes</b> are from {WEEKDAYS[weekday]}s, so they "
                        f"count extra right now ({lean.lower()}).")
            elif lean != "Off":
                text = (f"None of your likes are from a {WEEKDAYS[weekday]} yet, so the day changes nothing. "
                        f"Like something now, or set a like's day under <i>When you liked them</i>.")
            else:
                text = "Day weighting is off: every like counts the same, whatever the day."
            note = ("Day weighting is your preference. On MovieLens 1M it gave no measurable gain in "
                    "top-10 accuracy.") if lean != "Off" else ""
            st.markdown(f'<div class="ctx-banner"><span class="ctx-icon">{DAYPART_ICONS[part]}</span><div>'
                        f'It\'s <b>{when}</b>. {text}<span class="note">{note}</span></div></div>',
                        unsafe_allow_html=True)
        personal = user is not None or bool(likes)
        mood_note = f" · mood: {', '.join(mood)}" if mood else ""
        if personal:
            basis = f"MovieLens user {user}" if user is not None else f"{len(likes)} like{'s' if len(likes) != 1 else ''}"
            section("Picked for you", f"Based on {basis}{mood_note}" + (" · discovery on" if discover else ""))
        else:
            section("Popular in the community", "Like a movie to personalise" + mood_note)
        cards(rows, "recommendation", explanations, ranked=True,
              empty="No movies fit this mood. Try Nudge or Strong instead of Only these.")

st.markdown('<div class="footer">CineCue · Movie discovery, with a reason · Data: MovieLens 1M, GroupLens Research '
            '(Harper &amp; Konstan, 2015) · Posters: Wikipedia</div>', unsafe_allow_html=True)
