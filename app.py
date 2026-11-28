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
from cinecue.accounts import AccountError, AccountStore, database_url, read_env_file  # noqa: E402
from cinecue.artifacts import load_bundle, save_bundle  # noqa: E402
from cinecue.context import DAYPARTS, WEEKDAYS, ContextRanker, daypart  # noqa: E402
from cinecue.data import table_path  # noqa: E402
from cinecue.demo import context_tables, fit_demo, fit_tables, recommend_in_context, taste_match  # noqa: E402
from cinecue.first_run import official_tables  # noqa: E402
from cinecue.hub import download_models, model_repo  # noqa: E402
from cinecue.likes import load_likes, make_like, save_likes  # noqa: E402
from cinecue.search import TitleSearch, display_title, label  # noqa: E402

LIKES_FILE = ROOT / "data/likes.json"  # local only; see .gitignore
PAGES = ["Discover", "Search", "Movie library", "Watchlist", "Project results"]
PAGE_ICONS = {"Discover": ":material/explore:", "Search": ":material/manage_search:",
              "Movie library": ":material/video_library:", "Watchlist": ":material/bookmarks:",
              "Project results": ":material/insights:"}
MOOD_STRENGTHS = {"Nudge": "nudge", "Strong": "strong", "Only these": "only"}
# Widget values that should survive moving between pages (Streamlit drops the state of widgets
# a page does not draw). restore() before the widget, keep() after it.
REMEMBERED = {"mood": [], "mood_strength": "Strong", "ctx_lean": "Strong", "search_query": "",
              "search_order": "Best title match"}
DAYPART_ICONS = {0: "☾", 1: "☀", 2: "◐", 3: "☾"}
# Context weights you choose in Discover. MovieLens showed no measurable gain from any of them
# (results/context_summary.json), so "Off" equals the validated final ranker.
LEAN = {"Off": [1.0, 0, 0, 0, 0], "Light": [1.0, 0.5, 0.2, 0.25, 0], "Strong": [1.0, 1.5, 0.5, 0.75, 0]}

st.set_page_config(page_title="CineCue | Movie discovery", page_icon="🎬", layout="wide")
if toast := st.session_state.pop("toast", None):
    st.toast(toast[0], icon=toast[1])



# ---------- Accounts (PostgreSQL tables from prisma/schema.prisma) ----------
read_env_file(ROOT / ".env")  # local DATABASE_URL; the file is git-ignored
DATABASE = database_url()


@st.cache_resource(ttl=60, show_spinner=False)
def accounts_available(url):
    """(True, None) when the account database is reachable and migrated, else (False, reason)."""
    try:
        AccountStore(url).check()
        return True, None
    except Exception as error:  # unreachable server, wrong password, or tables not migrated yet
        return False, (str(error).strip().splitlines() or [type(error).__name__])[0][:200]


ACCOUNTS, ACCOUNTS_ERROR = accounts_available(DATABASE) if DATABASE else (False, None)


def store():
    """This browser session's database connection."""
    if "account_store" not in st.session_state:
        st.session_state.account_store = AccountStore(DATABASE)
    return st.session_state.account_store


def current_user():
    return st.session_state.get("user") if ACCOUNTS else None


# ---------- Likes and the current moment ----------
# With accounts, guests keep likes and the watchlist for this visit and signed-in users keep them
# in the database. Without a database (local use), likes go to data/likes.json as before.
st.session_state.setdefault("watchlist", {})
if "likes" not in st.session_state:
    st.session_state.likes = [] if ACCOUNTS else load_likes(LIKES_FILE)
    st.session_state.likes_rev = 0


def set_likes(likes, message=None):
    """Replace the likes, persist them, and tell widgets that mirror them to refresh."""
    user = current_user()
    if user:
        try:
            store().sync_likes(user.id, likes)
        except Exception as error:  # keep the screen consistent with what was saved
            st.session_state.toast = (f"Your likes could not be saved: {error}", ":material/error:")
            return
    elif not ACCOUNTS:
        save_likes(LIKES_FILE, likes)
    st.session_state.likes = likes
    st.session_state.likes_rev += 1
    if message:
        st.session_state.toast = (message, ":material/favorite:")


def set_saved(movie, saved):
    """Add or remove a movie from the watchlist, in the account when signed in."""
    key = str(movie.get("movie_id", movie["title"]))
    user = current_user()
    if user and movie.get("movie_id") is not None:
        try:
            if saved:
                store().save(user.id, movie["movie_id"], movie["title"], movie.get("genres", ""))
            else:
                store().unsave(user.id, movie["movie_id"])
        except Exception as error:
            st.session_state.toast = (f"Your watchlist could not be saved: {error}", ":material/error:")
            return
    name = label(movie["title"])
    if saved:
        st.session_state.watchlist[key] = movie
        st.session_state.toast = (f"Saved {name} to your watchlist", ":material/bookmark_added:")
    else:
        st.session_state.watchlist.pop(key, None)
        st.session_state.toast = (f"Removed {name} from your watchlist", ":material/bookmark_remove:")


def finish_login(user, verb):
    """Load the account's likes and watchlist, adding anything liked or saved as a guest this visit."""
    accounts = store()
    likes = accounts.likes(user.id)
    known = {like["movie_id"] for like in likes}
    guest_likes = [like for like in st.session_state.likes if like["movie_id"] not in known]
    if guest_likes:
        likes = likes + guest_likes
        accounts.sync_likes(user.id, likes)
    saved = {str(item["movie_id"]): item for item in accounts.watchlist(user.id)}
    guest_saved = [movie for key, movie in st.session_state.watchlist.items()
                   if key not in saved and movie.get("movie_id") is not None]
    for movie in guest_saved:
        accounts.save(user.id, movie["movie_id"], movie["title"], movie.get("genres", ""))
        saved[str(movie["movie_id"])] = movie
    st.session_state.user = user
    st.session_state.likes = likes
    st.session_state.likes_rev += 1
    st.session_state.watchlist = saved
    merged = len(guest_likes) + len(guest_saved)
    st.session_state.toast = (f"{verb}, {user.display_name}." + (f" Added {merged} item{'s' if merged != 1 else ''} "
                              "from this visit to your account." if merged else ""), ":material/waving_hand:")


def log_out():
    name = st.session_state.pop("user").display_name
    st.session_state.likes = []
    st.session_state.likes_rev += 1
    st.session_state.watchlist = {}
    st.session_state.toast = (f"Logged out. See you soon, {name}.", ":material/logout:")


def restore(key):
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(f"kept_{key}", REMEMBERED[key])


def keep(key):
    st.session_state[f"kept_{key}"] = st.session_state.get(key)


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


# ---------- Sidebar ----------
weekday, part = moment()
when_now = f"{WEEKDAYS[weekday]} {DAYPARTS[part]}"
with st.sidebar:
    st.markdown('<div class="brand">CINECUE<small>Movie picks, with a reason</small></div>', unsafe_allow_html=True)
    if ACCOUNTS:
        user = current_user()
        with st.container(key="account"):
            if user:
                st.markdown(f'<div class="account"><span class="avatar" aria-hidden="true">'
                            f'{html.escape(user.display_name[:1].upper())}</span><div><b>{html.escape(user.display_name)}</b>'
                            f'<span>{html.escape(user.email)}</span></div></div>', unsafe_allow_html=True)
                if st.button("Log out", key="logout", icon=":material/logout:", width="stretch"):
                    log_out()
                    st.rerun()
            else:
                with st.popover("Log in or sign up", icon=":material/account_circle:", width="stretch"):
                    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
                    with login_tab, st.form("login_form", clear_on_submit=True, border=False):
                        email = st.text_input("Email", autocomplete="email")
                        password = st.text_input("Password", type="password", autocomplete="current-password")
                        if st.form_submit_button("Log in", type="primary", width="stretch"):
                            try:
                                finish_login(store().log_in(email, password), "Welcome back")
                                st.rerun()
                            except AccountError as error:
                                st.error(str(error))
                            except Exception:
                                st.error("The account service is not responding. Try again in a moment.")
                    with signup_tab, st.form("signup_form", clear_on_submit=True, border=False):
                        name = st.text_input("Name", autocomplete="name", max_chars=60)
                        email = st.text_input("Email", autocomplete="email")
                        password = st.text_input("Password", type="password", autocomplete="new-password",
                                                 help="At least 8 characters. Stored only as a salted scrypt hash.")
                        confirm = st.text_input("Repeat password", type="password", autocomplete="new-password")
                        if st.form_submit_button("Create account", type="primary", width="stretch"):
                            if password != confirm:
                                st.error("The two passwords do not match.")
                            else:
                                try:
                                    finish_login(store().sign_up(email, name, password), "Welcome")
                                    st.rerun()
                                except AccountError as error:
                                    st.error(str(error))
                                except Exception:
                                    st.error("The account service is not responding. Try again in a moment.")
                st.markdown('<div class="ctx-hint">Browsing as a guest: likes and saves last for this visit. '
                            'Log in to keep them.</div>', unsafe_allow_html=True)
    elif DATABASE:
        st.caption(f":material/cloud_off: Accounts are unavailable right now ({ACCOUNTS_ERROR}).")
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
    dark_mode = st.toggle(":material/dark_mode: Dark mode", value=True, key="dark_mode")
page_phase = phase("page", page)

# ---------- Theme ----------
# Flat and dark by default: near-black surfaces, one red accent for actions and the active page,
# green for strong matches. No gradients, glows or decorative motion.
palette = (
    dict(bg="#141414", side="#0e0e0e", surface="#1c1c1c", control="#262626", text="#f2f2f2", muted="#a0a0a0",
         border="#2c2c2c", accent="#e0262f", accent_soft="rgba(224,38,47,.14)", bar="#333333", good="#46d369",
         lv3="#8bc34a", lv2="#e0b341", lv1="#e8853d", bad="#e5484d", like="#ff4d5a")
    if dark_mode else
    # Match colours are darker here so they keep at least 4.5:1 contrast on the light background.
    dict(bg="#f4f4f4", side="#ffffff", surface="#ffffff", control="#ebebeb", text="#141414", muted="#5e5e5e",
         border="#dddddd", accent="#c3161e", accent_soft="rgba(195,22,30,.08)", bar="#e2e2e2", good="#12773a",
         lv3="#4d7c0f", lv2="#935708", lv1="#c2410c", bad="#c42b30", like="#c3161e")
)
tokens = ";".join(f"--cue-{name.replace('_', '-')}:{value}" for name, value in palette.items())
st.markdown(f"<style>:root{{{tokens};color-scheme:{'dark' if dark_mode else 'light'}}}</style>",
            unsafe_allow_html=True)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp {background:var(--cue-bg);color:var(--cue-text)}
html,body,[data-testid="stApp"],.stApp button,.stApp input {font-family:'Inter',system-ui,sans-serif}
.block-container {max-width:1320px;padding-top:1.4rem;padding-bottom:3rem}
[data-testid="stHeader"] {background:var(--cue-bg)}
[data-testid="stHeader"] button {color:var(--cue-text)}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp summary,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:var(--cue-text)}
[data-testid="stWidgetLabel"] p {font-weight:500;font-size:.88rem}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {color:var(--cue-muted)!important}
[data-testid="stTooltipIcon"] svg {fill:none!important;stroke:var(--cue-muted)}
[data-testid="stPopover"] button {background:var(--cue-control)!important;color:var(--cue-text)!important;border-radius:4px;
  border:1px solid var(--cue-border)!important}
[data-testid="stPopover"] button * {color:var(--cue-text)!important}
[data-testid="stPopoverBody"] {background:var(--cue-surface)!important;border:1px solid var(--cue-border)}
[data-testid="stTabs"] button p {color:var(--cue-muted)} [data-testid="stTabs"] button[aria-selected="true"] p {color:var(--cue-text)}
/* Popover content is rendered outside .stApp, so the login and sign-up forms need their own colours. */
[data-testid="stPopoverBody"] [data-testid="stWidgetLabel"] p,[data-testid="stPopoverBody"] label,
[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p {color:var(--cue-text)!important}
[data-testid="stPopoverBody"] [data-testid="stTextInputRootElement"],[data-testid="stPopoverBody"] input {
  background:var(--cue-control)!important;color:var(--cue-text)!important;border-color:var(--cue-border)!important}
[data-testid="stPopoverBody"] input::placeholder {color:var(--cue-muted)}
[data-testid="stPopoverBody"] [data-testid="stTextInputRootElement"] button,
[data-testid="stPopoverBody"] [data-testid="stTooltipIcon"] svg {color:var(--cue-muted)!important;fill:var(--cue-muted)}
[data-testid="stPopoverBody"] [data-testid="stTabs"] button p {color:var(--cue-muted)!important}
[data-testid="stPopoverBody"] [data-testid="stTabs"] button[aria-selected="true"] p {color:var(--cue-text)!important}
[data-testid="stBaseButton-primaryFormSubmit"] {background:var(--cue-accent)!important;border-color:var(--cue-accent)!important}
[data-testid="stPopoverBody"] [data-testid="stBaseButton-primaryFormSubmit"] [data-testid="stMarkdownContainer"] p,
[data-testid="stBaseButton-primaryFormSubmit"] p {color:#fff!important;font-weight:600}

/* ---------- Controls: flat, square-ish, no glow ---------- */
.stButton button,.stDownloadButton button,.stLinkButton a {border-radius:4px;background:var(--cue-control);
  color:var(--cue-text);border:1px solid transparent;font-weight:600;transition:background-color .15s ease}
.stButton button:hover,.stDownloadButton button:hover,.stLinkButton a:hover {background:var(--cue-bar);color:var(--cue-text);
  border-color:transparent}
.stButton button:focus-visible {outline:2px solid var(--cue-text);outline-offset:1px}
[data-testid="stBaseButton-primary"] {background:var(--cue-accent)!important;color:#fff!important}
[data-testid="stToast"] {background:var(--cue-surface);color:var(--cue-text);border:1px solid var(--cue-border)}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],.stApp input,
[data-baseweb="popover"],[role="listbox"],[role="option"] {background:var(--cue-control)!important;color:var(--cue-text)!important}
[data-baseweb="select"] span,[data-baseweb="select"] input {color:var(--cue-text)!important}
[data-baseweb="select"] div {background-color:var(--cue-control)!important;border-color:var(--cue-border)!important}
[data-baseweb="select"] svg,[data-baseweb="input"] svg {fill:var(--cue-text)}
[data-baseweb="popover"] *,[role="option"] * {color:var(--cue-text)}
.stApp input::placeholder {color:var(--cue-muted)}
[data-baseweb="tag"] {background:var(--cue-bar)!important;color:var(--cue-text);border-radius:3px!important}
[data-testid="stExpander"] details {background:var(--cue-surface);border:1px solid var(--cue-border);border-radius:6px}
[data-testid="stExpander"] summary p {font-weight:600}
/* The base theme in config.toml is light, so Streamlit paints these from light colours unless they are set here:
   an open expander's header (#f0f0f0), the "Press Enter to …" input hint and the sidebar collapse arrows. */
[data-testid="stExpander"] summary {background:var(--cue-surface)!important;color:var(--cue-text)!important}
[data-testid="stExpander"] summary:hover {background:var(--cue-control)!important}
[data-testid="stExpander"] summary [data-testid="stExpanderIcon"] {color:var(--cue-muted)!important}
[data-testid="InputInstructions"] {color:var(--cue-muted)!important}
/* The login and sign-up fields are too narrow for the hint: it covered the password and the show-password eye. */
[data-testid="stPopoverBody"] [data-testid="InputInstructions"] {display:none}
[data-testid="stSidebarCollapseButton"] button,[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {color:var(--cue-muted)!important}
[data-testid="stAlert"],[data-testid="stAlertContainer"] {background:var(--cue-surface)!important;color:var(--cue-text)!important;
  border-radius:6px}
[data-testid="stAlertContainer"] p,[data-testid="stAlertContainer"] em {color:var(--cue-text)!important}
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"] {color:var(--cue-accent)!important}
[data-testid="stCheckbox"]:not([data-selected="true"]) label > div:not([data-testid]) {background:var(--cue-bar)!important}
[data-testid="stSelectbox"] [role="group"],[data-testid="stMultiSelect"] [role="group"] {background:var(--cue-control)!important;
  border-color:var(--cue-border)!important}
[data-testid="stSelectbox"] button,[data-testid="stMultiSelect"] button {background:transparent!important}
[data-testid="stSelectbox"] svg,[data-testid="stMultiSelect"] svg {fill:var(--cue-text);color:var(--cue-text)}
[data-testid="stNumberInput"] input,[data-testid="stNumberInput"] button {background:var(--cue-control)!important;color:var(--cue-text)!important}
/* Pills and segmented controls: selected is simply inverted. */
[data-testid="stButtonGroup"] button[data-variant] {background:transparent!important;color:var(--cue-text)!important;
  border-color:var(--cue-border)!important;border-radius:4px!important}
[data-testid="stButtonGroup"] button[data-variant] * {color:inherit!important}
[data-testid="stButtonGroup"] button[data-variant][data-selected="true"] {background:var(--cue-text)!important;
  color:var(--cue-bg)!important;border-color:var(--cue-text)!important}
[data-testid="stButtonGroup"] button[data-variant]:hover:not(:disabled):not([data-selected="true"]) {border-color:var(--cue-muted)!important}
[data-testid="stButtonGroup"] button[data-variant]:disabled {opacity:.45}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {background:var(--cue-side);border-right:1px solid var(--cue-border);min-width:280px;max-width:280px}
[data-testid="stSidebarUserContent"] {padding-top:.2rem}
[data-testid="stSidebarHeader"] {height:2.4rem;margin-bottom:0}
/* Streamlit pulls each markdown block up by 1rem; sidebar blocks here carry their own spacing. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {margin-bottom:0}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.5rem}
.brand {font-size:1.6rem;font-weight:800;letter-spacing:.02em;color:var(--cue-accent);line-height:1;padding:.1rem 0 1rem}
.brand small {display:block;font-size:.74rem;font-weight:500;letter-spacing:0;color:var(--cue-muted);margin-top:.45rem}
.account {display:flex;align-items:center;gap:.65rem;padding:.5rem .55rem;margin-bottom:.4rem;border-radius:6px;
  background:var(--cue-control)}
.account .avatar {flex:none;width:2rem;height:2rem;border-radius:4px;display:grid;place-items:center;font-weight:700;
  background:var(--cue-accent);color:#fff}
.account b {display:block;font-size:.9rem;color:var(--cue-text);line-height:1.2}
.account span:not(.avatar) {display:block;font-size:.76rem;color:var(--cue-muted);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;max-width:11rem}
.st-key-account {margin-bottom:.2rem}
.side-label {display:flex;align-items:center;gap:.45rem;font-size:.7rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--cue-muted);margin:1rem 0 .05rem}
.side-label b {font-size:.7rem;font-weight:600;color:var(--cue-text)}
/* Navigation: plain rows; the active page is bold with a red bar. */
[data-testid="stSidebar"] [role="radiogroup"] {gap:.05rem}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:.5rem .7rem;border-radius:4px;width:100%;margin:0;
  position:relative;cursor:pointer;transition:background-color .15s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:var(--cue-control)}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:not([data-testid]) {display:none}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {font-size:.95rem;font-weight:500;display:flex;align-items:center;
  gap:.65rem;color:var(--cue-muted)}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p span[role="img"] {font-size:1.2rem;color:inherit}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {color:var(--cue-text);font-weight:700}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before {content:"";position:absolute;left:0;
  top:20%;bottom:20%;width:3px;background:var(--cue-accent)}
.st-key-ctxcard {background:var(--cue-control);border-radius:6px;padding:.7rem .8rem .5rem;gap:.35rem!important}
.st-key-ctxcard [data-testid="stMarkdownContainer"] p {margin:0}
.ctx-now {display:flex;align-items:center;gap:.6rem}
.ctx-now b {display:block;font-size:1rem;line-height:1.2;color:var(--cue-text)}
.ctx-now span:not(.ctx-icon) {display:block;font-size:.8rem;line-height:1.2;color:var(--cue-muted)}
.ctx-icon {display:grid;place-items:center;width:1.9rem;height:1.9rem;border-radius:4px;background:var(--cue-bar);
  color:var(--cue-text);font-size:1rem}
.ctx-hint {font-size:.78rem;color:var(--cue-muted);margin-top:.4rem;line-height:1.4}
.like-list {list-style:none;margin:0;padding:0;display:grid;grid-template-columns:minmax(0,1fr);gap:.1rem;width:100%}
.like-list li {display:flex;justify-content:space-between;align-items:center;gap:.5rem;padding:.3rem .4rem;
  border-radius:4px;font-size:.84rem;margin:0;min-width:0;max-width:100%;box-sizing:border-box}
.like-list li:hover {background:var(--cue-control)}
.like-list .t {flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--cue-text)}
.like-list .d {flex-shrink:0;font-size:.72rem;color:var(--cue-muted)}
.like-list li.same .d {color:var(--cue-text);font-weight:600}
.like-list li.same .t::before {content:"";display:inline-block;width:3px;height:.8em;margin-right:.45rem;
  background:var(--cue-accent);vertical-align:-.05em}
.side-gap {height:.2rem;border-top:1px solid var(--cue-border);margin-top:.8rem}

/* ---------- Page headings ---------- */
.hero {padding:.2rem 0 1rem}
.eyebrow {color:var(--cue-muted);font-weight:600;font-size:.76rem;letter-spacing:.12em;text-transform:uppercase}
.hero h1 {font-size:2.5rem;line-height:1.08;font-weight:800;letter-spacing:-.025em;margin:.3rem 0 .45rem;padding:0}
.hero p {color:var(--cue-muted)!important;font-size:1rem;max-width:620px;margin:0}
.stats {font-size:.84rem;color:var(--cue-muted);margin:.5rem 0 .4rem}
.stats b {color:var(--cue-text);font-weight:600}
.stat {display:inline-block;background:var(--cue-control);border-radius:4px;padding:.3rem .65rem;font-size:.84rem;
  color:var(--cue-muted)}
.stat b {color:var(--cue-text);font-weight:600}
.section {display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:.4rem;margin:1.6rem 0 .7rem}
.stApp .section h3 {font-size:1.35rem!important;font-weight:700;margin:0;padding:0;letter-spacing:-.01em;color:var(--cue-text)}
.section .note {color:var(--cue-muted);font-size:.85rem}
.ctl-head {font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--cue-muted);margin:.2rem 0 -.2rem}
.ctx-banner {display:flex;align-items:center;gap:.75rem;padding:.6rem .8rem;border-radius:6px;background:var(--cue-surface);
  border-left:3px solid var(--cue-accent);font-size:.88rem;margin:.6rem 0 .2rem}
.ctx-banner b {color:var(--cue-text)}
.ctx-banner span.note {color:var(--cue-muted);font-size:.78rem;display:block}

/* ---------- Search bar ---------- */
.st-key-hero_search {margin:.2rem 0 .4rem;gap:.35rem!important}
.st-key-hero_search [data-baseweb="input"],.st-key-hero_search [data-baseweb="base-input"] {border-radius:4px!important}
.st-key-hero_search [data-testid="stTextInputRootElement"] {height:3.3rem;border-radius:4px;border:1px solid var(--cue-border)!important;
  background:var(--cue-control)!important}
.st-key-hero_search [data-testid="stTextInputRootElement"]:focus-within {border-color:var(--cue-text)!important}
.st-key-hero_search input {font-size:1.05rem!important;background:var(--cue-control)!important}
.st-key-hero_search [data-testid="stTextInputRootElement"] [data-testid="stIconMaterial"] {color:var(--cue-text);font-size:1.4rem}
.hero-search-hint {font-size:.8rem;color:var(--cue-muted)}
.hero-search-hint b {color:var(--cue-text);font-weight:600}

/* ---------- Movie tiles ---------- */
/* Cards are keyed (cuecard-*) so they can be styled; the grid wraps instead of squeezing tiles. */
[class*="st-key-cuecard"] {background:transparent;border:0!important;padding:0!important;gap:.45rem!important}
/* Every tile in every grid gets the same width: six per row, three or two in narrower windows. */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-cuecard"]) {flex-wrap:wrap!important;
  column-gap:1rem!important;row-gap:1.4rem;justify-content:flex-start}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-cuecard"]) > [data-testid="stColumn"] {
  flex:0 0 calc((100% - 5rem) / 6)!important;width:calc((100% - 5rem) / 6)!important;min-width:0!important;max-width:none!important}
@media (max-width:1180px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-cuecard"]) > [data-testid="stColumn"] {
    flex-basis:calc((100% - 2rem) / 3)!important;width:calc((100% - 2rem) / 3)!important}}
@media (max-width:640px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-cuecard"]) > [data-testid="stColumn"] {
    flex-basis:calc((100% - 1rem) / 2)!important;width:calc((100% - 1rem) / 2)!important}}
/* Streamlit pulls each markdown block up by 1rem; tiles carry their own spacing. */
[class*="st-key-cuecard"] [data-testid="stMarkdownContainer"],.st-key-billboard [data-testid="stMarkdownContainer"] {margin-bottom:0}
.poster {position:relative;aspect-ratio:2/3;width:100%;border-radius:4px;overflow:hidden;background:var(--cue-control);
  display:flex;align-items:flex-end;outline:none;transition:transform .2s ease,box-shadow .2s ease}
.poster img {position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
[class*="st-key-cuecard"]:hover .poster,.poster:focus-visible {transform:scale(1.035);box-shadow:0 10px 24px rgba(0,0,0,.45);z-index:2}
.poster .flat {position:absolute;inset:0;padding:.8rem;display:flex;flex-direction:column;justify-content:flex-end;color:#fff}
.poster .flat b {font-size:1.05rem;line-height:1.2;font-weight:700}
.poster .flat span {font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.8;margin-top:.3rem}
.poster .rank,.poster .lift,.poster .liked {position:absolute;z-index:1;background:rgba(0,0,0,.72);color:#fff;border-radius:3px;
  padding:.08rem .4rem;font-size:.72rem;font-weight:700;font-variant-numeric:tabular-nums}
.poster .rank {left:.45rem;top:.45rem}
.poster .lift {left:.45rem;bottom:.45rem;color:var(--cue-good)}
.poster .liked {right:.45rem;top:.45rem;color:#ff4d5a}
.movie-title {font-size:.95rem;font-weight:600;line-height:1.25;height:2.5em;margin:.15rem 0 0;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.meta {font-size:.76rem;color:var(--cue-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;height:1.2rem}
.meta .star {color:var(--cue-text);font-weight:600;margin-right:.35rem}
.reason {font-size:.78rem;line-height:1.35;color:var(--cue-muted);height:2.7em;overflow:hidden;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.reason b {color:var(--cue-text);font-weight:600}
[class*="st-key-cuecard"] .stButton button {min-height:2rem;padding:.2rem .4rem;font-size:.8rem}
[class*="-like-on-"] button {color:var(--cue-like)!important}
[class*="-btn-on-"] button {color:var(--cue-good)!important}

/* ---------- Taste match ---------- */
.lv5,.lv4 {--lv:var(--cue-good)} .lv3 {--lv:var(--cue-lv3)} .lv2 {--lv:var(--cue-lv2)} .lv1 {--lv:var(--cue-lv1)}
.lv0 {--lv:var(--cue-bad)}
.match {height:4.1rem;overflow:hidden}
.match .mv {display:flex;align-items:baseline;gap:.4rem;font-size:.8rem;white-space:nowrap}
.match .mv b {color:var(--lv);font-weight:700}
.match .mv .mr {color:var(--cue-text);font-weight:700;font-variant-numeric:tabular-nums}
.match .shift {font-size:.72rem;color:var(--cue-muted);margin-left:auto}
.match .shift.up {color:var(--cue-good)} .match .shift.down {color:var(--cue-bad)}
.match .meter {display:block;height:3px;margin:.3rem 0;background:var(--cue-bar)}
.match .meter i {display:block;height:100%;background:var(--lv)}
.match .mw {font-size:.76rem;color:var(--cue-muted);line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;
  -webkit-box-orient:vertical;overflow:hidden}
.match .mw b {color:var(--cue-text);font-weight:600}
.taste-strip {display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin:.3rem 0 .7rem}
.taste-strip .ts-label {font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--cue-muted);
  margin-right:.3rem}
.legend {display:flex;flex-wrap:wrap;gap:.45rem 1rem;margin:1.2rem 0 .4rem}
.match-key {display:inline-flex;align-items:center;gap:.4rem;font-size:.84rem;color:var(--cue-text)}
.match-key i {width:.7rem;height:.7rem;border-radius:2px;background:var(--lv)}

/* ---------- Hover details on posters ---------- */
.poster .tip {position:absolute;inset:0;z-index:3;display:flex;flex-direction:column;gap:.35rem;overflow-y:auto;
  scrollbar-width:none;padding:.7rem;background:rgba(14,14,14,.93);color:#f2f2f2;opacity:0;
  transition:opacity .18s ease;pointer-events:none;text-align:left}
[class*="st-key-cuecard"]:hover .poster .tip {pointer-events:auto}
[class*="st-key-cuecard"]:hover .poster .tip,.poster:focus-within .tip,.poster:focus .tip {opacity:1}
.tip .tip-title {font-weight:700;font-size:.9rem;line-height:1.25;margin-top:auto}
.tip .tip-title small {font-weight:500;opacity:.65;margin-left:.3rem}
.tip ul {list-style:none;margin:0;padding:0;display:grid;gap:.28rem}
.tip li {display:flex;gap:.45rem;align-items:flex-start;font-size:.74rem;line-height:1.35;margin:0;color:#cfcfcf}
.tip li i {font-style:normal;flex-shrink:0;width:1rem;text-align:center;color:#8c8c8c}
.tip li.hot i {color:#46d369}
.tip li b {color:#fff}

/* ---------- Featured pick (Discover) ---------- */
.st-key-billboard {background:var(--cue-surface);border-radius:6px;padding:1.1rem!important;margin:.4rem 0 .2rem}
.st-key-billboard .poster {max-width:220px}
.bb-label {font-size:.74rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--cue-accent)}
.bb-title {font-size:2.1rem;font-weight:800;letter-spacing:-.02em;line-height:1.1;margin:.35rem 0 .45rem}
.bb-meta {font-size:.88rem;color:var(--cue-muted);margin-bottom:.8rem}
.bb-meta .good {color:var(--cue-good);font-weight:700;margin-right:.5rem}
.bb-why {font-size:.98rem;line-height:1.5;max-width:560px;color:var(--cue-text)}
.bb-why b {font-weight:700}
.bb-tags {display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.8rem}
.bb-tags span {font-size:.76rem;background:var(--cue-control);border-radius:3px;padding:.2rem .5rem;color:var(--cue-muted)}
.st-key-billboard .stButton button {min-height:2.5rem;font-size:.92rem}
.st-key-billboard [class*="-like-off-"] button {background:var(--cue-text)!important}
.st-key-billboard [class*="-like-off-"] button p {color:var(--cue-bg)!important}
.st-key-billboard [class*="-like-off-"] button:hover {opacity:.85}

/* ---------- Results page ---------- */
.kpi {background:var(--cue-surface);border-radius:6px;padding:1rem 1.1rem;height:100%}
.kpi .name {font-size:.74rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--cue-muted)}
.kpi .value {font-size:2rem;font-weight:800;letter-spacing:-.02em;margin:.25rem 0 .1rem}
.kpi .note {font-size:.82rem;color:var(--cue-muted)}
.kpi .note b {color:var(--cue-text)}
.tbl-wrap {overflow-x:auto;border-radius:6px;background:var(--cue-surface);margin-bottom:1rem}
.tbl {width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;margin:0!important}
.tbl th,.tbl td {border-left:0!important;border-right:0!important;border-top:0!important;background:transparent}
.tbl th,.tbl td {padding:.6rem .85rem;border-bottom:1px solid var(--cue-border);text-align:right;white-space:nowrap}
.tbl thead th {color:var(--cue-muted);font-weight:600;font-size:.74rem;letter-spacing:.05em;text-transform:uppercase}
.tbl th:first-child {text-align:left;font-weight:500;color:var(--cue-text)}
.tbl thead th:first-child {color:var(--cue-muted)}
.tbl tbody tr:last-child th,.tbl tbody tr:last-child td {border-bottom:0}
.tbl tr.final {background:var(--cue-control)}
.tbl tr.final th {font-weight:700;box-shadow:inset 3px 0 0 var(--cue-accent)}
.bars {display:grid;gap:.55rem;margin:.4rem 0 .6rem}
.bar-row {display:grid;grid-template-columns:minmax(120px,220px) 1fr 3.2rem;align-items:center;gap:.75rem;font-size:.86rem}
.bar-row .name {color:var(--cue-muted)}
.bar-row .name.best {color:var(--cue-text);font-weight:600}
.bar-track {height:8px;background:var(--cue-bar);overflow:hidden}
.bar-track i {display:block;height:100%;background:var(--cue-muted)}
.bar-row:first-child .bar-track i {background:var(--cue-accent)}
.bar-row .num {text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.footer {border-top:1px solid var(--cue-border);margin-top:3rem;padding-top:1rem;color:var(--cue-muted);font-size:.8rem}
@media(max-width:640px){.hero h1{font-size:1.9rem}.bb-title{font-size:1.6rem}.kpi .value{font-size:1.6rem}}
/* ---------- Motion: short, quiet entrances ----------
   Each keyframe exists twice (-a/-b). phase() in Python flips the class between them when the
   content changes, which restarts the animation on DOM nodes Streamlit reuses. Fill mode is
   `backwards` so a finished animation stops holding `transform` and the hover scale still works. */
@keyframes cue-rise-a {from {opacity:0;transform:translateY(8px)} to {opacity:1;transform:none}}
@keyframes cue-rise-b {from {opacity:0;transform:translateY(8px)} to {opacity:1;transform:none}}
@keyframes cue-grow-a {from {transform:scaleX(0)} to {transform:scaleX(1)}}
@keyframes cue-grow-b {from {transform:scaleX(0)} to {transform:scaleX(1)}}
.ph-a,[class*="st-key-cuecard-a-"] {--rise:cue-rise-a;--grow:cue-grow-a}
.ph-b,[class*="st-key-cuecard-b-"] {--rise:cue-rise-b;--grow:cue-grow-b}
:root {--ease-out:cubic-bezier(.2,.7,.2,1)}
.hero > *,.kpi,.tbl-wrap,.bars,.taste-strip,.bb {animation:var(--rise) .45s var(--ease-out) backwards}
.hero > :nth-child(2) {animation-delay:.05s} .hero > :nth-child(3) {animation-delay:.1s}
[class*="st-key-cuecard"] {animation:var(--rise) .4s var(--ease-out) backwards;
  animation-delay:calc(var(--row,0) * 70ms + var(--col,0) * 40ms)}
[class*="st-key-cuecard"][class*="-row1-"] {--row:1} [class*="st-key-cuecard"][class*="-row2-"] {--row:2}
[class*="st-key-cuecard"][class*="-row3-"] {--row:3} [class*="st-key-cuecard"][class*="-row4-"] {--row:4}
.stColumn:nth-child(2) {--col:1} .stColumn:nth-child(3) {--col:2} .stColumn:nth-child(4) {--col:3}
.stColumn:nth-child(5) {--col:4} .stColumn:nth-child(6) {--col:5}
.match .meter i,.bar-track i {transform-origin:left;animation:var(--grow) .7s var(--ease-out) backwards;animation-delay:.15s}
@media (prefers-reduced-motion: reduce) {*,*::before,*::after {animation:none!important;transition:none!important}}
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
    st.markdown(f'<div class="section"><h3>{html.escape(title)}</h3><span class="note">{html.escape(note)}</span></div>',
                unsafe_allow_html=True)


def because_text(movie):
    """'Because you liked <b>Inside Man</b> on a Monday evening, like now · shares Crime' (html, plain)."""
    shared = movie.get("shared") or []
    when = f" {movie['because_when']}" if movie.get("because_when") else ""
    tail = f" · shares {', '.join(shared)}" if shared else ""
    plain = f"Because you liked {label(movie['because'])}{when}{tail}"
    rich = f"Because you liked <b>{html.escape(label(movie['because']))}</b>{html.escape(when + tail)}"
    return rich, plain


def tooltip_html(movie, name, year, genres, explanations=True):
    """Hover/focus panel over the poster: every signal behind the card, as short text lines.

    With explanations off ("Show why"), only facts remain: predicted stars and rating counts.
    """
    lines = match_lines(movie["match"]) if movie.get("match") else []
    if movie.get("prediction") is not None:
        lines.append(("★", f"<b>{movie['prediction']:.1f}</b> stars predicted for you", False))
    if not explanations:
        pass
    elif movie.get("moved", 0) > 0 and movie.get("context"):
        lines.append(("↑", f"Up <b>{movie['moved']}</b> place{'s' if movie['moved'] != 1 else ''} for "
                            f"{html.escape(movie['context'])}", True))
    if explanations and movie.get("because"):
        lines.append(("♥", because_text(movie)[0], bool(movie.get("because_when", "")
                                                          and "like now" in movie["because_when"])))
    if explanations and movie.get("context_tags"):
        lines.append(("◷", "Lifted by your <b>" + html.escape(", ".join(movie["context_tags"])) + "</b>", True))
    if explanations and movie.get("mood_match"):
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


def usual_shift(m):
    """How today's day weighting and mood move the movie compared with your usual taste."""
    rank, usual = m["rank"], m["usual_rank"]
    if usual > rank * 1.5 and usual - rank >= 20:
        return "up", f"↑ usually #{usual:,}"
    if rank > usual * 1.5 and rank - usual >= 20:
        return "down", f"↓ usually #{usual:,}"
    return "same", "about as usual"


SHORT_VERDICT = {"Top pick today": "Top pick", "Not your taste today": "Not for today"}  # tiles are narrow


def match_html(m):
    """Compact fit line under a tile: rank and verdict, a thin meter, and the main reason."""
    if m["liked"]:
        return ('<div class="match lv5"><div class="mv"><b>In your likes</b></div>'
                '<div class="meter"><i style="width:100%"></i></div>'
                '<div class="mw">You liked this one, so it shapes every match.</div></div>')
    direction, shift = usual_shift(m)
    if m.get("basis") == "general":
        why = "General appeal: no likes yet"
        if m["mood_match"]:
            why += f" · in your mood: {html.escape(', '.join(m['mood_match']))}"
        elif m["mood_miss"]:
            why += " · outside your mood"
        if not (m["mood_match"] or m["mood_miss"]):  # no mood: today equals usual, so no shift to show
            direction, shift = "same", ""
    elif m["closest"]:
        closest = m["closest"]
        why = f"Closest to <b>{html.escape(display_title(closest['title'])[0])}</b>"
        if closest["shared"]:
            why += f" · {html.escape(', '.join(closest['shared'][:2]))}"
    else:
        why = "Far from all of your likes"
    if m["mood_miss"] and not m["mood_match"] and m.get("basis") != "general":
        why += " · outside your mood"
    return (f'<div class="match lv{m["level"]}" title="Rank among {m["pool"]:,} comparable movies for your '
            f'{html.escape(when_now)} taste"><div class="mv"><span class="mr">#{m["rank"]:,}</span>'
            f'<b>{html.escape(SHORT_VERDICT.get(m["label"], m["label"]))}</b>'
            f'<span class="shift {direction}">{shift}</span></div>'
            f'<div class="meter"><i style="width:{max(m["percent"], 2):.0f}%"></i></div>'
            f'<div class="mw">{why}</div></div>')


def match_lines(m):
    """Tooltip lines for a taste match: verdict, change from usual, what pulls it closer or away."""
    if m["liked"]:
        return [("♥", "<b>In your likes</b>", True)]
    lines = [("◎", f"<b>{html.escape(m['label'])}</b>: #{m['rank']:,} of {m['pool']:,} for your "
                   f"{html.escape(when_now)} taste, ahead of {m['percent']:.0f}%", m["level"] >= 3)]
    direction, _ = usual_shift(m)
    if direction != "same":
        lines.append(("↕", f"Usually #{m['usual_rank']:,}. Today's day weighting and mood move it "
                           f"<b>{'up' if direction == 'up' else 'down'}</b>", direction == "up"))
    if m.get("basis") == "general":
        lines.append(("♡", "No likes yet: ranked by general appeal and your mood. <b>Like a few movies</b> "
                           "to see how close it is to your own taste", False))
    if m["closest"]:
        c = m["closest"]
        shared = f" · shares {html.escape(', '.join(c['shared']))}" if c["shared"] else ""
        lines.append(("♥", f"Pulled closer by <b>{html.escape(label(c['title']))}</b>, liked "
                           f"{html.escape(c['when'])}{shared}", "like now" in c["when"]))
    if m["furthest"]:
        lines.append(("✕", f"Pushed away by <b>{html.escape(label(m['furthest']['title']))}</b>", False))
    if m["total_likes"]:
        lines.append(("☷", f"Shares a genre with <b>{m['sharing_likes']}</b> of your {m['total_likes']} likes", False))
    if m["mood_match"]:
        lines.append(("✓", f"In your mood: <b>{html.escape(', '.join(m['mood_match']))}</b>", True))
    elif m["mood_miss"]:
        lines.append(("✕", f"Outside your mood ({html.escape(', '.join(m['mood_miss']))})", False))
    if m["low_data"]:
        lines.append(("!", "Fewer than 20 ratings, so Discover never shows it", False))
    return lines


def current_taste():
    """(mood, strength key, lean) as last set in Discover or Search."""
    get = lambda key: st.session_state.get(f"kept_{key}", REMEMBERED[key])  # noqa: E731
    return (list(get("mood") or []), MOOD_STRENGTHS.get(get("mood_strength") or "Strong"),
            get("ctx_lean") or "Off")


def with_match(rows, bundle, tables, likes, general=False):
    """Attach today's taste match to movie rows.

    Without likes it is a no-op unless `general` is set, in which case the fit uses general
    appeal and the mood (what Discover shows before you like anything).
    """
    if not rows or (not likes and not general):
        return rows
    mood, strength, lean = current_taste()
    found = taste_match(bundle, tables, ContextRanker(LEAN[lean]), likes, [int(r["movie_id"]) for r in rows],
                        weekday=weekday, part=part, mood=set(mood), mood_strength=strength)
    return [r | {"match": found.get(int(r["movie_id"]))} for r in rows]


def taste_controls(genres, lean_disabled=False):
    """Mood genres, mood strength and day weighting, shared by Discover and Search.

    Returns (mood, strength key, lean, third column) so a page can add its own control.
    """
    for key in ("mood", "mood_strength", "ctx_lean"):
        restore(key)
    mood = st.pills("In the mood for", genres, selection_mode="multi", key="mood",
                    help="Genres you are in the mood for. They are boosted on top of your taste.") or []
    mood_col, lean_col, third = st.columns([1.1, 1.1, 0.8])
    strength = mood_col.segmented_control(
        "Mood strength", list(MOOD_STRENGTHS), key="mood_strength", disabled=not mood,
        help="Nudge: a light boost. Strong: mood genres lead. Only these: nothing else.")
    lean = lean_col.segmented_control(
        f"Lean on {WEEKDAYS[weekday]} likes", list(LEAN), key="ctx_lean", disabled=lean_disabled,
        help=f"How much likes made on a {WEEKDAYS[weekday]} (and in the {DAYPARTS[part]}) count right now. "
             "Your preference: on MovieLens data this signal showed no measurable gain (see Project results).")
    for key in ("mood", "mood_strength", "ctx_lean"):
        keep(key)
    return mood, MOOD_STRENGTHS.get(strength or "Strong"), lean or "Off", third


def poster_html(movie, name, year, genres, rank=None, tip=True, explanations=True):
    """Poster tile: the image (or a flat genre-coloured block with the title), small corner badges,
    and the hover/focus details panel."""
    badge = f'<span class="rank" title="Rank in this list">#{rank}</span>' if rank else ""
    if explanations and movie.get("moved", 0) > 0 and movie.get("context"):
        badge += f'<span class="lift" title="Places gained for {html.escape(movie["context"])}">↑{movie["moved"]}</span>'
    if movie.get("movie_id") is not None and int(movie["movie_id"]) in liked_ids():
        badge += '<span class="liked" title="In your likes">♥</span>'
    details = tooltip_html(movie, name, year, genres, explanations) if tip else ""
    url = poster_url(movie.get("movie_id"))
    if url:
        body = f'<img src="{url}" alt="Poster: {html.escape(name)}" loading="lazy">'
    else:
        color = GENRE_COLORS.get(genres[0] if genres else "", "#2c2c2c")
        body = (f'<div class="flat" style="background:{color}"><b>{html.escape(name)}</b>'
                f'<span>{html.escape(genres[0] if genres else "Movie")}</span></div>')
    return f'<div class="poster" tabindex="0">{body}{badge}{details}</div>'


def meta_html(movie, year, genres):
    star = f'<span class="star">★ {movie["prediction"]:.1f}</span>' if movie.get("prediction") is not None else ""
    bits = ([year] if year else []) + genres[:3]
    return f'<div class="meta">{star}{html.escape(" · ".join(bits))}</div>'


def reason_html(movie):
    """One short line under a tile; the full reason is in the hover details."""
    if movie.get("because"):
        _, plain = because_text(movie)
        text = f"Because you liked <b>{html.escape(display_title(movie['because'])[0])}</b>"
    elif movie.get("mood_match"):
        plain = f"Matches your mood: {', '.join(movie['mood_match'])}"
        text = html.escape(plain)
    elif movie.get("ratings") and movie.get("prediction") is None:
        plain = f"{movie['ratings']:,} MovieLens ratings"
        text = html.escape(plain)
    else:
        plain = (movie.get("reason") or "Suggested from your ratings").rstrip(".")
        text = html.escape(plain)
    return f'<div class="reason" title="{html.escape(plain)}">{text}</div>'


def card_html(movie, explanations, rank=None):
    name, year = display_title(movie["title"])
    genres = [g for g in str(movie["genres"]).split("|") if g]
    parts = [poster_html(movie, name, year, genres, rank, explanations=explanations or bool(movie.get("match"))),
             f'<div class="movie-title" title="{html.escape(name)}">{html.escape(name)}</div>',
             meta_html(movie, year, genres)]
    if movie.get("match"):
        parts.append(match_html(movie["match"]))
    elif explanations:
        parts.append(reason_html(movie))
    else:  # "Show why" off: a neutral fact, so every tile keeps the same height
        fact = f"{movie['ratings']:,} MovieLens ratings" if movie.get("ratings") else ""
        parts.append(f'<div class="reason">{fact}</div>')
    return "".join(parts)


def card_buttons(movie, prefix, key, big=False):
    """Like and Save. A like is stamped with the current weekday and time of day."""
    saved = key in st.session_state.watchlist
    movie_id = movie.get("movie_id")
    liked = movie_id is not None and int(movie_id) in liked_ids()
    like_col, save_col = st.columns(2, gap="small")
    if movie_id is not None and like_col.button(
            ("♥ Liked" if liked else "♡ Like"), key=f"{prefix}-like-{'on' if liked else 'off'}-{key}",
            width="stretch", help="Unlike" if liked else
            f"Like it now. CineCue remembers it was a {WEEKDAYS[weekday]} {DAYPARTS[part]}."):
        toggle_like(int(movie_id), movie["title"])
        st.rerun()
    if save_col.button(("✓ Saved" if saved else ("＋ My list" if big else "＋ Save")),
                       key=f"{prefix}-btn-{'on' if saved else 'off'}-{key}", width="stretch",
                       help="Remove from your watchlist" if saved else "Save to your watchlist"):
        set_saved(movie, not saved)
        st.rerun()


PER_ROW = 6


def cards(rows, prefix, explanations=True, ranked=False, first_rank=1,
          empty="No movies here yet. Try another filter or save a movie from Discover."):
    """Poster grid, six to a row; tiles wrap rather than shrink in narrow windows."""
    if not rows:
        st.info(empty)
        return
    keys = [str(movie.get("movie_id", movie["title"])) for movie in rows]
    # New list -> new phase -> the tiles animate in again. Liking or saving keeps the list, so no replay.
    card_phase = phase(f"cards-{prefix}", tuple(keys))
    for start in range(0, len(rows), PER_ROW):
        row = min(start // PER_ROW, 4)  # the CSS stagger stops growing after the fifth row
        chunk = rows[start:start + PER_ROW]
        for offset, (column, movie) in enumerate(zip(st.columns(len(chunk), gap="small"), chunk)):
            key = keys[start + offset]
            with column, st.container(key=f"cuecard-{card_phase}-{prefix}-row{row}-{key}"):
                rank = first_rank + start + offset if ranked else None
                st.markdown(card_html(movie, explanations, rank), unsafe_allow_html=True)
                card_buttons(movie, prefix, key)


def billboard(movie, heading, explanations=True):
    """The top pick, large: poster, title, why it was picked, and Like / My list."""
    name, year = display_title(movie["title"])
    genres = [g for g in str(movie["genres"]).split("|") if g]
    key = str(movie.get("movie_id", movie["title"]))
    with st.container(key="billboard"):
        art, info = st.columns([1, 3], gap="large", vertical_alignment="center")
        art.markdown(poster_html(movie, name, year, genres, tip=False), unsafe_allow_html=True)
        with info:
            meta = []
            if movie.get("prediction") is not None:
                meta.append(f'<span class="good">★ {movie["prediction"]:.1f} predicted</span>')
            meta.append(html.escape(" · ".join(([year] if year else []) + genres[:4])))
            why = (because_text(movie)[0] + "." if movie.get("because") else
                   html.escape(movie.get("reason") or "Your strongest pick right now."))
            if not explanations:
                why = f"{movie['ratings']:,} MovieLens ratings." if movie.get("ratings") else ""
            tags = []
            if not explanations:
                pass
            elif movie.get("moved", 0) > 0 and movie.get("context"):
                tags.append(f"Up {movie['moved']} places for {movie['context']}")
            if movie.get("mood_match"):
                tags.append("In your mood: " + ", ".join(movie["mood_match"]))
            if explanations:
                tags += movie.get("context_tags") or []
            bb_phase = phase("billboard", movie.get("movie_id"))
            st.markdown(f'<div class="bb ph-{bb_phase}"><div class="bb-label">{html.escape(heading)}</div><div class="bb-title">{html.escape(name)}</div>'
                        f'<div class="bb-meta">{"".join(meta)}</div><div class="bb-why">{why}</div>'
                        + (('<div class="bb-tags">' + "".join(f"<span>{html.escape(t)}</span>" for t in tags) + "</div>")
                           if tags else "") + "</div>", unsafe_allow_html=True)
            buttons, _ = st.columns([1, 1.4])
            with buttons:
                card_buttons(movie, "billboard", key, big=True)


# ---------- Data and models ----------
required = ["train.csv", "validation.csv", "movies_clean.csv", "movie_features.csv",
            "rating_summary.json", "ranking_summary.json"]
processed = next((p for p in [ROOT / "results", ROOT / "data/processed"]
                  if all(table_path(p, n).is_file() for n in required)), None)
large_artifact = ROOT / "artifacts/recommender-32m.pkl"  # built by scripts/build_app_catalog.py
artifact = ROOT / "artifacts/recommender.pkl"


@st.cache_resource(show_spinner="Downloading the CineCue models. This happens once and takes a minute or two.")
def shared_models(repo):
    """Fetch missing models from the team's private GitHub repository (see cinecue.hub)."""
    try:
        return [path.name for path in download_models(ROOT / "artifacts", repo)], None
    except Exception as error:  # no login, no access, or offline: fall back to local models
        return [], str(error)


if model_repo() and not large_artifact.is_file():
    _, download_error = shared_models(model_repo())
    if download_error:
        st.sidebar.warning(f"Could not download the shared models ({download_error.splitlines()[0][:160]}). "
                           f"Run `gh auth login` or set GITHUB_TOKEN with read access to {model_repo()}.")
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
        user = current_user()
        st.caption(f"Saved to your account ({user.email})." if user else
                   "Saved for this visit only. Log in from the sidebar to keep it." if ACCOUNTS else
                   "Saved for this browser session only.")
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
        likes = [like for like in st.session_state.likes if like["movie_id"] in titles]
        tables, _ = context_model(signature, catalog)
        cards(with_match(filtered[(number - 1) * 12:number * 12], bundle, tables, likes), "catalog", False,
              empty="No movies match. Try a shorter search or fewer genres.")

    elif page == "Search":
        tables, _ = context_model(signature, catalog)
        likes = [like for like in st.session_state.likes if like["movie_id"] in titles]
        hero("Search", "Does it fit today?",
             f"Search any movie and see how close it is to your taste this {when_now}: "
             "your likes, the day, and your mood.")
        restore("search_query")
        with st.container(key="hero_search"):
            query = st.text_input("Search any movie", key="search_query", icon=":material/search:",
                                  label_visibility="collapsed",
                                  placeholder="Title, words or year, e.g. heat 1995")
            st.markdown('<div class="hero-search-hint">Every word must appear in the title, in any order. '
                        'Press Enter to search.</div>', unsafe_allow_html=True)
        keep("search_query")
        mood, strength, lean = current_taste()
        on_day = sum(like["weekday"] == weekday for like in likes)
        chips = [f"◷ <b>{when_now}</b>",
                 f"♥ <b>{len(likes)}</b> like{'s' if len(likes) != 1 else ''}"
                 + (f" · {on_day} from {WEEKDAYS[weekday]}s" if on_day else ""),
                 "◎ Mood: <b>" + (html.escape(", ".join(mood)) if mood else "any") + "</b>"
                 + (f" ({next(k for k, v in MOOD_STRENGTHS.items() if v == strength).lower()})" if mood else ""),
                 f"⚖ Day weighting: <b>{lean.lower()}</b>"]
        st.markdown(f'<div class="taste-strip ph-{page_phase}"><span class="ts-label">Today\'s taste</span>'
                    + "".join(f'<span class="stat">{c}</span>' for c in chips) + "</div>", unsafe_allow_html=True)
        with st.expander("Adjust mood and day weighting", icon=":material/tune:"):
            _, _, _, order_col = taste_controls(genres)
            restore("search_order")
            order_col.segmented_control("Order results by", ["Best title match", "Best fit today"],
                                        key="search_order")
            keep("search_order")
        if not likes:
            st.info("You have no likes yet, so each result's fit uses general appeal and your mood. Like a few "
                    "movies (♡ on any card, or *Movies you love* in Discover) to see how close each one is to "
                    "*your* taste today.", icon=":material/favorite:")
        if not query.strip():
            st.markdown('<div class="legend">' + "".join(
                f'<span class="match-key lv{level}"><i></i>{text}</span>' for level, text in
                [(5, "Top pick today · top 12"), (4, "Excellent · top 100"), (3, "Strong · top 500"),
                 (2, "Good · top 1,500"), (1, "Mixed · top 4,000"), (0, "Not your taste today")]) + "</div>",
                unsafe_allow_html=True)
            st.caption("Ranks compare a movie with every movie that has 20+ ratings, scored the way Discover "
                       "scores them right now. Hover a poster to see what pulls it closer or pushes it away.")
        else:
            by_id = catalog.assign(ratings=counts).rename_axis("movie_id").reset_index()[
                ["movie_id", "title", "genres", "ratings"]].set_index("movie_id", drop=False)
            ranked, fuzzy = title_search(signature, catalog)(query, dict(zip(by_id.index, by_id.ratings)))
            hits = [by_id.loc[m].to_dict() for m in ranked[:24]]
            rows = with_match(hits, bundle, tables, likes, general=True)
            by_fit = st.session_state.get("search_order") == "Best fit today"
            if by_fit:
                rows = sorted(rows, key=lambda r: (r["match"] is None, r["match"]["rank"] if r["match"] else 0))
            rows = rows[:12]
            note = "closest spellings" if fuzzy else f"{len(ranked):,} title{'s' if len(ranked) != 1 else ''} match"
            section(f"Results for “{query.strip()}”",
                    note + (f" · showing {len(rows)}" if len(ranked) > len(rows) else "")
                    + (" · best fit today first" if by_fit else ""))
            cards(rows, "search", False, empty="No titles match. Try fewer or shorter words.")

    else:
        hero("Discover", "Find your next movie.",
             "Pick a few movies you love, or start from a MovieLens profile. Every pick comes with the reason it was chosen.")

        def open_search():
            """Enter in the Discover search bar opens the Search page with that query."""
            query = st.session_state.get("discover_search", "").strip()
            if query:
                st.session_state.search_query = st.session_state.kept_search_query = query
                st.session_state.page = "Search"
                st.session_state.discover_search = ""

        with st.container(key="hero_search"):
            st.text_input("Search any movie", key="discover_search", on_change=open_search,
                          icon=":material/search:", label_visibility="collapsed",
                          placeholder=f"Search any movie and see how it fits your {when_now} taste…")
            st.markdown('<div class="hero-search-hint">Press Enter to search. Each result shows its rank for your '
                        'taste today, from <b>Top pick</b> to <b>Not your taste</b>.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stats"><b>{len(catalog):,}</b> movies from {int(years.min())} to {int(years.max())} · '
                    f'learned from <b>{len(history):,}</b> ratings by <b>{history.user_id.nunique():,}</b> MovieLens '
                    f'members</div>', unsafe_allow_html=True)
        tables, context_summary = context_model(signature, catalog)
        likes = [like for like in st.session_state.likes if like["movie_id"] in titles]
        when = f"{WEEKDAYS[weekday]} {DAYPARTS[part]}"
        mood_now = current_taste()[0]
        summary_bits = [f"{len(likes)} like{'s' if len(likes) != 1 else ''}", when]
        if mood_now:
            summary_bits.append("mood: " + ", ".join(mood_now))
        with st.expander("Your taste and tonight's mood · " + " · ".join(summary_bits), icon=":material/tune:",
                         expanded=not likes):
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
            mood, strength_key, lean, count_col = taste_controls(genres, lean_disabled=user is not None)
            count = count_col.segmented_control("Picks", [6, 12, 18], default=12, key="pick_count") or 12
            left, right = st.columns(2)
            discover = left.toggle("Discover less popular films", disabled=user is None and not likes)
            explanations = right.toggle("Show why", value=True)

        ranker = ContextRanker(LEAN[lean] if user is None else LEAN["Off"])
        personal = user is not None or bool(likes)
        # One extra pick when personal: the first is featured above the grid.
        rows = recommend_in_context(bundle, tables, ranker, likes=None if user is not None else likes,
                                    user_id=user, k=count + personal, discover=discover, weekday=weekday,
                                    part=part, mood=set(mood or []), mood_strength=strength_key)
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
        mood_note = f" · mood: {', '.join(mood)}" if mood else ""
        if personal and rows:
            billboard(rows[0], f"Top pick for your {when}", explanations)
            basis = f"MovieLens user {user}" if user is not None else f"{len(likes)} like{'s' if len(likes) != 1 else ''}"
            section("More picks for you", f"Based on {basis}{mood_note}" + (" · discovery on" if discover else ""))
            cards(rows[1:], "recommendation", explanations, ranked=True, first_rank=2)
        else:
            section("Popular on CineCue", "Like a movie to personalise" + mood_note)
            cards(rows, "recommendation", explanations, ranked=True,
                  empty="No movies fit this mood. Try Nudge or Strong instead of Only these.")

st.markdown('<div class="footer">CineCue · Movie discovery, with a reason · Data: MovieLens 1M, GroupLens Research '
            '(Harper &amp; Konstan, 2015) · Posters: Wikipedia</div>', unsafe_allow_html=True)
