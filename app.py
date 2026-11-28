"""CineCue website. Run with: python -m streamlit run app.py"""
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
from cinecue.data import table_path  # noqa: E402
from cinecue.demo import fit_demo, fit_tables, recommend  # noqa: E402
from cinecue.first_run import official_tables  # noqa: E402

st.set_page_config(page_title="CineCue | Movie discovery", page_icon="🎬", layout="wide")

# ---------- Sidebar ----------
st.session_state.setdefault("watchlist", {})
with st.sidebar:
    st.markdown('<div class="brand">Cine<span>Cue.</span><small>Movie picks, with a reason</small></div>',
                unsafe_allow_html=True)
    page = st.radio("Explore", ["Discover", "Movie library", "Watchlist", "Project results"],
                    label_visibility="collapsed", key="page")
    st.divider()
    dark_mode = st.toggle("Dark mode", key="dark_mode")
    saved_count = len(st.session_state.watchlist)
    st.caption(f"Watchlist: {saved_count} movie{'s' if saved_count != 1 else ''}")

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
[data-testid="stVerticalBlockBorderWrapper"]>div {background:var(--cue-surface);border-radius:14px}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--cue-border)!important;border-radius:14px}
.stButton button,.stDownloadButton button,.stLinkButton a {border-radius:8px;background:var(--cue-surface);
  color:var(--cue-text);border-color:var(--cue-border);font-weight:500}
.stButton button:hover,.stDownloadButton button:hover,.stLinkButton a:hover {border-color:var(--cue-accent);color:var(--cue-accent)}
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
.brand {font-size:1.65rem;font-weight:700;letter-spacing:-.05em;padding:.2rem 0 1.2rem;color:var(--cue-text)}
.brand span {color:var(--cue-accent)}
.brand small {display:block;font-size:.78rem;font-weight:500;letter-spacing:0;color:var(--cue-muted);margin-top:.15rem}
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

.poster {height:118px;border-radius:10px;position:relative;overflow:hidden;display:flex;align-items:flex-end;
  padding:.7rem .8rem;color:#fff}
.poster .mark {position:absolute;right:.55rem;top:.1rem;font-size:3.6rem;font-weight:700;opacity:.18;letter-spacing:-.05em}
.poster .genre {font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.92}
.poster .year {position:absolute;left:.7rem;top:.65rem;background:rgba(0,0,0,.32);border-radius:6px;padding:.1rem .45rem;
  font-size:.75rem;font-weight:600}
.movie-title {font-size:1.04rem;font-weight:600;line-height:1.3;margin:.75rem 0 .35rem;min-height:2.6em;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.chips {display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.55rem}
.chip {font-size:.72rem;border:1px solid var(--cue-border);border-radius:999px;padding:.08rem .5rem;color:var(--cue-muted)}
.score {display:flex;align-items:center;gap:.55rem;font-size:.85rem;color:var(--cue-muted);margin-bottom:.55rem}
.score b {color:var(--cue-text)}
.meter {flex:1;height:6px;border-radius:99px;background:var(--cue-bar);overflow:hidden}
.meter i {display:block;height:100%;background:var(--cue-accent);border-radius:99px}
.reason {background:var(--cue-accent-soft);border-radius:9px;padding:.55rem .65rem;font-size:.84rem;line-height:1.45;
  margin-bottom:.6rem;min-height:3.4em}
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
@media(max-width:640px){.hero h1{font-size:1.8rem}.kpi .value{font-size:1.6rem}}
</style>""", unsafe_allow_html=True)

# Poster colours per primary genre; all dark enough for white text.
GENRE_COLORS = {
    "Action": "#9a3c2a", "Adventure": "#8a5a17", "Animation": "#22735d", "Children's": "#2f6c94",
    "Comedy": "#8f6a12", "Crime": "#3d3f66", "Documentary": "#4c5f32", "Drama": "#5d3b78",
    "Fantasy": "#3e4f86", "Film-Noir": "#2f3438", "Horror": "#6b1d2b", "Musical": "#82386a",
    "Mystery": "#304a63", "Romance": "#963d5b", "Sci-Fi": "#17616f", "Thriller": "#55392a",
    "War": "#4f5034", "Western": "#80491f",
}
ARTICLE = re.compile(r"^(?P<name>.*), (?P<article>The|A|An|Les|La|Le|Il|L'|Das|Die|El)(?P<rest> \(.*\))?$")


def display_title(title):
    """'Matrix, The (1999)' -> ('The Matrix', '1999'): readable names for MovieLens titles."""
    match = re.match(r"^(?P<base>.*?)\s*\((?P<year>\d{4})\)\s*$", str(title))
    base, year = (match["base"], match["year"]) if match else (str(title), "")
    moved = ARTICLE.match(base)
    if moved:
        base = f"{moved['article']} {moved['name']}{moved['rest'] or ''}"
    return base, year


def label(title):
    name, year = display_title(title)
    return f"{name} ({year})" if year else name


def hero(eyebrow, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title, note=""):
    st.markdown(f'<div class="section"><h3>{html.escape(title)}</h3><span>{html.escape(note)}</span></div>',
                unsafe_allow_html=True)


def card_html(movie, explanations):
    name, year = display_title(movie["title"])
    genres = [g for g in str(movie["genres"]).split("|") if g]
    color = GENRE_COLORS.get(genres[0] if genres else "", "#34444f")
    second = GENRE_COLORS.get(genres[1], "#0b1114") if len(genres) > 1 else "#0b1114"
    initials = "".join(word[0] for word in re.findall(r"[A-Za-z0-9]+", name)[:2]).upper() or "C"
    parts = [
        f'<div class="poster" aria-hidden="true" style="background:linear-gradient(140deg,{color} 10%,'
        f'{second} 115%)">'
        + (f'<span class="year">{year}</span>' if year else "")
        + f'<span class="mark">{html.escape(initials)}</span>'
        f'<span class="genre">{html.escape(genres[0] if genres else "Movie")}</span></div>',
        f'<div class="movie-title" title="{html.escape(name)}">{html.escape(name)}</div>',
        '<div class="chips">' + "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genres[:4]) + "</div>",
    ]
    if movie.get("prediction") is not None:
        width = max(0.0, min(1.0, (movie["prediction"] - 1) / 4)) * 100
        parts.append(f'<div class="score"><span>Predicted <b>{movie["prediction"]:.1f}★</b></span>'
                     f'<span class="meter"><i style="width:{width:.0f}%"></i></span></div>')
    elif movie.get("ratings"):
        parts.append(f'<div class="score"><span><b>{movie["ratings"]:,}</b> ratings</span></div>')
    if explanations:
        if movie.get("because"):
            shared = movie.get("shared") or []
            text = f"Because you liked <b>{html.escape(label(movie['because']))}</b>"
            if shared:
                text += f" · shares {html.escape(', '.join(shared))}"
            parts.append(f'<div class="reason"><span class="label">Why this pick</span>{text}.</div>')
        elif movie.get("reason"):
            parts.append(f'<div class="reason"><span class="label">Why this pick</span>'
                         f'{html.escape(movie["reason"])}</div>')
    return "".join(parts)


def cards(rows, prefix, explanations=True, empty="No movies here yet. Try another filter or save a movie from Discover."):
    if not rows:
        st.info(empty)
        return
    for start in range(0, len(rows), 3):
        for column, movie in zip(st.columns(3, gap="medium"), rows[start:start + 3]):
            with column, st.container(border=True):
                st.markdown(card_html(movie, explanations), unsafe_allow_html=True)
                key = str(movie.get("movie_id", movie["title"]))
                saved = key in st.session_state.watchlist
                if st.button("✓ Saved — remove" if saved else "＋ Save to watchlist",
                             key=f"{prefix}-{key}", width="stretch"):
                    if saved:
                        del st.session_state.watchlist[key]
                    else:
                        st.session_state.watchlist[key] = movie
                    st.rerun()


# ---------- Data and models ----------
required = ["train.csv", "validation.csv", "movies_clean.csv", "movie_features.csv",
            "rating_summary.json", "ranking_summary.json"]
processed = next((p for p in [ROOT / "results", ROOT / "data/processed"]
                  if all(table_path(p, n).is_file() for n in required)), None)
artifact = ROOT / "artifacts/recommender.pkl"


@st.cache_resource(show_spinner="Preparing your library. The first launch can take a few minutes.")
def models(signature):
    if artifact.is_file():
        return load_bundle(artifact)
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
            column.markdown(f'<div class="kpi"><div class="name">{name}</div><div class="value">{value}</div>'
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
            f'<tr class="{"final" if name == "CineCue blend" else ""}"><th scope="row">{html.escape(name)}</th>'
            + "".join(f"<td>{formats[c].format(row[c])}</td>" for c in shown.columns) + "</tr>"
            for name, row in shown.iterrows())
        st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Model</th>{header}</tr></thead>'
                    f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)
        ndcg = shown["NDCG@10"].sort_values(ascending=False)
        st.markdown('<div class="bars" role="img" aria-label="NDCG@10 by model">' + "".join(
            f'<div class="bar-row"><span class="name{" best" if i == 0 else ""}">{html.escape(name)}</span>'
            f'<span class="bar-track"><i style="width:{value / ndcg.max() * 100:.1f}%"></i></span>'
            f'<span class="num">{value:.3f}</span></div>' for i, (name, value) in enumerate(ndcg.items())) + "</div>",
            unsafe_allow_html=True)
        st.caption("A hit is a movie the user rated 4 or 5 stars later. Variants with recency weighting were "
                   "evaluated in notebook 6 and are not part of CineCue.")
    else:
        st.info("Evaluation summaries appear here when notebook results are available.")
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
    if artifact.is_file():
        signature = ("saved-model", artifact.stat().st_mtime_ns)
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
    history, catalog, counts = bundle[0], bundle[1], bundle[4]
    titles = catalog.title.to_dict()
    genres = sorted({g for value in catalog.genres for g in str(value).split("|")})

    if page == "Movie library":
        hero("Movie library", "Find an old favorite.",
             f"Browse {len(catalog):,} movies. Search a title, filter by genre, and save anything for later.")
        left, right = st.columns([2, 1])
        query = left.text_input("Search movies", placeholder="Title or year, e.g. matrix or 1999")
        selected = right.multiselect("Genres", genres)
        records = catalog.assign(ratings=counts).rename_axis("movie_id").reset_index()[["movie_id", "title", "genres", "ratings"]]
        needle = query.strip().casefold()
        filtered = [r for r in records.to_dict("records")
                    if (needle in r["title"].casefold() or needle in label(r["title"]).casefold())
                    and (not selected or set(selected).intersection(str(r["genres"]).split("|")))]
        filtered.sort(key=lambda r: -r["ratings"])
        pages = max(1, (len(filtered) + 11) // 12)
        info, pager = st.columns([3, 1])
        number = pager.number_input("Page", min_value=1, max_value=pages, value=1)
        info.caption(f"{len(filtered):,} movies · most rated first · page {number} of {pages}")
        cards(filtered[(number - 1) * 12:number * 12], "catalog", False,
              empty="No movies match. Try a shorter search or fewer genres.")

    else:
        hero("Discover", "Find your next movie.",
             "Pick a few movies you love, or start from a MovieLens profile. Every pick comes with the reason it was chosen.")
        st.markdown(f'<div class="stats"><span class="stat"><b>{len(catalog):,}</b> movies</span>'
                    f'<span class="stat"><b>{history.user_id.nunique():,}</b> MovieLens profiles</span>'
                    f'<span class="stat"><b>{len(history):,}</b> ratings learned from</span></div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            mode = st.radio("Start from", ["My favorites", "MovieLens profile"], horizontal=True)
            liked, user = None, None
            if mode == "My favorites":
                liked = st.multiselect("Movies you love", sorted(titles, key=lambda m: label(titles[m])),
                                       format_func=lambda m: label(titles[m]), key="favorites",
                                       placeholder="Start typing a title, e.g. The Matrix")
            else:
                user = st.selectbox("MovieLens user", sorted(history.user_id.unique().tolist()))
            left, middle, right = st.columns(3)
            count = left.selectbox("Number of picks", [6, 12, 18], index=1)
            discover = middle.toggle("Discover less popular films", disabled=user is None and not liked)
            explanations = right.toggle("Show why", value=True)
        rows = recommend(bundle, user, count, discover, liked_movies=liked)
        personal = user is not None or bool(liked)
        if personal:
            basis = f"MovieLens user {user}" if user is not None else \
                f"{len(liked)} favorite{'s' if len(liked) != 1 else ''}"
            section("Picked for you", f"Based on {basis}" + (" · discovery on" if discover else ""))
        else:
            section("Popular in the community", "Add a favorite to personalise")
        cards(rows, "recommendation", explanations)

st.markdown('<div class="footer">CineCue · Movie discovery, with a reason · Data: MovieLens 1M, GroupLens Research '
            '(Harper &amp; Konstan, 2015)</div>', unsafe_allow_html=True)
