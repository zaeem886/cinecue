"""Run with python -m streamlit run app.py."""
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from cinecue.demo import fit_demo, fit_tables, recommend
from cinecue.first_run import official_tables
from cinecue.artifacts import load_bundle, save_bundle
from cinecue.data import table_path

st.set_page_config(page_title="CineCue | Movie discovery", page_icon="C", layout="wide")
dark_mode = st.sidebar.toggle("Dark mode", key="dark_mode")
palette = (
    ("#11181d", "#19232b", "#23323c", "#e5edf2", "#a6b7c3", "#34444f", "#60cfc1")
    if dark_mode else
    ("#f5f7f8", "#ffffff", "#edf2f3", "#17292e", "#52616b", "#dce4e7", "#007e77")
)
background, surface, control, text, muted, border, accent = palette
st.markdown(
    f"""<style>:root {{
    --cue-bg:{background}; --cue-surface:{surface}; --cue-control:{control};
    --cue-text:{text}; --cue-muted:{muted}; --cue-border:{border}; --cue-accent:{accent};
    color-scheme:{"dark" if dark_mode else "light"};
    }}</style>""",
    unsafe_allow_html=True,
)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp {background:var(--cue-bg);color:var(--cue-text)}
html,body,[data-testid="stApp"] {font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"] {background:var(--cue-surface);border-right:1px solid var(--cue-border)}
.block-container {max-width:1200px;padding-top:2rem;padding-bottom:3rem}
h1 {font-family:'DM Sans',sans-serif!important;font-weight:700!important;font-size:2.25rem!important;letter-spacing:-.04em}
h3 {font-size:1.2rem!important;line-height:1.4!important}
.brand {font-size:1.7rem;font-weight:700;letter-spacing:-.06em;padding-bottom:1.5rem}
.brand span {color:var(--cue-accent)}
[data-testid="stVerticalBlockBorderWrapper"]>div {background:var(--cue-surface);border-radius:12px}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"][style*="border"] {border-color:var(--cue-border)!important}
[data-testid="stHeader"] {background:var(--cue-bg)}
[data-testid="stSidebar"] {min-width:240px;max-width:270px}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.5rem}
[data-testid="stWidgetLabel"] p {font-weight:500}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:var(--cue-muted)!important}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp summary,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:var(--cue-text)}
.stButton button,.stDownloadButton button,.stLinkButton a {
border-radius:7px;background:var(--cue-surface);color:var(--cue-text);border-color:var(--cue-border)}
.stButton button:hover {border-color:var(--cue-accent);color:var(--cue-accent)}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],
.stApp input,[data-baseweb="popover"],[role="listbox"],[role="option"] {
background:var(--cue-control)!important;color:var(--cue-text)!important}
[data-baseweb="select"] span,[data-baseweb="select"] input {color:var(--cue-text)!important}
[data-baseweb="select"] div {background-color:var(--cue-control)!important;border-color:var(--cue-border)!important}
[data-baseweb="select"] svg,[data-baseweb="input"] svg {fill:var(--cue-text)}
[data-testid="stHeader"] button {color:var(--cue-text)}
[data-baseweb="popover"] *,[role="option"] * {color:var(--cue-text)}
.stApp input::placeholder {color:var(--cue-muted)}
[data-baseweb="tag"] {background:var(--cue-border);color:var(--cue-text)}
[data-testid="stExpander"] details {background:var(--cue-surface);border-color:var(--cue-border)}
[data-testid="stAlert"] {background:var(--cue-control);color:var(--cue-text)}
button[kind="primary"] {background:var(--cue-accent);border-color:var(--cue-accent)}
.footer {border-top:1px solid var(--cue-border);margin-top:3rem;padding-top:1rem;color:var(--cue-muted);font-size:.85rem}
@media(max-width:640px){h1{font-size:1.8rem!important}}
</style>""", unsafe_allow_html=True)
st.session_state.setdefault("watchlist", {})
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

with st.sidebar:
    st.markdown('<div class="brand">Cine<span>Cue.</span></div>', unsafe_allow_html=True)
    page = st.radio("Explore", ["Discover", "Movie library", "Watchlist", "Project results"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("A movie night starts with a good pick.")

def cards(rows, prefix, explanations=True):
    if not rows:
        st.info("No movies here yet. Try another filter or save a movie from Discover.")
    for start in range(0, len(rows), 3):
        for column, movie in zip(st.columns(3), rows[start:start + 3]):
            with column, st.container(border=True):
                st.subheader(movie["title"])
                st.caption(str(movie["genres"]).replace("|", " · "))
                if movie.get("prediction") is not None:
                    st.write(f'Predicted rating **{movie["prediction"]:.1f} / 5**')
                if explanations and movie.get("reason"):
                    with st.expander("Why this movie"):
                        st.write(movie["reason"])
                saved = movie["title"] in st.session_state.watchlist
                if st.button("Remove from watchlist" if saved else "Save to watchlist",
                             key=f'{prefix}-{movie["title"]}', use_container_width=True):
                    if saved:
                        del st.session_state.watchlist[movie["title"]]
                    else:
                        st.session_state.watchlist[movie["title"]] = movie
                    st.rerun()

if page == "Watchlist":
    st.title("Your next movie nights.")
    st.write("Keep your picks together and take the list with you.")
    saved = list(st.session_state.watchlist.values())
    st.caption("Saved for this browser session. Download your list to keep it.")
    if saved:
        st.download_button("Download watchlist", pd.DataFrame(saved).to_csv(index=False),
                           "cinecue-watchlist.csv", "text/csv")
    cards(saved, "saved")
elif page == "Project results":
    st.title("Behind the recommendations.")
    st.write("A data science project built on MovieLens 1M.")
    st.subheader("How your list is made")
    st.write("EASE learns which movies appear together in rating histories. Matrix factorization "
             "learns patterns in star ratings. CineCue combines their scores and excludes movies "
             "already in the selected profile. Favorite selections count as five-star ratings.")
    summary = next((p / "ranking_summary.json" for p in [ROOT / "results", ROOT / "data/processed"]
                    if (p / "ranking_summary.json").is_file()), None)
    if summary:
        table = pd.DataFrame(json.loads(summary.read_text())["test_known_users"]).T
        st.subheader("Measured ranking performance")
        st.caption("Test users known at the cutoff. Relevance means a later rating of four or five stars.")
        st.dataframe(table[[c for c in ["precision", "recall", "ndcg", "coverage", "long_tail_share"]
                            if c in table]], use_container_width=True)
        st.bar_chart(table["ndcg"], color="#007e77")
    else:
        st.info("Evaluation summaries appear here when notebook results are available.")
    st.subheader("Understanding the experiment")
    st.write("Discovery applies a popularity penalty to the candidate list. Explanations identify "
             "a positive contribution from your history when available. UTC and recency adjustments "
             "were evaluated in the notebooks and excluded from the final ranker. Timestamps record "
             "rating activity and cannot establish mood or viewing time.")
    st.link_button("View source and notebooks", "https://github.com/zaeem886/cinecue")
else:
    if artifact.is_file():
        signature = ("saved-model", artifact.stat().st_mtime_ns)
    else:
        signature = (str(processed), tuple(table_path(processed, n).stat().st_mtime_ns for n in required)) if processed else "ml-1m"
    try:
        bundle = models(signature)
    except Exception as error:
        st.error(f"The movie library could not load. {error}")
        if st.button("Try again"):
            st.rerun()
        st.stop()
    catalog = bundle[1]
    titles = catalog.title.to_dict()
    genres = sorted({g for value in catalog.genres for g in str(value).split("|")})
    if page == "Movie library":
        st.title("Find an old favorite.")
        st.write("Browse the catalog, search a title, or save something for later.")
        left, right = st.columns([2, 1])
        query = left.text_input("Search movies", placeholder="Search by title or year")
        selected = right.multiselect("Genres", genres)
        records = catalog[["title", "genres"]].to_dict("records")
        filtered = [r for r in records if query.strip().casefold() in r["title"].casefold()
                    and (not selected or set(selected).intersection(str(r["genres"]).split("|")))]
        st.caption(f"{len(filtered):,} movies")
        pages = max(1, (len(filtered) + 11) // 12)
        number = st.number_input("Page", min_value=1, max_value=pages, value=1)
        cards(filtered[(number - 1) * 12:number * 12], "catalog", False)
    else:
        st.title("Find your next movie")
        st.write("Choose a few favorites for recommendations based on your taste.")
        with st.container(border=True):
            mode = st.radio("Your starting point", ["My favorites", "MovieLens profile"], horizontal=True)
            liked, user = None, None
            if mode == "My favorites":
                liked = st.multiselect("Choose movies you love", sorted(titles, key=titles.get),
                                       format_func=titles.get, key="favorites")
            else:
                user = st.selectbox("MovieLens user", sorted(bundle[0].user_id.unique().tolist()))
            left, middle, right = st.columns(3)
            count = left.selectbox("Suggestions", [6, 12, 18], index=1)
            discover = middle.checkbox("Discover less popular films", disabled=user is None and not liked)
            explanations = right.checkbox("Show explanations", value=True)
        rows = recommend(bundle, user, count, discover, liked_movies=liked)
        st.subheader("Picked for you" if user is not None or liked else "Popular in the community")
        st.caption("Save a pick to your watchlist. Open its explanation to see a contributor to the score.")
        cards(rows, "recommendation", explanations)
st.markdown('<div class="footer">CineCue · Movie discovery, with a reason. &nbsp; MovieLens 1M / GroupLens</div>',
            unsafe_allow_html=True)
