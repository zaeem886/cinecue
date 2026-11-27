"""Run from the project root: python -m streamlit run app.py."""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from cinecue.demo import fit_demo, recommend

st.set_page_config(page_title='CineCue · Movie discovery', page_icon='🎬', layout='wide')
st.title('🎬 CineCue')
st.write('Pick movies you love to find something new, or explore an existing MovieLens profile.')

required = ['train.csv', 'validation.csv', 'movies_clean.csv', 'movie_features.csv',
            'rating_summary.json', 'ranking_summary.json']
locations = [ROOT / 'results', ROOT / 'data' / 'processed']
processed = next((p for p in locations if all((p / f).is_file() for f in required)), locations[0])
missing = [name for name in required if not (processed / name).is_file()]
if missing:
    st.info('Place the processed notebook outputs together in results/ or data/processed/.')
    st.write('Missing files: ' + ', '.join(missing))
    st.code('python scripts/download_data.py\n# Run notebooks 01 through 06 in order. See README.md.')
    st.stop()


@st.cache_resource(show_spinner='Fitting the selected models. This can take several minutes…')
def models(signature):
    return fit_demo(processed)


signature = (str(processed), tuple((name, (processed / name).stat().st_mtime_ns) for name in required))
try:
    bundle = models(signature)
except Exception as error:
    st.error(f'Could not prepare recommendations: {error}')
    st.stop()

with st.sidebar:
    st.header('Your movie list')
    mode = st.radio('Recommend for', ['My taste', 'MovieLens user'])
    user = None
    liked = None
    if mode == 'My taste':
        titles = bundle[1].title.to_dict()
        liked = st.multiselect('Movies you love', sorted(titles, key=titles.get), format_func=titles.get,
                               help='Each selection counts as a five-star rating for this session.')
    else:
        user = st.selectbox('MovieLens user', sorted(bundle[0].user_id.unique().tolist()))
    count = st.slider('Number of suggestions', 5, 20, 10)
    discover = st.checkbox('Discover less popular movies', disabled=user is None and not liked)
    explanations = st.checkbox('Show explanations', value=True)
    st.caption('Discovery applies the evaluated popularity penalty. It can reduce relevance.')

if user is None:
    st.subheader('Picked for you' if liked else 'Popular on MovieLens')
    st.caption('Based on your favorites.' if liked else 'Choose a favorite in the sidebar to personalize these picks.')
else:
    st.subheader(f'Movie picks for user {user}')
    st.caption('Previously rated movies are excluded.')
rows = recommend(bundle, user, count, discover, liked_movies=liked)
if not rows:
    st.info('No eligible unseen movies are available for this user.')
for start in range(0, len(rows), 2):
    columns = st.columns(2)
    for column, movie in zip(columns, rows[start:start + 2]):
        with column, st.container(border=True):
            st.subheader(movie['title'])
            st.caption(str(movie['genres']).replace('|', ' · '))
            if movie['prediction'] is not None:
                st.write(f"Predicted rating: **{movie['prediction']:.1f} / 5**")
            if explanations:
                st.write(movie['reason'])

with st.expander('About these recommendations'):
    st.write('CineCue blends EASE and matrix factorization using settings selected on validation data. '
             'Models are fitted on train plus validation; test ratings are never loaded by this app. '
             'Explanations name a positive contributor to the score, rather than claiming a single cause.')
    st.write('Time-of-day and recency ranking experiments are documented in the notebooks. '
             'This demo uses the final ranker, which does not include those signals. '
             'The explanation toggle changes presentation only, allowing the same list to be compared.')
