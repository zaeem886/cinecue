"""Demo adapter; fits on train + validation and never reads test feedback."""

import json

import numpy as np
import pandas as pd

from .data import load_movies
from .explain import blend_contributions
from .models import EASE, MatrixFactorization
from .ranking import blend, popularity_scale, rerank_by_popularity, top_k


def fit_demo(processed):
    history = pd.concat([pd.read_csv(processed / f'{s}.csv', usecols=['user_id', 'movie_id', 'rating'])
                         for s in ('train', 'validation')], ignore_index=True)
    ranking = json.loads((processed / 'ranking_summary.json').read_text())['selected']
    rating = json.loads((processed / 'rating_summary.json').read_text())['mf']
    args = (history.user_id, history.movie_id, history.rating)
    mf = MatrixFactorization(**rating).fit(*args)
    ease = EASE(reg=float(ranking['ease_reg'])).fit(*args[:2])
    if not mf.movies.ids.equals(ease.movies.ids):
        raise ValueError('Model catalogs differ.')
    catalog = load_movies(processed).set_index('movie_id').loc[mf.movies.ids]
    counts = history.movie_id.value_counts().reindex(mf.movies.ids, fill_value=0).to_numpy()
    return history, catalog, mf, ease, counts, ranking


def recommend(bundle, user_id, k=10, discover=False, liked_movies=None):
    history, catalog, mf, ease, counts, settings = bundle
    if liked_movies is None:
        profile = history.loc[history.user_id == user_id]
    else:
        profile = pd.DataFrame({'movie_id': list(dict.fromkeys(liked_movies)), 'rating': 5.0})
    positions = mf.movies.encode(profile.movie_id)
    stars = profile.rating.to_numpy(float)
    exclude = (counts < 20)[None, :].copy()
    exclude[0, positions] = True
    available = int((~exclude).sum())
    if not available:
        return []
    k = min(k, available)
    if not len(profile):
        ranked = top_k(counts[None, :], exclude, k)[0]
        return [dict(title=catalog.iloc[t].title, genres=catalog.iloc[t].genres,
                     prediction=None, reason='Popular among users in the training history.') for t in ranked]
    vector = mf.fold_in(positions, stars)
    predicted = mf.score_items(vector)
    ease_scores = ease.score(ease.history_matrix(np.zeros(len(positions), int), positions, 1))
    alpha = float(settings['alpha'])
    scores = blend(ease_scores, predicted[None, :], alpha)
    if discover:
        ranked = rerank_by_popularity(scores, exclude, popularity_scale(counts),
                                      float(settings['rerank_weight']), k,
                                      depth=min(100, available))[0]
    else:
        ranked = top_k(scores, exclude, k)[0]
    terms, _, _ = blend_contributions(ease, mf, alpha, positions, stars, ranked)
    rows = []
    for row, target in enumerate(ranked):
        eligible = np.flatnonzero((stars >= 4) & (terms[row] > 0))
        if len(eligible):
            source = eligible[np.argmax(terms[row, eligible])]
            source_movie = catalog.iloc[positions[source]]
            shared = sorted(set(str(source_movie.genres).split('|')) &
                            set(str(catalog.iloc[target].genres).split('|')))
            reason = f'Your positive rating for {source_movie.title} contributes to this recommendation.'
            if shared:
                reason += ' Shared genres: ' + ', '.join(shared) + '.'
        else:
            reason = 'Suggested from your combined rating history and general movie appeal.'
        rows.append(dict(title=catalog.iloc[target].title, genres=catalog.iloc[target].genres,
                         prediction=float(np.clip(predicted[target], 1, 5)), reason=reason))
    return rows
