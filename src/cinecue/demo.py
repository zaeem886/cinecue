"""Demo adapter; fits on train + validation and never reads test feedback."""

import json

import numpy as np
import pandas as pd

from .data import load_movies, table_path
from .explain import blend_contributions
from .models import EASE, MatrixFactorization
from .ranking import blend, popularity_scale, rerank_by_popularity, top_k


def fit_demo(processed):
    history = pd.concat([pd.read_csv(table_path(processed, f'{s}.csv'), usecols=['user_id', 'movie_id', 'rating'])
                         for s in ('train', 'validation')], ignore_index=True)
    ranking = json.loads((processed / 'ranking_summary.json').read_text())['selected']
    rating = json.loads((processed / 'rating_summary.json').read_text())['mf']
    return fit_tables(history, load_movies(processed), rating, ranking)


def fit_tables(history, catalog, rating, ranking):
    args = (history.user_id, history.movie_id, history.rating)
    mf = MatrixFactorization(**rating).fit(*args)
    ease = EASE(reg=float(ranking['ease_reg'])).fit(*args[:2])
    if not mf.movies.ids.equals(ease.movies.ids):
        raise ValueError('Model catalogs differ.')
    catalog = catalog.set_index('movie_id').loc[mf.movies.ids]
    counts = history.movie_id.value_counts().reindex(mf.movies.ids, fill_value=0).to_numpy()
    return history, catalog, mf, ease, counts, ranking


def recommend(bundle, user_id, k=10, discover=False, liked_movies=None, ratings=None):
    """Top-k picks for a MovieLens user, a list of liked movies (5 stars each), or a
    {movie_id: stars} profile such as a signed-in user's saved ratings."""
    history, catalog, mf, ease, counts, settings = bundle
    if ratings is not None:
        known = {int(m): float(s) for m, s in ratings.items() if int(m) in catalog.index}
        profile = pd.DataFrame({'movie_id': list(known), 'rating': list(known.values())})
    elif liked_movies is None:
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
        return [dict(movie_id=int(catalog.index[t]), title=catalog.iloc[t].title, genres=catalog.iloc[t].genres,
                     prediction=None, ratings=int(counts[t]), because=None, shared=[],
                     reason='Popular among users in the training history.') for t in ranked]
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
        because, shared = None, []
        if len(eligible):
            source = eligible[np.argmax(terms[row, eligible])]
            source_movie = catalog.iloc[positions[source]]
            shared = sorted(set(str(source_movie.genres).split('|')) &
                            set(str(catalog.iloc[target].genres).split('|')))
            because = source_movie.title
            reason = f'Your positive rating for {source_movie.title} contributes to this recommendation.'
            if shared:
                reason += ' Shared genres: ' + ', '.join(shared) + '.'
        else:
            reason = 'Suggested from your combined rating history and general movie appeal.'
        rows.append(dict(movie_id=int(catalog.index[target]), title=catalog.iloc[target].title,
                         genres=catalog.iloc[target].genres, prediction=float(np.clip(predicted[target], 1, 5)),
                         ratings=int(counts[target]), because=because, shared=shared, reason=reason))
    return rows


# ---------- Context-aware recommendations (weekday, daypart, mood) ----------

# Mood boosts in units of the gap between the best and the median of the top candidates:
# under 'strong' a full match lifts a typical candidate level with the best non-matching pick;
# 'nudge' only reorders near neighbours.
MOOD_STRENGTH = {"nudge": 0.3, "strong": 1.2, "only": 1.2}


def context_tables(catalog, summary=None):
    """Genre matrix for the catalog and the 7 x genres everyone-on-this-weekday lift table."""
    from .context import WEEKDAYS, genre_matrix
    G, vocabulary = genre_matrix(catalog.genres)
    lift = np.zeros((7, len(vocabulary)))
    table = (summary or {}).get('global_weekday_genre_lift', {})
    for d, day in enumerate(WEEKDAYS):
        for k, genre in enumerate(vocabulary):
            lift[d, k] = table.get(day, {}).get(genre, 0.0)
    return G, vocabulary, lift


def _subset_terms(ease, positions, mask, targets):
    """Per-like terms of the pull from the likes in `mask` (targets x likes), see context._z_or_zero.

    The z-score splits exactly into one term per like; the one-sided soft cap is shared out in
    proportion, so the terms still add up to the feature.
    """
    from .context import squash
    terms = np.zeros((len(targets), len(positions)))
    if not mask.any():
        return terms
    rows = np.asarray(ease.weights[positions[mask]], float)
    total = rows.sum(axis=0)
    std = total.std()
    if std > 0:
        raw = ((rows[:, targets] - rows.mean(axis=1, keepdims=True)) / std).T
        z = raw.sum(axis=1, keepdims=True)
        # Matches context._z_or_zero: one-sided and softly capped, shared out over the likes.
        ratio = np.divide(squash(np.maximum(z, 0.0)), z, out=np.zeros_like(z), where=z > 1e-12)
        terms[:, mask] = raw * ratio
    return terms


def _profile(bundle, likes=None, user_id=None):
    """Positions, stars, like weekdays and like dayparts for timed likes or a MovieLens profile."""
    history, _, mf = bundle[:3]
    if likes is not None:
        likes = list({like['movie_id']: like for like in likes}.values())
        positions = mf.movies.encode(np.array([like['movie_id'] for like in likes], dtype=np.int64))
        keep = positions >= 0
        likes = [like for like, ok in zip(likes, keep) if ok]
        return (positions[keep], np.full(len(likes), 5.0), np.array([like['weekday'] for like in likes], int),
                np.array([like['part'] for like in likes], int))
    profile = history.loc[history.user_id == user_id]
    positions = mf.movies.encode(profile.movie_id.to_numpy())
    unknown = np.full(len(positions), -1)
    return positions, profile.rating.to_numpy(float), unknown, unknown


def _mood_match(tables, mood):
    """Share of the chosen mood genres each movie has (0..1), and the mood restricted to known genres."""
    G, vocabulary, _ = tables
    mood = {m for m in mood if m in vocabulary}
    if not mood:
        return np.zeros(len(G)), mood
    return G[:, [vocabulary.index(m) for m in sorted(mood)]].sum(axis=1) / len(mood), mood


def _mood_boost(scores, match, exclude, strength, depth):
    """Mood match scaled by the best-to-median gap of the top candidates, so it bites at any scale."""
    if not match.any():
        return np.zeros(len(scores))
    top = np.sort(scores[~exclude])[-depth:]
    spread = top[-1] - np.median(top) if len(top) > 1 else 0.0
    return MOOD_STRENGTH[strength] * (spread if spread > 0 else 1.0) * match


def _like_terms(bundle, ranker, positions, stars, days, parts, weekday, part, targets):
    """Exact contribution of each like (columns) to each target's context score (rows)."""
    _, _, mf, ease, _, settings = bundle
    weights = dict(zip(('blend', 'same_weekday', 'same_daypart', 'weekday_genre', 'global_genre'), ranker.weights))
    blend_terms, _, _ = blend_contributions(ease, mf, float(settings['alpha']), positions, stars, targets)
    return (weights['blend'] * blend_terms
            + weights['same_weekday'] * _subset_terms(ease, positions, days == weekday, targets)
            + weights['same_daypart'] * _subset_terms(ease, positions, parts == part, targets))


def recommend_in_context(bundle, tables, ranker, likes=None, user_id=None, k=12, discover=False,
                         weekday=0, part=0, mood=(), mood_strength='strong'):
    """Recommendations for a set of timed likes (or a MovieLens profile) in a weekday/daypart context.

    likes: dicts with movie_id, weekday and part (see likes.make_like); each counts as five stars.
    A MovieLens profile has no like times in the website's bundle, so only the everyone-on-this-
    weekday signal and the mood apply to it. `mood` is a set of genre names to boost explicitly.
    Rows carry the same fields as `recommend` plus context indicators for the interface.
    """
    from .context import DAYPARTS, WEEKDAYS, context_features, genre_names
    history, catalog, mf, ease, counts, settings = bundle
    G, vocabulary, lift = tables
    positions, stars, days, parts = _profile(bundle, likes, user_id)
    exclude = counts < 20
    exclude[positions] = True
    match, mood = _mood_match(tables, mood)
    if mood and mood_strength == 'only':
        exclude = exclude | (match == 0)
    penalty = None
    if discover:
        from .ranking import popularity_scale
        penalty = float(settings['rerank_weight']) * popularity_scale(counts)
    when = f'{WEEKDAYS[weekday]} {DAYPARTS[part]}'

    if not len(positions):  # no profile yet: popularity, still shaped by the mood
        pop = np.log1p(counts.astype(float))
        pop = (pop - pop.mean()) / pop.std()
        score = np.where(exclude, -np.inf, pop + _mood_boost(pop, match, exclude, mood_strength, ranker.depth))
        ranked = np.argsort(-score, kind='stable')[:min(k, int(np.isfinite(score).sum()))]
        return [dict(movie_id=int(catalog.index[t]), title=catalog.iloc[t].title, genres=catalog.iloc[t].genres,
                     prediction=None, ratings=int(counts[t]), because=None, because_when=None, shared=[],
                     mood_match=sorted(mood & set(genre_names(catalog.iloc[t].genres))), context_tags=[],
                     moved=0, reason='Popular among MovieLens users.') for t in ranked]

    features = context_features(ease, mf, float(settings['alpha']), positions, stars, days, parts,
                                weekday, part, G, lift)
    boost = _mood_boost(ranker.score(features), match, exclude, mood_strength, ranker.depth)
    ranked = ranker.rank(features, exclude, k, extra=boost, penalty=penalty)
    if not len(ranked):
        return []
    terms = _like_terms(bundle, ranker, positions, stars, days, parts, weekday, part, ranked)
    # Rank each pick would have without context or mood, for the "moved up" indicator.
    plain = np.where(exclude, -np.inf, features[:, 0])
    predicted = mf.score_items(mf.fold_in(positions, stars))
    contribution = ranker.contributions(features[ranked])
    rows = []
    for row, target in enumerate(ranked):
        movie = catalog.iloc[target]
        names = set(genre_names(movie.genres))
        eligible = np.flatnonzero((stars >= 4) & (terms[row] > 0))
        because = because_when = None
        shared = []
        if len(eligible):
            source = eligible[np.argmax(terms[row, eligible])]
            source_movie = catalog.iloc[positions[source]]
            because = source_movie.title
            shared = sorted(set(genre_names(source_movie.genres)) & names)
            if days[source] >= 0:
                because_when = describe_like(days[source], parts[source], weekday, part)
        tags = []
        if contribution[row, 1] > 0.05 and (days == weekday).any():
            tags.append(f'{WEEKDAYS[weekday]} likes')
        if contribution[row, 2] > 0.05 and (parts == part).any():
            tags.append(f'{DAYPARTS[part]} likes')
        if contribution[row, 3] > 0.02:
            tags.append(f'{WEEKDAYS[weekday]} genres')
        if contribution[row, 4] > 0.02:
            tags.append(f'{WEEKDAYS[weekday]} crowd')
        mood_match = sorted(mood & names)
        moved = int((plain > plain[target]).sum()) + 1 - (row + 1)
        reason = (f'Your like of {because} contributes most to this pick.' if because else
                  'Suggested from your combined likes and general movie appeal.')
        rows.append(dict(movie_id=int(catalog.index[target]), title=movie.title, genres=movie.genres,
                         prediction=float(np.clip(predicted[target], 1, 5)), ratings=int(counts[target]),
                         because=because, because_when=because_when, shared=shared, mood_match=mood_match,
                         context_tags=tags, moved=moved, context=when, reason=reason))
    return rows


# ---------- Taste match for any movie (search) ----------

# Verdicts by today's rank among comparable movies. The top 12 is what Discover would show.
MATCH_LEVELS = ((12, 5, 'Top pick today'), (100, 4, 'Excellent match'), (500, 3, 'Strong match'),
                (1500, 2, 'Good match'), (4000, 1, 'Mixed match'))


def match_level(rank):
    """(level 0-5, verdict) for a rank: 5 is a top pick, 0 'not your taste today'."""
    return next(((level, text) for limit, level, text in MATCH_LEVELS if rank <= limit),
                (0, 'Not your taste today'))


def taste_match(bundle, tables, ranker, likes, movie_ids, weekday=0, part=0, mood=(), mood_strength='strong'):
    """How well each movie fits today's taste: likes + weekday/daypart context + mood.

    The score is the same one Discover ranks by. `rank` is the movie's place among comparable
    movies (20+ ratings, not already liked) today, and `percent` the share of them it beats;
    `usual_rank`/`usual` are the same for the plain final ranker, without the day context or
    mood, so the difference shows what today changes. Reasons name the like that pulls the movie
    closest and the one that pushes it furthest away, both exact terms of the score.

    Without likes there is no personal taste yet, so the score is the one Discover uses then:
    general appeal (log popularity) plus the mood; such results carry basis='general'.
    Returns {movie_id: dict}.
    """
    from .context import context_features, genre_names
    _, catalog, mf, ease, counts, settings = bundle
    G, vocabulary, lift = tables
    positions, stars, days, parts = _profile(bundle, likes)
    targets = mf.movies.encode(np.array(list(movie_ids), dtype=np.int64)) if len(movie_ids) else np.array([], int)
    if not len(targets):
        return {}
    pool = counts >= 20
    pool[positions] = False
    match, mood = _mood_match(tables, mood)
    known = targets >= 0
    terms = np.zeros((len(targets), len(positions)))
    if len(positions):
        basis = 'likes'
        features = context_features(ease, mf, float(settings['alpha']), positions, stars, days, parts,
                                    weekday, part, G, lift)
        context_score = ranker.score(features)
        usual = features[:, 0]
        terms[known] = _like_terms(bundle, ranker, positions, stars, days, parts, weekday, part, targets[known])
        predicted = mf.score_items(mf.fold_in(positions, stars))
    else:  # no likes yet: Discover's "popular in the community" score
        basis = 'general'
        pop = np.log1p(counts.astype(float))
        context_score = usual = (pop - pop.mean()) / pop.std()
        predicted = None
    today = context_score + _mood_boost(context_score, match, ~pool, mood_strength, ranker.depth)
    reference_today, reference_usual = np.sort(today[pool]), np.sort(usual[pool])
    liked_genres = [set(genre_names(catalog.iloc[p].genres)) for p in positions]
    liked_positions = set(positions.tolist())
    out = {}
    for row, (movie_id, target) in enumerate(zip(movie_ids, targets)):
        if target < 0:
            continue
        movie = catalog.iloc[target]
        names = set(genre_names(movie.genres))
        size = len(reference_today)
        rank = int(size - np.searchsorted(reference_today, today[target], side='right')) + 1
        usual_rank = int(size - np.searchsorted(reference_usual, usual[target], side='right')) + 1
        percent = 100.0 * np.searchsorted(reference_today, today[target]) / size
        before = 100.0 * np.searchsorted(reference_usual, usual[target]) / size
        level, verdict = match_level(rank)
        closest = int(np.argmax(terms[row])) if len(positions) and terms[row].max() > 0 else None
        furthest = int(np.argmin(terms[row])) if len(positions) and terms[row].min() < 0 else None

        def describe(k):
            if k is None:
                return None
            like_movie = catalog.iloc[positions[k]]
            return dict(title=like_movie.title, when=describe_like(days[k], parts[k], weekday, part),
                        shared=sorted(set(genre_names(like_movie.genres)) & names))
        sharing = sum(bool(g & names) for g in liked_genres)
        out[int(movie_id)] = dict(
            rank=rank, usual_rank=usual_rank, pool=size, level=level, label=verdict, basis=basis,
            percent=round(float(percent), 1), usual=round(float(before), 1),
            liked=int(target) in liked_positions, low_data=bool(counts[target] < 20),
            closest=describe(closest), furthest=describe(furthest) if furthest != closest else None,
            sharing_likes=int(sharing), total_likes=len(positions),
            mood_match=sorted(mood & names), mood_miss=sorted(mood - names) if mood else [],
            prediction=None if predicted is None else float(np.clip(predicted[target], 1, 5)))
    return out


def describe_like(like_day, like_part, weekday, part):
    """'on a Monday evening, like now' style wording for when a like happened."""
    from .context import DAYPARTS, WEEKDAYS
    text = f'on a {WEEKDAYS[like_day]} {DAYPARTS[like_part]}'
    if like_day == weekday and like_part == part:
        return text + ', like now'
    if like_day == weekday:
        return text + f', also a {WEEKDAYS[weekday]}'
    return text
