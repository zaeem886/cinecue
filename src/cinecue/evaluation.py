"""Rating errors broken down by how much history a user or movie had."""

import numpy as np
import pandas as pd

from .models import mae, rmse

USER_BINS = ([-1, 0, 19, 99, 299, np.inf], ["0 (new)", "1-19", "20-99", "100-299", "300+"])
MOVIE_BINS = ([-1, 0, 19, 99, 499, np.inf], ["0 (new)", "1-19", "20-99", "100-499", "500+"])


def segment_labels(events, history):
    """Label each event by its user's and movie's number of ratings in `history`."""
    user_counts = events.user_id.map(history.user_id.value_counts()).fillna(0)
    movie_counts = events.movie_id.map(history.movie_id.value_counts()).fillna(0)
    return pd.DataFrame({"user history": pd.cut(user_counts, USER_BINS[0], labels=USER_BINS[1]),
                         "movie history": pd.cut(movie_counts, MOVIE_BINS[0], labels=MOVIE_BINS[1])},
                        index=events.index)


def overall(actual, predictions):
    """RMSE and MAE for each named prediction array."""
    return pd.DataFrame({name: {"RMSE": rmse(actual, values), "MAE": mae(actual, values)}
                         for name, values in predictions.items()}).T


def by_segment(actual, predictions, labels):
    """RMSE per segment value, with the row count, for each named prediction array."""
    actual = np.asarray(actual, dtype=float)
    labels = pd.Series(labels).reset_index(drop=True)
    rows = {}
    for segment in labels.dropna().unique():
        mask = (labels == segment).to_numpy()
        rows[segment] = {"rows": int(mask.sum()),
                         **{name: rmse(actual[mask], np.asarray(values)[mask])
                            for name, values in predictions.items()}}
    categorical = isinstance(labels.dtype, pd.CategoricalDtype)
    order = [value for value in (labels.cat.categories if categorical else sorted(rows)) if value in rows]
    table = pd.DataFrame(rows).T.loc[order]
    table["rows"] = table["rows"].astype(int)
    return table
