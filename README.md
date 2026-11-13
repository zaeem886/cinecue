# CineCue

CineCue is a context-aware and explainable movie recommendation project. Phase 1
prepares MovieLens 1M through cleaning, exploratory data analysis (EDA), and
preprocessing. The code and notes are written to be read one notebook at a time.

The problem is to recommend relevant movies while accounting for limited context
and eventually explaining suggestions. Before building a recommender, we need to
understand rating quality, sparse histories, and popularity bias. Training,
evaluation, re-ranking, and explanation generation are reserved for Phase 2.

## Dataset

Use the official [MovieLens 1M dataset from GroupLens](https://grouplens.org/datasets/movielens/1m/).
Read the [dataset README and usage terms](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
before use. Research use requires acknowledgement; redistribution and commercial
use require separate permission. This project does not redistribute the archive,
raw files, or generated row-level data.

Dataset citation: F. Maxwell Harper and Joseph A. Konstan (2015),
*The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19.
[DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

We use `ratings.dat` (`user_id`, `movie_id`, `rating`, `timestamp`) and `movies.dat`
(`movie_id`, `title`, `genres`). The delimiter is `::`, and the files are read with
Latin-1 encoding to preserve title characters. The target is the original integer
rating from 1 to 5. We do not use the demographic file `users.dat`.

## Setup and download

Run every command from the repository root, the directory containing this README.
Use Python 3.11–3.14. Direct dependency versions, including notebook execution tools,
are pinned in [requirements.txt](requirements.txt).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_data.py
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in all subsequent commands. Activation is only a convenience.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
```

For offline work, copy both `ratings.dat` and `movies.dat` into `data/raw/`.
The download script also accepts an extracted `data/raw/ml-1m/` folder or
`data/raw/ml-1m.zip`. It reuses existing files and does not contact the network
when both data files are present:

```bash
python scripts/download_data.py --offline
```

If Python cannot verify the local HTTPS certificate chain, download the
[official ZIP](https://files.grouplens.org/datasets/movielens/ml-1m.zip) in a browser
and place it in `data/raw/`. On Windows, this equivalent download uses the Windows
certificate store:

```powershell
Invoke-WebRequest -Uri 'https://files.grouplens.org/datasets/movielens/ml-1m.zip' -OutFile 'data/raw/ml-1m.zip' -UseBasicParsing
python scripts/download_data.py --offline
```

Google Colab: clone the repository, change to the cloned directory with `%cd`, and
run `!python -m pip install -r requirements.txt` followed by
`!python scripts/download_data.py`. Run the execution commands below with `!` at
the start of each line, or open the notebooks individually after cloning. Install
dependencies before importing them; restart the Colab runtime if older versions
were already imported. Paths use `pathlib`, so no personal drive paths are required.

## Run Phase 1

Execute the notebooks in order. These commands save the real output cells and inline
plots back into the notebooks so they can be viewed on GitHub:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebooks/01_cleaning.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebooks/02_eda.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=python3 notebooks/03_preprocessing.ipynb
python scripts/verify_outputs.py
```

For interactive use, run `python -m jupyterlab`, select a kernel from the installed
environment, and use **Restart Kernel and Run All Cells** in the same notebook order.
The notebooks support working directories at either the repository root or `notebooks/`.

## Cleaning decisions

- Profile shapes, schema, raw dtypes, nulls, blank strings, exact duplicates,
  repeated user/movie pairs, and duplicate movie keys before changing data.
- Validate positive integer IDs, whole-star ratings, and nonnegative whole Unix
  seconds before March 2003, consistent with this dataset's February 2003 release.
  Parse times in UTC. Save invalid rating rows locally with source row numbers.
- Keep valid extreme ratings and high-activity users and movies. Statistical
  outlier removal would erase useful recommendation behavior.
- Keep original titles; trim and collapse whitespace in the clean title. Normalize
  genre tokens against the 18 official genres. Stop for invalid metadata or
  conflicting movie keys rather than arbitrarily selecting movie details.
- Remove exact interaction duplicates. Retain valid repeat events at distinct
  timestamps. For conflicting ratings at the same user/movie/timestamp, retain
  the first source row and save all conflict records for inspection. Never replace
  an earlier event with a later interaction.
- Validate a many-to-one left join and stop if any rating lacks movie metadata.
  Account for all excluded rows. Every check reports its affected count, including zero.

On the official data, all 1,000,209 ratings were retained. Invalid IDs, ratings,
timestamps, and metadata each affected **0 rows**. Exact rating duplicates,
repeated user/movie pairs, same-time conflicts, duplicate movie keys, and ratings
without movie metadata were also **0**. All 3,883 movie records were retained.

## Measured findings

The downloaded MovieLens files contain **1,000,209 ratings**, **6,040 active users**,
and **3,706 rated movies** out of **3,883 catalog movies**. The other 177 catalog
movies have no ratings. Rating activity spans 2000-04-25 23:05:32 UTC to
2003-02-28 17:49:50 UTC. The notebooks calculate these findings from the data:

| Stars | Ratings | Share |
| --- | ---: | ---: |
| 1 | 56,174 | 5.62% |
| 2 | 107,557 | 10.75% |
| 3 | 261,197 | 26.11% |
| 4 | 348,971 | 34.89% |
| 5 | 226,310 | 22.63% |

- **Positive skew:** 57.52% of ratings are 4 or 5; the mean is 3.582/5. Phase 2
  should compare against a simple rating baseline. Unrated movies are not negative examples.
- **Uneven activity:** the median user has 96 ratings; the median rated movie has
  123.5 ratings. Report future performance for sparse histories and infrequent movies.
- **Sparse feedback:** 95.53% of cells are empty in the 6,040 × 3,706 active
  user/movie matrix, counting unique pairs rather than repeat events. Including all
  3,883 catalog movies increases sparsity to 95.74%.
- **Popularity concentration:** the top 371 movies (rounded-up top 10% of rated
  movies) receive 44.45% of rating events. Phase 2 should examine popularity bias
  and catalog coverage.
- **Overlapping genres:** Drama is the largest catalog genre, with 1,603 movies.
  Genre mean ratings range from 3.215 for Horror (76,386 rating events) to 4.075
  for Film-Noir (18,261). One multigenre interaction can enter several groups;
  full-data genre means must not become Phase 2 features.
- **Uneven UTC activity:** 21:00–21:59 UTC is the busiest hour, with 65,491 ratings;
  Monday has the most submissions, with 173,931. Weekend activity accounts for
  25.36%. These describe submission behavior and cannot establish viewing habits.

## Features and target

| Field | Role |
| --- | --- |
| `user_id`, `movie_id` | Categorical identities; do not scale as continuous numbers. |
| `rating` | Target: original integer 1–5, unchanged. |
| `release_year` | Final parenthesized year in the title, restricted to 1800–2003; nullable. |
| `release_year_missing` | Binary indicator for an unsupported or missing release year. |
| 18 `genre_*` columns | Multihot genre flags, 0 or 1; no scaling needed. |
| `hour_utc`, `weekday_utc` | Categorical rating-submission context; hour 0–23, Monday=0 through Sunday=6. |
| `is_weekend_utc` | Binary Saturday/Sunday flag; no scaling needed. |
| `interaction_id`, `source_row` | Audit fields only, never predictive features. |
| `timestamp`, `rated_at` | Audit and time ordering; Unix seconds and UTC dates. |

There are no session features. Any future sessions must be labelled rating-session
proxies. No continuous transformer is needed in Phase 1. If a future model needs
release-year imputation or scaling, fit it on training rows only and apply it unchanged
to validation and test. Fit learned categorical encoders on training rows with an
explicit unknown-category policy. CSV readers should parse `rated_at` in UTC and
`release_year` with pandas nullable `Int64`.

The split is global and chronological, aiming for 80/10/10. Cutoffs use the nearest
cumulative timestamp-group size to 80% and 90%, with ties resolved toward the earlier
group. Equal timestamps always stay in the same split. Unseen users and movies stay
in later splits. Full-data EDA aggregates are never used as model inputs. Any future
history or popularity feature must use events strictly before the predicted interaction.

The realized split is:

| Split | Rows | Share | First rating (UTC) | Last rating (UTC) |
| --- | ---: | ---: | --- | --- |
| Train | 800,168 | 80.00008% | 2000-04-25 23:05:32 | 2000-12-02 14:52:18 |
| Validation | 100,020 | 9.99991% | 2000-12-02 14:52:28 | 2000-12-29 23:42:47 |
| Test | 100,021 | 10.00001% | 2000-12-29 23:43:34 | 2003-02-28 17:49:50 |

The inclusive Unix-second cutoffs are `975768738` for training and `978133367`
for validation. Equal timestamps do not cross boundaries. All 1,000,209 interactions
are present exactly once across the splits.

| Later split | Reference data | Unseen users | Unseen movies | Rows with either unseen |
| --- | --- | ---: | ---: | ---: |
| Validation | Train | 611 | 16 | 76,180 |
| Test | Train | 200 | 33 | 19,413 |
| Test | Train + validation | 29 | 28 | 4,298 |

These users and movies remain in the files. Phase 2 will need cold-start fallbacks,
especially for validation, where many users arrive after the training cutoff.
All release years were parseable in this dataset, so both release-year missing
counts are zero; the nullable field and missing indicator still support other inputs.

## Limitations

These are voluntary ratings, not a random sample of all viewers or all watched movies.
The dataset is old, and the catalog and users may differ from a current service.
Missing feedback does not mean dislike. Genre groups overlap, and genre averages
are associations rather than causal effects. We omit correlations of numeric IDs,
Unix seconds, and cyclic time categories because these would be misleading.

Timestamps describe **rating activity**, not watch times. Local time zones, moods,
viewing sessions, and historical catalog availability are not available. Release year
does not establish when a title entered the service. UTC context should therefore
be treated as a limited signal in Phase 2.

## Verification

All three notebooks were executed in order on the official data using Windows and
Python 3.14.8 with the pinned dependencies. All 17 code cells finished without error
outputs, and the EDA notebook contains five inline figures. The saved outputs are
included in the notebooks for GitHub viewing.

`scripts/verify_outputs.py` checks saved table schemas and counts, unchanged targets,
feature values, source-to-feature row preservation, split row conservation, strict
time boundaries, cold-start counts, summary consistency, and executed notebook
outputs. Additional small input checks covered invalid values, repeated events,
conflicts, accented text, missing metadata, malformed records, missing years, and
timestamp ties. Local-file, extracted-folder, and ZIP offline download paths passed.
Installed dependency versions match `requirements.txt`, and `pip check` passed.
The official source, terms, and download links were checked; the DOI is the citation
provided by GroupLens. Git ignore checks exclude the archive and generated data.

On this Windows setup, Jupyter prints a Proactor event-loop warning and starts an
extra selector thread; the notebooks still complete and pass output verification.
There are no outstanding Phase 1 blockers. Linux and Colab commands use portable
paths but were not executed in this verification environment.

## Repository layout

```text
cinecue/
  README.md
  requirements.txt
  .gitignore
  scripts/
    download_data.py
    verify_outputs.py
  notebooks/
    01_cleaning.ipynb
    02_eda.ipynb
    03_preprocessing.ipynb
  data/
    raw/                    # local download or existing files
    processed/              # local generated tables and summaries
```

The processed directory contains `ratings_clean.csv`, `movies_clean.csv`,
`invalid_ratings.csv`, `same_time_conflicts.csv`, `movie_features.csv`,
`interaction_features.csv`, `train.csv`, `validation.csv`, `test.csv`, and the
aggregate `cleaning_summary.json`, `eda_summary.json`, and `split_summary.json`.
Only directory placeholders are tracked under `data/`. Figures remain inline.

Suggested group milestones:

1. `Add MovieLens download setup and cleaning notebook`
2. `Explore ratings, popularity, genres, and UTC activity`
3. `Prepare context features and chronological data splits`
