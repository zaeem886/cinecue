"""The website visitor's likes, each stamped with the weekday and daypart it happened.

Likes live in a small local JSON file so they survive browser refreshes and restarts.
The file holds only movie ids and times; nothing is sent anywhere.
"""

from datetime import datetime
import json
import os
from tempfile import NamedTemporaryFile

from .context import daypart


def make_like(movie_id, when=None, weekday=None, part=None, title=""):
    """A like at `when` (default: now, local time). `weekday`/`part` override the clock."""
    when = when or datetime.now().astimezone()
    return {"movie_id": int(movie_id), "title": str(title), "liked_at": when.isoformat(timespec="seconds"),
            "weekday": int(when.weekday() if weekday is None else weekday),
            "part": int(daypart(when.hour) if part is None else part)}


def load_likes(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    likes = []
    for item in data.get("likes", []):
        try:
            like = {"movie_id": int(item["movie_id"]), "title": str(item.get("title", "")),
                    "liked_at": str(item.get("liked_at", "")),
                    "weekday": int(item["weekday"]), "part": int(item["part"])}
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= like["weekday"] < 7 and 0 <= like["part"] < 4:
            likes.append(like)
    return likes


def save_likes(path, likes):
    """Write atomically, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp") as output:
        json.dump({"version": 1, "likes": likes}, output, indent=2)
        temporary = output.name
    os.replace(temporary, path)
