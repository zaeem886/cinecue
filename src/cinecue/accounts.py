"""User accounts, timed likes, watchlists and activity stored in PostgreSQL.

The tables are defined in prisma/schema.prisma and created with `npm run db:migrate`.
This module reads and writes them with psycopg; the app needs only DATABASE_URL.

Passwords are hashed with scrypt (Python standard library) and a random salt per user.
Only the hash is stored. All queries are parameterised.
"""

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import re
import secrets

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
SCRYPT = {"n": 2**14, "r": 8, "p": 1}


class AccountError(ValueError):
    """A problem the user can fix, with a message safe to show on screen."""


@dataclass(frozen=True)
class User:
    id: int
    email: str
    display_name: str


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    encode = lambda raw: base64.b64encode(raw).decode()  # noqa: E731
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${encode(salt)}${encode(digest)}"


def verify_password(password, stored):
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# A fixed hash checked when an email is unknown, so failed logins take the same time either way.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def read_env_file(path):
    """Load KEY=VALUE lines from a local .env file into os.environ (existing variables win)."""
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def database_url():
    return os.environ.get("DATABASE_URL", "").strip() or None


class AccountStore:
    """Accounts, timed likes, watchlist and activity for one website session."""

    def __init__(self, url):
        self.url = url
        self._connection = None

    def _connect(self):
        import psycopg

        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(self.url, autocommit=True, connect_timeout=10)
        return self._connection

    def _run(self, query, params=(), fetch=None):
        import psycopg

        for attempt in (1, 2):  # reconnect once if the server dropped an idle connection
            try:
                with self._connect().cursor() as cursor:
                    cursor.execute(query, params)
                    if fetch == "one":
                        return cursor.fetchone()
                    if fetch == "all":
                        return cursor.fetchall()
                    return None
            except psycopg.OperationalError:
                self._connection = None
                if attempt == 2:
                    raise

    def close(self):
        if self._connection is not None and not self._connection.closed:
            self._connection.close()
        self._connection = None

    def check(self):
        """Raise if the database or its tables are not reachable."""
        try:
            self._run("SELECT 1 FROM users, likes, watchlist, events LIMIT 1")
        finally:
            self.close()

    # ---- accounts ----
    def sign_up(self, email, display_name, password):
        email, display_name = email.strip().lower(), " ".join(display_name.split())
        if not EMAIL.match(email):
            raise AccountError("Enter a valid email address.")
        if not 1 <= len(display_name) <= 60:
            raise AccountError("Enter a name of up to 60 characters.")
        if len(password) < MIN_PASSWORD:
            raise AccountError(f"Use a password of at least {MIN_PASSWORD} characters.")
        import psycopg

        try:
            row = self._run("INSERT INTO users (email, display_name, password_hash, last_login_at) "
                            "VALUES (%s, %s, %s, now()) RETURNING id",
                            (email, display_name, hash_password(password)), fetch="one")
        except psycopg.errors.UniqueViolation:
            raise AccountError("An account with this email already exists. Log in instead.") from None
        user = User(row[0], email, display_name)
        self.log(user.id, "signup")
        return user

    def log_in(self, email, password):
        row = self._run("SELECT id, email, display_name, password_hash FROM users WHERE email = %s",
                        (email.strip().lower(),), fetch="one")
        if not verify_password(password, row[3] if row else _DUMMY_HASH) or not row:
            raise AccountError("Email or password is incorrect.")
        self._run("UPDATE users SET last_login_at = now() WHERE id = %s", (row[0],))
        self.log(row[0], "login")
        return User(row[0], row[1], row[2])

    def delete_account(self, user_id):
        self._run("DELETE FROM users WHERE id = %s", (user_id,))

    # ---- taste profile: timed likes (see cinecue.likes) ----
    def likes(self, user_id):
        """The user's likes in the app's format, oldest first."""
        rows = self._run("SELECT movie_id, title, liked_at, weekday, part FROM likes WHERE user_id = %s "
                         "ORDER BY liked_at", (user_id,), fetch="all")
        return [{"movie_id": int(movie), "title": title, "liked_at": when.isoformat(timespec="seconds"),
                 "weekday": int(weekday), "part": int(part)} for movie, title, when, weekday, part in rows]

    def sync_likes(self, user_id, likes):
        """Make the stored likes equal `likes`: insert new ones, delete removed ones, and update
        changed weekday/daypart. All in one transaction; each change is logged as an event."""
        from datetime import datetime

        wanted = {int(like["movie_id"]): like for like in likes}
        for like in wanted.values():
            if not (0 <= int(like["weekday"]) < 7 and 0 <= int(like["part"]) < 4):
                raise AccountError("A like has an invalid weekday or time of day.")
        connection = self._connect()
        with connection.transaction(), connection.cursor() as cursor:
            cursor.execute("SELECT movie_id, weekday, part FROM likes WHERE user_id = %s FOR UPDATE", (user_id,))
            stored = {movie: (weekday, part) for movie, weekday, part in cursor.fetchall()}
            events = []
            for movie in stored.keys() - wanted.keys():
                cursor.execute("DELETE FROM likes WHERE user_id = %s AND movie_id = %s", (user_id, movie))
                events.append(("unlike", movie, None))
            for movie, like in wanted.items():
                moment = (int(like["weekday"]), int(like["part"]))
                if movie not in stored:
                    when = like.get("liked_at")
                    cursor.execute("INSERT INTO likes (user_id, movie_id, title, liked_at, weekday, part) "
                                   "VALUES (%s, %s, %s, %s, %s, %s)",
                                   (user_id, movie, str(like.get("title", ""))[:300],
                                    datetime.fromisoformat(when) if when else datetime.now().astimezone(), *moment))
                    events.append(("like", movie, {"weekday": moment[0], "part": moment[1]}))
                elif stored[movie] != moment:
                    cursor.execute("UPDATE likes SET weekday = %s, part = %s WHERE user_id = %s AND movie_id = %s",
                                   (*moment, user_id, movie))
                    events.append(("retime", movie, {"weekday": moment[0], "part": moment[1]}))
            for kind, movie, detail in events:
                cursor.execute("INSERT INTO events (user_id, kind, movie_id, detail) VALUES (%s, %s, %s, %s)",
                               (user_id, kind, movie, None if detail is None else json.dumps(detail)))
        return len(events)

    # ---- watchlist ----
    def watchlist(self, user_id):
        """Saved movies, newest first, as {"movie_id", "title", "genres"}."""
        rows = self._run("SELECT movie_id, title, genres FROM watchlist WHERE user_id = %s ORDER BY added_at DESC",
                         (user_id,), fetch="all")
        return [{"movie_id": int(movie), "title": title, "genres": genres} for movie, title, genres in rows]

    def save(self, user_id, movie_id, title="", genres=""):
        self._run("INSERT INTO watchlist (user_id, movie_id, title, genres) VALUES (%s, %s, %s, %s) "
                  "ON CONFLICT (user_id, movie_id) DO NOTHING",
                  (user_id, int(movie_id), str(title)[:300], str(genres)[:300]))
        self.log(user_id, "save", movie_id)

    def unsave(self, user_id, movie_id):
        self._run("DELETE FROM watchlist WHERE user_id = %s AND movie_id = %s", (user_id, int(movie_id)))
        self.log(user_id, "unsave", movie_id)

    # ---- activity ----
    def log(self, user_id, kind, movie_id=None, detail=None):
        self._run("INSERT INTO events (user_id, kind, movie_id, detail) VALUES (%s, %s, %s, %s)",
                  (user_id, kind, None if movie_id is None else int(movie_id),
                   None if detail is None else json.dumps(detail)))

    def activity(self, user_id):
        """Counts per activity kind, for the profile page."""
        rows = self._run("SELECT kind, count(*) FROM events WHERE user_id = %s GROUP BY kind", (user_id,),
                         fetch="all")
        return dict(rows)
