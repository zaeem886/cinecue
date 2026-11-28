"""Checks for CineCue accounts. Run: python -m unittest discover -s tests

Password tests always run. Database tests need an empty, migrated PostgreSQL database in
TEST_DATABASE_URL (never the real one: every test deletes the users it creates), e.g.
    createdb cinecue_test
    DATABASE_URL=postgresql://localhost:5432/cinecue_test npx prisma migrate deploy
    TEST_DATABASE_URL=postgresql://localhost:5432/cinecue_test python -m unittest tests.test_accounts
"""

import os
from pathlib import Path
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cinecue.accounts import AccountError, AccountStore, hash_password, verify_password  # noqa: E402

TEST_URL = os.environ.get("TEST_DATABASE_URL")


class PasswordTests(unittest.TestCase):
    def test_hash_round_trip_and_salting(self):
        first, second = hash_password("correct horse"), hash_password("correct horse")
        self.assertNotEqual(first, second)  # a new salt every time
        self.assertTrue(verify_password("correct horse", first))
        self.assertFalse(verify_password("wrong horse", first))
        self.assertNotIn("correct horse", first)

    def test_malformed_hashes_never_verify(self):
        for stored in ["", "plain-text", "bcrypt$1$2$3$4$5", "scrypt$x$8$1$AA==$AA=="]:
            self.assertFalse(verify_password("anything", stored))


@unittest.skipUnless(TEST_URL, "set TEST_DATABASE_URL to a migrated test database")
class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = AccountStore(TEST_URL)
        self.email = f"test-{uuid.uuid4().hex[:10]}@example.com"
        self.user = self.store.sign_up(self.email.upper(), "  Test   User ", "password123")

    def tearDown(self):
        self.store.delete_account(self.user.id)
        self.store.close()

    def test_sign_up_normalises_and_logs_in(self):
        self.assertEqual(self.user.email, self.email)
        self.assertEqual(self.user.display_name, "Test User")
        self.assertEqual(self.store.log_in(f"  {self.email} ", "password123"), self.user)
        self.assertEqual(self.store.activity(self.user.id), {"signup": 1, "login": 1})

    def test_rejections(self):
        with self.assertRaises(AccountError):
            self.store.sign_up(self.email, "Someone", "password123")  # duplicate email
        for email, name, password in [("not-an-email", "A", "password123"), ("a@b.co", "", "password123"),
                                      ("a@b.co", "A", "short")]:
            with self.assertRaises(AccountError):
                self.store.sign_up(email, name, password)
        with self.assertRaises(AccountError):
            self.store.log_in(self.email, "wrong password")
        with self.assertRaises(AccountError):
            self.store.log_in("nobody@example.com", "password123")

    def test_likes_sync_inserts_retimes_and_removes(self):
        likes = [{"movie_id": 2571, "title": "Matrix, The (1999)", "liked_at": "2026-11-28T21:15:00+05:00",
                  "weekday": 4, "part": 3},
                 {"movie_id": 1, "title": "Toy Story (1995)", "liked_at": "2026-11-29T09:00:00+05:00",
                  "weekday": 5, "part": 1}]
        self.assertEqual(self.store.sync_likes(self.user.id, likes), 2)
        stored = self.store.likes(self.user.id)
        self.assertEqual([like["movie_id"] for like in stored], [2571, 1])  # oldest first
        self.assertEqual((stored[0]["weekday"], stored[0]["part"], stored[0]["title"]), (4, 3, "Matrix, The (1999)"))
        self.assertEqual(self.store.sync_likes(self.user.id, likes), 0)  # nothing changed
        moved = [likes[0] | {"weekday": 6, "part": 0}]                    # retime one, drop the other
        self.assertEqual(self.store.sync_likes(self.user.id, moved), 2)
        self.assertEqual([(l["movie_id"], l["weekday"], l["part"]) for l in self.store.likes(self.user.id)],
                         [(2571, 6, 0)])
        with self.assertRaises(AccountError):
            self.store.sync_likes(self.user.id, [likes[0] | {"weekday": 9}])
        self.assertEqual(len(self.store.likes(self.user.id)), 1)  # the bad sync changed nothing
        activity = self.store.activity(self.user.id)
        self.assertEqual((activity["like"], activity["retime"], activity["unlike"]), (2, 1, 1))

    def test_watchlist_and_activity(self):
        self.store.save(self.user.id, 260, "Star Wars: Episode IV - A New Hope (1977)", "Action|Adventure|Sci-Fi")
        self.store.save(self.user.id, 260)  # saving twice keeps one entry
        self.store.save(self.user.id, 1196)
        saved = self.store.watchlist(self.user.id)
        self.assertEqual(sorted(item["movie_id"] for item in saved), [260, 1196])
        self.assertEqual(next(i for i in saved if i["movie_id"] == 260)["genres"], "Action|Adventure|Sci-Fi")
        self.store.unsave(self.user.id, 260)
        self.assertEqual([item["movie_id"] for item in self.store.watchlist(self.user.id)], [1196])
        self.assertEqual(self.store.activity(self.user.id)["save"], 3)

    def test_deleting_an_account_removes_its_data(self):
        self.store.sync_likes(self.user.id, [{"movie_id": 2571, "weekday": 0, "part": 2}])
        self.store.save(self.user.id, 260)
        other = AccountStore(TEST_URL)
        other.delete_account(self.user.id)
        self.assertEqual(other.likes(self.user.id), [])
        self.assertEqual(other.watchlist(self.user.id), [])
        other.close()
        self.user = self.store.sign_up(f"again-{self.email}", "Again", "password123")  # for tearDown


if __name__ == "__main__":
    unittest.main()
