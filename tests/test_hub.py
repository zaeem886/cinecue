"""Checks for downloading shared models from GitHub releases. Run: python -m unittest discover -s tests"""

import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cinecue import hub  # noqa: E402

PAYLOAD = b"model bytes " * 1000
PARTS = {"big.pkl.part-a": PAYLOAD[:5000], "big.pkl.part-b": PAYLOAD[5000:10000], "big.pkl.part-c": PAYLOAD[10000:]}


def release(files):
    """Fake release assets plus a fetch that writes each asset's bytes."""
    assets = {name: {"url": name, "size": len(data)} for name, data in files.items()}

    def fetch(asset, token, output):
        output.write(files[asset["url"]])

    return assets, fetch


class DownloadModelsTest(unittest.TestCase):
    def run_download(self, files, names=("big.pkl",)):
        assets, fetch = release(files)
        with TemporaryDirectory() as folder, \
                mock.patch.object(hub, "github_token", return_value="token"), \
                mock.patch.object(hub, "release_assets", return_value=assets), \
                mock.patch.object(hub, "fetch_asset", side_effect=fetch):
            present = hub.download_models(folder, "owner/models", names)
            contents = {path.name: path.read_bytes() for path in present}
            leftovers = [p.name for p in Path(folder).iterdir() if p.name.startswith(".")]
        return contents, leftovers

    def test_parts_are_joined_in_order_and_checked(self):
        checksum = f"{hashlib.sha256(PAYLOAD).hexdigest()}  big.pkl\n".encode()
        contents, leftovers = self.run_download({**PARTS, "big.pkl.sha256": checksum})
        self.assertEqual(contents, {"big.pkl": PAYLOAD})
        self.assertEqual(leftovers, [])

    def test_whole_file_without_checksum(self):
        contents, _ = self.run_download({"big.pkl": PAYLOAD})
        self.assertEqual(contents, {"big.pkl": PAYLOAD})

    def test_bad_checksum_is_rejected(self):
        checksum = f"{'0' * 64}  big.pkl\n".encode()
        with self.assertRaises(ValueError):
            self.run_download({**PARTS, "big.pkl.sha256": checksum})

    def test_files_absent_from_release_are_skipped(self):
        contents, _ = self.run_download({"big.pkl": PAYLOAD}, names=("big.pkl", "other.pkl"))
        self.assertEqual(list(contents), ["big.pkl"])

    def test_missing_token_is_reported(self):
        with TemporaryDirectory() as folder, mock.patch.object(hub, "github_token", return_value=None):
            with self.assertRaises(PermissionError):
                hub.download_models(folder, "owner/models", ("big.pkl",))

    def test_present_files_need_no_token(self):
        with TemporaryDirectory() as folder, mock.patch.object(hub, "github_token") as token:
            (Path(folder) / "big.pkl").write_bytes(PAYLOAD)
            self.assertEqual([p.name for p in hub.download_models(folder, "owner/models", ("big.pkl",))],
                             ["big.pkl"])
            token.assert_not_called()


if __name__ == "__main__":
    unittest.main()
