"""Download MovieLens 1M, or use files already in data/raw/."""

import argparse
from pathlib import Path
import shutil
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Never use the network")
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    names = ("ratings.dat", "movies.dat")
    if all((raw / name).is_file() for name in names):
        print(f"Using local files in {raw}")
        return
    # Also accept the directory layout from manually unzipping the official archive.
    if all((raw / "ml-1m" / name).is_file() for name in names):
        for name in names:
            shutil.copyfile(raw / "ml-1m" / name, raw / name)
        print("Copied local ml-1m files into data/raw/.")
        return
    archive = raw / "ml-1m.zip"
    if not archive.is_file():
        if args.offline:
            raise SystemExit("Place ratings.dat and movies.dat, ml-1m/, or ml-1m.zip in data/raw/.")
        print(f"Downloading {URL}")
        temporary = archive.with_suffix(".zip.part")
        try:
            with urlopen(URL, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(archive)
        except URLError as error:
            raise SystemExit(
                f"Download failed: {error.reason}. Download ml-1m.zip from the official "
                "GroupLens page into data/raw/, then run this script with --offline. "
                "See README.md for a Windows download alternative."
            ) from error
        finally:
            temporary.unlink(missing_ok=True)
    # Read only known members; do not extract arbitrary archive paths.
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive checksum check failed; download it again.")
        for name in (*names, "README"):
            with bundle.open(f"ml-1m/{name}") as source, (raw / name).open("wb") as output:
                shutil.copyfileobj(source, output)
    print(f"Ready: {raw / 'ratings.dat'} and {raw / 'movies.dat'}")


if __name__ == "__main__":
    main()
