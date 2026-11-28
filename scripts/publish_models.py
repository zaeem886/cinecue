"""Upload the website's fitted models to a release of a private GitHub repository.

Run once after building the models (scripts/save_models.py, scripts/build_app_catalog.py),
and again whenever they are rebuilt. Teammates then get the models automatically
(see src/cinecue/hub.py).

    gh auth login                                   # once, with access to the models repository
    python scripts/publish_models.py                # defaults to cinecue.hub.DEFAULT_REPO

The repository is created private if it does not exist, and the script refuses to upload to
a public one: the models contain MovieLens ratings, which may not be redistributed. Models
larger than --part-mb are uploaded as `<name>.part-a`, `.part-b`, ... so a dropped connection
only repeats one part. Each model gets a `<name>.sha256` file that the website checks. The
release stays a draft until every file is uploaded and verified, so the website never sees a
half-uploaded release.
"""

import argparse
import base64
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinecue.hub import DEFAULT_REPO, MODEL_FILES  # noqa: E402

README = """# CineCue models (private, course project)

Fitted EASE + matrix factorization models used by the CineCue website
(https://github.com/zaeem886/cinecue). They contain MovieLens ratings
(F. M. Harper and J. A. Konstan, 2015, ACM TiiS 5(4): 19). GroupLens does not allow
redistribution, so **this repository must stay private to the team**.

The models are attached to the latest release, not committed. The website downloads them
on first use; large models are split into `.part-*` files and checked against `.sha256`.

| File | Catalog |
| --- | --- |
| `recommender-32m.pkl` | 9,000 most-rated MovieLens 32M films (1902–2023), built by `scripts/build_app_catalog.py` |
| `recommender.pkl` | MovieLens 1M catalog (films up to 2000), built by `scripts/save_models.py` |

These are Python pickles. Load them only with the CineCue code, and only from this repository.
"""


def gh(*args, check=True):
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])} failed: {result.stderr.strip()}")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split(path, folder, part_bytes):
    """Write `path` as <name>.part-a, -b, ... in `folder` and return the part paths."""
    count = -(-path.stat().st_size // part_bytes)
    if count > 26:
        raise SystemExit(f"{path.name} needs more than 26 parts; raise --part-mb.")
    parts = []
    with path.open("rb") as source:
        for index in range(count):
            part = folder / f"{path.name}.part-{chr(ord('a') + index)}"
            part.write_bytes(source.read(part_bytes))
            parts.append(part)
    return parts


def ensure_private_repo(repo):
    if gh("repo", "view", repo, "--json", "visibility", check=False).returncode != 0:
        print(f"Creating private repository {repo}")
        gh("repo", "create", repo, "--private",
           "--description", "Private fitted models for the CineCue website (contain MovieLens ratings)")
        # A release needs at least one commit; the README is it.
        gh("api", "-X", "PUT", f"repos/{repo}/contents/README.md", "-f", "message=Describe the CineCue models",
           "-f", f"content={base64.b64encode(README.encode()).decode()}")
    visibility = json.loads(gh("repo", "view", repo, "--json", "visibility").stdout)["visibility"]
    if visibility != "PRIVATE":
        raise SystemExit(f"{repo} is {visibility.lower()}. Make it private before uploading MovieLens-derived models.")


def release_assets(repo, tag):
    """Map asset name -> (id, size, state) for the release tagged `tag`."""
    release = json.loads(gh("api", f"repos/{repo}/releases/tags/{tag}").stdout)
    assets = json.loads(gh("api", f"repos/{repo}/releases/{release['id']}/assets?per_page=100").stdout)
    return release["id"], {a["name"]: (a["id"], a["size"], a["state"]) for a in assets}


def upload(repo, tag, path, attempts=5):
    for attempt in range(1, attempts + 1):
        started = time.time()
        result = gh("release", "upload", tag, str(path), "--repo", repo, "--clobber", check=False)
        if result.returncode == 0:
            print(f"  {path.name}: uploaded in {time.time() - started:.0f}s", flush=True)
            return
        print(f"  {path.name}: attempt {attempt} failed ({result.stderr.strip().splitlines()[-1:]})", flush=True)
        time.sleep(10)
    raise SystemExit(f"Gave up uploading {path.name}. Rerun the script to try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"owner/name (default {DEFAULT_REPO})")
    parser.add_argument("--tag", default=f"models-{date.today().isoformat()}", help="release tag (default today)")
    parser.add_argument("--part-mb", type=int, default=90, help="split models larger than this (default 90 MB)")
    args = parser.parse_args()
    gh("auth", "status")
    ensure_private_repo(args.repo)

    models = [ROOT / "artifacts" / name for name in MODEL_FILES]
    for path in models:
        if not path.is_file():
            print(f"Skipping {path.name}: not built on this machine")
    models = [path for path in models if path.is_file()]
    if not models:
        raise SystemExit("No models to upload.")

    if gh("release", "view", args.tag, "--repo", args.repo, check=False).returncode != 0:
        gh("release", "create", args.tag, "--repo", args.repo, "--draft", "--title", f"CineCue models {args.tag}",
           "--notes", "Fitted CineCue models. Private: they contain MovieLens ratings.")
    part_bytes = args.part_mb * 2**20
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        expected = {}
        for path in models:
            print(f"{path.name} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)
            files = split(path, folder, part_bytes) if path.stat().st_size > part_bytes else [path]
            checksum = folder / f"{path.name}.sha256"
            checksum.write_text("".join(f"{sha256(f)}  {f.name}\n" for f in files if f != path)
                                + f"{sha256(path)}  {path.name}\n")
            # Remove this model's assets from an earlier run of the same tag that this run won't replace.
            _, assets = release_assets(args.repo, args.tag)
            keep = {f.name for f in files} | {checksum.name}
            for name, (asset_id, _, _) in assets.items():
                if (name == path.name or name.startswith(f"{path.name}.part-")) and name not in keep:
                    gh("api", "-X", "DELETE", f"repos/{args.repo}/releases/assets/{asset_id}")
            for file in [checksum, *files]:
                upload(args.repo, args.tag, file)
                expected[file.name] = file.stat().st_size
            for file in files:
                if file != path:
                    file.unlink()

        _, assets = release_assets(args.repo, args.tag)
        bad = [name for name, size in expected.items()
               if name not in assets or assets[name][1] != size or assets[name][2] != "uploaded"]
        if bad:
            raise SystemExit(f"These files did not upload completely: {', '.join(bad)}. Rerun the script.")
    gh("release", "edit", args.tag, "--repo", args.repo, "--draft=false", "--latest")
    print(f"Done: https://github.com/{args.repo}/releases/tag/{args.tag} (private)")


if __name__ == "__main__":
    main()
