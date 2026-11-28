"""Share the website's trained models through a private GitHub repository.

The fitted models (artifacts/*.pkl) are too large for the main repository and contain
MovieLens ratings, so they are attached to the latest release of a private repository
that only the team can read. The website and scripts download any missing file on first
use. A large model may be stored as `<name>.part-a`, `<name>.part-b`, ... which are joined
in order; a `<name>.sha256` asset, when present, is checked before the file is used.

Access needs a GitHub token that can read the repository: set GITHUB_TOKEN (or GH_TOKEN),
for example as a secret on a hosted server, or run `gh auth login` once locally. The
repository comes from the CINECUE_MODEL_REPO environment variable, falling back to
DEFAULT_REPO.
"""

import hashlib
import json
import os
import shutil
import ssl
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

DEFAULT_REPO = "ahmedsaalman/cinecue-models"
MODEL_FILES = ("recommender-32m.pkl", "recommender.pkl")
API = "https://api.github.com"
ATTEMPTS = 3


def model_repo():
    """The model repository ("owner/name") to use, or None when sharing is not configured."""
    return os.environ.get("CINECUE_MODEL_REPO", DEFAULT_REPO).strip() or None


def github_token():
    """A token from GITHUB_TOKEN or GH_TOKEN, else from the GitHub CLI login, else None."""
    for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(variable, "").strip():
            return os.environ[variable].strip()
    if shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _open(url, token, accept):
    request = Request(url, headers={"Accept": accept, "X-GitHub-Api-Version": "2022-11-28",
                                    "User-Agent": "cinecue-models"})
    # Unredirected, so the token is not forwarded to the storage host GitHub redirects to.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urlopen(request, timeout=60, context=_ssl_context())


def release_assets(repo, token):
    """Map asset name -> {"url", "size"} for the repository's latest published release."""
    with _open(f"{API}/repos/{repo}/releases/latest", token, "application/vnd.github+json") as response:
        release = json.load(response)
    return {asset["name"]: {"url": asset["url"], "size": asset["size"]} for asset in release["assets"]}


def fetch_asset(asset, token, output):
    """Stream one asset into the open binary file `output`, retrying a dropped connection."""
    start = output.tell()
    for attempt in range(1, ATTEMPTS + 1):
        output.seek(start)
        output.truncate()
        try:
            with _open(asset["url"], token, "application/octet-stream") as response:
                shutil.copyfileobj(response, output, 1 << 20)
        except OSError:
            if attempt == ATTEMPTS:
                raise
            continue
        if output.tell() - start == asset["size"]:
            return
        if attempt == ATTEMPTS:
            raise OSError(f"Downloaded {output.tell() - start} of {asset['size']} bytes")


def _checksums(assets, name, token):
    """Expected SHA-256 digests from `<name>.sha256` ("digest  filename" lines), if any."""
    if f"{name}.sha256" not in assets:
        return {}
    with NamedTemporaryFile() as buffer:
        fetch_asset(assets[f"{name}.sha256"], token, buffer)
        buffer.seek(0)
        lines = buffer.read().decode().splitlines()
    return {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}


def _download(assets, name, folder, token):
    """Download (or join the parts of) `name` into `folder`. Returns False if it is absent."""
    pieces = [name] if name in assets else sorted(n for n in assets if n.startswith(f"{name}.part-"))
    if not pieces:
        return False
    expected = _checksums(assets, name, token).get(name)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=folder, prefix=f".{name}.", delete=False) as output:
            temporary = Path(output.name)
            for piece in pieces:
                fetch_asset(assets[piece], token, output)
        if expected:
            digest = hashlib.sha256()
            with temporary.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"{name} failed its SHA-256 check; the download is corrupt.")
        os.replace(temporary, folder / name)
        temporary = None
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return True


def download_models(folder, repo_id=None, files=MODEL_FILES):
    """Download the listed files into `folder` unless they are already there.

    Returns the files now present. Files missing from the release are skipped, so a
    release holding only the 32M model still works.
    """
    repo_id = repo_id or model_repo()
    folder = Path(folder)
    missing = [name for name in files if not (folder / name).is_file()]
    if missing and repo_id:
        token = github_token()
        if not token:
            raise PermissionError(f"No GitHub token. Run `gh auth login` or set GITHUB_TOKEN "
                                  f"with read access to {repo_id}.")
        assets = release_assets(repo_id, token)
        for name in missing:
            _download(assets, name, folder, token)
    return [folder / name for name in files if (folder / name).is_file()]
