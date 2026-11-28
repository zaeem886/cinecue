"""Deploy the CineCue website to a private Hugging Face Docker Space.

    pip install huggingface_hub                     # only this script needs it
    hf auth login
    python scripts/publish_models.py                # models -> private GitHub release
    python scripts/deploy_space.py --space <organisation>/cinecue --github-token <token>

Uploads only what the website needs: app.py, src/cinecue/, .streamlit/config.toml,
requirements-app.txt, the results summaries (*.json), results/posters.csv, and the
Dockerfile and Space card from space/. No ratings are uploaded; the website downloads the
models at startup from the private GitHub repository (see src/cinecue/hub.py).

The Space reads that repository with a GitHub token stored as its GITHUB_TOKEN secret. Use a
fine-grained token limited to the models repository with read-only Contents access
(https://github.com/settings/personal-access-tokens/new). Pass it with --github-token or the
CINECUE_SPACE_GITHUB_TOKEN environment variable. Your own `gh` login token is deliberately
not used: it can write to every repository you can.
"""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cinecue.hub import DEFAULT_REPO  # noqa: E402

INCLUDE = ["app.py", "requirements-app.txt", ".streamlit/config.toml", "src/cinecue/*.py",
           "results/*.json", "results/posters.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="<organisation>/cinecue")
    parser.add_argument("--models", default=DEFAULT_REPO, help=f"GitHub owner/name (default {DEFAULT_REPO})")
    parser.add_argument("--github-token", default=os.environ.get("CINECUE_SPACE_GITHUB_TOKEN"),
                        help="read-only token for the models repository (stored as the Space secret GITHUB_TOKEN)")
    args = parser.parse_args()
    if not args.github_token:
        raise SystemExit("Pass --github-token (a read-only fine-grained token for the models repository).")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("This script needs huggingface_hub: pip install huggingface_hub")

    api = HfApi()
    print(f"Logged in to Hugging Face as {api.whoami()['name']}")
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    if not api.repo_info(args.space, repo_type="space").private:
        raise SystemExit(f"{args.space} is public. Make it private first.")
    api.add_space_secret(args.space, "GITHUB_TOKEN", args.github_token,
                         description="Reads the private CineCue models repository on GitHub")
    api.add_space_variable(args.space, "CINECUE_MODEL_REPO", args.models)
    for name in ["Dockerfile", "README.md"]:
        api.upload_file(path_or_fileobj=ROOT / "space" / name, path_in_repo=name, repo_id=args.space,
                        repo_type="space", commit_message=f"Update {name}")
    api.upload_folder(folder_path=ROOT, repo_id=args.space, repo_type="space", allow_patterns=INCLUDE,
                      commit_message="Deploy CineCue website")
    print(f"Deployed: https://huggingface.co/spaces/{args.space} (private; first start takes a few minutes)")


if __name__ == "__main__":
    main()
