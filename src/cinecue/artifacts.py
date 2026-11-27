"""Persist locally generated recommendation bundles across server restarts."""

import os
import pickle
from tempfile import NamedTemporaryFile

VERSION = 1


def save_bundle(path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary_path = output.name
            pickle.dump({"version": VERSION, "bundle": bundle}, output, protocol=5)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_bundle(path):
    # Only open this app's local artifact. Pickle files from unknown sources can execute code.
    with path.open("rb") as source:
        artifact = pickle.load(source)
    if artifact.get("version") != VERSION:
        raise ValueError("Saved model format is unsupported. Rebuild with scripts/save_models.py.")
    return artifact["bundle"]
