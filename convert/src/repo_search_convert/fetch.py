"""Download the model files at the pinned revision and verify them against the
committed manifest.

LFS files carry a sha256 in Hub metadata; small files are plain git blobs and the Hub
publishes only their blob oid, so those are verified as git blob sha1
(sha1 of "blob <size>\\0" + contents). Either way the digest is checked against Hub
metadata for the pinned revision, never against a prior local copy.
"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

_MANIFEST = Path(__file__).resolve().parents[2] / "manifest.json"
_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path, size: int) -> str:
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(work_dir: Path) -> Path:
    """Download every file in the committed manifest into work_dir/hf and verify it."""
    manifest = json.loads(_MANIFEST.read_text())
    repo_id = manifest["repo_id"]
    revision = manifest["revision"]
    dest = work_dir / "hf"
    dest.mkdir(parents=True, exist_ok=True)
    for name, meta in manifest["files"].items():
        path = Path(
            hf_hub_download(repo_id=repo_id, filename=name, revision=revision, local_dir=dest)
        )
        size = path.stat().st_size
        if size != meta["size"]:
            raise RuntimeError(f"{name}: size {size} != manifest {meta['size']}")
        if "sha256" in meta:
            digest = _sha256(path)
            if digest != meta["sha256"]:
                raise RuntimeError(f"{name}: sha256 {digest} != manifest {meta['sha256']}")
        elif "git_blob_sha1" in meta:
            digest = _git_blob_sha1(path, size)
            if digest != meta["git_blob_sha1"]:
                raise RuntimeError(
                    f"{name}: git blob sha1 {digest} != manifest {meta['git_blob_sha1']}"
                )
        else:
            raise RuntimeError(f"{name}: manifest entry has no hash")
        print(f"fetched {name} ({size} bytes, hash ok)", flush=True)
    return dest
