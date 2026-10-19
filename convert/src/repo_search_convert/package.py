"""Deterministic tar.zst packaging of the compiled model and its metadata.

Entries are sorted by name and stamped with fixed mtime/uid/gid/mode, so two runs over
the same compiled model produce identical bytes.
"""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import zstandard

from .spec import (
    BUCKETS,
    DOCUMENT_FORMAT,
    EMBEDDING_DIM,
    FUNCTION_PREFIX,
    DEFAULT_FUNCTION,
    QUERY_FORMAT,
    REPO_ID,
    REVISION,
    SHORT_REVISION,
)

LICENSE_SRC = Path(__file__).resolve().parents[3] / "LICENSE-APACHE"

NOTICE = f"""EmbeddingGemma 2, by Google DeepMind.
Source: https://huggingface.co/{REPO_ID}
Pinned revision: {REVISION}
Licensed under the Apache License, Version 2.0 (see LICENSE).
"""

ARCHIVE_MANIFEST = {
    "model": REPO_ID,
    "revision": REVISION,
    "functions": {f"{FUNCTION_PREFIX}{b}": b for b in BUCKETS},
    "default_function": DEFAULT_FUNCTION,
    "inputs": {
        "input_ids": {"dtype": "int32", "shape": [1, "length"]},
        "attention_mask": {"dtype": "float16", "shape": [1, "length"]},
    },
    "outputs": {"embedding": {"dtype": "float16", "shape": [1, EMBEDDING_DIM]}},
    "embedding_dim": EMBEDDING_DIM,
    "prompts": {"document": DOCUMENT_FORMAT, "query": QUERY_FORMAT},
}


def _info(name: str, size: int, dir_: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if dir_:
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.mode = 0o644
        info.size = size
    return info


def _iter_mlmodelc(mlmodelc: Path) -> list[tuple[str, Path | None]]:
    """(archive name, fs path or None for a directory) entries, sorted."""
    entries = [(mlmodelc.name + "/", None)]
    for p in sorted(mlmodelc.rglob("*")):
        rel = p.relative_to(mlmodelc.parent)
        if p.is_dir():
            entries.append((f"{rel}/", None))
        elif p.is_file():
            entries.append((str(rel), p))
        else:
            raise RuntimeError(f"unsupported entry in mlmodelc (not a file or directory): {p}")
    return entries


def build_archive(work_dir: Path, mlmodelc: Path, hf_dir: Path) -> Path:
    if not LICENSE_SRC.is_file():
        raise RuntimeError(f"LICENSE-APACHE not found at {LICENSE_SRC}")
    out = work_dir / f"eg2-coreml-{SHORT_REVISION}.tar.zst"

    static_files = [
        ("tokenizer.json", hf_dir / "tokenizer.json"),
        ("LICENSE", LICENSE_SRC),
    ]
    static_bytes = [
        ("NOTICE", NOTICE.encode()),
        ("manifest.json", (json.dumps(ARCHIVE_MANIFEST, indent=2) + "\n").encode()),
    ]

    entries: list[tuple[str, object]] = _iter_mlmodelc(mlmodelc)
    entries += [(n, p) for n, p in static_files]
    entries += [(n, b) for n, b in static_bytes]
    entries.sort(key=lambda e: e[0])

    with out.open("wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw) as zstd:
            with tarfile.open(fileobj=zstd, mode="w|") as tar:
                for name, payload in entries:
                    if payload is None:
                        tar.addfile(_info(name, 0, dir_=True))
                    elif isinstance(payload, bytes):
                        tar.addfile(_info(name, len(payload)), io.BytesIO(payload))
                    else:
                        with Path(payload).open("rb") as f:
                            tar.addfile(_info(name, Path(payload).stat().st_size), f)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"wrote {out} sha256={digest}", flush=True)
    return out
