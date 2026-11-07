"""Build the off-macOS ONNX archive: fetch the pinned fp16 ONNX build,
rewrite its MatMul -> Split pairs (see `onnx_unsplit`), check the rewritten
graph against the source on the fixed sample, and package it.
"""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import zstandard
from huggingface_hub import hf_hub_download

from .fetch import sha256_file
from .onnx_unsplit import unsplit
from .package import write_archive
from .samples import SampleTokenizer, fixed_samples
from .spec import (
    BUCKETS,
    ONNX_MIN_COSINE,
    ONNX_SOURCE_FILE,
    ONNX_SOURCE_MEMBERS,
    ONNX_SOURCE_REPO,
    ONNX_SOURCE_REVISION,
    ONNX_SOURCE_SHA256,
    SHORT_REVISION,
)


def _fetch_source(work_dir: Path) -> Path:
    archive = Path(
        hf_hub_download(
            repo_id=ONNX_SOURCE_REPO,
            filename=ONNX_SOURCE_FILE,
            revision=ONNX_SOURCE_REVISION,
            local_dir=work_dir / "onnx-source",
        )
    )
    digest = sha256_file(archive)
    if digest != ONNX_SOURCE_SHA256:
        raise RuntimeError(f"{archive}: sha256 {digest} != pinned {ONNX_SOURCE_SHA256}")
    dest = work_dir / "onnx-source" / "model"
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    with (
        archive.open("rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as zstd,
        tarfile.open(fileobj=zstd, mode="r|") as tar,
    ):
        for member in tar:
            names.append(member.name)
            tar.extract(member, dest, filter="data")
    if sorted(names) != sorted(ONNX_SOURCE_MEMBERS):
        raise RuntimeError(f"{archive}: members {names} != {list(ONNX_SOURCE_MEMBERS)}")
    return dest


def _embed(session: ort.InferenceSession, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feeds = {"input_ids": ids, "attention_mask": mask}
    for i in session.get_inputs():
        if i.name not in feeds:
            feeds[i.name] = np.zeros((0, i.shape[-1]), dtype=np.float32)
    out = session.run(["sentence_embedding"], feeds)[0].astype(np.float64)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _verify(source: Path, rewritten: Path, tokenizer: Path) -> float:
    """The lowest cosine between the two graphs' embeddings over the fixed
    sample, each bucket's documents run as one padded batch."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    a = ort.InferenceSession(str(source), options, providers=providers)
    b = ort.InferenceSession(str(rewritten), options, providers=providers)
    samples = fixed_samples(SampleTokenizer(tokenizer), BUCKETS)
    worst = 1.0
    for bucket, docs in samples.items():
        ids = np.concatenate([d["input_ids"].numpy() for d in docs]).astype(np.int64)
        mask = np.concatenate([d["attention_mask"].numpy() for d in docs]).astype(np.int64)
        cos = (_embed(a, ids, mask) * _embed(b, ids, mask)).sum(axis=1)
        print(f"bucket {bucket}: min cosine {cos.min():.9f}", flush=True)
        worst = min(worst, float(cos.min()))
    if worst < ONNX_MIN_COSINE:
        raise RuntimeError(f"rewritten graph diverges: min cosine {worst} < {ONNX_MIN_COSINE}")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(prog="repo-search-convert-onnx")
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    source_dir = _fetch_source(work_dir)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    graph_name = manifest["model"]
    model = onnx.load(str(source_dir / graph_name))
    stats = unsplit(model)
    print(f"unsplit: {stats}", flush=True)

    out_dir = work_dir / "onnx"
    out_dir.mkdir(exist_ok=True)
    graph = out_dir / graph_name
    data = out_dir / f"{graph_name}_data"
    data.unlink(missing_ok=True)
    onnx.save(
        model,
        str(graph),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data.name,
        size_threshold=1024,
    )
    del model
    worst = _verify(source_dir / graph_name, graph, source_dir / "tokenizer.json")

    manifest["model_id"] = f"onnx-{sha256_file(graph)}"
    staging = work_dir / "eg2-onnx-fp16.tar.zst"
    digest = write_archive(
        staging,
        [
            ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()),
            ("tokenizer.json", source_dir / "tokenizer.json"),
            (graph.name, graph),
            (data.name, data),
        ],
    )
    out = work_dir / f"eg2-onnx-fp16-{SHORT_REVISION}-{digest[:8]}.tar.zst"
    staging.rename(out)
    staging.with_suffix(staging.suffix + ".sha256").unlink()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(
        f"archive {out.name} size={out.stat().st_size} sha256={digest} "
        f"model_id={manifest['model_id']} min_cosine={worst:.9f}",
        flush=True,
    )
