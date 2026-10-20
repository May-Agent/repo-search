"""Emit Core ML reference embeddings for the fixed sample set as JSON.

The Rust embedder's parity test runs this module and compares vectors
(cosine >= 0.9999). Everything handed to ``predict`` is a numpy-owned copy kept
alive past the call: Core ML releases inputs on its own dispatch queue (see
verify.py for the off-GIL decref crash this avoids).
"""

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
from tokenizers import Tokenizer

from .samples import fixed_samples
from .spec import BUCKETS, DOCUMENT_FORMAT, FUNCTION_PREFIX, QUERY_FORMAT

QUERIES = [
    "where does the daemon mount the repo index",
    "function that merges bm25 and knn scores",
    "retry policy for a locked database",
    # Over the largest bucket: truncation must keep the post-processed <eos>.
    "how does the indexer schedule embedding work across every repository "
    * 60,
]


class _Tokenizer:
    """The HF-shaped interface ``fixed_samples`` wants, over tokenizer.json."""

    def __init__(self, path: Path, truncate: int | None = None):
        self._t = Tokenizer.from_file(str(path))
        if truncate is not None:
            self._t.enable_truncation(truncate)
        self.pad_token_id = self._t.token_to_id("<pad>")
        if self.pad_token_id is None:
            raise RuntimeError("tokenizer has no <pad> token")

    def __call__(self, text: str) -> dict:
        return {"input_ids": self._t.encode(text).ids}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._t.decode(list(ids), skip_special_tokens=skip_special_tokens)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def main() -> None:
    parser = argparse.ArgumentParser(prog="repo_search_convert.reference")
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir
    mlmodelc = next(model_dir.glob("*.mlmodelc"), None)
    if mlmodelc is None:
        raise RuntimeError(f"no .mlmodelc in {model_dir}")

    tok = _Tokenizer(model_dir / "tokenizer.json")
    pad = tok.pad_token_id
    samples = fixed_samples(tok, BUCKETS)
    keepalive = []
    models = {}
    out = {"samples": [], "documents": [], "queries": []}

    def embed_ids(ids: list[int]) -> list[float]:
        b = min(b for b in BUCKETS if len(ids) <= b)
        if b not in models:
            models[b] = ct.models.CompiledMLModel(
                str(mlmodelc),
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                function_name=f"{FUNCTION_PREFIX}{b}",
            )
        feed = {
            "input_ids": np.array([ids + [pad] * (b - len(ids))], dtype=np.int32),
            "attention_mask": np.array(
                [[1] * len(ids) + [0] * (b - len(ids))], dtype=np.float16
            ),
        }
        keepalive.append(feed)
        return _unit(models[b].predict(feed)["embedding"][0]).tolist()

    for b in sorted(BUCKETS):
        for s in samples[b]:
            ids = s["input_ids"][0, : s["tokens"]].tolist()
            out["samples"].append(
                {
                    "bucket": b,
                    "file": s["file"],
                    "tokens": s["tokens"],
                    "input_ids": ids,
                    "embedding": embed_ids(ids),
                }
            )
            # embed_document path: the same decoded window re-formatted through
            # the document prompt.
            prefix = DOCUMENT_FORMAT.format(title=s["file"], text="")
            window_text = s["text"][len(prefix) :]
            doc_ids = tok(
                DOCUMENT_FORMAT.format(title=s["file"], text=window_text)
            )["input_ids"]
            if len(doc_ids) <= max(BUCKETS):
                out["documents"].append(
                    {
                        "title": s["file"],
                        "text": window_text,
                        "embedding": embed_ids(doc_ids),
                    }
                )
    # Queries truncate at the largest bucket inside the tokenizer, the same
    # path the Rust query tokenizer takes.
    qtok = _Tokenizer(model_dir / "tokenizer.json", truncate=max(BUCKETS))
    for q in QUERIES:
        prompt = QUERY_FORMAT.format(query=q)
        ids = qtok(prompt)["input_ids"]
        out["queries"].append(
            {
                "query": q,
                "tokens": len(tok(prompt)["input_ids"]),
                "embedding": embed_ids(ids),
            }
        )

    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()