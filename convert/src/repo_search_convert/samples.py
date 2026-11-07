"""The fixed verification sample.

Four documents per bucket, each `title: <file name> | text: <window>` where the window
is a prefix of the document's token stream cut at a target length. Targets spread
evenly across (L/2, L] — (1, 128] for the smallest bucket. Source text is this
package's own committed Python, so the sample is reproducible without a fixture file.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from .spec import DOCUMENT_FORMAT, SAMPLES_PER_BUCKET


class SampleTokenizer:
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


def _doc_ids(tok, path: Path) -> list[int]:
    doc = DOCUMENT_FORMAT.format(title=path.name, text=path.read_text())
    return tok(doc)["input_ids"]


def fixed_samples(tok, buckets) -> dict[int, list[dict]]:
    files = sorted(Path(__file__).parent.glob("*.py"))
    docs = [(f.name, _doc_ids(tok, f)) for f in files]
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad_token_id")

    samples = {}
    for b in buckets:
        lo = 1 if b <= 128 else b // 2
        targets = [lo + (b - lo) * (j + 1) // SAMPLES_PER_BUCKET for j in range(SAMPLES_PER_BUCKET)]
        bucket_samples = []
        for j, n in enumerate(targets):
            candidates = [d for d in docs if len(d[1]) >= n]
            if not candidates:
                raise RuntimeError(
                    f"no committed source file tokenizes to >= {n} tokens for bucket {b}"
                )
            name, ids = candidates[j % len(candidates)]
            window = ids[:n]
            bucket_samples.append(
                {
                    "file": name,
                    "tokens": n,
                    "text": DOCUMENT_FORMAT.format(
                        title=name, text=tok.decode(window, skip_special_tokens=True)
                    ),
                    "input_ids": torch.tensor(
                        [window + [pad_id] * (b - n)], dtype=torch.int32
                    ),
                    "attention_mask": torch.tensor(
                        [[1] * n + [0] * (b - n)], dtype=torch.float32
                    ),
                }
            )
        samples[b] = bucket_samples
    return samples