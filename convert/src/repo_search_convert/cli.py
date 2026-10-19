"""Pipeline entry point: fetch, convert, verify, package."""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2Model,
)

from .convert import compile_plan_variants, convert_all
from .fetch import fetch_weights
from .package import build_archive
from .samples import fixed_samples
from .spec import BUCKETS
from .verify import verify


def main() -> None:
    parser = argparse.ArgumentParser(prog="repo-search-convert")
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    build_dir = work_dir / "build"
    build_dir.mkdir(exist_ok=True)

    hf_dir = fetch_weights(work_dir)
    tok = AutoTokenizer.from_pretrained(hf_dir)
    full = EmbeddingGemma2Model.from_pretrained(
        hf_dir, attn_implementation="eager", dtype=torch.float32
    )
    lm = full.language_model.eval()
    del full
    gc.collect()

    samples = fixed_samples(tok, BUCKETS)
    mlpackage, mlmodelc, timings = convert_all(lm, samples, build_dir)
    plan_variants = compile_plan_variants(mlpackage, build_dir)
    verify(mlmodelc, plan_variants, lm, samples, timings, work_dir)
    build_archive(work_dir, mlmodelc, hf_dir)