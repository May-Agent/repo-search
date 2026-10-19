"""Verification: ANE op coverage per function and cosine to the fp32 reference.

Cosines are computed on L2-normalised vectors. The fp32 reference is the stock transformers model output, masked mean pooled and
L2-normalized. The encoder's own fp32 output is reported too as a port-faithfulness
diagnostic (it should sit at ~1.0). Any miss fails the run.

Notes on the harness, both dictated by Core ML realities:
- MLComputePlan reports compute-device usage only for the compiled model's default
  function, so ``convert.compile_plan_variants`` produces one compiled copy of the
  multifunction model per function (identical programs and weights, differing only in
  the spec's defaultFunctionName) and this module counts ops on each variant.
- Every array handed to ``predict`` is a numpy-owned copy. A numpy view of a torch
  tensor (``tensor.numpy()``) makes the Core ML feature value free torch memory on a
  Core ML thread without the GIL, which segfaults the process.
"""

import gc
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from coremltools.models.compute_plan import MLComputePlan

from .encoder import Encoder
from .spec import BUCKETS, FUNCTION_PREFIX, MIN_ANE_OPS, MIN_COSINE, REPO_ID, REVISION


def _device_name(usage) -> str:
    return (
        type(usage.preferred_compute_device)
        .__name__.replace("ML", "")
        .replace("ComputeDevice", "")
    )


def _op_counts(plan, function: str) -> dict:
    ops = plan.model_structure.program.functions[function].block.operations
    counts = {}
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            if op.operator_name == "const":
                continue
            raise RuntimeError(f"no compute device usage for non-const op {op.operator_name}")
        name = _device_name(usage)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _unit(v: np.ndarray) -> np.ndarray:
    """Cosine needs both vectors normalised: the fp16 output is not exactly unit."""
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _torch_reference(lm, sample) -> np.ndarray:
    with torch.no_grad():
        hs = lm(
            input_ids=sample["input_ids"].long(),
            attention_mask=sample["attention_mask"].long(),
        ).last_hidden_state
        m = sample["attention_mask"].unsqueeze(-1).float()
        return F.normalize((hs * m).sum(1) / m.sum(1), dim=-1)[0].numpy()


def verify(
    mlmodelc: Path,
    plan_variants: dict[int, Path],
    lm,
    samples,
    timings: dict,
    work_dir: Path,
) -> dict:
    report = {
        "model": {"repo_id": REPO_ID, "revision": REVISION},
        "checks": {
            "min_ane_ops": MIN_ANE_OPS,
            "min_cosine": MIN_COSINE,
            "compute_units": "CPU_AND_NE",
            "op_count_note": "MLComputePlan reports usage only for the compiled "
            "model's default function; each function was counted on a variant "
            "compiled with it as default.",
        },
        "buckets": {},
    }
    failures = []

    for b in BUCKETS:
        fn = f"{FUNCTION_PREFIX}{b}"
        plan = MLComputePlan.load_from_path(
            path=str(plan_variants[b]), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        counts = _op_counts(plan, fn)
        del plan
        gc.collect()
        total = sum(counts.values())
        ane = counts.get("NeuralEngine", 0)
        print(f"{fn}: {ane}/{total} ops on NeuralEngine ({counts})", flush=True)
        if ane < MIN_ANE_OPS:
            failures.append(f"{fn}: {ane}/{total} ops on NeuralEngine < {MIN_ANE_OPS}")

        encoder = Encoder(lm, b).eval()
        cm = ct.models.CompiledMLModel(
            str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name=fn
        )
        entry = {
            "function": fn,
            "convert_seconds": timings[b],
            "ops_by_device_cpu_and_ne": counts,
            "total_ops": total,
            "samples": [],
        }
        # Feeds outlive the model: Core ML releases them on its own queue.
        keepalive = []
        for s in samples[b]:
            ref = _torch_reference(lm, s)
            feed = {
                # numpy-owned copies: views of torch tensors freed on a Core ML
                # thread segfault the process.
                "input_ids": np.array(s["input_ids"].numpy(), dtype=np.int32),
                "attention_mask": np.array(
                    s["attention_mask"].numpy(), dtype=np.float16
                ),
            }
            keepalive.append(feed)
            out = _unit(cm.predict(feed)["embedding"][0])
            cos = float(out @ ref)
            with torch.no_grad():
                enc_out = _unit(
                    encoder(s["input_ids"], s["attention_mask"])[0].numpy()
                )
            enc_cos = float(enc_out @ ref)
            sample_entry = {
                "file": s["file"],
                "tokens": s["tokens"],
                "cosine": cos,
                "encoder_fp32_vs_transformers": enc_cos,
            }
            if cos < MIN_COSINE:
                sample_entry["text"] = s["text"]
                failures.append(
                    f"{fn}: cosine {cos} < {MIN_COSINE} for {s['file']} ({s['tokens']} tokens)"
                )
            entry["samples"].append(sample_entry)
        entry["min_cosine"] = min(s["cosine"] for s in entry["samples"])
        del encoder, cm, keepalive
        gc.collect()
        report["buckets"][str(b)] = entry

    report["passed"] = not failures
    report["failures"] = failures
    out_path = work_dir / "verification.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)
    if failures:
        raise RuntimeError("verification failed: " + "; ".join(failures))
    return report
