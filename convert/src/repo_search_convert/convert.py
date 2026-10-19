"""Trace the encoder at each bucket length, convert to fp16 ML programs, merge into
one multifunction package (weights stored once), and compile to .mlmodelc.
"""

import gc
import shutil
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.proto import Model_pb2

from .encoder import Encoder
from .spec import BUCKETS, DEFAULT_FUNCTION, FUNCTION_PREFIX


def _rewrite_spec_deterministic(mlpackage: Path, default_function: str | None = None) -> None:
    """Re-serialise the package spec with deterministic map ordering.

    protobuf randomises map order per process, and the compiled model embeds
    base64(sha256(model.mlmodel)), so a non-deterministic spec makes the packaged
    archive differ between runs. ``default_function`` also sets the default function.
    """
    spec_path = mlpackage / "Data/com.apple.CoreML/model.mlmodel"
    spec = Model_pb2.Model()
    spec.ParseFromString(spec_path.read_bytes())
    if default_function is not None:
        spec.description.defaultFunctionName = default_function
    spec_path.write_bytes(spec.SerializeToString(deterministic=True))


def convert_all(lm, samples, build_dir: Path) -> tuple[Path, Path, dict]:
    """Convert every bucket, merge into the multifunction package, compile it.

    Returns (mlpackage path, mlmodelc path, per-bucket conversion timings). One bucket
    at a time: each traced graph and converted model is freed before the next.
    """
    pkg_paths = {}
    timings = {}
    for b in BUCKETS:
        model = Encoder(lm, b).eval()
        s0 = samples[b][0]
        t0 = time.time()
        with torch.no_grad():
            traced = torch.jit.trace(
                model, (s0["input_ids"].int(), s0["attention_mask"].float())
            )
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, b), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, b), dtype=np.float16),
            ],
            outputs=[ct.TensorType(name="embedding", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        pkg = build_dir / f"eg2_{FUNCTION_PREFIX}{b}.mlpackage"
        mlmodel.save(str(pkg))
        pkg_paths[b] = pkg
        timings[b] = time.time() - t0
        del model, traced, mlmodel
        gc.collect()
        print(f"converted bucket {b} in {timings[b]:.1f}s", flush=True)

    desc = ct.utils.MultiFunctionDescriptor()
    for b in BUCKETS:
        desc.add_function(str(pkg_paths[b]), "main", f"{FUNCTION_PREFIX}{b}")
    desc.default_function_name = DEFAULT_FUNCTION
    mlpackage = build_dir / "eg2-coreml.mlpackage"
    ct.utils.save_multifunction(desc, str(mlpackage))
    _rewrite_spec_deterministic(mlpackage)

    mlmodelc = build_dir / "eg2-coreml.mlmodelc"
    ct.utils.compile_model(str(mlpackage), str(mlmodelc))
    print(f"compiled {mlmodelc}", flush=True)
    return mlpackage, mlmodelc, timings


def compile_plan_variants(mlpackage: Path, build_dir: Path) -> dict[int, Path]:
    """One compiled copy per function with that function as the default.

    MLComputePlan reports device usage only for the compiled model's default
    function, so verifying per-function op placement on the multifunction model
    needs one variant per function that differs solely in
    ``description.defaultFunctionName``. The function programs and the shared
    weight blob are identical across variants.
    """
    paths = {}
    for b in BUCKETS:
        fn = f"{FUNCTION_PREFIX}{b}"
        if fn == DEFAULT_FUNCTION:
            paths[b] = build_dir / "eg2-coreml.mlmodelc"
            continue
        variant = build_dir / f"plan_{fn}.mlpackage"
        shutil.copytree(mlpackage, variant)
        _rewrite_spec_deterministic(variant, fn)
        compiled = build_dir / f"plan_{fn}.mlmodelc"
        ct.utils.compile_model(str(variant), str(compiled))
        paths[b] = compiled
        print(f"compiled plan variant {fn}", flush=True)
    return paths