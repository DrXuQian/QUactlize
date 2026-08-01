"""Build quactlize's extension.

TWO HALVES, AND THEY BUILD DIFFERENTLY. The preprocessing half (csrc/preprocess) is pure host C++ -- zero __global__,
zero device intrinsics -- so it compiles with an ordinary C++ compiler anywhere torch is installed, including a
developer machine with no PPU. The kernel half is device code that only hgcc can compile, through actlize's
PPUToolchain.

So the default build produces the HOST half alone. That is not a placeholder: it is the half that owns the physical
weight layout, it is what an importer needs, and being able to build and test it without the accelerator is the
difference between a format's reorder being checkable in CI and only on the box.

    pip install -e .                    host half only (works without the PPU SDK)
    QUACTLIZE_WITH_DEVICE=1 pip install -e .    also build the kernels (needs PPU_SDK_ROOT and hgcc)

The device half is not wired here yet; setting the variable reports that rather than silently building half a library
and letting the missing ops surface as an AttributeError at the first call.
"""
import os, sys
from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent.resolve()
WITH_DEVICE = os.environ.get("QUACTLIZE_WITH_DEVICE", "0") not in ("0", "", "false", "False")

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError:
    sys.exit("quactlize's extension needs torch installed to build")

if WITH_DEVICE:
    sys.exit("QUACTLIZE_WITH_DEVICE is not implemented yet: the kernels are built through actlize's PPUToolchain by "
             "build.sh, and moving them into this extension is the open structural work. See docs/CHECKPOINT.md.")

# HEADERS ARE NOT TRACKED BY setuptools. Editing weight_layout.h and rebuilding produced a .so with the OLD table
# and a test failure that read like a logic error. `depends` makes the headers part of the extension's dependency
# set so a change to one forces a recompile.
HEADERS = sorted(str(p) for p in (ROOT / "quactlize/csrc/preprocess").rglob("*.h"))

cuda_inc = [p for p in ("/usr/local/cuda/include", os.environ.get("CUDA_HOME", "") + "/include") if p and Path(p).exists()]

ext = CppExtension(
    name="quactlize._C",
    sources=[
        "quactlize/csrc/preprocess/cutlass_kernels/cutlass_preprocessors.cpp",
        "quactlize/csrc/preprocess/thop/weight_preprocess_ops.cpp",
    ],
    include_dirs=[
        str(ROOT / "quactlize/csrc/preprocess"),
        str(ROOT / "third_party/cutlass/include"),
        *cuda_inc,
    ],
    # USE_AIU compiles in the PPU column interleave. Without it the whole AIU branch vanishes from the preprocessor
    # and preprocess_weights_for_mixed_gemm(..., use_aiu_interleaved=true) would return the ordinary layout; the op now
    # refuses that request rather than returning wrongly-ordered bytes, but the flag is what makes it answerable.
    define_macros=[("USE_AIU", "1")],
    depends=HEADERS,
    extra_compile_args=["-std=c++17", "-O2", "-Wno-unused-function", "-Wno-sign-compare"],
)

setup(
    name="quactlize",
    version="0.1.0",
    packages=["quactlize"],
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
