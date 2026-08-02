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
# EVERY HEADER THE EXTENSION COMPILES, not just the .h files under csrc. This list is passed as `depends`, so what
# it misses is what setuptools will not rebuild for -- and it used to miss both the .hpp extension and the whole of
# quactlize/include, which is where the pre-pass arithmetic under test actually lives. The failure that produces is a
# golden suite reporting green against a stale .so, i.e. the oracle certifying code nobody is running.
HEADERS = sorted(str(p) for d in ("quactlize/csrc/preprocess", "quactlize/include")
                 for ext in ("*.h", "*.hpp", "*.cuh")
                 for p in (ROOT / d).rglob(ext))

cuda_inc = [p for p in ("/usr/local/cuda/include", os.environ.get("CUDA_HOME", "") + "/include") if p and Path(p).exists()]

ext = CppExtension(
    name="quactlize._C",
    sources=[
        "quactlize/csrc/preprocess/cutlass_kernels/cutlass_preprocessors.cpp",
        "quactlize/csrc/preprocess/thop/weight_preprocess_ops.cpp",
        # THE ONLINE SCALE PRE-PASS. Host-only and portable: gguf_scale_decode.hpp's packed-unit
        # re-exports are behind __has_include, so this builds against stock cutlass with no PPU SDK.
        "quactlize/csrc/preprocess/thop/gguf_prepass_ops.cpp",
    ],
    include_dirs=[
        str(ROOT / "quactlize/csrc/preprocess"),
        str(ROOT / "quactlize/include"),
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
