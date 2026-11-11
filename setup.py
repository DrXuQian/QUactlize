"""Legacy host extension or a complete precompiled PPU wheel.

Use bash build.sh --wheel for releases. QUACTLIZE_WITH_DEVICE=1 also supports
PEP 517 builds, with QUACTLIZE_PPU_BUNDLE for verified reuse. Without that flag,
the existing host-only build is unchanged. QUACTLIZE_WITH_HOST=1 adds the legacy
extension to a PPU wheel using installed platform Torch. Never fetch Torch here.
"""
import os, sys
from pathlib import Path
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).parent.resolve()
WITH_DEVICE = os.environ.get("QUACTLIZE_WITH_DEVICE", "0") not in ("0", "", "false", "False")
WITH_HOST = not WITH_DEVICE or os.environ.get("QUACTLIZE_WITH_HOST", "0") == "1"


class PpuBuildPy(build_py):
    def run(self):
        import json
        import shutil
        import tempfile
        from tools.wheel_support import wheel_metadata
        from tools.build_ppu_library import build

        super().run()
        supplied = os.environ.get('QUACTLIZE_PPU_BUNDLE')
        with tempfile.TemporaryDirectory(prefix='quactlize-wheel-') as temporary:
            bundle = Path(supplied).resolve(strict=True) if supplied else Path(temporary) / 'package'
            if not supplied:
                build(Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')), bundle,
                      int(os.environ.get('JOBS', '4')))
            metadata = wheel_metadata(bundle, ROOT)
            destination = Path(self.build_lib) / 'quactlize/_native'
            destination.mkdir(parents=True, exist_ok=False)
            manifest = json.loads((bundle / 'manifest.json').read_text())
            for name in ['manifest.json', *manifest['payloads']]:
                out = destination / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bundle / name, out)
            (destination / 'wheel.json').write_text(json.dumps(metadata, indent=2) + '\n')


class PpuWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        python, abi, platform = super().get_tag()
        return (python, abi, platform) if WITH_HOST else ('py3', 'none', platform)


class PpuDistribution(Distribution):
    def has_ext_modules(self):
        # Precompiled C-ABI DSOs are platform libraries, even without a CPython extension.
        return True


if WITH_DEVICE and not WITH_HOST:
    setup(ext_modules=[], distclass=PpuDistribution,
          cmdclass={'build_py': PpuBuildPy, 'bdist_wheel': PpuWheel})
    sys.exit(0)

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError:
    sys.exit("quactlize's extension needs torch installed to build")

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
        # dlopen seam to the hgcc-built device .so; host-only, links nothing but libdl.
        "quactlize/csrc/preprocess/thop/ppu_backend.cpp",
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
    libraries=["dl"],
)

setup(
    name="quactlize",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension, **({'build_py': PpuBuildPy, 'bdist_wheel': PpuWheel} if WITH_DEVICE else {})},
    python_requires=">=3.9",
)
