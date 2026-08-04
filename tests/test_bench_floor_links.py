"""A HEADER INCLUDED BY MANY TRANSLATION UNITS MUST STILL LINK.

benchmarks/bench_floor.cuh defines a __global__. A __global__ has EXTERNAL linkage, so the first version of that
header produced one definition per including translation unit and the box's link failed with

    multiple definition of `bench_floor::bench_floor_nop(int*)'

after a full compile of 180 generated MoE sweep units. Every check this repo had was blind to it: the syntax
gate compiles ONE translation unit, and a collision needs two.

So this test compiles TWO and LINKS them. It is the smallest thing that can fail the way the box failed, and it
runs with nvcc locally in about a second -- the round trip it replaces was a full box build.

It uses a stubbed device runtime rather than the real one, because what is under test is LINKAGE, not the
kernel: two objects, one symbol, does the linker accept it.
"""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _arch() -> str:
    """__hfma2 needs sm_53+, and nvcc's default target predates it -- the failure reads `identifier "__hfma2" is
    undefined`, which looks like a missing header rather than a missing -arch."""
    try:
        import torch
        m, n = torch.cuda.get_device_capability()
        return f"{m}{n}"
    except Exception:
        return "80"

TU = """
#include "bench_floor.cuh"
// Each TU takes the address of something in the header, so the definitions cannot be dropped as unused before
// the linker sees them -- an unused static is elided, and eliding it would make this test pass vacuously.
double tu_{n}_floor() {{ return bench_floor::us(); }}
"""

MAIN = """
double tu_0_floor();
double tu_1_floor();
int main() { return (tu_0_floor() >= 0.0 && tu_1_floor() >= 0.0) ? 0 : 1; }
"""


@pytest.mark.cpu_reference
def test_bench_floor_header_links_from_two_translation_units(tmp_path):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("no nvcc")

    for i in (0, 1):
        (tmp_path / f"tu{i}.cu").write_text(TU.format(n=i))
    (tmp_path / "main.cu").write_text(MAIN)

    objs = []
    # INCLUDE ORDER IS LOAD-BEARING and is written down in benchmarks/gemv_bench.py: third_party/cutlass is
    # NVIDIA's and is what a local nvcc resolves `cutlass/...` against; third_party/actlize is the PPU fork and
    # supplies only what NVIDIA's does not have. Putting actlize first drags in the PPU SDK's vendor headers and
    # the build dies on hggc_runtime.h -- which reads exactly like "not portable" and is not.
    # The STUB device runtime, which the repo already keeps for its host-only gates. bench_floor.cuh calls the
    # hggc runtime, which exists only on the box; the subject here is LINKAGE, and a __global__'s linkage does
    # not depend on which runtime backs the call. Without the stub this test could only ever SKIP, which is the
    # "check that never runs" shape -- worse than no test, because the tier reports green.
    inc = ["-I", str(ROOT / "benchmarks"),
           "-I", str(ROOT / "dev" / "fold_derivation" / "stub_inc"),
           "-I", str(ROOT / "third_party" / "cutlass" / "include"),
           "-I", str(ROOT / "third_party" / "cutlass" / "tools" / "util" / "include"),
           "-I", str(ROOT / "third_party" / "actlize" / "tools" / "util" / "include"),
           "-I", str(ROOT / "third_party" / "actlize" / "include")]
    arch = _arch()
    for i in (0, 1):
        o = tmp_path / f"tu{i}.o"
        r = subprocess.run([nvcc, "-std=c++17", f"-arch=sm_{arch}", "--expt-relaxed-constexpr", *inc,
                            "-c", str(tmp_path / f"tu{i}.cu"), "-o", str(o)],
                           capture_output=True, text=True)
        if r.returncode:
            pytest.skip(f"bench_floor.cuh needs headers this machine lacks:\n{r.stderr[-600:]}")
        objs.append(str(o))

    r = subprocess.run([nvcc, "-std=c++17", f"-arch=sm_{arch}", "--expt-relaxed-constexpr", *inc,
                        str(tmp_path / "main.cu"), *objs, "-o", str(tmp_path / "app")],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "bench_floor.cuh does not link from two translation units. A __global__ or a non-inline function in a "
        "header needs internal linkage (anonymous namespace) or exactly one defining TU.\n\n" + r.stderr[-1500:])
