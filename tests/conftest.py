"""WHICH ARM A TEST RUNS AGAINST IS PART OF WHAT IT TESTS, and until this file existed nothing enforced that.

THE FAILURE THIS COMES FROM. The first run of the python tier on ppu001 reported 17 failures. Sixteen of them were
one cause: four test functions that exist to validate the CPU REFERENCE ARM were silently handed the device arm,
because quactlize dlopens libquactlize_ppu.so when it is available and every one of these tests just calls the op.

They did not fail in a way that said so. test_vecdot_matches_llama_cpp came back at relative error 1.5e+00 and
test_native_gemv_matches_the_oracle at 1.5e-03 against a 1e-6 bound, which reads as "the kernel is wrong" when what
actually happened is that a test whose own comment says it "resolves to the CPU arm deliberately" did not.

Only ONE of the four said it out loud -- test_cuda_vecdot_cooperative_matches_cpu_reference asserts
gguf_backend().startswith("cpu"). That assert was right and the other three lacked it. This hook is that assert,
applied by marker, so a new test joins the family by declaring it rather than by remembering to copy a line.

WHY A HARD FAILURE AND NOT A SKIP. A skip here would be indistinguishable from "the CPU arm is fine", and the CPU
arm is the ORACLE these tests establish -- skipping it on the one machine that has a device is exactly backwards.
The message names the command instead.
"""
import os

import pytest


def _backend():
    """None when quactlize cannot be imported at all -- then the test's own importorskip handles it."""
    try:
        import quactlize
        return quactlize.gguf_backend()
    except Exception:
        return None


# pytest_runtest_setup, NOT pytest_collection_modifyitems + usefixtures. The first version of this file used the
# latter and THE CHECK NEVER RAN: by the time collection_modifyitems fires, each item's fixture closure is already
# computed, so a usefixtures marker added there is inert. It was caught by planting a fake "ppu" backend and
# watching all sixteen tests pass anyway -- a gate that cannot fail, which is the exact shape this file exists to
# stop. Any future change here must be re-checked the same way, not by reading it.
def pytest_runtest_setup(item):
    if item.get_closest_marker("cpu_reference") is None:
        return
    backend = _backend()
    if backend is None or backend.startswith("cpu"):
        return
    pytest.fail(
        f"this test needs gguf_vecdot's CPU REFERENCE ARM, but the backend is '{backend}'.\n"
        f"  The device library was dlopened, so the op forwarded to it -- and these tests compare our arithmetic\n"
        f"  against the official gguf package in the SAME SUMMATION ORDER at fp32. The device arm accumulates in\n"
        f"  fp16 in a different order, so the numbers disagree for a reason that is not a defect in either.\n"
        f"  Run them with the device disabled:\n"
        f"      QUACTLIZE_PPU_LIB=/nonexistent python -m pytest -m cpu_reference tests/\n"
        f"  benchmarks/run_batch.sh's pytest step already runs them as a separate pass.\n"
        f"  QUACTLIZE_PPU_LIB currently = {os.environ.get('QUACTLIZE_PPU_LIB', '<unset>')}")
