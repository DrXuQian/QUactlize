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


# ------------------------------------------------------------------------------------------------------------
# A STALE EXTENSION MUST STOP THE SESSION, not produce results.
#
# This happened TWICE on ppu001 in one afternoon, the same way both times: codex commits C++, the box pulls,
# somebody runs `pytest tests`, and the whole tier reports on a .so built before the change. The second run
# produced "worst nan" on all five formats and a planted fault that went uncaught -- readings that look like
# five real defects and are about code nobody was running.
#
# The guard already existed in tests/test_layouts.py, but only for tests taking its Q fixture, so it surfaced as
# one ERROR among many AFTER everything else had already been measured. run_batch's pytest step rebuilds
# automatically now, and that only helps whoever remembers to use run_batch.
#
# So it is a SESSION-LEVEL refusal: nothing runs, and the first thing on screen is the command to fix it.
#
# ABSENT IS NOT STALE. A machine that never built the extension is a legitimate state -- the individual tests
# skip with their own reasons and that is correct. Only PRESENT-BUT-OLDER is always wrong, and that is the only
# case this refuses.
def pytest_sessionstart(session):
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parent.parent
    so = sorted((root / "quactlize").glob("_C*.so"))
    if not so:
        return                                   # absent: let the per-test skips handle it
    built = so[0].stat().st_mtime
    newer = sorted(p.name for p in (root / "quactlize" / "csrc").rglob("*")
                   if p.suffix in (".cpp", ".h", ".hpp") and p.stat().st_mtime > built)
    if not newer:
        return
    raise pytest.UsageError(
        "THE BUILT EXTENSION IS OLDER THAN ITS SOURCES -- refusing to run, because every result below would be "
        "about code nobody is running.\n"
        f"  newer than {so[0].name}: {', '.join(newer[:6])}{' ...' if len(newer) > 6 else ''}\n"
        "  fix:  python3 setup.py build_ext --inplace\n"
        "  or:   ./benchmarks/run_batch.sh pytest   (rebuilds by itself, and splits the cpu_reference pass)\n"
        "  This has produced two full runs of plausible-looking garbage already -- 'worst nan' on all five "
        "formats and a planted fault going uncaught, both from a .so predating the commit under test.")
