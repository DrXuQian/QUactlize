#!/usr/bin/env python3
"""Device-free contract for the exact Stream-K fixup CTA cohort.

Stream-K's named barrier and its block-striped FP32 scratch indexing must use
the *same* thread count as the CTA which owns the accumulator fragment.  A
fixed 128-thread cohort deadlocks a 64-thread CTA; accepting 128 threads while
indexing modulo 64 aliases the two half-CTAs.  Both errors compile cleanly, so
this checker pins the vendor policy seam, both owned wrappers, and the real-type
64/128 host oracle before a device round trip.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"
DENSE = ROOT / (
    "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/"
    "ppu_aiu_gemm_mixed_input_streamk.hpp"
)
GROUPED = ROOT / (
    "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/"
    "ppu_aiu_gemm_mixed_input_group_streamk.hpp"
)
L122 = ROOT / "dev/fold_derivation/l122_streamk_fixup_cohort.cu"


_CPP_TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.MULTILINE | re.DOTALL,
)


def without_cpp_comments(text: str) -> str:
    """Remove comments but preserve literals and line structure."""
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(('"', "'")):
            return token
        return "\n" * token.count("\n")

    return _CPP_TOKEN.sub(replace, text)


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def one_regex(text: str, pattern: str, label: str, bad: list[str]) -> None:
    count = len(re.findall(pattern, text, re.MULTILINE | re.DOTALL))
    if count != 1:
        bad.append(f"{label}: expected one match, found {count}")


def audit(vendor: str, dense: str, grouped: str, oracle: str) -> list[str]:
    bad: list[str] = []
    vendor = without_cpp_comments(vendor)
    dense = without_cpp_comments(dense)
    grouped = without_cpp_comments(grouped)
    oracle = without_cpp_comments(oracle)

    # The fourth parameter is defaulted so every legacy two/three-argument
    # actlize spelling retains the old 128-thread source behaviour.
    one_regex(
        vendor,
        r"template\s*<(?:(?!>\s*class\s+PersistentTileSchedulerPPUStreamK).)*"
        r"uint32_t\s+FixupThreadsPerCta\s*=\s*NumThreadsPerWarpGroup\s*>\s*"
        r"class\s+PersistentTileSchedulerPPUStreamK",
        "vendor default-compatible fourth cohort parameter",
        bad,
    )
    one_regex(
        vendor,
        r"static\s+constexpr\s+uint32_t\s+FixupThreadCount\s*=\s*"
        r"FixupThreadsPerCta\s*;",
        "vendor exposed cohort constant",
        bad,
    )
    one_regex(
        vendor,
        r"using\s+BarrierManager\s*=\s*NamedBarrierManager\s*<\s*"
        r"FixupThreadCount\s*,",
        "fixup named-barrier cohort",
        bad,
    )
    # First version is deliberately exact: no sub-warp cohort and no CTA with
    # more than one 128-thread warp group.
    allowed = re.compile(
        r"(?:FixupThreadCount|FixupThreadsPerCta)\s*==\s*64(?:(?!;).){0,240}"
        r"\|\|(?:(?!;).){0,240}"
        r"(?:FixupThreadCount|FixupThreadsPerCta)\s*==\s*128",
        re.DOTALL,
    )
    if len(allowed.findall(vendor)) != 1:
        bad.append("vendor cohort must fail closed to exactly 64 or 128 threads")

    for label, wrapper in (("dense", dense), ("grouped", grouped)):
        one_regex(
            wrapper,
            r"using\s+TileScheduler\s*=\s*(?:typename\s+)?(?:detail::)?"
            r"PersistentTileSchedulerPPUStreamK\s*<\s*TileShape\s*,\s*"
            r"ClusterShape\s*,[^,>]+,\s*MaxThreadsPerBlock\s*>\s*;",
            f"{label} scheduler carries the CTA cohort in its type",
            bad,
        )
        one_regex(
            wrapper,
            r"TileScheduler::FixupThreadCount\s*==\s*MaxThreadsPerBlock",
            f"{label} exact scheduler/CTA equality",
            bad,
        )
        wrapper_allowed = re.compile(
            r"MaxThreadsPerBlock\s*==\s*64(?:(?!;).){0,240}\|\|"
            r"(?:(?!;).){0,240}MaxThreadsPerBlock\s*==\s*128",
            re.DOTALL,
        )
        if len(wrapper_allowed.findall(wrapper)) != 1:
            bad.append(f"{label} wrapper must accept exactly 64 or 128 CTA threads")
        if re.search(
            r"static_assert\s*\(\s*MaxThreadsPerBlock\s*==\s*128\s*,",
            wrapper,
        ):
            bad.append(f"{label} wrapper retained the obsolete fixed-128 gate")

    # L122 binds two real grouped Operations, asks the actual accumulator
    # fragment/BlockStripedReduce for its stripe count, and enumerates the same
    # workspace address used by fixup.  Merely checking (q,thread) would miss
    # holes in fragments with more than one stripe.
    for token, label in (
        ("Operation64", "real 64-thread operation"),
        ("Operation128", "real 128-thread operation"),
        ("BlockStripedReduce", "vendor block-striped reduction type"),
        ("FixupThreadCount", "scheduler cohort type witness"),
        ("MaxThreadsPerBlock", "actual CTA thread witness"),
        ("kStripes", "real fragment stripe count"),
        ("holes", "hole counter"),
        ("duplicate_visits", "duplicate counter"),
    ):
        if oracle.count(token) < 1:
            bad.append(f"L122 is missing {label} ({token!r})")
    one_regex(
        oracle,
        r"cohort_thread\s*=\s*thread\s*%\s*CohortThreads\s*;",
        "L122 reviewed thread-within-cohort formula",
        bad,
    )
    one_regex(
        oracle,
        r"local\s*=\s*stripe\s*\*\s*CohortThreads\s*\+\s*cohort_thread\s*;",
        "L122 reviewed workspace slot formula",
        bad,
    )
    one_regex(
        oracle,
        r"DefaultLegacy::FixupThreadCount\s*==\s*128",
        "L122 legacy-default cohort witness",
        bad,
    )
    for token, label in (
        ("red64_as_128", "actual-64/wrong-128 red arm"),
        ("red128_as_64", "actual-128/wrong-64 red arm"),
    ):
        if oracle.count(token) < 3:
            bad.append(f"L122 lacks the {label}")

    return bad


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"cannot plant {label}: anchor count is {text.count(old)}")
    return text.replace(old, new, 1)


def source_plants(texts: list[str]) -> list[str]:
    """Return failures of the checker itself, not failures of planted source."""
    bad: list[str] = []
    plants: list[tuple[int, str, str, str]] = [
        (
            0,
            "NamedBarrierManager<FixupThreadCount,",
            "NamedBarrierManager<NumThreadsPerWarpGroup,",
            "vendor barrier falls back to the global 128-thread cohort",
        ),
        (
            1,
            ", MaxThreadsPerBlock>;",
            ">;",
            "dense wrapper drops the cohort template argument",
        ),
        (
            2,
            ", MaxThreadsPerBlock>;",
            ">;",
            "grouped wrapper drops the cohort template argument",
        ),
        (
            1,
            "TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
            "TileScheduler::FixupThreadCount % MaxThreadsPerBlock == 0",
            "dense wrapper weakens exact cohort equality",
        ),
        (
            2,
            "TileScheduler::FixupThreadCount == MaxThreadsPerBlock",
            "TileScheduler::FixupThreadCount % MaxThreadsPerBlock == 0",
            "grouped wrapper weakens exact cohort equality",
        ),
        (
            3,
            "stripe * CohortThreads + cohort_thread",
            "stripe * CohortThreads + (cohort_thread % (CohortThreads / 2))",
            "L122 aliases the workspace address across half-cohorts",
        ),
    ]
    for index, old, new, label in plants:
        planted = list(texts)
        try:
            planted[index] = replace_once(planted[index], old, new, label)
        except ValueError as exc:
            bad.append(str(exc))
            continue
        if not audit(*planted):
            bad.append(f"checker accepted planted regression: {label}")
    return bad


def compile_and_run_l122(selected_cohort: int = 64) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="qz-l122-out-") as td:
        exe = Path(td) / "l122"
        cmd = [
            "nvcc", "-std=c++17", "-x", "cu", "-arch=sm_80", "-w",
            "-D__HGGCCC__", "--expt-relaxed-constexpr",
            f"-DL122_SELECTED_COHORT={selected_cohort}",
            "-I", str(ROOT / "dev/fold_derivation/stub_inc"),
            "-I", str(ROOT / "third_party/actlize/include"),
            "-I", str(ROOT / "third_party/actlize/tools/util/include"),
            "-I", str(ROOT / "quactlize/include"),
            "-I", str(ROOT / "tests"),
            "-I", str(ROOT / "benchmarks"),
            "-o", str(exe), str(L122),
        ]
        compiled = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log = compiled.stdout + compiled.stderr
        if compiled.returncode != 0:
            return compiled.returncode, log
        ran = subprocess.run([str(exe)], cwd=ROOT, capture_output=True, text=True)
        return ran.returncode, log + ran.stdout + ran.stderr


def main() -> int:
    paths = (VENDOR, DENSE, GROUPED, L122)
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        print("[streamk-fixup-cohort] FAIL: missing " + ", ".join(missing))
        return 1
    texts = [path.read_text() for path in paths]
    bad = audit(*texts)
    bad.extend(source_plants(texts))
    rc, log = compile_and_run_l122()
    if rc != 0 or log.count("L122 exact-fixup-cohort=PASS") != 1:
        tail = "\n".join(log.splitlines()[-12:])
        bad.append(f"L122 did not compile/run to PASS (rc={rc}):\n{tail}")
    expected_invalid = "PPU Stream-K fixup supports exactly 64- or 128-thread CTAs"
    for cohort in (32, 256):
        invalid_rc, invalid_log = compile_and_run_l122(cohort)
        if invalid_rc == 0 or expected_invalid not in invalid_log:
            tail = "\n".join(invalid_log.splitlines()[-12:])
            bad.append(
                f"L122 invalid cohort {cohort} did not fail at the exact vendor gate "
                f"(rc={invalid_rc}):\n{tail}"
            )
    if bad:
        print("[streamk-fixup-cohort] FAIL: " + "; ".join(bad))
        return 1
    print(
        "[streamk-fixup-cohort] PASS -- exact 64/128 vendor cohort, dense/grouped "
        "type wiring, real-fragment workspace coverage; six planted regressions and "
        "invalid 32/256 cohorts rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
