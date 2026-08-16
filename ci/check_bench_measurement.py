#!/usr/bin/env python3
"""Dense and MoE must consume one measured-quantity contract, not merely include its header."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / "benchmarks" / "bench_select.hpp"

CONSUMERS = {
    "benchmarks/test_lowbit_dense_bench.cu": (
        "bench_measure::make_traffic(",
        "bench_measure::measure(",
        "bench_measure::format_metrics(",
        "bench_measure::read_reps()",
        "upd(best, dense_tactic(c), us)",
        "_a.bc = c.b_chunk",
        "_a.bc_eff = c.b_chunk_effective",
        "kDenseFixedBChunkRequest",
        "Policy::Descriptor::atom_at_a_time",
        "r.avg_runtime_ms * 1e3",
        "options.k * options.l",
    ),
    "benchmarks/lowbit_dense_unit.inc": (
        "bench_measure::format_tag(",
        "FN##_effective()",
        "dense_tactic(cfg)",
    ),
    "benchmarks/lowbit_moe_bench.hpp": (
        "bench_measure::make_traffic(",
        "bench_measure::measure(",
        "bench_measure::format_metrics(",
        "bench_measure::format_tag(",
        "upd(BEST, _cfg, u, _tim.wall_us)",
    ),
    "benchmarks/test_lowbit_moe_bench.cu": (
        'bench_measure::read_reps("MOE_REPS")',
        "bench_measure::ridge_flops_per_byte()",
        "bench_measure::mfu_pct(tf)",
        "e.tactic.tm",
    ),
    "benchmarks/moe_only_filter.hpp": (
        "bench_measure::format_tag(",
    ),
    "benchmarks/moe_splitk_bench_common.hpp": (
        "bench_measure::nameplate_pct(",
        "bench_measure::kHbmGBPerSecond",
    ),
    "benchmarks/test_moe_splitk_bench.cu": (
        "bench_measure::kHbmGBPerSecond",
    ),
    # THE GEMV BENCH, added 2026-08-11. It cannot include bench_select.hpp -- both define `Best` at global scope
    # and the generator emits that name into every GEMV unit signature -- so its nameplate is a MIRROR, and the
    # mirror is checked numerically below rather than asserted in a comment. Listed here so the file is READ
    # (texts is built from CONSUMERS) and so its own constant name cannot quietly disappear.
    "benchmarks/gemv_perf_common.hpp": (
        "HBM_GBS",
    ),
}

# These expressions are the old two-copy implementation. Comments are stripped before matching so an explanation
# may name the failure without making the gate reject itself.
BANNED = {
    "benchmarks/test_lowbit_dense_bench.cu": (
        r"\b500\.0\b",
        r"\b2766\.0\b",
        r'getenv\s*\(\s*"BENCH_REPS"',
        r"sscanf\s*\(\s*label",
    ),
    "benchmarks/lowbit_moe_bench.hpp": (
        r"static\s+constexpr\s+double\s+(?:PEAK|HBM_GBS)",
        r"/\s*PEAK\b",
        r"/\s*HBM_GBS\b",
    ),
    "benchmarks/test_lowbit_moe_bench.cu": (
        r"/\s*PEAK\b",
        r"/\s*HBM_GBS\b",
        r"sscanf\s*\(\s*e\.tag",
    ),
}


def uncomment(text: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def under_preprocessor_guard(text: str, needle: str, guard: str) -> bool:
    """Every live-code occurrence of needle must be nested under a matching #if/#ifdef/#ifndef line."""
    stack: list[str] = []
    found = False
    for line in uncomment(text).splitlines():
        stripped = line.strip()
        if re.match(r"#\s*(?:if|ifdef|ifndef)\b", stripped):
            stack.append(stripped)
        if needle in line:
            found = True
            if not any(guard in frame for frame in stack):
                return False
        if re.match(r"#\s*endif\b", stripped):
            if stack:
                stack.pop()
    return found


# EVERY BENCH IN THE TREE, for the one check below that is about a SHAPE rather than a symbol.
ALL_BENCH_SOURCES = (
    "benchmarks/bench_select.hpp",
    "benchmarks/lowbit_moe_bench.hpp",
    "benchmarks/gemv_perf_common.hpp",
    "benchmarks/moe_splitk_bench_common.hpp",
    "benchmarks/test_lowbit_dense_bench.cu",
    "benchmarks/test_lowbit_moe_bench.cu",
    "benchmarks/test_gemv_perf.cu",
    "benchmarks/test_moe_splitk_bench.cu",
)

# `gbs > nameplate` on the same line as a control-transfer, or immediately followed by one.
_OVER_PEAK = re.compile(r">\s*(?:bench_measure::kHbmGBPerSecond|HBM_GBS)\b")
_ESCAPE = re.compile(r"\b(?:return|continue|break)\b\s*;")


def over_peak_must_not_drop_the_row(texts: dict[str, str]) -> list[str]:
    """A ROW OVER THE NAMEPLATE MUST BE KEPT. Found three times in three benches, each worse than the last:

        lowbit_moe_bench.hpp        printed DID NOT RUN and excluded the row from the verdict
        gemv_perf_common.hpp        `if (gbs <= HBM_GBS) upd(best, ...)` -- deleted the FASTEST rows from the winner
        moe_splitk_bench_common.hpp `if (gbs > peak) return;` -- skipped the rest of the row's handling, and with
                                    a DIRECTION: its pb term charges 2*slices*total*N*2, so modelled bytes grow
                                    with S while a working split lowers us, and gbs rises on both counts. The
                                    rows most likely to trip it were the SUCCESSFUL high-S rows, in the bench
                                    whose whole purpose is the split-K ladder.

    Exceeding the nameplate indicts the TRAFFIC MODEL -- a weight charged once per grid_m, an L2-served
    reduction charged to DRAM -- not the measurement. Flag the model; keep the row. This bans the shape rather
    than the three instances, because the instances were written independently and a fourth bench would write
    it again.
    """
    problems = []
    for rel in ALL_BENCH_SOURCES:
        lines = uncomment(texts.get(rel, "")).splitlines()
        for i, line in enumerate(lines):
            if not _OVER_PEAK.search(line):
                continue
            window = " ".join(lines[i:i + 3])
            if _ESCAPE.search(window):
                problems.append(
                    f"{rel}:{i+1}: a row over the nameplate is dropped (return/continue/break within 3 lines of "
                    f"the comparison). Over the nameplate indicts the traffic model, not the measurement -- "
                    f"flag the model and KEEP the row: {line.strip()[:70]}")
    return problems


def dense_selection_must_settle_before_admission(texts: dict[str, str]) -> list[str]:
    """The dense caller must resolve accumulated samples before testing ``Best.tag``.

    ``upd()`` intentionally appends samples without maintaining a provisional
    leader.  Consequently an empty tag before ``settle(best)`` means only
    "not settled", not "no row passed".  This is a caller-ordering contract;
    the host probe for ``bench_select.hpp`` alone cannot catch it because that
    probe already calls ``settle`` in the correct order.
    """

    rel = "benchmarks/test_lowbit_dense_bench.cu"
    body = uncomment(texts.get(rel, ""))
    settles = list(re.finditer(r"\bsettle\s*\(\s*best\s*\)", body))
    admissions = list(re.finditer(
        r"\bif\s*\(\s*best\.tag\s*\[\s*0\s*\]\s*==\s*'\\0'\s*\)",
        body,
    ))
    problems: list[str] = []
    if len(settles) != 1:
        problems.append(
            f"{rel}: expected one dense settle(best) call, found {len(settles)}"
        )
    if len(admissions) != 1:
        problems.append(
            f"{rel}: expected one empty-winner admission check, found {len(admissions)}"
        )
    if len(settles) == 1 and len(admissions) == 1:
        if settles[0].start() > admissions[0].start():
            problems.append(
                f"{rel}: checks Best.tag before settle(best); passing rows remain "
                "only in Best.seen and are falsely reported as 'no config passed'"
            )
    return problems


def audit(texts: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for rel, needles in CONSUMERS.items():
        body = uncomment(texts.get(rel, ""))
        for needle in needles:
            if needle not in body:
                problems.append(f"{rel}: shared measurement call/field is absent: {needle}")
    for rel, patterns in BANNED.items():
        body = uncomment(texts.get(rel, ""))
        for pattern in patterns:
            if re.search(pattern, body):
                problems.append(f"{rel}: private measurement expression returned: {pattern}")
    # THE MIRRORED NAMEPLATE MUST EQUAL THE SHARED ONE. gemv_perf_common.hpp keeps its own literal because it
    # cannot include bench_select.hpp (a global `Best` in both, and the CMake generator emits that name into
    # every GEMV unit signature). A mirror is only acceptable while something compares it, so: parse both and
    # require equality. This is NOT "two copies agreeing is a reference" -- bench_select.hpp remains the source
    # and this fails the moment the mirror drifts from it.
    shared = re.search(r"kHbmGBPerSecond\s*=\s*([0-9.eE+-]+)", HEADER.read_text())
    mirror = re.search(r"HBM_GBS\s*=\s*([0-9.eE+-]+)",
                       uncomment(texts.get("benchmarks/gemv_perf_common.hpp", "")))
    if not shared or not mirror:
        problems.append("cannot read the nameplate from bench_select.hpp and/or gemv_perf_common.hpp; the "
                        "mirror check cannot run, which is a failure and not a skip")
    elif float(shared.group(1)) != float(mirror.group(1)):
        problems.append(f"gemv_perf_common.hpp mirrors the nameplate as {mirror.group(1)} while "
                        f"bench_select.hpp says {shared.group(1)}: the GEMV bench's percentages are against a "
                        f"different machine than the dense and MoE ones")

    problems += over_peak_must_not_drop_the_row(texts)
    problems += dense_selection_must_settle_before_admission(texts)

    dense = texts.get("benchmarks/test_lowbit_dense_bench.cu", "")
    if not under_preprocessor_guard(
            dense, "kDenseFixedBChunkRequest", "!defined(LOWBIT_DENSE_UNIT_BUILD)"):
        problems.append("benchmarks/test_lowbit_dense_bench.cu: fixed bc witness leaks into bc0/bc1 generated "
                        "units with different inline definitions")
    return problems


PROBE = r'''
#include <cmath>
#include <cstdio>
#include <cstring>
#include "bench_select.hpp"

static int close(double a, double b) { return std::fabs(a - b) < 1.0e-9 * (1.0 + std::fabs(b)); }
static int fail(char const* why) { std::fprintf(stderr, "%s\n", why); return 1; }

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "reps") == 0) {
    std::printf("%d\n", bench_measure::read_reps());
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "moe-reps") == 0) {
    std::printf("%d\n", bench_measure::read_reps("MOE_REPS"));
    return 0;
  }
  if (bench_measure::kPeakFlopsPerSecond != 500.0e12 || bench_measure::kHbmGBPerSecond != 2766.0)
    return fail("machine constants drifted");

  char tag[bench_measure::kTagBytes];
  bench_measure::format_tag(tag, sizeof tag,
      {"i4",64,128,64,64,32,3,1,0,true});
  if (std::strcmp(tag, "i4 64x128:64 w64x32 s3 bc1->0 B")) return fail("MoE canonical tag drifted");
  bench_measure::format_tag(tag, sizeof tag,
      {nullptr,64,128,64,64,32,3,1,0,false});
  if (std::strcmp(tag, "64x128:64 w64x32 s3 bc1->0")) return fail("dense canonical tag drifted");
  Best best{};
  bench_measure::Tactic slower{"i4",32,64,64,32,32,3,0,0,false};
  bench_measure::Tactic leader{"i4",64,128,64,64,32,3,0,0,false};
  upd(best, slower, 200.0); upd(best, leader, 100.0);
  if (best.tag[0] != '\0')
    return fail("upd unexpectedly resolved a provisional winner before settle");
  settle(best);
  if (!best.has_tactic || best.tactic.tm != 64 || best.tactic.tn != 128)
    return fail("selection discarded the structured winning tactic");

  auto traffic = bench_measure::make_traffic({1.0e6,0.5e6,0.1e6,0.2e6,3.0,2.0,6.0});
  if (!close(traffic.distinct.total(), 3.0e6) || !close(traffic.tile.total(), 5.8e6))
    return fail("traffic channels/copy counts do not produce the planted distinct and tile totals");

  // THE SPLIT-K C TERM, which no control could previously reach. Two independent properties:
  //   splitk == 1 must be BYTE-IDENTICAL to the unsplit model, or every row ever recorded moves under a change
  //     that was supposed to affect only split runs;
  //   splitk == S must carry the reduction round trip 2*W*(S-1) + D. Planted: D = 0.2e6, W = 2D, S = 3
  //     -> 0.2e6 + 2*(0.4e6)*2 = 1.8e6 = 9D, in BOTH models (the workspace is written once and read once, so it
  //     is distinct traffic, not a re-read).
  auto sk1 = bench_measure::make_traffic({1.0e6,0.5e6,0.1e6,0.2e6,3.0,2.0,6.0,1,2.0});
  auto sk3 = bench_measure::make_traffic({1.0e6,0.5e6,0.1e6,0.2e6,3.0,2.0,6.0,3,2.0});
  if (!close(sk1.distinct.total(), traffic.distinct.total()) ||
      !close(sk1.tile.total(), traffic.tile.total()))
    return fail("splitk=1 is no longer identical to the unsplit traffic model");
  if (!close(sk3.distinct.output, 9.0 * 0.2e6) || !close(sk3.tile.output, 9.0 * 0.2e6))
    return fail("the split-K reduction round trip is missing from the C term");
  if (close(sk3.distinct.total(), traffic.distinct.total()))
    return fail("splitk did not change total traffic at all");
  auto m = bench_measure::measure(2.0, 1.0e9, traffic);
  if (!close(m.compute.tflops, 500.0) || !close(m.compute.mfu_pct, 100.0))
    return fail("useful FLOP -> TF/s/MFU arithmetic drifted");
  if (!close(m.hbm.distinct_gbs, 1500.0) || !close(m.hbm.tile_gbs, 2900.0) ||
      !close(m.hbm.distinct_nameplate_pct, 100.0 * 1500.0 / 2766.0) ||
      !close(m.hbm.tile_reuse, 5.8 / 3.0) || !close(m.hbm.distinct_metadata_share, 0.1) ||
      !m.hbm.tile_l2_served)
    return fail("HBM/reuse named fields drifted");

  bench_measure::Traffic twice{{1000.0,0,0,0},{2000.0,0,0,0}};
  bench_measure::Traffic four {{1000.0,0,0,0},{4000.0,0,0,0}};
  auto m2 = bench_measure::measure(7.0, 9.0e8, twice);
  auto m4 = bench_measure::measure(7.0, 9.0e8, four);
  if (!close(m2.compute.mfu_pct, m4.compute.mfu_pct) ||
      !close(m2.hbm.distinct_gbs, m4.hbm.distinct_gbs) ||
      !close(m4.hbm.tile_gbs, 2.0 * m2.hbm.tile_gbs) ||
      !close(m2.hbm.tile_reuse, 2.0) || !close(m4.hbm.tile_reuse, 4.0))
    return fail("tile/reuse was deleted, aliased to distinct, or fed back into MFU");

  char fragment[256];
  bench_measure::format_metrics(fragment, sizeof fragment, m);
  if (!std::strstr(fragment, "MFU") || !std::strstr(fragment, "distinct") ||
      !std::strstr(fragment, "nameplate") || !std::strstr(fragment, "tile") ||
      !std::strstr(fragment, "x distinct") || !std::strstr(fragment, "L2-served"))
    return fail("common printed fragment lost a named measurement field");
  std::puts(fragment);
  return 0;
}
'''


def compile_probe() -> pathlib.Path:
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cxx:
        raise RuntimeError("no host C++ compiler; shared measurement arithmetic was not exercised")
    work = pathlib.Path(tempfile.mkdtemp(prefix="bench-measurement-"))
    source, binary = work / "probe.cpp", work / "probe"
    source.write_text(PROBE)
    result = subprocess.run(
        [cxx, "-std=c++17", "-I", str(ROOT / "benchmarks"), str(source), "-o", str(binary)],
        capture_output=True, text=True,
    )
    if result.returncode:
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError("host measurement probe does not compile:\n" + result.stderr[-2000:])
    return binary


def run_probe(binary: pathlib.Path, arg: str | None = None, env_overrides: dict[str, str] | None = None) -> str:
    env = dict(os.environ)
    env.pop("BENCH_REPS", None)
    env.pop("MOE_REPS", None)
    env.update(env_overrides or {})
    command = [str(binary)] + ([arg] if arg else [])
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"host measurement probe failed ({arg or 'metrics'}):\n"
                           + (result.stdout + result.stderr)[-2000:])
    return result.stdout.strip()


def check_tune_tag_consumer() -> None:
    source = ROOT / "tools" / "tune.py"
    source_text = source.read_text()

    # Compile the bytes we just read. importlib may reuse a same-second, same-size .pyc; that exact false-green was
    # reproduced with this checker's winner regex changed on disk while exec_module still ran the old bytecode.
    def load(text: str, suffix: str) -> dict:
        namespace = {"__file__": str(source), "__name__": f"quactlize_tune_{suffix}"}
        exec(compile(text, str(source), "exec"), namespace)
        return namespace

    module = load(source_text, "measurement")
    expected = (64, 128, 64, 64, 32, 3, 0)
    names = (
        "64x128:64 w64x32 s3 bc0->0",
        "64x128x64:64x32:s3:bc0->0",
        "i4 64x128x64:64x32:s3 bc0",
    )
    got = tuple(module["canonical"](name) for name in names)
    if got != (expected, expected, expected):
        raise RuntimeError(f"tune canonicalizer does not bridge new/legacy/analyser tags: {got}")
    tk128 = module["canonical"]("64x128:128 w64x32 s3 bc0->0")
    bc1 = module["canonical"]("64x128:64 w64x32 s3 bc1->1")
    if tk128 == expected or bc1 == expected or tk128 == bc1:
        raise RuntimeError(f"tune canonicalizer aliases TileK/B-chunk row axes: {expected}, {tk128}, {bc1}")

    rows = [{"name": name} for name in (names[0],
            "64x128:128 w64x32 s3 bc0->0", "64x128:64 w64x32 s3 bc1->1")]
    index = module["config_index"](rows)
    if any(module["indexed_config"](index, row["name"]) is not row for row in rows):
        raise RuntimeError("complete config inventory does not preserve distinct TileK/B-chunk rows")
    legacy_row = {"name": "64x128:64x32:s3"}
    legacy_index = module["config_index"]([legacy_row])
    if module["indexed_config"](legacy_index, names[0]) is not None:
        raise RuntimeError("legacy geometry-only inventory falsely validated a complete TileK/B-chunk winner")
    if module["indexed_config"](legacy_index, "64x128:64x32:s3") is not legacy_row:
        raise RuntimeError("legacy geometry-only winner no longer matches the same legacy shipping identity")
    partial_row = {"name": "64x128:64 w64x32 s3"}
    partial_index = module["config_index"]([partial_row])
    if module["indexed_config"](partial_index, "64x128:128 w64x32 s3") is not None:
        raise RuntimeError("partial config identities alias different TileK values through legacy geometry")
    old_partial_row = {"name": "i4 64x128x64:64x32:s3"}
    old_partial_index = module["config_index"]([old_partial_row])
    if module["indexed_config"](old_partial_index, "i4 64x128x256:64x32:s3") is not None:
        raise RuntimeError("old partial config identities alias different TileK values")

    class Completed:
        returncode = 0
        stdout = "==== WINNER: 64x128:64 w64x32 s3 bc0->0 at 321.5 TFLOP/s (separated)\n"
        stderr = ""

    real_run = module["subprocess"].run
    module["subprocess"].run = lambda *args, **kwargs: Completed()
    try:
        config, tflops, note = module["bench_run"]("unused", 1, 1, 1, 1, 3, "unused")
    finally:
        module["subprocess"].run = real_run
    if (config, tflops, note) != ("64x128:64 w64x32 s3 bc0->0", 321.5, "separated"):
        raise RuntimeError(f"tune winner parser truncated the canonical tag: {(config, tflops, note)}")

    # Same-length planted mutation: proves this gate executes current source bytes instead of a stale pyc.
    winner_pattern = (r'r"====\s+WINNER:\s+(.+?)\s+at\s+([0-9.]+)\s+TFLOP/s\s+'
                      r'\(separated\)"')
    mutated_pattern = winner_pattern.replace("(.+?)", r"(\S+)")
    if source_text.count(winner_pattern) != 1:
        raise RuntimeError("cannot locate the unique tune winner regex for its same-size control")
    mutated_text = source_text.replace(winner_pattern, mutated_pattern, 1)
    if len(mutated_text) != len(source_text):
        raise RuntimeError("cannot plant the same-size tune winner-regex control")
    mutated = load(mutated_text, "measurement_mutant")
    mutated_real_run = mutated["subprocess"].run
    mutated["subprocess"].run = lambda *args, **kwargs: Completed()
    try:
        mutated_result = mutated["bench_run"]("unused", 1, 1, 1, 1, 3, "unused")
    finally:
        mutated["subprocess"].run = mutated_real_run
    if mutated_result == ("64x128:64 w64x32 s3 bc0->0", 321.5, "separated"):
        raise RuntimeError("same-size planted winner-regex mutation was not observed")


def main() -> int:
    if not HEADER.is_file():
        print("[bench-measurement] FAIL: benchmarks/bench_select.hpp is missing")
        return 1
    texts = {rel: (ROOT / rel).read_text() for rel in set(CONSUMERS) | set(ALL_BENCH_SOURCES)}
    problems = audit(texts)
    if problems:
        print("[bench-measurement] FAIL: " + problems[0])
        return 1

    # Every required call is live: deleting any one in memory must make the same audit red.
    controls = 0
    for rel, needles in CONSUMERS.items():
        for needle in needles:
            planted = dict(texts)
            planted[rel] = planted[rel].replace(needle, "BENCH_MEASURE_PLANTED_DROP")
            if not audit(planted):
                print(f"[bench-measurement] FAIL: deletion control was accepted: {rel} / {needle}")
                return 1
            controls += 1
    for rel, patterns in BANNED.items():
        planted = dict(texts)
        # Each literal is chosen to match the corresponding regex and is appended as live code, not a comment.
        samples = {
            r"\b500\.0\b": "\ndouble planted_peak = 500.0;\n",
            r"\b2766\.0\b": "\ndouble planted_hbm = 2766.0;\n",
            r'getenv\s*\(\s*"BENCH_REPS"': '\nauto planted_reps = getenv("BENCH_REPS");\n',
            r"sscanf\s*\(\s*label": '\nsscanf(label, "%d", &planted_peak);\n',
            r"static\s+constexpr\s+double\s+(?:PEAK|HBM_GBS)": "\nstatic constexpr double PEAK = 1;\n",
            r"/\s*PEAK\b": "\ndouble planted_mfu = 1 / PEAK;\n",
            r"/\s*HBM_GBS\b": "\ndouble planted_pct = 1 / HBM_GBS;\n",
            r"sscanf\s*\(\s*e\.tag": '\nsscanf(e.tag, "%d", &planted_mfu);\n',
        }
        for pattern in patterns:
            planted[rel] = texts[rel] + samples[pattern]
            if not audit(planted):
                print(f"[bench-measurement] FAIL: private-expression control was accepted: {rel} / {pattern}")
                return 1
            controls += 1

    guard_open = "#if !defined(LOWBIT_DENSE_UNIT_BUILD)\n#if defined(PPU_B_CHUNK)"
    if guard_open not in texts["benchmarks/test_lowbit_dense_bench.cu"]:
        print("[bench-measurement] FAIL: cannot plant the generated-unit ODR guard control")
        return 1
    planted = dict(texts)
    planted["benchmarks/test_lowbit_dense_bench.cu"] = planted[
        "benchmarks/test_lowbit_dense_bench.cu"].replace(guard_open, "#if 1\n#if defined(PPU_B_CHUNK)", 1)
    if not audit(planted):
        print("[bench-measurement] FAIL: fixed bc witness was accepted outside the generated-unit guard")
        return 1
    controls += 1

    # Same caller, same statements, only their order is wrong.  This is the
    # exact regression that made a sweep with passing rows print
    # ``no config passed``: upd() had populated Best.seen, but Best.tag is
    # deliberately unresolved until settle(best).
    dense_source = texts["benchmarks/test_lowbit_dense_bench.cu"]
    settle_line = "    const int ties = settle(best);\n"
    admission_line = (
        "    if (best.tag[0] == '\\0') { std::fprintf(stderr, "
        '"no config passed\\n"); return 1; }\n'
    )
    if dense_source.count(settle_line) != 1 or dense_source.count(admission_line) != 1:
        print("[bench-measurement] FAIL: cannot plant settle/admission ordering control")
        return 1
    reordered = dense_source.replace(settle_line, "", 1)
    reordered = reordered.replace(admission_line, admission_line + settle_line, 1)
    planted = dict(texts)
    planted["benchmarks/test_lowbit_dense_bench.cu"] = reordered
    if not audit(planted):
        print("[bench-measurement] FAIL: pre-settle empty-winner admission was accepted")
        return 1
    controls += 1

    binary = compile_probe()
    work = binary.parent
    try:
        fragment = run_probe(binary)
        reps = (
            run_probe(binary, "reps"),
            run_probe(binary, "reps", {"BENCH_REPS": "3"}),
            run_probe(binary, "moe-reps", {"BENCH_REPS": "3", "MOE_REPS": "5"}),
            run_probe(binary, "reps", {"BENCH_REPS": "0"}),
        )
        if reps != ("1", "3", "5", "1"):
            raise RuntimeError(f"repetitions reader returned {reps}, expected unset/BENCH/MoE/clamp = 1/3/5/1")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    check_tune_tag_consumer()

    print(f"[bench-measurement] PASS: shared constants/tag/reps/MFU and distinct+tile traffic; "
          f"dense+MoE route reports through the shared layer; tune preserves tag axes and legacy input; "
          f"{controls} deletion/private-expression controls fired; {fragment}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"[bench-measurement] FAIL: {exc}")
        raise SystemExit(1)
