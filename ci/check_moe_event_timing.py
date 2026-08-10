#!/usr/bin/env python3
"""Static contract for the MoE event timer.

This is intentionally a source-order gate.  The defect is an interval boundary: all code compiles if the start
event drifts above initialize or the blocking prefix copy, and every output remains numerically correct while the
performance number becomes wall/setup time again.  A device-free gate can prove the ordering and batching protocol;
the PPU run in BOX.md owns the numerical 11.1 us anchor.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "quactlize/include/moe_grouped_ppu.cuh"
HARNESS = ROOT / "benchmarks/lowbit_moe_bench.hpp"
MAIN = ROOT / "benchmarks/test_lowbit_moe_bench.cu"


def section(text: str, begin: str, end: str) -> str:
    if text.count(begin) != 1 or text.count(end) < 1:
        raise ValueError(f"cannot isolate {begin!r} .. {end!r}")
    return text.split(begin, 1)[1].split(end, 1)[0]


def audit(launcher: str, harness: str, main: str) -> list[str]:
    bad: list[str] = []
    try:
        launch = section(launcher, "  Gemm gemm;", "\ntemplate <QuantMode QuantOp, int TM")
    except ValueError as e:
        return [str(e)]

    ordered = [
        "gemm.initialize(args, workspace, stream)",
        "hggcMemcpy(workspace, pfx.data()",
        "hggcEventRecord(kernel_span->start, stream)",
        "gemm.run(stream)",
        "hggcEventRecord(kernel_span->stop, stream)",
    ]
    if any(launch.count(x) != 1 for x in ordered):
        bad.append("launch interval anchors are missing or duplicated")
    elif [launch.index(x) for x in ordered] != sorted(launch.index(x) for x in ordered):
        bad.append("launch order is not initialize -> blocking H2D -> start -> run -> stop")
    if all(x in launch for x in ordered[2:]):
        bracket = launch.split(ordered[2], 1)[1].split(ordered[4], 1)[0]
        if any(x in bracket for x in ("Synchronize", "hggcMemcpy", "gemm.initialize")):
            bad.append("the start/run/stop device interval contains setup, a copy, or synchronization")
    if "kernel_span->recorded = true" not in launch:
        bad.append("successful stop event does not mark the pair queryable")
    if launcher.count("false,kernel_span)") != 1:
        bad.append("filter_and_run does not forward its event pair through the one MOEG_CALL seam")

    try:
        timed = section(harness, "// S068 exposed why subtraction is not a timer:",
                        "\n// A CHECKSUM BEFORE/AFTER")
    except ValueError as e:
        return bad + [str(e)]
    required = (
        "MoeKernelEventBatch batch(iters + 1);",
        "f(batch.at(0));",
        "for (int i = 0; i < iters; ++i) f(batch.at(i + 1));",
        "auto const& e = *batch.at(i + 1);",
        "auto t1 = std::chrono::high_resolution_clock::now();",
        "hggcEventElapsedTime(&ms, e.start, e.stop)",
        "if (!std::isfinite(us) || us <= 0.0) continue;",
        "100.0 * (out.max_us - out.min_us) / out.kernel_span_us",
    )
    for needle in required:
        if timed.count(needle) != 1:
            bad.append(f"timed overload must contain exactly one {needle!r}")
    if timed.count("f(nullptr);") != 1:
        bad.append("timed overload needs exactly one uninstrumented ACU cold call")
    if timed.count("hggcDeviceSynchronize()") != 3:
        bad.append("timed overload must sync ACU once, then once after warmup and once after all timed launches")
    if "hggcEventSynchronize" in timed:
        bad.append("per-launch event synchronization serialises the timed loop")
    anchors = ("MoeKernelEventBatch batch(iters + 1);", "f(batch.at(0));", "auto t0 =",
               "for (int i = 0; i < iters; ++i) f(batch.at(i + 1));", "auto t1 =",
               "hggcEventElapsedTime(&ms")
    if all(x in timed for x in anchors):
        pos = [timed.index(x) for x in anchors]
        if pos != sorted(pos):
            bad.append("protocol order is event pool -> instrumented warmup -> t0 -> 20 launches -> t1 -> query")
        else:
            warmup_sync = timed.find("hggcDeviceSynchronize()", pos[1])
            final_sync = timed.find("hggcDeviceSynchronize()", pos[3])
            if not (pos[1] < warmup_sync < pos[2] and pos[3] < final_sync < pos[4]):
                bad.append("warmup and timed batch do not each end at their one required synchronization")
    if "std::enable_if_t<std::is_invocable_v<F>, int>" not in harness:
        bad.append("the zero-argument time_it overload used by split-K disappeared")
    if harness.count("u = _tim.kernel_span_us") != 2:
        bad.append("both one- and two-plane rows must use event span as primary us")
    if harness.count("time_it(_go, 20)") != 2:
        bad.append("both row families must request exactly 20 timed launches")
    if harness.count("upd(BEST, _cfg, u, _tim.wall_us)") != 2:
        bad.append("both row families must retain the matching host wall in selection state")
    if harness.count("moe_abcast(), _kev)") != 2:
        bad.append("both row families must pass their per-launch event pair")
    if "timing = \"event-kernel-span-upper-v1\"" not in harness:
        bad.append("sample records do not name the new timing protocol")
    if "if (timing.expected > 0 && !timing.complete())" not in harness or "no wall fallback" not in harness:
        bad.append("an incomplete event batch can fall back to wall instead of being excluded")
    if harness.count("else { moe_excluded(") != 2 or "incomplete device-event batch" not in harness:
        bad.append("a rejected timed row can leave an unmatched JSON attempt instead of an exclusion record")
    try:
        report = section(harness, "inline void report(", "\n// DID THIS ROW ACTUALLY RUN?")
    except ValueError as e:
        bad.append(str(e))
    else:
        primary = "const double us = timing.complete() ? timing.kernel_span_us : timing.wall_us;"
        if report.count(primary) != 1:
            bad.append("report metrics do not select complete event span as primary with ACU wall as the only fallback")

    if "event-kernel-span-upper-v1" not in main or "acu-cold-host-wall-v1" not in main or "timing=%s" not in main:
        bad.append("run identity would allow old wall-us and new event-us samples to merge")
    if "bench_floor::launch_bound(e.wall_us)" not in main:
        bad.append("host launch floor is not compared with the same-clock host wall")
    for label in ("kernel-span-upper", "host-wall", "spread=(max-min)/mean"):
        if label not in harness + main:
            bad.append(f"output no longer labels {label!r}")
    return bad


def main() -> int:
    launcher, harness, main_src = LAUNCHER.read_text(), HARNESS.read_text(), MAIN.read_text()
    bad = audit(launcher, harness, main_src)
    if bad:
        print("[moe-event-timing] FAIL: " + "; ".join(bad))
        return 1

    # Negative controls prove the inspection sees each silent regression class rather than merely finding files.
    # Move the complete guarded start block above the blocking copy. The planted source remains valid C++ and
    # null-safe; only its timing boundary is wrong, which is the silent regression this control must prove visible.
    start_block = (
        "  if (kernel_span != nullptr) {\n"
        "    hggcError_t const err = hggcEventRecord(kernel_span->start, stream);\n"
        "    if (err != hggcSuccess) {\n"
        "      std::printf(\"[moe_grouped] start-event record failed: %s\\n\", hggcGetErrorString(err));\n"
        "      ++moeg_fail_count();\n"
        "      return false;\n"
        "    }\n"
        "  }\n"
    )
    planted_order = launcher.replace(start_block, "", 1).replace(
        "  if (args.mtiles_uniform == 0 && workspace != nullptr && !prefix_ready) {\n",
        start_block + "  if (args.mtiles_uniform == 0 && workspace != nullptr && !prefix_ready) {\n", 1)
    if not audit(planted_order, harness, main_src):
        print("[moe-event-timing] FAIL: order gate accepted start-before-prefix planted control")
        return 1

    planted_sync = harness.replace(
        "for (int i = 0; i < iters; ++i) f(batch.at(i + 1));",
        "for (int i = 0; i < iters; ++i) { f(batch.at(i + 1)); hggcEventSynchronize(batch.at(i + 1)->stop); }", 1)
    if not audit(launcher, planted_sync, main_src):
        print("[moe-event-timing] FAIL: batching gate accepted per-launch synchronization")
        return 1

    warmup = "  f(batch.at(0));\n  CUTLASS_PPU_CHECK(hggcDeviceSynchronize());\n"
    planted_warmup = harness.replace(warmup, "", 1).replace(
        "  auto t0 = std::chrono::high_resolution_clock::now();\n",
        "  auto t0 = std::chrono::high_resolution_clock::now();\n" + warmup, 1)
    if not audit(launcher, planted_warmup, main_src):
        print("[moe-event-timing] FAIL: warmup-order gate accepted a warmup inside the host interval")
        return 1

    planted_primary = harness.replace("u = _tim.kernel_span_us", "u = _tim.wall_us", 1)
    if not audit(launcher, planted_primary, main_src):
        print("[moe-event-timing] FAIL: primary-time gate accepted a wall fallback")
        return 1

    planted_incomplete = harness.replace(
        "if (timing.expected > 0 && !timing.complete())",
        "if (false && timing.expected > 0 && !timing.complete())", 1)
    if not audit(launcher, planted_incomplete, main_src):
        print("[moe-event-timing] FAIL: fail-close gate accepted an incomplete-event wall fallback")
        return 1

    report_primary = "const double us = timing.complete() ? timing.kernel_span_us : timing.wall_us;"
    planted_report = harness.replace(report_primary, "const double us = timing.wall_us;", 1)
    if not audit(launcher, planted_report, main_src):
        print("[moe-event-timing] FAIL: report gate accepted wall-derived MFU/MBU")
        return 1

    planted_interval_sync = launcher.replace(
        "  gemm.run(stream);", "  gemm.run(stream);\n  hggcStreamSynchronize(stream);  // planted", 1)
    if not audit(planted_interval_sync, harness, main_src):
        print("[moe-event-timing] FAIL: interval gate accepted a synchronization inside start/run/stop")
        return 1

    planted_zero = harness.replace(
        "if (!std::isfinite(us) || us <= 0.0) continue;",
        "if (false && (!std::isfinite(us) || us <= 0.0)) continue;", 1)
    if not audit(launcher, planted_zero, main_src):
        print("[moe-event-timing] FAIL: sample-validity gate accepted zero/non-finite event elapsed")
        return 1

    values = [10.0, 11.0, 12.0, 13.0]
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / mean * 100.0
    if abs(mean - 11.5) > 1e-12 or abs(spread - 26.08695652173913) > 1e-12:
        print("[moe-event-timing] FAIL: planted spread oracle is wrong")
        return 1

    print("[moe-event-timing] PASS -- setup < start < run < stop, one warmup + batched events, "
          "event us primary, wall audit retained; eight planted regressions rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
