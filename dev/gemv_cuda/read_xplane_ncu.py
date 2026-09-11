#!/usr/bin/env python3
"""Extract named, unit-preserving counters from the bounded Q4 comparison."""

import argparse
import csv
import io
import json
import math
from pathlib import Path


CORE = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum", "dram__bytes.sum.per_second",
    "dram__bytes.sum.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct", "lts__t_sectors.sum",
    "smsp__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
)
EXTRA = (
    "l1tex__t_sector_hit_rate.pct",
    "sm__pipe_aluheavy_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warp_latency_per_inst_issued.ratio",
)


def parse(text, expected_kernels):
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3 or "Kernel Name" not in rows[0]:
        raise ValueError("missing NCU raw header")
    header, units, *records = rows
    if len(records) != expected_kernels or len(set(header)) != len(header):
        raise ValueError("NCU kernel count or column identity differs")
    if any(name not in header for name in CORE):
        raise ValueError("missing required NCU metric")
    selected = list(CORE) + [x for x in EXTRA if x in header]
    selected += [x for x in header if x.startswith("smsp__average_warps_issue_stalled_") and x.endswith(".ratio")]
    output = []
    for index, record in enumerate(records):
        if len(record) != len(header):
            raise ValueError("truncated NCU raw record")
        kernel = record[header.index("Kernel Name")]
        if (index == 0 and "reduce" in kernel) or (index > 0 and "kpack_gemv_reduce" not in kernel):
            raise ValueError("unexpected producer/reducer order")
        metrics, warnings = {}, []
        for name in selected:
            col = header.index(name)
            try:
                value = float(record[col].replace(",", ""))
            except ValueError as exc:
                raise ValueError("unavailable NCU metric: " + name) from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid NCU metric: " + name)
            metrics[name] = dict(value=value, unit=units[col])
            if name.endswith("_sector_hit_rate.pct") and value > 100:
                # Preserve profiler evidence; never clamp a replay-inconsistent
                # hit ratio to 100% or use it as a performance conclusion.
                metrics[name]["usable_for_conclusion"] = False
                warnings.append("OUT_OF_RANGE_REPLAY_HIT_RATIO: " + name)
        if metrics["gpu__time_duration.sum"]["value"] <= 0:
            raise ValueError("invalid kernel duration")
        output.append(dict(kernel=kernel, role="producer" if index == 0 else "reducer",
                           metrics=metrics, warnings=warnings))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    manifest = json.loads((args.directory/"manifest.json").read_text())
    if manifest["status"] != "PASS" or len(manifest["profiles"]) != 8:
        raise ValueError("incomplete profile campaign")
    result = dict(scope="RTX5070_WSL_FIXED_5090_WINNERS_NOT_PPU_OR_5090_COUNTERS",
                  replay_mode=manifest["replay_mode"], cache_control=manifest["cache_control"],
                  clock_control=manifest["clock_control"], authority=manifest["authority"], profiles=[])
    for profile in manifest["profiles"]:
        key = profile["key"]
        expected = 1 + (profile["arm"] == "kpack" and profile["recipe"][2] > 1)
        kernels = parse((args.directory/f"{key}.raw.csv").read_text(), expected)
        timing = (args.directory/f"{key}.timing.log").read_text().strip()
        if "status=PASS" not in timing:
            raise ValueError("missing separate unprofiled correctness/timing")
        result["profiles"].append(dict(key=key, recipe=profile["recipe"], kernels=kernels,
                                       unprofiled_timing_line=timing,
                                       report_sha256=profile["report_sha256"]))
        for kernel in kernels:
            m = kernel["metrics"]
            print(key, kernel["role"], {name:m[name]["value"] for name in (
                "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
                "dram__bytes.sum.pct_of_peak_sustained_elapsed",
                "dram__bytes.sum.per_second", "smsp__inst_executed.sum")})
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
