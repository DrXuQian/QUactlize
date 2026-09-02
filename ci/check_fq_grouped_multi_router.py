#!/usr/bin/env python3
"""Static source-contract gate for the grouped multi-router pilot."""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(path: str, needles: tuple[str, ...]) -> None:
    text = (ROOT / path).read_text()
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"{path} misses {needle!r}")


def main() -> int:
    for command in (
        [sys.executable, "-B", "tools/fq_grouped_multi_router.py"],
        [sys.executable, "-B", "tools/plan_fq_grouped_multi_router.py", "self-test"],
        [sys.executable, "-B", "tools/analyze_fq_grouped_multi_router.py", "self-test"],
        [
            sys.executable,
            "-B",
            "tools/fq_grouped_multi_router_manifest.py",
            "self-test",
        ],
    ):
        subprocess.run(command, cwd=ROOT, check=True)
    require(
        "benchmarks/test_fq_grouped_multi_router_perf.cu",
        (
            "FQ_GROUPED_ROUTER_CELL",
            "FQ_GROUPED_ROUTER_RUN",
            "make_weights(family.first.first, family.first.second, 256, true)",
            "grouped_configs(route.total",
            "work_tm128=%d",
            "i == 5",
            "nonnegative(fields[std::size_t(i + 1)]",
            "parsed < 0",
            "UINT64_C(14695981039346656037)",
            "UINT64_C(1099511628211)",
            "rows_hash == uint64_t(expected_hash)",
            "rows_hash=0x%016llx",
        ),
    )
    from tools.fq_grouped_multi_router import materialize

    balanced = materialize()["balanced"]
    if balanced["zero"] != 0 or balanced["active"] != 256:
        raise AssertionError("balanced zero=0 parser control differs")
    require(
        "quactlize/csrc/fq_grouped_multi_router_perf.cmake.in",
        (
            "test_fq_grouped_multi_router_perf",
            "FQ_KQUANT_PERF_QTYPE",
            "layout=K-pack-only",
        ),
    )
    require(
        "quactlize/csrc/CMakeLists.txt.in", ("fq_grouped_multi_router_perf.cmake.in",)
    )
    require(
        "tools/build_fq_grouped_multi_router_bundle.sh",
        (
            "/root/autodl-tmp",
            "test_fq_grouped_multi_router_perf",
            "quactlize.fq-grouped-multi-router-prebuilt.v1",
            "PPU_BUNDLE_JOBS",
            "wait_worker_batch",
            "setsid bash",
            "terminate_worker_groups",
            'kill -TERM -- "-$pgid"',
            "rerun with RESUME=1",
        ),
    )
    print(
        "[fq-grouped-multi-router:static] PASS target/plan/analyzer/bundle source contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
