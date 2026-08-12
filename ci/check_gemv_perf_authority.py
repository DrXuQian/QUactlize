#!/usr/bin/env python3
"""Pin GEMV perf to real S068--S071 projections at T=1/2/4 and byte pitches."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.fixtures import dedup, fixtures  # noqa: E402
from benchmarks.workloads import MODELS  # noqa: E402

MAIN = ROOT / "benchmarks/test_gemv_perf.cu"
COMMON = ROOT / "benchmarks/gemv_perf_common.hpp"
FIXTURE = ROOT / "benchmarks/gemv_perf_fixture.hpp"
ORACLE = ROOT / "dev/fold_derivation/l135_gemv_perf_authority.cpp"


def expected_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    out = []
    for tokens in (1, 2, 4):
        band = []
        for model_index, (model, cfg) in enumerate(MODELS.items()):
            rows = dedup([r for r in fixtures(model, cfg) if r[0] == "moe" and r[4] == tokens])
            for _, label, n, k, _, extra in rows:
                # The formal inventory orders gate/up for both models before
                # down for both models; derive that order from the projection label.
                band.append(("expert_down" in label, model_index, extra["experts"], n, k, extra["topk"]))
        for index, (_, _, experts, n, k, topk) in enumerate(sorted(band)):
            out.append((68 + index, experts, tokens, n, k, topk,
                        {1: 8, 2: 15, 4: 30}[tokens]))
    return out


def source_shapes(text: str) -> list[tuple[int, int, int, int, int, int, int]]:
    pat = re.compile(
        r'\{"S0(6[8-9]|7[0-1])[^\"]*",\s*(\d+),\s*(\d+),\s*'
        r'(\d+),\s*(\d+),\s*32,\s*QuantOp::FinegrainedScaleZero,\s*(\d+),\s*(\d+)\}'
    )
    return [tuple(map(int, m.groups())) for m in pat.finditer(text)]


def audit(main: str, common: str) -> list[str]:
    bad = []
    got, want = source_shapes(main), expected_shapes()
    if got != want:
        bad.append(f"S068--S071 x T=1/2/4 mirror drift: got={got}, want={want}")
    tokens = (
        "gemv_perf_fixture::make_route(sh.experts, sh.rows, sh.topk)",
        "int active = 0;   // grouped: expected distinct active experts; independent of E",
        "int(route.active_ids.size()) != sh.active",
        "route.active_slot_for_expert",
        "gemv_perf_fixture::plane_seed(e, active, false)",
        "gemv_perf_fixture::plane_seed(e, active, true)",
        "gemv_perf_fixture::scale_value(e, g, n, active)",
        "gemv_perf_fixture::zero_value(e, g, n, active)",
        "gemv_perf_fixture::packed_plane_expert_offset(",
        "gemv_perf_fixture::packed_plane_bytes(sh.N, sh.K, LoBits)",
        "p.max_rows = b.max_rows",
        "for (int e : b.active_ids)",
        "if (!verify_witnesses(b, sh.N, tag)) return;",
        "WRONG EXPERT DATA",
    )
    for token in tokens:
        if token not in common:
            bad.append(f"common harness lost {token!r}")
    for token in ("gemv_perf_fixture::plane_seed(e, active, false)",
                  "gemv_perf_fixture::plane_seed(e, active, true)"):
        if common.count(token) != 2:
            bad.append(f"real-expert seed must feed packer and witness exactly twice: {token!r}")
    for forbidden in (
        "b.offs[e + 1] = b.offs[e] + sh.rows",
        "std::memcpy(wl.data() + size_t(e) * plo.size(), plo.data(), plo.size())",
        "double(experts) * (wb + sb)",
    ):
        if forbidden in common:
            bad.append(f"uniform/identical expert fixture returned: {forbidden!r}")
    return bad


def compile_run(include_root: Path) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="quactlize-l135-bin-") as td:
        exe = Path(td) / "l135"
        build = subprocess.run(
            ["g++", "-std=c++17", "-I", str(include_root), "-I", str(ROOT),
             str(ORACLE), "-o", str(exe)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if build.returncode:
            return build
        return subprocess.run([str(exe)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> int:
    missing = [p for p in (MAIN, COMMON, FIXTURE, ORACLE) if not p.is_file()]
    if missing:
        print("[gemv-perf-authority] FAIL missing: " + ", ".join(map(str, missing)))
        return 1
    bad = audit(MAIN.read_text(), COMMON.read_text())
    if bad:
        print("[gemv-perf-authority] FAIL: " + "; ".join(bad))
        return 1
    green = compile_run(ROOT)
    if (green.returncode or "pitch_checks=4096" not in green.stdout or
            "old_pitch_wrong_witnesses=24/24 PASS" not in green.stdout):
        print("[gemv-perf-authority] FAIL positive:\n" + green.stdout)
        return 1

    # Compile the real oracle against a copied fixture header with the exact
    # historical unit error planted: logical sub-byte codes used as uint8 bytes.
    source = FIXTURE.read_text()
    old = "std::uint64_t(n) * std::uint64_t(k) * std::uint64_t(bits) / 8u"
    new = "std::uint64_t(n) * std::uint64_t(k)"  # logical codes advanced as bytes
    if source.count(old) != 1:
        print(f"[gemv-perf-authority] FAIL cannot plant pitch; matches={source.count(old)}")
        return 1
    with tempfile.TemporaryDirectory(prefix="quactlize-l135-plant-") as td:
        root = Path(td)
        target = root / "benchmarks"
        target.mkdir()
        (target / FIXTURE.name).write_text(source.replace(old, new, 1))
        # The fixture's relative include must resolve to the production router authority.
        (target / "moe_router_fixture.hpp").symlink_to(ROOT / "benchmarks/moe_router_fixture.hpp")
        red = compile_run(root)
    if (red.returncode != 1 or "pitch_checks=4096" not in red.stdout or
            "old_pitch_wrong_witnesses=24/24 FAIL" not in red.stdout):
        print("[gemv-perf-authority] FAIL planted logical-code pitch did not red:\n" + red.stdout)
        return 1

    # Structural plants prove the audit does not silently stop asking for
    # routed rows or real-expert salting while the pitch oracle stays green.
    plants = (
        ("gemv_perf_fixture::make_route(sh.experts, sh.rows, sh.topk)",
         "gemv_perf_fixture::Route{}", "router"),
        ("gemv_perf_fixture::plane_seed(e, active, false)",
         "gemv_perf_fixture::plane_seed(active_slot[e], active, false)", "real expert id"),
        ("if (!verify_witnesses(b, sh.N, tag)) return;",
         "if (false && !verify_witnesses(b, sh.N, tag)) return;", "device witness"),
    )
    for old, new, label in plants:
        planted = COMMON.read_text().replace(old, new, 1)
        if not audit(MAIN.read_text(), planted):
            print(f"[gemv-perf-authority] FAIL {label} plant escaped audit")
            return 1

    print("[gemv-perf-authority] PASS: S068--S071 x T=1/2/4 derive from workloads/fixtures; "
          "E256 ragged routes 8/15/30 active; 4096 byte-pitch checks; logical-code pitch planted red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
