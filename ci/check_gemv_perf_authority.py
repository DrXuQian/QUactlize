#!/usr/bin/env python3
"""Pin GEMV perf geometry, format semantics, identities and byte pitches."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.fixtures import dedup, fixtures  # noqa: E402
from benchmarks.workloads import MODELS, projections  # noqa: E402

MAIN = ROOT / "benchmarks/test_gemv_perf.cu"
COMMON = ROOT / "benchmarks/gemv_perf_common.hpp"
FIXTURE = ROOT / "benchmarks/gemv_perf_fixture.hpp"
MANIFEST = ROOT / "benchmarks/gemv_perf_manifest.hpp"
PITCH_ORACLE = ROOT / "dev/fold_derivation/l135_gemv_perf_authority.cpp"
MANIFEST_ORACLE = ROOT / "dev/fold_derivation/l144_gemv_perf_manifest.cpp"


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
            out.append((68 + 4 * (tokens.bit_length() - 1) + index,
                        experts, tokens, n, k, topk,
                        {1: 8, 2: 15, 4: 30}[tokens]))
    return out


def audit(common: str) -> list[str]:
    bad = []
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


def compile_run(oracle: Path, include_root: Path) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="quactlize-l135-bin-") as td:
        exe = Path(td) / "l135"
        build = subprocess.run(
            ["g++", "-std=c++17", "-I", str(include_root), "-I", str(ROOT),
             "-I", str(ROOT / "quactlize/include"), str(oracle), "-o", str(exe)],
            cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if build.returncode:
            return build
        return subprocess.run([str(exe)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def manifest_records(include_root: Path) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    proc = compile_run(MANIFEST_ORACLE, include_root)
    records: list[dict] = []
    if proc.returncode == 0:
        try:
            records = [json.loads(line) for line in proc.stdout.splitlines() if line]
        except (json.JSONDecodeError, TypeError):
            proc = subprocess.CompletedProcess(proc.args, 1, proc.stdout + "\ninvalid JSON output")
    return proc, records


def expected_geometry_dicts() -> list[dict]:
    s_rows = []
    for sid, experts, tokens, n, k, topk, active in expected_shapes():
        s_rows.append(dict(id=f"S{sid:03d}", route="grouped", experts=experts,
                           m=tokens, n=n, k=k, topk=topk, active=active))
    extras = [
        dict(id="H-G8-2048", route="grouped", experts=8, m=1,
             n=2048, k=2048, topk=8, active=8),
        dict(id="D-EXT-O", route="dense", experts=0, m=1,
             n=5120, k=8192, topk=0, active=1),
        dict(id="D-EXT-K1024", route="dense", experts=0, m=1,
             n=5120, k=1024, topk=0, active=1),
        dict(id="D-EXT-Q", route="dense", experts=0, m=1,
             n=8192, k=5120, topk=0, active=1),
        dict(id="D-4096", route="dense", experts=0, m=1,
             n=4096, k=4096, topk=0, active=1),
    ]
    return s_rows + extras


def audit_manifest(records: list[dict]) -> list[str]:
    bad: list[str] = []
    geometries = [r for r in records if r.get("rec") == "geometry"]
    cases = [r for r in records if r.get("rec") == "case"]
    summaries = [r for r in records if r.get("rec") == "summary"]
    if geometries != [dict(rec="geometry", **x) for x in expected_geometry_dicts()]:
        bad.append(f"geometry authority drift: got={geometries}, want={expected_geometry_dicts()}")
    if len(cases) != 86:
        bad.append(f"case count={len(cases)}, expected 17*5+1=86")
    if len(summaries) != 1 or summaries[0].get("errors") != 0:
        bad.append(f"manifest oracle summary invalid: {summaries}")

    ids: dict[str, str] = {}
    json_to_id: dict[str, str] = {}
    by_geometry: dict[str, list[dict]] = {}
    for record in cases:
        sid = record.get("shape_id")
        shape = record.get("shape")
        fmt = record.get("format")
        if not isinstance(sid, str) or not isinstance(shape, dict):
            bad.append(f"malformed case record: {record}")
            continue
        canonical = json.dumps(shape, sort_keys=True, separators=(",", ":"))
        if sid in ids and ids[sid] != canonical:
            bad.append(f"shape id collision: {sid}")
        if canonical in json_to_id and json_to_id[canonical] != sid:
            bad.append(f"shape JSON alias: {json_to_id[canonical]} vs {sid}")
        ids[sid], json_to_id[canonical] = canonical, sid
        if shape.get("format") != fmt:
            bad.append(f"case format missing from shape identity: {sid}")
        geometry_id = sid.split("/", 1)[0]
        by_geometry.setdefault(geometry_id, []).append(record)

        expected = {
            "int4": (32, "finegrained_scale_zero", "shipping"),
            "int2": (16, "finegrained_scale_zero", "shipping"),
            "int1": (16, "finegrained_scale_zero", "controlled-unshipped"),
            "q3": (16, "finegrained_scale_zero", "shipping"),
            "q6": (16, "finegrained_scale_zero", "shipping"),
        }.get(fmt)
        if expected and (shape.get("group_size"), shape.get("quant_op"),
                         shape.get("semantic")) != expected:
            # The one explicit reference is checked separately below.
            if not (geometry_id == "D-4096" and fmt == "int4" and
                    (shape.get("group_size"), shape.get("quant_op"),
                     shape.get("semantic")) ==
                    (128, "finegrained_scale_only", "reference")):
                bad.append(f"format semantic drift: {sid}: {shape}")

    for geometry in expected_geometry_dicts():
        rows = by_geometry.get(geometry["id"], [])
        primary = [r for r in rows if r["shape"].get("semantic") != "reference"]
        if sorted(r.get("format") for r in primary) != ["int1", "int2", "int4", "q3", "q6"]:
            bad.append(f"{geometry['id']} primary format set drift")
        refs = [r for r in rows if r["shape"].get("semantic") == "reference"]
        if geometry["id"] == "D-4096":
            if len(refs) != 1 or refs[0].get("format") != "int4" or \
                    refs[0]["shape"].get("group_size") != 128 or \
                    refs[0]["shape"].get("quant_op") != "finegrained_scale_only":
                bad.append("D-4096 must carry exactly one int4 gs128 ScaleOnly reference")
        elif refs:
            bad.append(f"reference semantics leaked onto {geometry['id']}")

    if summaries:
        summary = summaries[0]
        config = summary.get("config", {})
        want_keys = {"chunk", "cta_m", "cta_n", "format", "layout", "route",
                     "step_k", "threads", "tile_size_k"}
        if set(config) != want_keys:
            bad.append(f"config identity dropped/added an axis: {sorted(config)}")
        job = summary.get("job", {})
        if not isinstance(job, dict) or job.get("shape") != cases[0].get("shape") or \
                not job.get("expected") or job["expected"][0].get("config") != config:
            bad.append("job helper does not preserve complete shape/config identities")

    # The two model-derived external anchors must still be actual Qwen3-32B
    # projection shapes; D-EXT-K1024 is a separately requested external anchor.
    qwen = {(name, n, k) for name, n, k, _ in projections(MODELS["Qwen3-32B"])}
    if ("o", 5120, 8192) not in qwen or ("q", 8192, 5120) not in qwen:
        bad.append("Qwen3-32B external dense projection derivation drift")
    return bad


def main() -> int:
    missing = [p for p in (MAIN, COMMON, FIXTURE, MANIFEST,
                            PITCH_ORACLE, MANIFEST_ORACLE) if not p.is_file()]
    if missing:
        print("[gemv-perf-authority] FAIL missing: " + ", ".join(map(str, missing)))
        return 1
    bad = audit(COMMON.read_text())
    if bad:
        print("[gemv-perf-authority] FAIL: " + "; ".join(bad))
        return 1
    manifest_green, records = manifest_records(ROOT)
    manifest_bad = audit_manifest(records)
    if manifest_green.returncode or manifest_bad:
        print("[gemv-perf-authority] FAIL manifest:\n" + manifest_green.stdout)
        if manifest_bad:
            print("; ".join(manifest_bad))
        return 1

    green = compile_run(PITCH_ORACLE, ROOT)
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
        red = compile_run(PITCH_ORACLE, root)
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
        if not audit(planted):
            print(f"[gemv-perf-authority] FAIL {label} plant escaped audit")
            return 1

    # The manifest gate must reject semantic drift, aliasing and shape drift,
    # not merely parse its own positive output.
    manifest_source = MANIFEST.read_text()
    manifest_plants = (
        ("{Format::Int4, 32, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,",
         "{Format::Int4, 128, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,",
         "int4 group size"),
        ("{Format::Int1, 16, QuantOp::FinegrainedScaleZero, SemanticClass::ControlledUnshipped,",
         "{Format::Int1, 16, QuantOp::FinegrainedScaleZero, SemanticClass::Shipping,",
         "int1 shipping label"),
        ("Route::Dense, 0, 1, 5120, 1024, 0, 1},",
         "Route::Dense, 0, 1, 5120, 2048, 0, 1},",
         "external dense geometry"),
        ("out += \",\\\"group_size\\\":\" + std::to_string(s.group_size) +",
         "out += \",\\\"format\\\":\"; detail::append_json_string(out, \"constant\"); "
         "out += \",\\\"group_size\\\":\" + std::to_string(s.group_size) +",
         "shape identity format axis"),
    )
    for old, new, label in manifest_plants:
        if manifest_source.count(old) != 1:
            print(f"[gemv-perf-authority] FAIL cannot plant {label}; matches={manifest_source.count(old)}")
            return 1
        with tempfile.TemporaryDirectory(prefix="quactlize-gemv-manifest-plant-") as td:
            target = Path(td) / "benchmarks"
            target.mkdir()
            (target / MANIFEST.name).write_text(manifest_source.replace(old, new, 1))
            proc, planted_records = manifest_records(Path(td))
        if proc.returncode == 0 and not audit_manifest(planted_records):
            print(f"[gemv-perf-authority] FAIL {label} plant escaped manifest oracle")
            return 1

    print("[gemv-perf-authority] PASS: 17 geometries / 86 semantic cases; "
          "S068--S079 derive from workloads/fixtures; shipping gs=int4:32,int2/q3/q6:16; "
          "int1=controlled-unshipped; D-4096 int4 gs128 ScaleOnly is reference-only; "
          "shape/config JSON identities complete; E256 ragged routes 8/15/30 active; "
          "4096 byte-pitch checks; eight semantic/fixture plants red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
