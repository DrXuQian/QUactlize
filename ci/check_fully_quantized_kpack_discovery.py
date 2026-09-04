#!/usr/bin/env python3
"""Fail-closed local contract for FullyQuantized canonical K-pack discovery."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import fully_quantized_kpack_discovery_matrix as matrix  # noqa: E402
import fully_quantized_kpack_bundle_index as bundle_index  # noqa: E402
import gen_fully_quantized_grouped_kpack_units as grouped_gen  # noqa: E402
import gen_fully_quantized_kpack_discovery_units as dense_gen  # noqa: E402
import plan_fq_kpack_route_optimal as route_plan  # noqa: E402


DENSE_HEADER = ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp"
DENSE_DRIVER = ROOT / "benchmarks/test_fully_quantized_internal_sweep.cu"
GROUPED_HEADER = ROOT / "benchmarks/fully_quantized_grouped_kpack_discovery.hpp"
DENSE_UNIT = ROOT / "benchmarks/fully_quantized_splitk_producer_unit.inc"
GROUPED_UNIT = ROOT / "benchmarks/fully_quantized_grouped_kpack_discovery_unit.inc"
DENSE_TYPE = ROOT / "dev/fold_derivation/l252_fully_quantized_kpack_discovery_types.cu"
GROUPED_TYPE = ROOT / "dev/fold_derivation/l253_fully_quantized_grouped_kpack_discovery_types.cu"
GROUPED_DRIVER = ROOT / "benchmarks/test_fully_quantized_grouped_kpack_discovery.cu"
BUNDLE_BUILDER = ROOT / "tools/build_fully_quantized_kpack_discovery_bundle.sh"
BOX_EXECUTOR = ROOT / "tools/run_fully_quantized_kpack_discovery_box.sh"
BUNDLE_INDEX = ROOT / "tools/fully_quantized_kpack_bundle_index.py"
FQ_CMAKE = ROOT / "quactlize/csrc/fq_internal_sweep.cmake.in"
BUILD_SH = ROOT / "build.sh"


def require_once(text: str, needle: str, what: str) -> None:
    if text.count(needle) != 1:
        raise ValueError(f"{what}: expected one {needle!r}, got {text.count(needle)}")


def dense_fixture_contract(text: str) -> None:
    expected = {
        "high_bits ? f.high_native.data() : nullptr": 2,
        "high_bits ? f.high.data() : nullptr": 4,
        "high_bits ? high_back.data() : nullptr": 2,
        "high_back == f.high_native": 2,
    }
    for needle, count in expected.items():
        if text.count(needle) != count:
            raise ValueError(
                "two-plane layout2 prepare/recover roundtrip changed " +
                f"{needle}: {text.count(needle)}/{count}")


def builder_sdk_contract(text: str) -> None:
    if text.count('PPU_SDK="$sdk" PPU_PRESERVE_STALE_BUILD_TREES=1') != 2:
        raise ValueError("dense/grouped build.sh calls are not bound to the receipted SDK")


def source_contract() -> None:
    dense = DENSE_HEADER.read_text()
    grouped = GROUPED_HEADER.read_text()
    dense_driver = DENSE_DRIVER.read_text()
    dense_unit = DENSE_UNIT.read_text()
    grouped_unit = GROUPED_UNIT.read_text()
    grouped_driver = GROUPED_DRIVER.read_text()
    builder = BUNDLE_BUILDER.read_text()
    executor = BOX_EXECUTOR.read_text()
    cmake = FQ_CMAKE.read_text()
    build_sh = BUILD_SH.read_text()
    for path in (DENSE_TYPE, GROUPED_TYPE):
        if not path.is_file():
            raise ValueError(f"missing type-only gate {path.relative_to(ROOT)}")
    require_once(dense, "using KPack = fpa_intb_ppu::DenseKPackKernelTypes<",
                 "dense generic K-pack type")
    require_once(dense, "using KPack4 = fpa_intb_ppu::DenseQ4KPack4KernelTypes<",
                 "dense Q4 K-pack type")
    if "WeightLayout == 2" not in dense or \
            "MainloopUsesPackedMetadata" not in dense:
        raise ValueError("dense type does not bind layout2 packed metadata")
    if "FQ_TC_WEIGHT_LAYOUT" not in dense_unit:
        raise ValueError("generated dense unit does not bind its canonical layout")
    if "bool Persistent" not in grouped or \
            "GROUPED_PERSISTENT" not in matrix.GROUPED_ALGORITHMS:
        raise ValueError("grouped persistent type axis is absent")
    if "true, false, false, 0, Persistent, typename T::Policy, 0" not in grouped:
        raise ValueError("grouped launch does not activate packed metadata/persistent axis")
    if "Descriptor::packed_metadata" not in grouped or \
            "!Descriptor::interleaved_metadata" not in grouped:
        raise ValueError("grouped packed metadata publication is not exact")
    for needle in ("in.empty < 0", "in.empty != in.experts - in.active"):
        if needle not in grouped:
            raise ValueError("grouped type rejects the all-active router: " + needle)
    correctness = grouped.find("for (int repeat = 0; repeat < options.correctness_repeats")
    timing = grouped.find("if (options.measure)")
    if correctness < 0 or timing < 0 or correctness >= timing or \
            "RAW_FP16_MISMATCH" not in grouped:
        raise ValueError("grouped raw-bit correctness does not precede timing")
    if "PERSIST != 0" not in grouped_unit:
        raise ValueError("generated grouped unit erased its algorithm axis")
    for needle in (
            "first_bad_expert", "bad_first_m_tile", "bad_later_m_tiles",
            "bad_got_zero", "bad_got_poison", "bad_by_local_m_mod16",
            "bad_by_n_mod64_n16",
            "inspect(in, int(cute::size<0>(typename T::Tile{})), result)"):
        if needle not in grouped:
            raise ValueError(
                "grouped raw-bit failure lost its exact coordinate map: " + needle)
    for macro in (
            "#define FQ_GROUPED_DECLARE(FN,Q,L,TM,TN,TK,WM,WN,ST,DN,PERSIST)",
            "#define FQ_GROUPED_REGISTER(FN,Q,L,TM,TN,TK,WM,WN,ST,DN,PERSIST)"):
        if macro not in grouped_driver:
            raise ValueError(
                "grouped driver does not consume the generated DeliveryN axis: " + macro)
    if "{#FN,Q,L,TM,TN,TK,WM,WN,ST,DN,(PERSIST != 0)," not in grouped_driver:
        raise ValueError("grouped registry row does not preserve DeliveryN before persistence")
    for needle in ("moe_router_fixture::route(", "out.empty < 0",
                   "kpack_grouped_fixture_rows::load(",
                   "kpack_grouped_fixture_rows::rows_fnv64(",
                   "--rows-file=", "--workload-key=", "--router-profile=",
                   "--schedule-seed=", "cli.schedule_seed ^ rows_hash",
                   "out.offsets.back() != out.total", "out.active <= 0",
                   "roundtrip=PASS",
                   "correctness=RAW_FP16", "top_n=NONE"):
        if needle not in grouped_driver:
            raise ValueError("grouped driver lost fixture/confirmation contract: " + needle)
    for needle in ("FQ_GROUPED_KPACK_MISMATCH_MAP", "m_tile=[first:%llu,later:%llu]",
                   "local_m_mod16=[", "n_mod64_n16=["):
        if needle not in grouped_driver:
            raise ValueError(
                "grouped raw-bit failure report lost its coordinate evidence: " + needle)
    for needle in ("--schedule-seed=", "std::shuffle(execution_rows.begin()",
                   "cli.schedule_seed"):
        if needle not in dense_driver:
            raise ValueError("dense driver lost round-order control: " + needle)
    if ("/root/autodl-tmp" not in builder or "ensure_owned_scratch" not in builder or
            "clear_owned_shard_scratch" not in builder or
            "shard_receipt validate" not in builder or
            "FQ_KPACK_MAX_PARENTS_PER_BINARY" not in builder or
            "--parent-begin" not in builder or "--parent-count" not in builder or
            "check_free_space" not in builder):
        raise ValueError("local bundle builder lost bounded scratch/resume authority")
    if ('scratch_build="$out/scratch/$key"' not in builder or
            'PPU_BUILD_RESUME="$build_resume"' not in builder or
            ".quactlize-source-head" not in builder):
        raise ValueError("local bundle builder lost resumable per-shard scratch")
    if builder.count("PPU_PRESERVE_STALE_BUILD_TREES=1") != 2 or \
            'TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"' not in builder or \
            "PYTHONDONTWRITEBYTECODE=1" not in builder:
        raise ValueError("local bundle builder lost isolated build environment")
    builder_sdk_contract(builder)
    planted_builder = builder.replace('PPU_SDK="$sdk" ', "", 1)
    try:
        builder_sdk_contract(planted_builder)
    except ValueError:
        pass
    else:
        raise ValueError("explicit build SDK negative stayed green")
    for needle in ('stage="$out/payloads/.$key.current.$$"',
                   'mv "$stage" "$payload"',
                   "existing OUT lacks build input authority",
                   "existing OUT scratch lacks ownership marker",
                   '"host_cxx"', '"resolved_path"'):
        if needle not in builder:
            raise ValueError("bundle publication/provenance contract lost: " + needle)
    for forbidden in ("./build.sh", " bin/hgcc", " cmake ",
                      "gen_fully_quantized_kpack_discovery_units.py",
                      "gen_fully_quantized_grouped_kpack_units.py"):
        if forbidden in executor:
            raise ValueError("box executor is not prebuilt-only: " + forbidden)
    if ("ALL_RAW_BIT_CLEAN" not in executor or "top_n=NONE" not in executor or
            "parent-union=EXACT" not in executor or
            "--runtime-probe-binary" not in executor or
            'done <"$shard_index"' not in executor):
        raise ValueError("box executor introduced candidate pruning")
    for needle in ("plan_fq_kpack_route_optimal.py",
                   "materialize_kpack_discovery_workloads.py",
                   '"format_cells":1381', '"router_controls":120',
                   '"q4_historical_anchors":286',
                   "expected_dense=429 if qtype==12 else 143",
                   "len(grouped_lines)!=77", "--workload-key=",
                   "--router-profile=", "--rows-file=",
                   "PILOT_WORKLOAD_LIMIT", "SCREEN_DIAGNOSTIC",
                   "CONFIRM_REQUIRED_FOR_HEURISTIC", "DIAGNOSTIC_ONLY",
                   "canonical_route_plan_sha256", '"phase":phase',
                   "map(int,sys.argv[5:9])"):
        if needle not in executor:
            raise ValueError("box executor lost canonical workload authority: " + needle)
    if "plan_fq_kquant_kpack_perf.py" in executor:
        raise ValueError("box executor reused the obsolete 77/24 workload plan")
    for needle in ("device_identity_sha256", '"runtime_sdk"',
                   "receipt_sha256", "inspector_sha256",
                   "libhggc_candidates", "loaded_libhggc",
                   "top-level bundle SDK differs from build authority",
                   "payload/identity-probe loaded libhggc sets differ",
                   "payload_runtime_linkage_sha256",
                   "binary receipt authority chain differs",
                   "build_input_authority_sha256",
                   "bundle path escapes or is missing",
                   "resumed result authority differs"):
        if needle not in executor:
            raise ValueError("box result authority lost runtime binding: " + needle)
    for needle in (
            "FQ_GROUPED_KPACK_GENERATED_DIR",
            "FQ_GROUPED_KPACK_QTYPE",
            "FQ_GROUPED_KPACK_WEIGHT_LAYOUT",
            "FQ_GROUPED_KPACK_PACKED_FORMAT"):
        if needle not in cmake or needle not in build_sh:
            raise ValueError("grouped FQ build axis is not forwarded: " + needle)
    if "test_fully_quantized_grouped_kpack_discovery" not in cmake or \
            "PPU_PACKED_SCALE=1" not in cmake:
        raise ValueError("grouped FQ CMake target lost packed metadata")
    dense_fixture_contract(dense_driver)
    planted_driver = dense_driver.replace(
        "high_bits ? f.high_native.data() : nullptr", "nullptr", 1)
    try:
        dense_fixture_contract(planted_driver)
    except ValueError:
        pass
    else:
        raise ValueError("two-plane roundtrip negative stayed green")


def generated_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="fq-kpack-check-") as temporary:
        root = pathlib.Path(temporary)
        total_dense = total_grouped = 0
        seen: dict[tuple[int, str], list[int]] = {}
        for shard in bundle_index.plan(False, 32):
            qtype, operator = shard["qtype"], shard["operator"]
            path = root / shard["shard_key"]
            if operator == "dense":
                document = dense_gen.generate(
                    qtype, path, 4, parent_begin=shard["parent_begin"],
                    parent_count=shard["parent_count"])
                dense_gen.validate_manifest(document)
                rows = document["dense_tc_parents"]
                total_dense += document["denominator"]["compiled_rows"]
            else:
                document = grouped_gen.generate(
                    qtype, path, 4, parent_begin=shard["parent_begin"],
                    parent_count=shard["parent_count"])
                grouped_gen.validate_manifest(document)
                rows = document["grouped_parents"]
                total_grouped += document["denominator"]["compiled_rows"]
            if len(rows) > 32 or [row["static_candidate_id"] for row in rows] != shard["parent_ids"]:
                raise ValueError(f"{shard['shard_key']} parent range differs")
            seen.setdefault((qtype, operator), []).extend(
                row["parent_ordinal"] for row in rows)
        if total_dense != 14750 or total_grouped != 27412:
            raise ValueError(
                f"compiled denominator differs dense={total_dense} grouped={total_grouped}")
        for (qtype, operator), ordinals in seen.items():
            if ordinals != list(range(bundle_index.authority_count(qtype, operator))):
                raise ValueError(f"q{qtype} {operator} parent union differs")


def negative_contract() -> None:
    document = matrix.make_manifest(True)
    plants = []
    missing_source = copy.deepcopy(document)
    missing_source["formats"][4]["source_rows"].pop(100)
    plants.append(missing_source)
    fake_availability = copy.deepcopy(document)
    fake_availability["formats"][0]["algorithms"]["GROUPED_BC_FULL_OUTPUT"] = {
        "status": "AVAILABLE", "reason": "TC"}
    plants.append(fake_availability)
    wrong_layout = copy.deepcopy(document)
    wrong_layout["formats"][2]["weight_layout"] = 2
    plants.append(wrong_layout)
    hidden_top = copy.deepcopy(document)
    hidden_top["confirmation"]["top_n"] = 32
    plants.append(hidden_top)
    for planted in plants:
        try:
            matrix.validate_manifest(planted, expanded=True)
        except (TypeError, ValueError):
            continue
        raise ValueError("checker negative plant stayed green")


def workload_contract() -> None:
    document = route_plan.materialize()
    route_plan.validate_plan(document)
    for qtype in matrix.QTYPES:
        dense = [cell for cell in document["cells"]
                 if cell["qtype"] == qtype and cell["operator"] == "dense"
                 and cell["source_class"] == "real-inventory"]
        grouped = [cell for cell in document["cells"]
                   if cell["qtype"] == qtype and cell["operator"] == "grouped"
                   and cell["source_class"] == "real-inventory"]
        if len(dense) != 143 or len(grouped) != 52:
            raise ValueError(
                f"q{qtype} canonical real inventory differs "
                f"dense={len(dense)} grouped={len(grouped)}")


def run_type_gate(command: str | None) -> None:
    if command is None:
        return
    result = subprocess.run(command, cwd=ROOT, shell=True, text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise ValueError("type-only command failed:\n" + result.stdout[-4000:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type-gate-command",
        help="optional caller-owned command that compiles l252/l253 for all five PPU_PACKED_FORMAT values")
    args = parser.parse_args()
    try:
        matrix.self_test()
        bundle_index.self_test()
        dense_gen.self_test()
        grouped_gen.self_test()
        source_contract()
        for script in (BUNDLE_BUILDER, BOX_EXECUTOR):
            syntax = subprocess.run(["bash", "-n", str(script)], cwd=ROOT,
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            if syntax.returncode:
                raise ValueError(
                    f"{script.name} shell syntax failed:\n{syntax.stdout}")
        generated_contract()
        workload_contract()
        negative_contract()
        run_type_gate(args.type_gate_command)
        print("[fq-kpack-discovery-check] PASS formats=5 raw=115200 "
              "dense=14750/59000 grouped=27412 APxDN binaries=1323<=32-parents "
              "workloads=143-dense+52-grouped/q "
              "packed-metadata=1 raw-bit-before-timing "
              "BC+grouped-splitk=STRUCTURAL_UNAVAILABLE "
              "top-N=FORBIDDEN negatives=RED")
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"[fq-kpack-discovery-check] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
