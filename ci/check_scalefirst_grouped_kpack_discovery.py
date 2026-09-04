#!/usr/bin/env python3
"""Local source/manifest contract for grouped canonical K-pack discovery."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
HEADER = ROOT / "benchmarks/scalefirst_grouped_kpack_discovery.hpp"
DRIVER = ROOT / "benchmarks/test_scalefirst_grouped_kpack_discovery.cu"
UNIT = ROOT / "benchmarks/scalefirst_grouped_kpack_discovery_unit.inc"
CMAKE = ROOT / "quactlize/csrc/scalefirst_internal_sweep.cmake.in"
BUILDER = TOOLS / "build_scalefirst_kpack_discovery_bundle.sh"
PREBUILT = TOOLS / "run_scalefirst_kpack_discovery_box.sh"
ANALYZER = TOOLS / "analyze_scalefirst_kpack_discovery.py"
SHARDS = TOOLS / "scalefirst_kpack_binary_shards.py"


def require(text: str, tokens: tuple[str, ...], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise AssertionError(f"{label} lacks {missing}")


def run(command: list[str], expected: int = 0) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != expected:
        raise AssertionError(
            f"rc={result.returncode}, expected={expected}: {' '.join(command)}\n"
            + result.stdout[-4000:])
    return result.stdout


def main() -> int:
    try:
        header = HEADER.read_text()
        driver = DRIVER.read_text()
        unit = UNIT.read_text()
        cmake = CMAKE.read_text()
        builder = BUILDER.read_text()
        prebuilt = PREBUILT.read_text()
        analyzer = ANALYZER.read_text()
        require(header, (
            "Q4KPack4MainloopPolicy<", "KPackMainloopPolicy<",
            "QuantMode::FinegrainedScaleZero",
            "Descriptor::kpack_transpose", "!Descriptor::packed_metadata",
            "ppu_tactics::GroupedSpace", "moe_grouped_ppu::launch<",
            "PersistentGemm::maximum_active_blocks()", "grid_space(",
            "occupancy < 0", "in.empty < 0",
            "in.empty != in.experts - in.active",
            "correctness_repeats", "hggcMemset(in.output, 0x7b",
        ), "grouped type/runtime seam")
        require(driver, (
            "moe_router_fixture::route(", "fixture.offsets",
            "kpack_grouped_fixture_rows::load(",
            "kpack_grouped_fixture_rows::rows_fnv64(", "host.empty < 0",
            "--rows-file=", "--workload-key=", "--router-profile=",
            "--schedule-seed=", "cli.schedule_seed ^ rows_hash",
            "--symbol-file=", "SF_GROUPED_SELECTION_FAIL",
            "q4_kpack4::prepare(", "transform_generic<false>",
            "center_correction()", "RAW_FP16", "timing=AFTER_CORRECTNESS",
            "status=STRUCTURAL_UNAVAILABLE",
            "NO_GROUPED_SPLITK_KERNEL_OR_REDUCER",
        ), "ragged fixture/verdict seam")
        require(unit, ("SCALEFIRST_GROUPED_UNIT_ROWS",
                       "scalefirst_grouped_kpack::run_row<"), "unit seam")
        require(cmake, ("SCALEFIRST_GROUPED_KPACK_GENERATED_DIR",
                        "test_scalefirst_grouped_kpack_discovery",
                        "-DPPU_PACKED_SCALE=0"), "CMake seam")
        require(builder, ("test_scalefirst_internal_sweep",
                          "test_scalefirst_grouped_kpack_discovery",
                          "quactlize.scalefirst_kpack_prebuilt_bundle.v2",
                          "tracked build source is dirty",
                          "untracked build source is not in the clean commit",
                          "recursive submodule is dirty",
                          "ONE_SHARD_THEN_COMPACT_PAYLOAD",
                          "build-input-authority.json",
                          "binary-receipt.json",
                          "PPU_BUILD_RESUME=",
                          "PPU_PRESERVE_STALE_BUILD_TREES=1",
                          'payloads/$shard_id',
                          "PYTHONDONTWRITEBYTECODE=1",
                          "PILOT", "parents_per_binary",
                          '"route":"scalefirst"', '"parent_ids":parent_ids',
                          'TMPDIR=',
                          "one-shard preflight",
                          "validate_owned_path", "partial scratch lacks exact resume markers",
                          '"tree"', '"submodules"',
                          '"compiler"', '"inspector"',
                          '"runtime_libraries"',
                          '"device_arch"',
                          "binary_sha256", "manifest_sha256"),
                "local bundle builder")
        require(prebuilt, ("BUNDLE is required", "binary hash differs",
                           "checkout HEAD differs from prebuilt source",
                           "untracked build/validation source",
                           "recursive submodule identity differs",
                           "build input authority chain differs",
                           "binary receipt chain differs",
                           "binary ELF architecture differs",
                           "runtime_sdk", "--runtime-probe-binary",
                           "libhggc_candidates", "loaded_libhggc",
                           '"soname":match.group(1)',
                           "loaded_libhggc(identity_probe)",
                           "payload/identity-probe libhggc sets differ",
                           "ALL_EQUAL_TO_IDENTITY_PROBE",
                           "cannot identify loaded libhggc runtime libraries",
                           "shard-index.tsv", "REQUIRE_FULL",
                           "plan_fq_kpack_route_optimal.py",
                           "materialize_kpack_discovery_workloads.py",
                           '"format_cells":1381', "PILOT_WORKLOAD_LIMIT",
                           "--workload-key=", "--router-profile=",
                           "--rows-file=", "top_n=NONE",
                           "device-identity.json",
                           "--algorithm=full-output", " retain ",
                           "SF_GROUPED_COMPLETE"), "prebuilt-only runner")
        try:
            require(prebuilt.replace("loaded_libhggc(identity_probe)", "[]", 1),
                    ("loaded_libhggc(identity_probe)",),
                    "planted runtime-link authority")
        except AssertionError:
            pass
        else:
            raise AssertionError("identity-probe ldd deletion stayed green")
        if any(token in prebuilt for token in ("./build.sh", "TARGET=", "JOBS=")):
            raise AssertionError("prebuilt box runner contains a compiler/build path")
        if any(token in prebuilt for token in (
                "${DENSE_SHAPE", "${GROUPED_TOKENS", "${GROUPED_TOPK",
                "${GROUPED_EXPERTS", "${GROUPED_N", "${GROUPED_K")):
            raise AssertionError(
                "prebuilt runner retained a one-workload demo control")
        if "for q in 10 11 12 13 14" in prebuilt:
            raise AssertionError("prebuilt runner bypasses the bundle shard index")
        if ('box SDK lacks hgcc' in prebuilt or 'SDK identity differs' in prebuilt or
                'verify_sdk(' in prebuilt):
            raise AssertionError(
                "box runner incorrectly requires the build SDK/compiler identity")
        if ("--top " in prebuilt or "--top=" in prebuilt or
                "ranked[:" in analyzer):
            raise AssertionError("timing rank/top-N entered the safe screen elimination")
        if any(token in builder for token in (
                '"sources":', 'source_files =', 'SOURCE_AUTHORITY_FILES')):
            raise AssertionError(
                "bundle builder uses a hand-picked source list as authority")
        require(analyzer, ("timing_rank_used_for_elimination\": False",
                           "product census differs",
                           "not explicitly structural-unavailable",
                           "EXCLUDED_DIAGNOSTIC_ONLY",
                           "MEASURED_FULL_OUTPUT_ONLY",
                           "CANDIDATE_ONLY_NOT_WINNER"),
                "safe retention/product denominator")
        require(analyzer, ("def measurement_contract(",
                           "result authority changed on resume",
                           "stale measurement parameters stayed green"),
                "measurement resume authority")
        joined = "\n".join((header, driver, unit, cmake,
                            builder, prebuilt, analyzer))
        if "MODELED_REDUCER" in "\n".join((header, driver, unit)) or \
                "quactlize_ppu_grouped_lowbit" in joined:
            raise AssertionError(
                "discovery graph borrowed a modeled reducer or legacy grouped ABI")

        run([sys.executable, "-B",
             str(TOOLS / "scalefirst_grouped_kpack_matrix.py"), "self-test"])
        run([sys.executable, "-B", str(SHARDS), "self-test"])
        with tempfile.TemporaryDirectory(prefix="qz-sfg-kpack-") as temp:
            root = pathlib.Path(temp)
            for qtype, layout in ((10, 2), (11, 2), (12, 1), (13, 2), (14, 2)):
                out = root / f"q{qtype}"
                command = [
                    sys.executable, "-B",
                    str(TOOLS / "gen_scalefirst_grouped_kpack_units.py"),
                    "--qtype", str(qtype), "--out-dir", str(out),
                    "--per-unit", "100000"]
                run(command)
                manifest = json.loads((out / "manifest.json").read_text())
                identity = manifest["identity"]
                denominator = manifest["denominator"]
                if identity != {
                        "qtype": qtype, "weight_layout": layout,
                        "weight_layout_name": (
                            "q4-kpack4-transpose-v1" if qtype == 12 else
                            "kquant-kpack-transpose-v1"),
                        "artifact_tile_k": 0,
                        "quant_mode": "ScaleZero", "metadata_planes": 2} or \
                        denominator["compiled_rows"] != \
                        denominator["authority_typed_rows"] or \
                        denominator["compiled_rows"] <= 0 or \
                        len(manifest["units"]) != 1 or \
                        manifest["algorithms"]["full_output"] != {
                            "nonpersistent": "RAW_BIT_THEN_TIMING",
                            "persistent": "RAW_BIT_THEN_TIMING"} or \
                        manifest["algorithms"]["persistent"] != \
                        "AVAILABLE_RUNTIME_EXACT_OCCUPANCY_CAPACITY_BALANCED" or \
                        manifest["algorithms"]["cuda"] != {
                            "status": "STRUCTURAL_UNAVAILABLE",
                            "reason": "NO_CANONICAL_KPACK_CUDA_READER"} or \
                        manifest["algorithms"]["split_k"] != {
                            "S2": "STRUCTURAL_UNAVAILABLE",
                            "S4": "STRUCTURAL_UNAVAILABLE",
                            "S8": "STRUCTURAL_UNAVAILABLE"}:
                    raise AssertionError(f"q{qtype} full generated denominator differs")
                if qtype == 10:
                    watched = [out / "manifest.json",
                               out / "scalefirst_grouped_registry.inc",
                               pathlib.Path(manifest["units"][0])]
                    before = [(path.stat().st_ino, path.stat().st_mtime_ns)
                              for path in watched]
                    run(command)
                    after = [(path.stat().st_ino, path.stat().st_mtime_ns)
                             for path in watched]
                    if before != after:
                        raise AssertionError(
                            "identical grouped regeneration invalidated resumable objects")
            dense_resume = root / "dense-resume"
            dense_command = [
                sys.executable, "-B",
                str(TOOLS / "gen_scalefirst_internal_units.py"),
                "--qtype", "10", "--artifact-tk", "0", "--bchunk", "0",
                "--weight-layout", "2", "--out-dir", str(dense_resume),
                "--per-unit", "100000"]
            run(dense_command)
            dense_doc = json.loads((dense_resume / "manifest.json").read_text())
            watched = [dense_resume / "manifest.json",
                       dense_resume / "scalefirst_registry.inc",
                       pathlib.Path(dense_doc["units"][0])]
            before = [(path.stat().st_ino, path.stat().st_mtime_ns)
                      for path in watched]
            run(dense_command)
            after = [(path.stat().st_ino, path.stat().st_mtime_ns)
                     for path in watched]
            if before != after:
                raise AssertionError(
                    "identical dense regeneration invalidated resumable objects")
            # A binary-range manifest carries no repeated 10k-row rejection
            # payload. Stable IDs and the compact rejection hash bind it to
            # the same full authority.
            dense_range = root / "dense-range"
            range_command = dense_command.copy()
            range_command[range_command.index(str(dense_resume))] = str(dense_range)
            range_command.extend(["--parent-begin", "32", "--parent-count", "32"])
            run(range_command)
            range_doc = json.loads((dense_range / "manifest.json").read_text())
            if (range_doc["selection"]["mode"] != "parent-range" or
                    range_doc["parent_range"]["begin"] != 32 or
                    range_doc["parent_range"]["end"] != 64 or
                    [row["parent_id"] for row in range_doc["compiled_parents"]] !=
                    list(range(32, 64)) or "non_typed_rows" in range_doc or
                    range_doc["non_typed_authority"]["count"] !=
                    range_doc["denominator"]["non_typed_rows"]):
                raise AssertionError("dense compact parent range identity differs")
            out_of_range = run(range_command[:-4] + [
                "--parent-begin", "999999", "--parent-count", "1"], expected=2)
            if "outside authority" not in out_of_range:
                raise AssertionError("out-of-range parent shard stayed green")
            planted = run([
                sys.executable, "-B",
                str(TOOLS / "gen_scalefirst_grouped_kpack_units.py"),
                "--qtype", "10", "--out-dir", str(root / "drop"),
                "--per-unit", "100000", "--plant-drop-last"], expected=2)
            if "typed denominator" not in planted:
                raise AssertionError("drop-one grouped type negative did not fire")
            receipt_root = root / "receipt"
            receipt_root.mkdir()
            authority = receipt_root / "authority.json"
            manifest = receipt_root / "manifest.json"
            binary = receipt_root / "binary"
            receipt = receipt_root / "binary-receipt.json"
            authority.write_text('{"source":"clean-tree"}\n')
            manifest.write_text('{"typed_rows":1}\n')
            binary.write_bytes(b"linked-ppu-image-v1")
            invoke = [
                "bash", "-c",
                'source "$1"; binary_receipt "$2" "$3" "$4" "$5" "$6"',
                "sf-receipt-test", str(BUILDER),
            ]
            missing = run(invoke + ["verify", str(authority), str(manifest),
                                    str(binary), str(receipt)], expected=1)
            if "receipt is missing" not in missing:
                raise AssertionError("missing binary receipt negative did not fire")
            run(invoke + ["record", str(authority), str(manifest),
                          str(binary), str(receipt)])
            run(invoke + ["verify", str(authority), str(manifest),
                          str(binary), str(receipt)])
            binary.write_bytes(b"stale-or-substituted-ppu-image")
            stale = run(invoke + ["verify", str(authority), str(manifest),
                                  str(binary), str(receipt)], expected=1)
            if "receipt differs" not in stale:
                raise AssertionError("stale binary receipt negative did not fire")
            owned = root / "owned"
            owned.mkdir()
            outside = root / "outside"
            outside.mkdir()
            symlink = owned / "shard"
            symlink.symlink_to(outside, target_is_directory=True)
            owned_check = [
                "bash", "-c",
                'source "$1"; validate_owned_path "$2" "$3" scratch',
                "sf-owned-path-test", str(BUILDER), str(owned), str(symlink)]
            escaped = run(owned_check, expected=1)
            if "symlink" not in escaped:
                raise AssertionError("scratch symlink negative did not fire")
        print("[sf-grouped-kpack-discovery:self-test] PASS formats=5 "
              "full-generated-denominator=13679 delivery=16/32/64 ragged-empty-router=BOUND "
              "correctness-before-timing=BOUND persistent=EXACT-OCCUPANCY "
              "splitk=STRUCTURAL_UNAVAILABLE resume-mtime=STABLE "
              "binary-shards=DISJOINT-EXACT compact-reject-authority=BOUND "
              "receipt-negative=RED")
        return 0
    except (AssertionError, OSError, ValueError) as error:
        print(f"[sf-grouped-kpack-discovery] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
