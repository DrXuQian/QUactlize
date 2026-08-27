#!/usr/bin/env python3
"""Validate a completed K-pack4 pilot bundle for measurement reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile


MAPPING_ID = "0x51344b5034540001"
MEASUREMENT_REL = (
    "benchmarks/fq_q4k_decode_real_shapes_policy.json",
    "benchmarks/test_fully_quantized_internal_sweep.cu",
    "benchmarks/fully_quantized_splitk_producer_bench.hpp",
    "benchmarks/fully_quantized_splitk_producer_unit.inc",
    "quactlize/include/q4_kpack4_offline.hpp",
    "quactlize/include/fpA_intB_ppu.cuh",
    "quactlize/include/ppu_mixed_policy.hpp",
    "quactlize/include/ppu_placed_arrangement.hpp",
    "quactlize/include/actlize_extensions/cutlass/gemm/collective/"
    "quactlize_mma_mixed_input.hpp",
    "quactlize/csrc/device/ppu_dense_layout.cu",
    "quactlize/csrc/fq_internal_sweep.cmake.in",
    "tools/fully_quantized_internal_matrix.py",
    "tools/gen_fully_quantized_splitk_producer_units.py",
    "tools/analyze_fq_q4k_decode_real_shapes.py",
    "ci/check_fq_q4k_kpack4_generator.py",
    "build.sh",
)


class BundleError(ValueError):
    pass


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority_rows(path: pathlib.Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise BundleError(f"{path}: authority row is malformed")
        rows.append((match.group(2), match.group(1)))
    return rows


def one_suffix(rows: list[tuple[str, str]], suffix: str, label: str) -> str:
    values = [digest for path, digest in rows if path.endswith(suffix)]
    if len(values) != 1:
        raise BundleError(f"{label} authority lacks exactly one {suffix}")
    return values[0]


def validate(root: pathlib.Path, bundle: pathlib.Path) -> dict[str, str]:
    root = root.resolve()
    bundle = bundle.resolve()
    manifest = bundle / "generated/manifest.json"
    binary = bundle / "build/ppu_targets/test_fully_quantized_internal_sweep"
    source_authority = bundle / "source-authority.sha256"
    binary_authority = bundle / "results/binary.sha256"
    summary = bundle / "results/summary.json"
    result_authority = bundle / "results/authority.sha256"
    required = (manifest, binary, source_authority, binary_authority, summary,
                result_authority)
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not os.access(binary, os.X_OK) or binary.is_symlink():
        raise BundleError(f"pilot bundle is incomplete: missing={missing}")

    source_lines = source_authority.read_text().splitlines()
    if len(source_lines) < 3 or \
            not re.fullmatch(r"[0-9a-f]{40}", source_lines[0]) or \
            not re.fullmatch(r"[0-9a-f]{40}", source_lines[1]):
        raise BundleError("pilot source authority header is malformed")
    actlize = subprocess.check_output(
        ["git", "-C", str(root / "third_party/actlize"), "rev-parse", "HEAD"],
        text=True).strip()
    if actlize != source_lines[1]:
        raise BundleError("pilot actlize authority differs")
    source_records = authority_rows_from_lines(source_lines[2:], source_authority)
    for rel in MEASUREMENT_REL:
        matches = [(path, digest) for path, digest in source_records
                   if path == rel or path.endswith("/" + rel)]
        if len(matches) != 1:
            raise BundleError(f"pilot source authority lacks exact {rel}")
        if sha256(root / rel) != matches[0][1]:
            raise BundleError(f"measurement source changed: {rel}")

    manifest_value = json.loads(manifest.read_text())
    expected_identity = {
        "qtype": 12, "format": "Q4_K", "artifact_tile_k": 0,
        "bchunk": 0, "tile_m_filter": 8, "weight_layout": "q4-kpack4",
    }
    if manifest_value.get("identity") != expected_identity or \
            manifest_value.get("denominator", {}).get("typed_rows") != 72 or \
            manifest_value.get("weight_mapping", {}).get("mapping_id") != MAPPING_ID:
        raise BundleError("pilot manifest identity differs")

    binary_sha = sha256(binary)
    binary_fields = binary_authority.read_text().split()
    if len(binary_fields) < 1 or binary_fields[0] != binary_sha:
        raise BundleError("pilot binary hash differs")
    summary_value = json.loads(summary.read_text())
    if summary_value.get("schema") != "quactlize.fq_q4k_kpack4_pilot.v1" or \
            summary_value.get("shape") != [1, 1024, 5120] or \
            summary_value.get("typed_rows") != 72 or \
            summary_value.get("layout") != "q4-kpack4-transpose-v1" or \
            summary_value.get("weight_mapping_id") != MAPPING_ID:
        raise BundleError("pilot result did not close the admitted identity")
    manifest_sha = sha256(manifest)
    summary_sha = sha256(summary)
    if summary_value.get("manifest_sha256") != manifest_sha:
        raise BundleError("pilot summary/manifest hash differs")

    final_records = authority_rows(result_authority)
    if one_suffix(final_records, "/generated/manifest.json", "pilot result") != manifest_sha or \
            one_suffix(final_records, "/results/summary.json", "pilot result") != summary_sha or \
            one_suffix(final_records,
                       "/ppu_targets/test_fully_quantized_internal_sweep",
                       "pilot result") != binary_sha:
        raise BundleError("pilot final manifest/binary/summary authority differs")
    return {
        "measurement_git_sha": source_lines[0],
        "actlize_git_sha": source_lines[1],
        "manifest_sha256": manifest_sha,
        "binary_sha256": binary_sha,
        "summary_sha256": summary_sha,
    }


def authority_rows_from_lines(lines: list[str], path: pathlib.Path
                              ) -> list[tuple[str, str]]:
    rows = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise BundleError(f"{path}: authority row is malformed")
        rows.append((match.group(2), match.group(1)))
    return rows


def write_authority(path: pathlib.Path,
                    rows: list[tuple[str, pathlib.Path]]) -> None:
    path.write_text("".join(f"{sha256(source)}  {label}\n"
                            for label, source in rows))


def self_test(root: pathlib.Path) -> None:
    root = root.resolve()
    actlize = subprocess.check_output(
        ["git", "-C", str(root / "third_party/actlize"), "rev-parse", "HEAD"],
        text=True).strip()
    with tempfile.TemporaryDirectory(prefix="qz-kpack4-pilot-bundle-") as temp:
        bundle = pathlib.Path(temp) / "moved-pilot"
        manifest = bundle / "generated/manifest.json"
        binary = bundle / "build/ppu_targets/test_fully_quantized_internal_sweep"
        results = bundle / "results"
        manifest.parent.mkdir(parents=True)
        binary.parent.mkdir(parents=True)
        results.mkdir(parents=True)
        manifest.write_text(json.dumps({
            "identity": {"qtype": 12, "format": "Q4_K",
                         "artifact_tile_k": 0, "bchunk": 0,
                         "tile_m_filter": 8, "weight_layout": "q4-kpack4"},
            "denominator": {"typed_rows": 72},
            "weight_mapping": {"mapping_id": MAPPING_ID},
        }, sort_keys=True) + "\n")
        binary.write_bytes(b"synthetic-kpack4-binary\n")
        binary.chmod(0o755)
        summary = results / "summary.json"
        summary.write_text(json.dumps({
            "schema": "quactlize.fq_q4k_kpack4_pilot.v1",
            "shape": [1, 1024, 5120], "typed_rows": 72,
            "layout": "q4-kpack4-transpose-v1",
            "weight_mapping_id": MAPPING_ID,
            "manifest_sha256": sha256(manifest),
        }, sort_keys=True) + "\n")
        source_rows = [(f"/original/repo/{rel}", root / rel)
                       for rel in MEASUREMENT_REL]
        source_rows.append(("/original/pilot/generated/manifest.json", root / "build.sh"))
        source = bundle / "source-authority.sha256"
        source.write_text("1" * 40 + "\n" + actlize + "\n")
        with source.open("a") as stream:
            for label, path in source_rows:
                stream.write(f"{sha256(path)}  {label}\n")
        (results / "binary.sha256").write_text(
            f"{sha256(binary)}  /original/pilot/build/ppu_targets/"
            "test_fully_quantized_internal_sweep\n")
        write_authority(results / "authority.sha256", [
            ("/original/pilot/generated/manifest.json", manifest),
            ("/original/pilot/build/ppu_targets/test_fully_quantized_internal_sweep", binary),
            ("/original/pilot/results/summary.json", summary),
        ])
        validate(root, bundle)

        final_authority = results / "authority.sha256"
        original_final = final_authority.read_text()
        negatives = []
        negatives.append((final_authority, original_final.replace(
            sha256(manifest), "0" * 64, 1), "final-manifest"))
        original_summary = summary.read_text()
        negatives.append((summary, original_summary.replace(
            sha256(manifest), "0" * 64, 1), "summary-manifest"))
        binary_hash = results / "binary.sha256"
        original_binary_hash = binary_hash.read_text()
        negatives.append((binary_hash, original_binary_hash.replace(
            sha256(binary), "0" * 64, 1), "binary"))
        for path, broken, label in negatives:
            original = path.read_text()
            path.write_text(broken)
            try:
                validate(root, bundle)
            except BundleError:
                pass
            else:
                raise AssertionError(f"{label} negative stayed green")
            path.write_text(original)
    print("[fq-kpack4-pilot-bundle:self-test] PASS moved/prebuild-manifest "
          "positive; final manifest, summary binding and binary negatives RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    self_parser = sub.add_parser("self-test")
    self_parser.add_argument("--root", type=pathlib.Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--root", type=pathlib.Path, required=True)
    validate_parser.add_argument("--bundle", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test(args.root)
        else:
            result = validate(args.root, args.bundle)
            print("[fq-kpack4-pilot-bundle] PASS "
                  f"measurement_sha={result['measurement_git_sha']} "
                  f"manifest={result['manifest_sha256']} "
                  f"binary={result['binary_sha256']}")
        return 0
    except (AssertionError, BundleError, json.JSONDecodeError, OSError,
            subprocess.CalledProcessError) as error:
        print(f"[fq-kpack4-pilot-bundle] FAIL: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
