#!/usr/bin/env python3
"""Prove that an FQ dense shard contains only shared-memory negatives.

The proof is intentionally independent of the linked payload's device image.
It recompiles a host census from the generated registry with the exact hgcc
compile flags used by the shard, then requires every generated parent to take
the shipping shared-memory guard.  Grouped and ScaleFirst shards are outside
this exception by construction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


TOOLS = pathlib.Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fully_quantized_kpack_bundle_index as bundle_index  # noqa: E402


PROOF_SCHEMA = "quactlize.fully_quantized_kpack_structural_proof.v1"
PAYLOAD_KIND = bundle_index.STRUCTURAL_PAYLOAD_KIND
SMEM_LIMIT = 262144
SHA256 = re.compile(r"[0-9a-f]{64}")
SYMBOL = re.compile(r"fqk_tc_q\d+_[A-Za-z0-9_]+")

SHARD_FIELDS = (
    "shard_key", "qtype", "operator", "route", "parent_begin",
    "parent_end", "parent_count", "authority_count", "parent_ids",
)

CENSUS_SOURCE = r'''#include <cstddef>
#include <cstdio>
#include "fully_quantized_splitk_producer_bench.hpp"
#include "fq_tc_registry.inc"

template <int Q, int A, int TM, int TN, int TK, int WM, int WN,
          int ST, int BC, int AP, int DN>
void fq_emit(char const* symbol) {
  using T = fq_internal_sweep::TcRowTypes<
      Q,A,TM,TN,TK,WM,WN,ST,BC,AP,FQ_TC_WEIGHT_LAYOUT,DN>;
  static_assert(
      T::Shipping::SharedStorageSize > ppu_tactics::kBlockSmemBytes,
      "structural census admitted a device-bearing FQ dense parent");
  std::printf(
      "FQ_STRUCTURAL_ROW symbol=%s q=%d A=%d tm=%d tn=%d tk=%d "
      "wm=%d wn=%d stages=%d bchunk=%d ap=%d dn=%d "
      "shipping_smem=%zu split_smem=%zu limit=%zu\n",
      symbol,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN,
      std::size_t(T::Shipping::SharedStorageSize),
      std::size_t(T::SplitKernel::SharedStorageSize),
      std::size_t(ppu_tactics::kBlockSmemBytes));
}
#define FQ_STRUCTURAL_EMIT_12(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN) \
  fq_emit<Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN>(#FN); ++count;
#define FQ_STRUCTURAL_EMIT_11(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP) \
  FQ_STRUCTURAL_EMIT_12(FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,0)
#define FQ_STRUCTURAL_PICK(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,NAME,...) NAME
#define FQ_STRUCTURAL_EMIT(...) \
  FQ_STRUCTURAL_PICK(__VA_ARGS__,FQ_STRUCTURAL_EMIT_12, \
                     FQ_STRUCTURAL_EMIT_11)(__VA_ARGS__)
int main() {
  int count = 0;
  FQ_TC_REGISTRY_ROWS(FQ_STRUCTURAL_EMIT)
  std::printf("FQ_STRUCTURAL_DONE rows=%d limit=%zu\n", count,
              std::size_t(ppu_tactics::kBlockSmemBytes));
  return 0;
}
'''

STUB_SOURCE = r'''extern "C" void** __hggcRegisterFatBinary(void*) { return nullptr; }
extern "C" void __hggcUnregisterFatBinary(void**) {}
extern "C" void __hggcRegisterVar(...) {}
extern "C" int hggcMemsetAsync(void*, int, unsigned long, void*) { return 0; }
extern "C" int hggcGetLastError() { return 0; }
'''


class ProofError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ProofError(f"{label} is not a sha256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProofError(f"{label} is not an integer >= {minimum}")
    return value


def _native_shard(document: dict[str, Any]) -> dict[str, Any]:
    try:
        return {field: copy.deepcopy(document[field]) for field in SHARD_FIELDS}
    except KeyError as error:
        raise ProofError(f"native shard lacks {error.args[0]}") from error


def validate_structural_proof(
        proof_doc: dict[str, Any], native: dict[str, Any],
        manifest_sha256: str, binary_sha256: str,
        receipt_doc: dict[str, Any] | None = None) -> None:
    """Validate an already loaded proof without trusting its filesystem."""
    if not isinstance(proof_doc, dict) or set(proof_doc) != {
            "schema", "payload_kind", "shard", "manifest_sha256",
            "binary_sha256", "source_authority", "repair_authority",
            "compile_authority", "shared_memory_limit_bytes", "rows",
            "all_rows_shipping_shared_storage"}:
        raise ProofError("FQ dense structural proof schema fields differ")
    if (proof_doc["schema"] != PROOF_SCHEMA or
            proof_doc["payload_kind"] != PAYLOAD_KIND or
            proof_doc["shard"] != _native_shard(native)):
        raise ProofError("FQ dense structural proof shard identity differs")
    shard = proof_doc["shard"]
    if shard["route"] != "fully-quantized" or shard["operator"] != "dense":
        raise ProofError("structural proof is not an FQ dense shard")
    if (proof_doc["manifest_sha256"] != manifest_sha256 or
            proof_doc["binary_sha256"] != binary_sha256):
        raise ProofError("structural proof payload hash differs")
    if proof_doc["shared_memory_limit_bytes"] != SMEM_LIMIT:
        raise ProofError("structural proof block shared-memory limit differs")
    if proof_doc["all_rows_shipping_shared_storage"] is not True:
        raise ProofError("structural proof did not close every row")

    source = proof_doc["source_authority"]
    if not isinstance(source, dict) or set(source) != {
            "build_input_authority_sha256", "source_sha", "source_tree",
            "submodules", "sdk_compiler_sha256", "sdk_inspector_sha256",
            "host_cxx_sha256"}:
        raise ProofError("structural proof source authority differs")
    for field in ("build_input_authority_sha256", "sdk_compiler_sha256",
                  "sdk_inspector_sha256", "host_cxx_sha256"):
        _sha(source.get(field), f"source_authority.{field}")
    if (not isinstance(source.get("source_sha"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", source["source_sha"]) or
            not isinstance(source.get("source_tree"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", source["source_tree"]) or
            not isinstance(source.get("submodules"), list)):
        raise ProofError("structural proof Git authority differs")

    repair = proof_doc["repair_authority"]
    if not isinstance(repair, dict) or set(repair) != {
            "source_sha", "source_tree", "tool_path", "tool_sha256"}:
        raise ProofError("structural proof repair authority differs")
    if (not re.fullmatch(r"[0-9a-f]{40}", repair.get("source_sha", "")) or
            not re.fullmatch(r"[0-9a-f]{40}", repair.get("source_tree", "")) or
            repair.get("tool_path") != "tools/fq_dense_structural_proof.py"):
        raise ProofError("structural proof repair Git identity differs")
    _sha(repair.get("tool_sha256"), "repair_authority.tool_sha256")

    compile_authority = proof_doc["compile_authority"]
    required_compile = {
        "build_make_sha256", "payload_inspector_output_sha256",
        "registry_sha256", "unit_sources", "unit_objects",
        "link_file_sha256", "link_argv_sha256",
        "census_source_sha256",
        "census_compile_argv_sha256", "census_object_sha256",
        "stub_source_sha256", "stub_compile_argv_sha256",
        "stub_object_sha256", "census_link_argv_sha256",
        "census_binary_sha256", "census_stdout_sha256", "nm_path",
        "nm_sha256", "nm_output_sha256",
    }
    if not isinstance(compile_authority, dict) or \
            set(compile_authority) != required_compile:
        raise ProofError("structural proof compile authority differs")
    for field in required_compile - {
            "unit_sources", "unit_objects", "nm_path"}:
        _sha(compile_authority.get(field), f"compile_authority.{field}")
    if (not isinstance(compile_authority["nm_path"], str) or
            not pathlib.PurePosixPath(compile_authority["nm_path"]).is_absolute()):
        raise ProofError("structural proof nm path differs")
    units = compile_authority["unit_sources"]
    if (not isinstance(units, list) or not units or
            any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or
                not SHA256.fullmatch(str(row["sha256"])) for row in units)):
        raise ProofError("structural proof unit source authority differs")
    objects = compile_authority["unit_objects"]
    if (not isinstance(objects, list) or len(objects) != len(units) or
            any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or
                not SHA256.fullmatch(str(row["sha256"]))
                for row in objects)):
        raise ProofError("structural proof unit object authority differs")

    rows = proof_doc["rows"]
    if not isinstance(rows, list) or len(rows) != shard["parent_count"]:
        raise ProofError("structural proof row denominator differs")
    expected_ids = shard["parent_ids"]
    observed_ids, observed_symbols = [], []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "parent_id", "symbol", "runtime_variants",
                "shipping_smem", "split_smem"}:
            raise ProofError(f"structural proof row {ordinal} differs")
        observed_ids.append(row["parent_id"])
        observed_symbols.append(row["symbol"])
        if (not isinstance(row["symbol"], str) or
                not SYMBOL.fullmatch(row["symbol"]) or
                row["runtime_variants"] !=
                ["TC_S1", "TC_S2", "TC_S4", "TC_S8"] or
                _integer(row["shipping_smem"], "shipping_smem", minimum=1)
                <= SMEM_LIMIT or
                _integer(row["split_smem"], "split_smem", minimum=1) <= 0):
            raise ProofError(f"row {ordinal} is not shipping-structural")
    if (observed_ids != expected_ids or
            len(set(observed_ids)) != len(observed_ids) or
            len(set(observed_symbols)) != len(observed_symbols)):
        raise ProofError("structural proof parent/symbol union differs")

    if receipt_doc is not None:
        if bundle_index.receipt_kind(receipt_doc) != PAYLOAD_KIND:
            raise ProofError("structural proof receipt kind differs")
        bundle_index.validate_receipt(
            receipt_doc, native, manifest_sha256, binary_sha256)
        if receipt_doc.get("device_arch") != "NO_DEVICE_KERNEL":
            raise ProofError("structural proof receipt device marker differs")
        shared_authority = {
            "build_input_authority_sha256":
                "build_input_authority_sha256",
            "source_sha": "source_sha",
            "source_tree": "source_tree",
            "submodules": "submodules",
            "sdk_compiler_sha256": "sdk_compiler_sha256",
            "sdk_inspector_sha256": "sdk_inspector_sha256",
        }
        for proof_field, receipt_field in shared_authority.items():
            if source.get(proof_field) != receipt_doc.get(receipt_field):
                raise ProofError(
                    f"structural proof/receipt {proof_field} differs")
        if (compile_authority["payload_inspector_output_sha256"] !=
                receipt_doc.get("inspector_output_sha256")):
            raise ProofError("structural proof/receipt inspection differs")


def _load_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProofError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProofError(f"{label} must be an object")
    return value


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True).strip()


def _payload_source_clean(root: pathlib.Path) -> bool:
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    for line in tracked.splitlines():
        path = line[3:]
        if path not in (".coord/BOX.md", ".coord/INBOX.md"):
            return False
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    guarded = ("benchmarks/", "ci/", "cmake/", "dev/", "quactlize/",
               "tools/", "third_party/")
    for path in untracked.splitlines():
        if path.startswith(guarded) or path in ("CMakeLists.txt", "build.sh"):
            return False
    return True


def _write_new(path: pathlib.Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProofError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.current.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise ProofError(f"cannot publish {path}: {error}") from error


def _generated_authority(source_root: pathlib.Path, manifest: pathlib.Path,
                         build: dict[str, Any], temporary: pathlib.Path
                         ) -> tuple[dict[str, Any], pathlib.Path, list[pathlib.Path]]:
    document = _load_json(manifest, "dense manifest")
    identity = document.get("identity") or {}
    parent_range = document.get("parent_range") or {}
    if identity.get("qtype") is None:
        raise ProofError("dense manifest lacks qtype")
    command = [
        sys.executable,
        str(source_root / "tools/gen_fully_quantized_kpack_discovery_units.py"),
        "--qtype", str(identity["qtype"]),
        "--per-unit", str(build["configuration"]["configs_per_unit"]),
        "--parent-begin", str(parent_range.get("begin")),
        "--parent-count", str(parent_range.get("count")),
        "--out-dir", str(temporary / "expected"),
    ]
    run = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, cwd=source_root)
    if run.returncode:
        raise ProofError("cannot regenerate dense authority:\n" + run.stdout[-4000:])
    expected_root = temporary / "expected"
    expected = _load_json(expected_root / "manifest.json", "expected manifest")
    actual_cmp, expected_cmp = copy.deepcopy(document), copy.deepcopy(expected)
    expected_cmp["units"] = actual_cmp.get("units")
    if actual_cmp != expected_cmp:
        raise ProofError("dense manifest differs from frozen generator")
    registry = manifest.parent / "fq_tc_registry.inc"
    if registry.read_bytes() != (expected_root / "fq_tc_registry.inc").read_bytes():
        raise ProofError("dense registry differs from frozen generator")
    expected_units = sorted((expected_root / "units").glob("*.cu"))
    actual_units = sorted((manifest.parent / "units").glob("*.cu"))
    if ([path.name for path in actual_units] !=
            [path.name for path in expected_units] or
            any(actual.read_bytes() != expected_unit.read_bytes()
                for actual, expected_unit in zip(actual_units, expected_units))):
        raise ProofError("dense unit sources differ from frozen generator")
    if document.get("units") != [str(path) for path in actual_units]:
        raise ProofError("dense manifest unit paths differ")
    return document, registry, actual_units


def _compile_argv(build_dir: pathlib.Path, sdk: pathlib.Path,
                  manifest: dict[str, Any], units: list[pathlib.Path],
                  census_source: pathlib.Path, census_object: pathlib.Path
                  ) -> tuple[list[str], pathlib.Path, list[pathlib.Path]]:
    build_make = (build_dir / "ppu_targets/CMakeFiles/"
                  "test_fully_quantized_internal_sweep.dir/build.make")
    lines = build_make.read_text(encoding="utf-8").splitlines()
    candidates: list[list[str]] = []
    for line in lines:
        if "fq_kpack_dense_unit_" not in line or " && " not in line:
            continue
        words = shlex.split(line.split(" && ", 1)[1])
        if words and pathlib.Path(words[0]).resolve() == (sdk / "bin/hgcc").resolve():
            candidates.append(words)
    if len(candidates) != len(units):
        raise ProofError("cannot recover exact dense unit compile command")
    required = {
        "-x", "hg", "-DFQ_SWEEP_ARTIFACT_TK=0",
        "-DFQ_SWEEP_BCHUNK=0", "-DPPU_PACKED_SCALE=1",
        f"-DFQ_SWEEP_QTYPE={manifest['identity']['qtype']}",
        f"-DFQ_SWEEP_WEIGHT_LAYOUT={manifest['identity']['weight_layout']}",
        f"-DFQ_TC_WEIGHT_LAYOUT={manifest['identity']['weight_layout']}",
    }
    expected_sources = {path.resolve() for path in units}
    observed_sources: set[pathlib.Path] = set()
    objects_by_source: dict[pathlib.Path, pathlib.Path] = {}
    normalized: list[list[str]] = []
    for words in candidates:
        if words.count("-c") != 1 or words.count("-o") != 1 or \
                not required.issubset(set(words)):
            raise ProofError("dense unit compile command shape differs")
        source_index = words.index("-c") + 1
        output_index = words.index("-o") + 1
        original_source = pathlib.Path(words[source_index]).resolve()
        if original_source not in expected_sources:
            raise ProofError("dense unit compile source differs")
        original_object = pathlib.Path(words[output_index]).resolve()
        object_root = (build_dir / "ppu_targets/ppu_obj").resolve()
        if (original_object.parent != object_root or
                not original_object.is_file()):
            raise ProofError("dense unit compile object differs")
        observed_sources.add(original_source)
        objects_by_source[original_source] = original_object
        candidate = list(words)
        candidate[source_index] = "<SOURCE>"
        candidate[output_index] = "<OBJECT>"
        normalized.append(candidate)
    if observed_sources != expected_sources or any(
            words != normalized[0] for words in normalized[1:]):
        raise ProofError("dense unit compile flag union differs")
    words = list(candidates[0])
    source_index, output_index = words.index("-c") + 1, words.index("-o") + 1
    words[source_index] = str(census_source)
    words[output_index] = str(census_object)
    return words, build_make, [objects_by_source[path.resolve()]
                               for path in units]


def _link_authority(build_dir: pathlib.Path,
                    unit_objects: list[pathlib.Path],
                    binary: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    link_file = (build_dir / "ppu_targets/CMakeFiles/"
                 "test_fully_quantized_internal_sweep.dir/link.txt")
    lines = link_file.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ProofError("dense shard link command differs")
    words = shlex.split(lines[0])
    if words.count("-o") != 1:
        raise ProofError("dense shard link output differs")
    cwd = (build_dir / "ppu_targets").resolve()
    output = pathlib.Path(words[words.index("-o") + 1])
    output = (cwd / output).resolve() if not output.is_absolute() else output.resolve()
    if output != (cwd / "test_fully_quantized_internal_sweep") or \
            not output.is_file() or file_sha(output) != file_sha(binary):
        raise ProofError("dense shard linked binary differs")
    linked_objects = []
    for word in words:
        if not word.endswith(".o"):
            continue
        path = pathlib.Path(word)
        linked_objects.append(
            (cwd / path).resolve() if not path.is_absolute() else path.resolve())
    if (len(unit_objects) != len(set(unit_objects)) or
            any(linked_objects.count(path) != 1 for path in unit_objects)):
        raise ProofError("dense shard unit/link object union differs")
    return link_file, words
def _parse_census(stdout: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    done = []
    for line in stdout.splitlines():
        fields = dict(token.split("=", 1) for token in line.split()[1:]
                      if "=" in token)
        if line.startswith("FQ_STRUCTURAL_ROW "):
            rows.append(fields)
        elif line.startswith("FQ_STRUCTURAL_DONE "):
            done.append(fields)
        elif line.strip():
            raise ProofError("structural census emitted an unknown line")
    expected = manifest["dense_tc_parents"]
    if len(done) != 1 or done[0] != {
            "rows": str(len(expected)), "limit": str(SMEM_LIMIT)}:
        raise ProofError("structural census completion differs")
    if len(rows) != len(expected):
        raise ProofError("structural census row count differs")
    result = []
    numeric = ("q", "A", "tm", "tn", "tk", "wm", "wn", "stages",
               "bchunk", "ap", "dn", "shipping_smem", "split_smem", "limit")
    for ordinal, (observed, wanted) in enumerate(zip(rows, expected)):
        if set(observed) != {"symbol", *numeric}:
            raise ProofError(f"structural census row {ordinal} fields differ")
        try:
            values = {field: int(observed[field]) for field in numeric}
        except ValueError as error:
            raise ProofError(f"structural census row {ordinal} is malformed") from error
        expected_values = {
            "q": wanted["qtype"], "A": wanted["artifact_tile_k"],
            "tm": wanted["tile_m"], "tn": wanted["tile_n"],
            "tk": wanted["tactic_tile_k"], "wm": wanted["warp_m"],
            "wn": wanted["warp_n"], "stages": wanted["stages"],
            "bchunk": wanted["bchunk"], "ap": wanted["a_provider"],
            "dn": wanted["resolved_delivery_n"], "limit": SMEM_LIMIT,
        }
        if (observed["symbol"] != wanted["symbol"] or
                any(values[field] != value
                    for field, value in expected_values.items()) or
                values["shipping_smem"] <= SMEM_LIMIT or
                values["split_smem"] <= 0):
            raise ProofError(f"structural census row {ordinal} is not exact")
        result.append({
            "parent_id": wanted["static_candidate_id"],
            "symbol": wanted["symbol"],
            "runtime_variants": wanted["runtime_variants"],
            "shipping_smem": values["shipping_smem"],
            "split_smem": values["split_smem"],
        })
    return result


def create_structural_proof(
        *, source_root: pathlib.Path, sdk: pathlib.Path,
        build_authority_path: pathlib.Path, shard_key: str,
        manifest_path: pathlib.Path, binary_path: pathlib.Path,
        build_dir: pathlib.Path, output_path: pathlib.Path) -> dict[str, Any]:
    source_root, sdk, build_dir = (path.resolve(strict=True) for path in
                                   (source_root, sdk, build_dir))
    manifest_path, binary_path = (path.resolve(strict=True) for path in
                                  (manifest_path, binary_path))
    build = _load_json(build_authority_path, "build input authority")
    if (build.get("source_sha") != _git(source_root, "rev-parse", "HEAD") or
            build.get("source_tree") != _git(
                source_root, "rev-parse", "HEAD^{tree}")):
        raise ProofError("frozen source differs from build authority")
    if not _payload_source_clean(source_root):
        raise ProofError("frozen source used by structural proof is dirty")
    submodules = build.get("submodules")
    if not isinstance(submodules, list):
        raise ProofError("structural proof submodule authority is absent")
    for row in submodules:
        if not isinstance(row, dict) or set(row) != {
                "path", "gitlink", "current"}:
            raise ProofError("structural proof submodule record differs")
        checkout = source_root / row["path"]
        if (_git(checkout, "rev-parse", "HEAD") != row["current"] or
                row["current"] != row["gitlink"] or
                _git(checkout, "status", "--porcelain",
                     "--untracked-files=all")):
            raise ProofError(
                f"structural proof submodule differs: {row['path']}")
    if build.get("sdk", {}).get("compiler", {}).get("sha256") != \
            file_sha(sdk / "bin/hgcc"):
        raise ProofError("structural proof compiler differs")

    inspector_run = subprocess.run(
        [str(sdk / "bin/hgobjdump"), "-lelf", str(binary_path)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if inspector_run.returncode:
        raise ProofError("structural payload inspection failed")
    if re.search(r"ELF FILE \d+ \(PPU [^)]+\)", inspector_run.stdout):
        raise ProofError("device-bearing payload must use the ordinary receipt")

    with tempfile.TemporaryDirectory(prefix="fq-dense-structural-") as name:
        temporary = pathlib.Path(name)
        manifest, registry, units = _generated_authority(
            source_root, manifest_path, build, temporary)
        parent_range = manifest["parent_range"]
        native = {
            "shard_key": shard_key, "qtype": manifest["identity"]["qtype"],
            "operator": "dense", "route": "fully-quantized",
            "parent_begin": parent_range["begin"],
            "parent_end": parent_range["end"],
            "parent_count": parent_range["count"],
            "authority_count": parent_range["authority_count"],
            "parent_ids": [row["static_candidate_id"]
                           for row in manifest["dense_tc_parents"]],
        }
        expected_key = bundle_index.shard_key(
            native["qtype"], "dense", native["parent_begin"],
            native["parent_end"])
        if shard_key != expected_key:
            raise ProofError("structural shard key differs")

        source = temporary / "census.cu"
        source.write_text(CENSUS_SOURCE, encoding="utf-8")
        census_object = temporary / "census.o"
        compile_argv, build_make, unit_objects = _compile_argv(
            build_dir, sdk, manifest, units, source, census_object)
        link_file, original_link_argv = _link_authority(
            build_dir, unit_objects, binary_path)
        compile_run = subprocess.run(
            compile_argv, cwd=build_dir / "ppu_targets", text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if compile_run.returncode or not census_object.is_file():
            raise ProofError("structural census compile failed:\n" +
                             compile_run.stdout[-4000:])

        host = build.get("host_cxx") or {}
        host_words = host.get("command")
        if not isinstance(host_words, list) or not host_words:
            raise ProofError("structural proof host compiler is absent")
        resolved = shutil.which(host_words[0])
        if (resolved is None or pathlib.Path(resolved).resolve().as_posix() !=
                host.get("resolved_path") or
                file_sha(pathlib.Path(resolved).resolve()) != host.get("sha256")):
            raise ProofError("structural proof host compiler differs")
        stub = temporary / "hggc_stub.cpp"
        stub.write_text(STUB_SOURCE, encoding="utf-8")
        stub_object = temporary / "hggc_stub.o"
        stub_argv = [*host_words, "-x", "c++", "-c", str(stub),
                     "-o", str(stub_object)]
        stub_run = subprocess.run(stub_argv, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        if stub_run.returncode or not stub_object.is_file():
            raise ProofError("structural host stub compile failed:\n" +
                             stub_run.stdout[-2000:])
        census_binary = temporary / "census"
        link_argv = [*host_words, str(census_object), str(stub_object),
                     "-o", str(census_binary)]
        link_run = subprocess.run(link_argv, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        if link_run.returncode or not census_binary.is_file():
            raise ProofError("structural census link failed:\n" +
                             link_run.stdout[-2000:])
        census = subprocess.run([str(census_binary)], text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if census.returncode:
            raise ProofError("structural census execution failed")
        rows = _parse_census(census.stdout, manifest)

        nm_path_text = shutil.which("nm")
        if nm_path_text is None:
            raise ProofError("nm is required for structural wrapper closure")
        nm_path = pathlib.Path(nm_path_text).resolve()
        nm_run = subprocess.run(
            [str(nm_path), "-C", "--defined-only", str(binary_path)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if nm_run.returncode:
            raise ProofError("cannot inspect structural host wrappers")
        wrappers = sorted(set(re.findall(
            r"fq_internal_sweep_generated::(fqk_tc_[A-Za-z0-9_]+)",
            nm_run.stdout)))
        expected_wrappers = sorted(row["symbol"] for row in rows)
        if wrappers != expected_wrappers:
            raise ProofError("structural payload wrapper union differs")

        repair_root = pathlib.Path(_git(TOOLS, "rev-parse", "--show-toplevel"))
        repair_path = pathlib.Path(__file__).resolve()
        repair_relative = repair_path.relative_to(repair_root).as_posix()
        if repair_relative != "tools/fq_dense_structural_proof.py":
            raise ProofError("repair tool repository path differs")
        try:
            _git(repair_root, "ls-files", "--error-unmatch", repair_relative)
        except subprocess.CalledProcessError as error:
            raise ProofError("repair tool is not committed") from error
        if _git(repair_root, "status", "--porcelain", "--", repair_relative):
            raise ProofError("repair tool authority is dirty")
        committed_tool = subprocess.check_output(
            ["git", "-C", str(repair_root), "show",
             f"HEAD:{repair_relative}"])
        if committed_tool != repair_path.read_bytes():
            raise ProofError("repair tool bytes differ from committed authority")
        proof = {
            "schema": PROOF_SCHEMA,
            "payload_kind": PAYLOAD_KIND,
            "shard": native,
            "manifest_sha256": file_sha(manifest_path),
            "binary_sha256": file_sha(binary_path),
            "source_authority": {
                "build_input_authority_sha256":
                    file_sha(build_authority_path),
                "source_sha": build["source_sha"],
                "source_tree": build["source_tree"],
                "submodules": build["submodules"],
                "sdk_compiler_sha256": build["sdk"]["compiler"]["sha256"],
                "sdk_inspector_sha256":
                    build["sdk"]["inspector"]["sha256"],
                "host_cxx_sha256": build["host_cxx"]["sha256"],
            },
            "repair_authority": {
                "source_sha": _git(repair_root, "rev-parse", "HEAD"),
                "source_tree": _git(repair_root, "rev-parse", "HEAD^{tree}"),
                "tool_path": repair_relative,
                "tool_sha256": file_sha(repair_path),
            },
            "compile_authority": {
                "build_make_sha256": file_sha(build_make),
                "payload_inspector_output_sha256":
                    hashlib.sha256(inspector_run.stdout.encode()).hexdigest(),
                "registry_sha256": file_sha(registry),
                "unit_sources": [
                    {"path": path.relative_to(manifest_path.parent).as_posix(),
                     "sha256": file_sha(path)} for path in units],
                "unit_objects": [
                    {"path": path.relative_to(build_dir).as_posix(),
                     "sha256": file_sha(path)} for path in unit_objects],
                "link_file_sha256": file_sha(link_file),
                "link_argv_sha256": digest(original_link_argv),
                "census_source_sha256":
                    hashlib.sha256(CENSUS_SOURCE.encode()).hexdigest(),
                "census_compile_argv_sha256": digest(compile_argv),
                "census_object_sha256": file_sha(census_object),
                "stub_source_sha256":
                    hashlib.sha256(STUB_SOURCE.encode()).hexdigest(),
                "stub_compile_argv_sha256": digest(stub_argv),
                "stub_object_sha256": file_sha(stub_object),
                "census_link_argv_sha256": digest(link_argv),
                "census_binary_sha256": file_sha(census_binary),
                "census_stdout_sha256":
                    hashlib.sha256(census.stdout.encode()).hexdigest(),
                "nm_path": nm_path.as_posix(),
                "nm_sha256": file_sha(nm_path),
                "nm_output_sha256":
                    hashlib.sha256(nm_run.stdout.encode()).hexdigest(),
            },
            "shared_memory_limit_bytes": SMEM_LIMIT,
            "rows": rows,
            "all_rows_shipping_shared_storage": True,
        }
        validate_structural_proof(
            proof, native, file_sha(manifest_path), file_sha(binary_path))
        _write_new(output_path, json.dumps(
            proof, indent=2, sort_keys=True).encode() + b"\n")
        return proof


def self_test() -> None:
    native = {
        "shard_key": "q14-dense-p00000-00002", "qtype": 14,
        "operator": "dense", "route": "fully-quantized",
        "parent_begin": 0, "parent_end": 2, "parent_count": 2,
        "authority_count": 2, "parent_ids": ["p0", "p1"],
    }
    hashes = {
        "build_make_sha256": "1" * 64,
        "payload_inspector_output_sha256": "2" * 64,
        "registry_sha256": "3" * 64,
        "unit_sources": [{"path": "units/u.cu", "sha256": "4" * 64}],
        "unit_objects": [{"path": "ppu_targets/ppu_obj/u.o",
                          "sha256": "4" * 64}],
        "link_file_sha256": "4" * 64,
        "link_argv_sha256": "4" * 64,
        "census_source_sha256": "5" * 64,
        "census_compile_argv_sha256": "6" * 64,
        "census_object_sha256": "7" * 64,
        "stub_source_sha256": "8" * 64,
        "stub_compile_argv_sha256": "9" * 64,
        "stub_object_sha256": "a" * 64,
        "census_link_argv_sha256": "b" * 64,
        "census_binary_sha256": "c" * 64,
        "census_stdout_sha256": "d" * 64,
        "nm_path": "/usr/bin/nm", "nm_sha256": "e" * 64,
        "nm_output_sha256": "f" * 64,
    }
    proof = {
        "schema": PROOF_SCHEMA, "payload_kind": PAYLOAD_KIND,
        "shard": native, "manifest_sha256": "a" * 64,
        "binary_sha256": "b" * 64,
        "source_authority": {
            "build_input_authority_sha256": "0" * 64,
            "source_sha": "1" * 40, "source_tree": "2" * 40,
            "submodules": [], "sdk_compiler_sha256": "3" * 64,
            "sdk_inspector_sha256": "8" * 64,
            "host_cxx_sha256": "4" * 64,
        },
        "repair_authority": {
            "source_sha": "5" * 40, "source_tree": "6" * 40,
            "tool_path": "tools/fq_dense_structural_proof.py",
            "tool_sha256": "7" * 64,
        },
        "compile_authority": hashes,
        "shared_memory_limit_bytes": SMEM_LIMIT,
        "rows": [{
            "parent_id": f"p{i}", "symbol": f"fqk_tc_q14_row{i}",
            "runtime_variants": ["TC_S1", "TC_S2", "TC_S4", "TC_S8"],
            "shipping_smem": 280576, "split_smem": 280576,
        } for i in range(2)],
        "all_rows_shipping_shared_storage": True,
    }
    receipt = {
        "schema": bundle_index.STRUCTURAL_RECEIPT_SCHEMA, **native,
        "payload_kind": PAYLOAD_KIND, "device_arch": "NO_DEVICE_KERNEL",
        "manifest_sha256": "a" * 64, "binary_sha256": "b" * 64,
        "build_input_authority_sha256": "0" * 64,
        "source_sha": "1" * 40, "source_tree": "2" * 40,
        "submodules": [], "sdk_compiler_sha256": "3" * 64,
        "sdk_inspector_sha256": "8" * 64,
        "inspector_output_sha256": "2" * 64,
        "structural_proof":
            "payloads/q14-dense-p00000-00002/structural-proof.json",
        "structural_proof_sha256": "c" * 64,
    }
    validate_structural_proof(
        proof, native, "a" * 64, "b" * 64, receipt)
    plants = []
    grouped = copy.deepcopy(proof); grouped["shard"]["operator"] = "grouped"
    plants.append(grouped)
    missing = copy.deepcopy(proof); missing["rows"].pop()
    plants.append(missing)
    fits = copy.deepcopy(proof); fits["rows"][0]["shipping_smem"] = SMEM_LIMIT
    plants.append(fits)
    stale = copy.deepcopy(proof); stale["binary_sha256"] = "9" * 64
    plants.append(stale)
    duplicate = copy.deepcopy(proof); duplicate["rows"][1] = copy.deepcopy(
        duplicate["rows"][0])
    plants.append(duplicate)
    failures = 0
    for planted in plants:
        try:
            validate_structural_proof(
                planted, native, "a" * 64, "b" * 64, receipt)
        except (ProofError, ValueError):
            failures += 1
    if failures != len(plants):
        raise ProofError("structural proof negative plant stayed green")
    cxx = shutil.which("c++")
    if cxx is None:
        raise ProofError("structural census self-test requires c++")
    with tempfile.TemporaryDirectory(prefix="fq-structural-self-test-") as name:
        root = pathlib.Path(name)
        (root / "fully_quantized_splitk_producer_bench.hpp").write_text(
            "#include <cstddef>\n"
            "namespace ppu_tactics { inline constexpr std::size_t "
            "kBlockSmemBytes=262144; }\n"
            "namespace fq_internal_sweep {\n"
            "template<int Q,int A,int TM,int TN,int TK,int WM,int WN,"
            "int ST,int BC,int AP,int L,int DN> struct TcRowTypes {\n"
            " struct Shipping { static constexpr std::size_t "
            "SharedStorageSize=TM==128?280576:262144; };\n"
            " struct SplitKernel { static constexpr std::size_t "
            "SharedStorageSize=Shipping::SharedStorageSize; }; }; }\n",
            encoding="utf-8")
        source = root / "census.cpp"
        source.write_text(CENSUS_SOURCE, encoding="utf-8")
        for label, tm, expected in (("over", 128, 0), ("fits", 64, 1)):
            (root / "fq_tc_registry.inc").write_text(
                "#define FQ_TC_REGISTRY_ROWS(X) "
                f"X(fqk_tc_q14_{label},14,0,{tm},256,256,64,64,2,0,0,64)\n",
                encoding="utf-8")
            run = subprocess.run(
                [cxx, "-std=c++17", "-DFQ_TC_WEIGHT_LAYOUT=2",
                 "-I", str(root), str(source), "-o", str(root / label)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if (run.returncode == 0) != (expected == 0):
                raise ProofError(
                    f"structural census compile-time {label} plant differs")
    print("[fq-dense-structural-proof:self-test] PASS exact-dense-only "
          "missing+fits+stale+duplicate+grouped=RED compile-fit=RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    create = commands.add_parser("create")
    for name in ("source-root", "sdk", "build-authority", "manifest",
                 "binary", "build-dir", "output"):
        create.add_argument(f"--{name}", type=pathlib.Path, required=True)
    create.add_argument("--shard-key", required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            proof = create_structural_proof(
                source_root=args.source_root, sdk=args.sdk,
                build_authority_path=args.build_authority,
                shard_key=args.shard_key, manifest_path=args.manifest,
                binary_path=args.binary, build_dir=args.build_dir,
                output_path=args.output)
            print("FQ_DENSE_STRUCTURAL_PROOF PASS "
                  f"shard={args.shard_key} rows={len(proof['rows'])} "
                  f"limit={proof['shared_memory_limit_bytes']} "
                  f"output={args.output}")
        return 0
    except (OSError, ProofError, subprocess.CalledProcessError,
            KeyError, TypeError, ValueError) as error:
        print(f"[fq-dense-structural-proof] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
