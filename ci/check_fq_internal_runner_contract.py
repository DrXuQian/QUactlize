#!/usr/bin/env python3
"""Fail-closed source-graph contract for the FullyQuantized sweep runner."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = pathlib.Path("/workspace").resolve()


DIRECT_SOURCES = (
    "quactlize/csrc/device/ppu_dense_layout.cu",
    "quactlize/include/dense_splitk_parallel_ppu.cuh",
    "quactlize/include/gguf_bc_vecdot.hpp",
    "quactlize/include/gguf_packed_unit.hpp",
    "quactlize/include/ppu_dense_shipping_policy.hpp",
    "quactlize/include/ppu_group_schedule.hpp",
    "tests/helper.h",
)


def check_m8_shape_admission(bench: str, analyzer: str) -> None:
    required = (
        '#include "ppu_dense_shipping_policy.hpp"',
        'ppu_dense_shipping::default_config_for_m(in.m)',
        'ppu_dense_shipping::kDecodeDefault',
        'M8_DECODE_ONLY_M_GE_8',
    )
    missing = [token for token in required if token not in bench]
    if missing:
        raise ValueError(f"FQ benchmark lost TM8 decode-only admission: {missing}")
    if '"M8_DECODE_ONLY_M_GE_8"' not in analyzer:
        raise ValueError("FQ analyzer lost TM8 decode-only terminal")
    for token in ("first_bad_index", "RAW_FP16_MISMATCH",
                  "CORRECTNESS_SYNCHRONIZE"):
        if token not in bench:
            raise ValueError(f"FQ benchmark lost exact failure witness: {token}")


def resume_validator_source(runner: str) -> str:
    """Extract the exact Python decision used before a resumed shard can run."""
    command = runner.find('existing_rc="$(python3 -B - "$run_log"')
    if command < 0:
        raise AssertionError("runner resume validator invocation is missing")
    marker = "<<'PY'\n"
    start = runner.find(marker, command)
    if start < 0:
        raise AssertionError("runner resume validator heredoc is missing")
    start += len(marker)
    end = runner.find("\nPY\n", start)
    if end < 0 or ')" || return 2' not in runner[end:end + 80]:
        raise AssertionError("runner resume validator no longer fails before remeasurement")
    binary_run = runner.find('"$binary" "${shape_args[@]}"', end)
    if binary_run < 0:
        raise AssertionError("runner binary invocation is missing after resume validation")
    return runner[start:end]


def runtime_resume_decision(script: str, mutation) -> tuple[str, str]:
    """Return REUSE/REMEASURE/REJECT using the runner's exact validator."""
    root = WORKSPACE / f"quactlize-fq-run-authority-{os.getpid()}"
    if root.exists() or root.is_symlink():
        raise AssertionError(f"refusing pre-existing self-test directory {root}")
    root.mkdir()
    try:
        shard = "q12-a64-bc0"
        directory = root / shard
        directory.mkdir()
        paths = {
            "log": directory / "run.log",
            "rc": directory / "run.rc",
            "commit": directory / "run.commit.json",
            "contract": root / "run-contract.json",
            "source": root / "source-hashes.json",
            "binary": root / "binary-hashes.json",
        }
        paths["log"].write_text("committed runtime evidence\n")
        paths["rc"].write_text("0\n")
        paths["contract"].write_text("{}\n")
        generated_digest = "a" * 64
        binary_digest = "b" * 64
        paths["source"].write_text(json.dumps({
            "generated_shards": {shard: generated_digest},
        }) + "\n")
        paths["binary"].write_text(json.dumps({shard: binary_digest}) + "\n")
        paths["commit"].write_text(json.dumps({
            "schema": "quactlize.fully_quantized_internal_sweep.run_commit.v1",
            "rc": 0,
            "run_log_sha256": hashlib.sha256(paths["log"].read_bytes()).hexdigest(),
            "run_rc_sha256": hashlib.sha256(paths["rc"].read_bytes()).hexdigest(),
            "run_contract_sha256": hashlib.sha256(
                paths["contract"].read_bytes()).hexdigest(),
            "generated_source_sha256": generated_digest,
            "binary_sha256": binary_digest,
        }, sort_keys=True) + "\n")
        if mutation is not None:
            mutation(paths)
        result = subprocess.run(
            [sys.executable, "-B", "-", str(paths["log"]), str(paths["rc"]),
             str(paths["commit"]), str(paths["contract"]), str(paths["source"]),
             str(paths["binary"]), shard],
            input=script, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        diagnostic = result.stdout.strip()
        if result.returncode != 0:
            return "REJECT", diagnostic
        return ("REUSE" if diagnostic == "0" else "REMEASURE"), diagnostic
    finally:
        if (root.parent != WORKSPACE or
                not root.name.startswith("quactlize-fq-run-authority-") or
                root.is_symlink()):
            raise AssertionError(f"unsafe self-test cleanup target {root}")
        shutil.rmtree(root)


def check_runtime_resume_negatives(runner: str) -> None:
    """Deleted or replaced committed sidecars must reject, never rerun."""
    script = resume_validator_source(runner)
    baseline, diagnostic = runtime_resume_decision(script, None)
    if baseline != "REUSE":
        raise AssertionError(
            f"intact committed run is not reusable: {baseline}: {diagnostic}")
    mutations = (
        ("deleted run.log", lambda paths: paths["log"].unlink(),
         "incomplete run evidence triplet"),
        ("replaced run.log", lambda paths: paths["log"].write_text("replacement\n"),
         "run evidence authority changed"),
        ("deleted run.rc", lambda paths: paths["rc"].unlink(),
         "incomplete run evidence triplet"),
        ("replaced run.rc", lambda paths: paths["rc"].write_text("7\n"),
         "run evidence authority changed"),
        ("deleted run.commit.json", lambda paths: paths["commit"].unlink(),
         "incomplete run evidence triplet"),
        ("replaced run.commit.json",
         lambda paths: paths["commit"].write_text("{}\n"),
         "run evidence authority changed"),
    )
    for label, mutation, expected in mutations:
        decision, diagnostic = runtime_resume_decision(script, mutation)
        if decision != "REJECT" or expected not in diagnostic:
            raise AssertionError(
                f"{label} did not reject before remeasurement: "
                f"decision={decision} diagnostic={diagnostic!r}")


def check_binary_deletion_negative(runner: str) -> None:
    """Exercise the runner's exact committed-binary guard before build."""
    lines = runner.splitlines()
    assignment = next((index for index, line in enumerate(lines)
                       if 'binary="$out/build/$shard/ppu_targets/' in line), -1)
    guard_start = next((index for index in range(assignment + 1, len(lines))
                        if lines[index].strip() ==
                        'if [ "$shard_evidence" = 1 ] && \\'), -1)
    build_start = next((index for index in range(guard_start + 1, len(lines))
                        if lines[index].strip().startswith(
                            'if [ ! -f "$binary" ] ||')), -1)
    if assignment < 0 or guard_start < 0 or build_start < 0:
        raise AssertionError("runner committed-binary pre-build guard is missing")
    guard = "\n".join(
        line[8:] if line.startswith("        ") else line
        for line in lines[guard_start:build_start])
    script = (
        "set -u\n"
        "binary=$1\nshard=q12-a64-bc0\nshard_evidence=$2\n"
        "guard_then_build() {\n" + guard + "\nprintf 'BUILD\\n'\n}\n"
        "guard_then_build\n")
    root = WORKSPACE / f"quactlize-fq-binary-guard-{os.getpid()}"
    if root.exists() or root.is_symlink():
        raise AssertionError(f"refusing pre-existing self-test directory {root}")
    root.mkdir()
    try:
        binary = root / "device-binary"
        binary.write_text("fixture\n")
        binary.chmod(0o755)

        def decide(evidence: int) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["bash", "-c", script, "fq-binary-guard", str(binary),
                 str(evidence)], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False)

        intact = decide(1)
        if intact.returncode != 0 or intact.stdout.strip() != "BUILD":
            raise AssertionError(
                f"intact committed binary was rejected: {intact.stdout!r}")
        binary.unlink()
        deleted = decide(1)
        if (deleted.returncode == 0 or
                "lost its exact binary" not in deleted.stdout or
                "BUILD" in deleted.stdout):
            raise AssertionError(
                "deleted committed binary reached build/remeasure: " +
                deleted.stdout)
        fresh = decide(0)
        if fresh.returncode != 0 or fresh.stdout.strip() != "BUILD":
            raise AssertionError("fresh missing binary did not reach build")
    finally:
        if (root.parent != WORKSPACE or
                not root.name.startswith("quactlize-fq-binary-guard-") or
                root.is_symlink()):
            raise AssertionError(f"unsafe self-test cleanup target {root}")
        shutil.rmtree(root)


def check(runner: str, generator: str, cmake: str, build: str,
          root_cmake: str, analyzer: str) -> None:
    if 'parser.add_argument("--out-dir"' not in generator:
        raise ValueError("generator no longer declares --out-dir")
    invocation = re.search(
        r'gen_fully_quantized_splitk_producer_units\.py"\s*\\\n'
        r'(?:.*\\\n){0,4}.*--out-dir "\$generated"', runner)
    if invocation is None or '--out "$generated"' in runner:
        raise ValueError("runner does not pass the generator's exact --out-dir ABI")
    if '"-I${_FQ_SWEEP_GENERATED_DIR}"' not in cmake:
        raise ValueError("generated registry include is absent from hgcc device flags")
    if "${FQ_TC_GENERATED_UNIT_SOURCES}" not in cmake:
        raise ValueError("generated device units are absent from the target source graph")
    if "test_fully_quantized_internal_sweep" not in cmake:
        raise ValueError("FullyQuantized executable target is absent")
    if (root_cmake.count(
            'include("${QZ_INTERNAL_SWEEP_CMAKE_AUTHORITY}/fq_internal_sweep.cmake.in")') != 1 or
            root_cmake.count(
            'include("${QZ_INTERNAL_SWEEP_CMAKE_AUTHORITY}/scalefirst_internal_sweep.cmake.in")') != 1 or
            'QZ_INTERNAL_SWEEP_CMAKE_AUTHORITY must be an absolute' not in root_cmake or
            'internal-sweep CMake authority lacks ${_qz_internal_sweep_fragment}' not in root_cmake):
        raise ValueError(
            "root CMake does not bind both internal-sweep fragments to one fail-closed authority")
    for variable in (
        "FQ_SWEEP_GENERATED_DIR", "FQ_SWEEP_QTYPE", "FQ_SWEEP_ARTIFACT_TK",
        "FQ_SWEEP_BCHUNK", "FQ_SWEEP_PACKED_FORMAT",
    ):
        if variable not in build:
            raise ValueError(f"build.sh does not forward {variable}")
    if '--out-dir "$generated"' not in runner:
        raise ValueError("runner/generator output-directory seam is unbound")
    if 'plan-only cannot be COMPLETE' not in runner:
        raise ValueError("runner lost its no-device fail-close")
    if 'splitk_scope=PRODUCER_ONLY_REDUCER_UNTIMED_CORRECTNESS' not in runner:
        raise ValueError("runner lost its producer-only metric-scope disclosure")
    if ('plan_source_sha' not in runner or 'spec_sha' not in runner or
            'differs from materialized plan' not in runner):
        raise ValueError("runner resume no longer binds the current inventory bytes")
    empty_guard = 'if [ "$typed" -eq 0 ]; then'
    if (runner.count(empty_guard) != 1 or
            'static-only shard=%s; no binary required typed=0' not in runner or
            runner.index(empty_guard) > runner.index(
                "build shard=%s typed=%s")):
        raise ValueError("runner lost the typed-zero no-binary guard")
    if ('if output.exists():' not in runner or
            'previous.get("generated_shards")' not in runner or
            'source-hashes authority changed on resume' not in runner):
        raise ValueError("runner resets or fails to compare generated authority on resume")
    if ('--output "$identity_current"' not in runner or
            'device identity changed inside resumed bundle' not in runner or
            'if [ ! -s "$identity" ]' in runner):
        raise ValueError("runner does not re-measure device identity on every resume")
    if ('run-contract.json' not in runner or
            'run contract changed inside resumed bundle' not in runner or
            '"iterations": iterations' not in runner or
            '"correctness_repeats": repeats' not in runner or
            '"configs_per_unit": per_unit' not in runner or
            '"peak_tflops": peak_tflops' not in runner or
            '"hbm_gbs": hbm_gbs' not in runner or
            '"plan_sha256": sys.argv[8]' not in runner or
            '"source_state_sha256": sys.argv[9]' not in runner or
            '"identity_sha256"' not in runner):
        raise ValueError("runner does not bind completed logs to a run contract")
    if ('TMPDIR="$out/identity-probe"' not in runner or
            '"identity_probe_tmpdir": "identity-probe"' not in runner):
        raise ValueError("identity probe temporary files are not bundle-local/bound")
    for tree in (
        "tree/quactlize/include",
        "tree/third_party/actlize/include",
        "tree/third_party/actlize/tools/util/include",
    ):
        if tree not in runner or tree not in analyzer:
            raise ValueError(f"transitive dependency-tree authority missing: {tree}")
    for evidence in (
        "binary/run.log exists but saved device identity is missing",
        "binary/run.log exists but source-hashes authority is missing",
        "binary/run.log exists but binary-hashes authority is missing",
        "binary/run.log exists but run contract is missing",
        "binary/run.log shard lost generated authority",
        "binary/run.log shard lost binary authority",
        "committed shard=%s lost its exact binary",
        "incomplete run evidence triplet",
        "run evidence authority changed",
    ):
        if evidence not in runner:
            raise ValueError(f"completed-runtime authority deletion is not fail-closed: {evidence}")
    for write in (
        'atomic_text "$out/plan.sha256" "$plan_sha" || return 2',
        'atomic_text "$out/source-state.sha256" "$source_state" || return 2',
        'mv -f -- "$plan_current" "$plan" || return 2',
        'mv -f -- "$source_patch_current" "$out/source.patch" || return 2',
        'mv -f -- "$run_commit_current" "$run_commit" || return 2',
        'mv -f -- "$provenance_current" "$out/provenance.txt" || return 2',
    ):
        if write not in runner:
            raise ValueError(f"sidecar write is not atomic/fail-closed: {write}")
    if (runner.count("def atomic_write(path, text):") < 4 or
            runner.count("os.replace(temporary, path)") < 4):
        raise ValueError("JSON authority sidecars are not atomically published")
    for source in DIRECT_SOURCES:
        if runner.count(source) < 2 or source not in analyzer:
            raise ValueError(f"direct compiled dependency absent from source authority: {source}")
    if ('_require_runtime_parent(row, parent, "TC")' not in analyzer or
            '_require_runtime_parent(row, parent, "BC")' not in analyzer or
            'FQ_SHAPE_DONE set differs from requested shape set' not in analyzer):
        raise ValueError("analyzer does not bind runtime records to parent shard/shapes")
    if ('f"generated/{name}"' not in analyzer or
            '"generated_source_hashes": generated_hashes' not in analyzer):
        raise ValueError("generated source authority is not published in summary provenance")
    if ('cell_models != declared_models' not in analyzer or
            'set(cells_by_model) != set(by_id)' not in analyzer):
        raise ValueError("inventory/plan model-set equality is not fail-closed")
    if ('--run-contract' not in runner or '"run_contract": run_contract' not in analyzer or
            'checked_samples(runtime, expected_samples)' not in analyzer):
        raise ValueError("run contract/sample denominator is absent from final summary")
    if ('run_commit="$out/raw/$shard/run.commit.json"' not in runner or
            'runtime_authority(' not in analyzer or
            '"runtime_hashes": runtime_hashes' not in analyzer):
        raise ValueError("per-shard runtime log/rc authority is not committed/published")
    if ('"peak_tflops": peak_tflops' not in analyzer or
            '"hbm_gbs": hbm_gbs' not in analyzer):
        raise ValueError("summary does not publish its metric denominators")
    if ('--attempt-id "$attempt_id"' not in runner or
            '"orchestration_attempt_id": attempt_id' not in analyzer or
            'f".current.{os.getpid()}"' not in analyzer or
            'os.replace(current, output)' not in analyzer):
        raise ValueError("current attempt ID/atomic summary finalization is unbound")


def main() -> int:
    paths = {
        "runner": ROOT / "tools/run_fully_quantized_internal_sweep_box.sh",
        "generator": ROOT / "tools/gen_fully_quantized_splitk_producer_units.py",
        "cmake": ROOT / "quactlize/csrc/fq_internal_sweep.cmake.in",
        "build": ROOT / "build.sh",
        "root_cmake": ROOT / "quactlize/csrc/CMakeLists.txt.in",
        "analyzer": ROOT / "tools/analyze_fully_quantized_internal_sweep.py",
    }
    texts = {name: path.read_text() for name, path in paths.items()}
    check(**texts)
    bench = (ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp").read_text()
    check_m8_shape_admission(bench, texts["analyzer"])

    # Same compiled row, same analyzer: changing only the shape-admission
    # terminal must be caught.  Different fixture files would not isolate the
    # contract variable that failed on the real M=64 sweep.
    planted_m8 = bench.replace("M8_DECODE_ONLY_M_GE_8", "M8_ANY_M", 1)
    try:
        check_m8_shape_admission(planted_m8, texts["analyzer"])
    except ValueError as error:
        if "TM8" not in str(error):
            raise
    else:
        raise AssertionError("TM8 prefill admission plant stayed green")

    # Negative 1 changes only the accepted generator flag.  This is the exact
    # seam that previously let a runner survive local review yet fail before
    # generating its first device unit.
    planted_flag = texts["runner"].replace(
        '--out-dir "$generated"', '--out "$generated"', 1)
    try:
        check(planted_flag, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "out-dir" not in str(error):
            raise
    else:
        raise AssertionError("wrong generator flag stayed green")

    # Negative 2 removes only the device preprocessor include.  A host target
    # include is not evidence that hgcc can see the generated registry.
    planted_include = texts["cmake"].replace(
        '    "-I${_FQ_SWEEP_GENERATED_DIR}"\n', "", 1)
    try:
        check(texts["runner"], texts["generator"], planted_include,
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "hgcc" not in str(error):
            raise
    else:
        raise AssertionError("missing hgcc generated include stayed green")

    # Negative 3 plants the original resume reset while changing no generated
    # file.  It must be caught before a previous shard digest can be forgotten.
    planted_reset = texts["runner"].replace(
        'generated = previous.get("generated_shards")', 'generated = {}', 1)
    try:
        check(planted_reset, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "generated authority" not in str(error):
            raise
    else:
        raise AssertionError("generated-authority resume reset stayed green")

    # Negative 4 changes only the identity probe destination back to the saved
    # file, recreating the stale-resume behavior.
    planted_identity = texts["runner"].replace(
        '--output "$identity_current"', '--output "$identity"', 1)
    try:
        check(planted_identity, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "identity" not in str(error):
            raise
    else:
        raise AssertionError("stale resume identity stayed green")

    # Negative 5 removes one actual direct dependency from only the runner's
    # hash set.  Two independently complete-looking lists must not stay green.
    planted_source = texts["runner"].replace(
        ' "tests/helper.h",\n', '', 1)
    try:
        check(planted_source, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "direct compiled dependency" not in str(error):
            raise
    else:
        raise AssertionError("missing direct source authority stayed green")

    # Negative 6 removes only the parent-shard binding from TC runtime rows.
    planted_parent = texts["analyzer"].replace(
        '_require_runtime_parent(row, parent, "TC")', '', 1)
    try:
        check(texts["runner"], texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], planted_parent)
    except ValueError as error:
        if "parent shard" not in str(error):
            raise
    else:
        raise AssertionError("cross-shard runtime acceptance stayed green")

    # Negative 7 drops one semantic timing knob from the persisted contract.
    planted_contract = texts["runner"].replace(
        '    "correctness_repeats": repeats,\n', '', 1)
    try:
        check(planted_contract, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "run contract" not in str(error):
            raise
    else:
        raise AssertionError("unbound correctness-repeats stayed green")

    # Negative 8 validates publication, not merely private input checking.
    planted_publish = texts["analyzer"].replace(
        '        f"generated/{name}": digest\n', '', 1)
    try:
        check(texts["runner"], texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], planted_publish)
    except ValueError as error:
        if "published" not in str(error):
            raise
    else:
        raise AssertionError("unpublished generated authority stayed green")

    # Negative 9 removes the one saved-identity deletion guard while leaving
    # normal identity comparison intact.
    planted_deletion = texts["runner"].replace(
        '        raise SystemExit("binary/run.log exists but saved device identity is missing")\n',
        '', 1)
    try:
        check(planted_deletion, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "authority deletion" not in str(error):
            raise
    else:
        raise AssertionError("deleted saved identity stayed green on resume")

    # Negative 10 removes the per-shard commit marker while leaving run.log
    # and run.rc handling intact.  A partial pair must never be reusable.
    planted_run_commit = texts["runner"].replace(
        '        run_commit="$out/raw/$shard/run.commit.json"\n', '', 1)
    try:
        check(planted_run_commit, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "runtime log/rc authority" not in str(error) and \
                "authority deletion" not in str(error):
            raise
    else:
        raise AssertionError("deleted per-shard run commit stayed green")

    # Negative 11 restores the exact empty-shard failure from the real
    # three-model run: q12-a32-bc0 had typed_rows=0, yet the runner entered
    # build.sh.  Removing only the typed-zero branch must be detected before a
    # compiler can turn an empty runtime graph into an unrelated build red.
    planted_empty_build = texts["runner"].replace(
        'if [ "$typed" -eq 0 ]; then', 'if false; then', 1)
    try:
        check(planted_empty_build, texts["generator"], texts["cmake"],
              texts["build"], texts["root_cmake"], texts["analyzer"])
    except ValueError as error:
        if "typed-zero" not in str(error):
            raise
    else:
        raise AssertionError("typed-zero shard calling build stayed green")

    # Negative 12 deletes only one of the two fragment includes from the root
    # CMake graph.  A valid FQ fragment does not excuse a missing ScaleFirst
    # fragment (or vice versa); the overlay must bind the pair to one source
    # authority rather than silently configure a partial graph.
    planted_fragment = texts["root_cmake"].replace(
        'include("${QZ_INTERNAL_SWEEP_CMAKE_AUTHORITY}/scalefirst_internal_sweep.cmake.in")',
        '', 1)
    try:
        check(texts["runner"], texts["generator"], texts["cmake"],
              texts["build"], planted_fragment, texts["analyzer"])
    except ValueError as error:
        if "both internal-sweep fragments" not in str(error):
            raise
    else:
        raise AssertionError("missing component fragment stayed green")

    # Exercise the runner's exact pre-run resume decision.  The analyzer already
    # deletes run.rc dynamically; cover all three committed sidecars here, both
    # missing and replaced, and require rejection rather than a fresh benchmark.
    check_runtime_resume_negatives(texts["runner"])
    check_binary_deletion_negative(texts["runner"])

    print("[fq-internal-runner] PASS: exact --out-dir ABI, hgcc generated "
          "include, generated unit graph, five build variables, no-device "
          "fail-close, inventory/run/device resume identity, all-static guard, "
          "published generated/direct-source authority, parent-shard binding, "
          "atomic authorities, and producer-only disclosure; twelve seam plus "
          "six dynamic run-sidecar and one delete-binary negative red")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError) as error:
        print(f"[fq-internal-runner] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
