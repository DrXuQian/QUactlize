#!/usr/bin/env python3
"""Local fail-closed contract for the all-format ScaleFirst runner."""

from __future__ import annotations

import os
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNNER = TOOLS / "run_scalefirst_internal_sweep_box.sh"
ANALYZER = TOOLS / "analyze_scalefirst_internal_sweep.py"


def run(command: list[str], *, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != expect:
        raise AssertionError(
            f"rc={result.returncode}, expected={expect}: {' '.join(command)}\n"
            + result.stdout[-4000:])
    return result


def check_contract(runner: str, analyzer: str) -> None:
    if any(token in runner for token in ("/tmp", "mktemp", "rm -")):
        raise AssertionError("runner reintroduced /tmp, mktemp, or destructive cleanup")
    required = (
        'INTERNAL_SWEEP_ATTEMPT_ID', 'OUT must be a strict /workspace child',
        'TMPDIR="$out/identity-probe"', 'raw-log-hashes.json',
        '--attempt-id "$attempt_id"', 'SCALEFIRST_CONFIGS_PER_UNIT',
        'SCALEFIRST_SWEEP_GENERATED_DIR', '--schedule-seed="$schedule_seed"',
        'binary/run evidence exists but plan.json is missing',
        'binary/run evidence exists but plan.sha256 is missing',
        'binary/run evidence exists without source authority',
        'binary/run evidence exists without saved device identity',
        'binary/run evidence exists but %s authority is missing',
        'run_commit="$out/raw/$shard/run.commit.json"',
        'incomplete run evidence triplet', 'run evidence authority changed',
        'mv -f -- "$current_commit" "$run_commit" || return 2',
        '--validate-bind-binary "$binary" --binary-hashes "$binary_hashes"',
        '--binary-shard "$shard" --binary-evidence 1',
        '--binary-shard "$shard" --binary-evidence 0',
    )
    missing = [token for token in required if token not in runner]
    if missing:
        raise AssertionError(f"runner lost contract tokens: {missing}")
    analyzer_required = (
        'runtime_authority(', '"runtime_hashes": runtime_hashes',
        'def validate_bind_binary(', 'stat.S_ISLNK(mode)',
        'def _persistent_grid_space(', 'persistent exact grid/mask denominator',
        'missing persistent grid stayed green', 'wrong persistent grid stayed green',
        '"run_commit_sha256": sha256(commit_path)',
        '"orchestration_attempt_id": attempt_id',
        'atomic_json(output, summary, pretty=False)',
    )
    missing = [token for token in analyzer_required if token not in analyzer]
    if missing:
        raise AssertionError(f"analyzer lost authority/attempt tokens: {missing}")
    if runner.index("# Establish resume state before materializing") > \
            runner.index('python3 -B - "$requested_spec"'):
        raise AssertionError("resume evidence is computed after authority materialization")
    if runner.index('--binary-shard "$shard" --binary-evidence 1') > \
            runner.index("build shard=%s typed=%s"):
        raise AssertionError("resume binary admission occurs after rebuild branch")


def expect_contract_red(runner: str, analyzer: str, needle: str) -> None:
    try:
        check_contract(runner, analyzer)
    except AssertionError as error:
        if needle not in str(error):
            raise
    else:
        raise AssertionError(f"planted deletion stayed green: {needle}")


def main() -> int:
    run(["bash", "-n", str(RUNNER)])
    run([sys.executable, "-B", str(ANALYZER), "--self-test"])
    run([sys.executable, "-B", str(TOOLS / "scalefirst_internal_matrix.py"),
         "self-test"])
    text = RUNNER.read_text()
    analyzer_text = ANALYZER.read_text()
    check_contract(text, analyzer_text)

    # Three structural negatives keep resume fail-closed when an authority is
    # deleted without changing the otherwise-valid runner/analyzer.
    expect_contract_red(text.replace(
        'binary/run evidence exists but plan.json is missing',
        'plan missing', 1), analyzer_text, "plan.json")
    expect_contract_red(text.replace(
        '        run_commit="$out/raw/$shard/run.commit.json"\n', '', 1),
        analyzer_text, "run_commit")
    expect_contract_red(text, analyzer_text.replace(
        '            "run_commit_sha256": sha256(commit_path),\n', '', 1),
        "run_commit_sha256")

    # Generator denominator negative: removing one typed row must fail before
    # any compiler/device work.  The explicit /workspace child obeys the same
    # artifact policy as the production runner; no mktemp or /tmp seam exists.
    scratch = pathlib.Path("/workspace") / f"quactlize-scalefirst-contract-{os.getpid()}"
    if scratch.exists():
        raise AssertionError(f"refusing pre-existing self-test directory {scratch}")
    scratch.mkdir(parents=False)
    try:
        planted = run([
            sys.executable, "-B", str(TOOLS / "gen_scalefirst_internal_units.py"),
            "--qtype", "8", "--artifact-tk", "32", "--bchunk", "0",
            "--per-unit", "64", "--out-dir", str(scratch),
            "--plant-drop-last"], expect=2)
        if "typed denominator" not in planted.stdout:
            raise AssertionError("drop-one negative did not name its denominator")

        # Byte-level evidence negatives: a valid per-shard commit passes;
        # deleting the commit or substituting the log fails with the same
        # plan/run/generated/binary authority unchanged.
        sys.path.insert(0, str(TOOLS))
        import analyze_scalefirst_internal_sweep as analyzer

        # Dynamic production-helper negative.  A resumed ordinary executable
        # passes; deleting it, replacing it with a symlink, or substituting
        # bytes must reject rather than authorize a rebuild/remeasure.
        binary_dir = scratch / "binary"
        binary_dir.mkdir()
        binary = binary_dir / "test_scalefirst_internal_sweep"
        binary.write_bytes(b"shipping-binary")
        binary.chmod(0o755)
        binary_hashes_path = scratch / "binary-hashes.json"
        binary_hashes_path.write_text(json.dumps({
            "q12-a32-bc0": hashlib.sha256(binary.read_bytes()).hexdigest()
        }) + "\n")
        analyzer.validate_bind_binary(
            binary, binary_hashes_path, "q12-a32-bc0", True)
        saved_binary = binary.read_bytes()
        binary.unlink()
        try:
            analyzer.validate_bind_binary(
                binary, binary_hashes_path, "q12-a32-bc0", True)
        except ValueError as error:
            if "missing" not in str(error):
                raise
        else:
            raise AssertionError("deleted resume binary stayed green")
        real_binary = binary_dir / "real-binary"
        real_binary.write_bytes(saved_binary)
        real_binary.chmod(0o755)
        binary.symlink_to(real_binary)
        try:
            analyzer.validate_bind_binary(
                binary, binary_hashes_path, "q12-a32-bc0", True)
        except ValueError as error:
            if "non-symlink" not in str(error):
                raise
        else:
            raise AssertionError("symlink resume binary stayed green")
        binary.unlink()
        binary.write_bytes(b"substituted-binary")
        binary.chmod(0o755)
        try:
            analyzer.validate_bind_binary(
                binary, binary_hashes_path, "q12-a32-bc0", True)
        except ValueError as error:
            if "hash changed" not in str(error):
                raise
        else:
            raise AssertionError("substituted resume binary stayed green")

        raw = scratch / "raw"
        shard = "q12-a32-bc0"
        directory = raw / shard
        directory.mkdir(parents=True)
        log = directory / "run.log"
        rc_path = directory / "run.rc"
        commit = directory / "run.commit.json"
        contract = scratch / "run-contract.json"
        log.write_text("evidence\n")
        rc_path.write_text("0\n")
        contract.write_text('{"contract":1}\n')
        generated = {shard: "1" * 64}
        binaries = {shard: "2" * 64}
        raw_hashes = {shard: hashlib.sha256(log.read_bytes()).hexdigest()}
        commit_doc = {
            "schema": analyzer.RUN_COMMIT_SCHEMA,
            "rc": 0,
            "run_log_sha256": raw_hashes[shard],
            "run_rc_sha256": hashlib.sha256(rc_path.read_bytes()).hexdigest(),
            "run_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
            "generated_source_sha256": generated[shard],
            "binary_sha256": binaries[shard],
        }
        commit.write_text(json.dumps(commit_doc, sort_keys=True) + "\n")
        analyzer.runtime_authority(
            raw, {shard}, contract, generated, binaries, raw_hashes)
        saved_commit = commit.read_bytes()
        commit.unlink()
        try:
            analyzer.runtime_authority(
                raw, {shard}, contract, generated, binaries, raw_hashes)
        except ValueError as error:
            if "runtime evidence shard set" not in str(error) and \
                    "log/rc/commit" not in str(error):
                raise
        else:
            raise AssertionError("deleted run commit stayed green")
        commit.write_bytes(saved_commit)
        log.write_text("substituted\n")
        try:
            analyzer.runtime_authority(
                raw, {shard}, contract, generated, binaries, raw_hashes)
        except ValueError as error:
            if "authority changed" not in str(error):
                raise
        else:
            raise AssertionError("substituted run log stayed green")
    finally:
        # Exact, freshly-created, strict /workspace child; never a broad or
        # caller-controlled cleanup target.
        shutil.rmtree(scratch)
    print("[scalefirst-internal-runner] PASS atomic attempt/resume/run-commit "
          "authority; negatives=drop-one-typed-row+delete-plan+delete-commit+"
          "substitute-log+delete/symlink/substitute-binary")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as error:
        print(f"[scalefirst-internal-runner] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
