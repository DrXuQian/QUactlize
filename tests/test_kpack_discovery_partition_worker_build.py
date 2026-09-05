import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/build_kpack_discovery_partition_worker.sh"
RUNNER = ROOT / "tools/run_tm8_epilogue_selective_q4_box.sh"


def test_partition_worker_shell_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_partition_worker_uses_disjoint_lpt_assignment_and_both_routes() -> None:
    source = SCRIPT.read_text()
    assert "greedy-LPT" in source
    assert 'row["parents_by_route"][route]' in source
    assert 'worker=min(range(workers), key=lambda item: (loads[item], item))' in source
    assert 'action[0]=="move"' in source
    assert '("swap",left,right,li,ri,trial)' in source
    assert 'assignments[left][li],assignments[right][ri]' in source
    assert source.count('done <<<"$assigned_tasks"') == 2
    assert '("scalefirst","fully-quantized")' in source
    assert "build_scalefirst_kpack_discovery_bundle.sh" in source
    assert "build_fully_quantized_kpack_discovery_bundle.sh" in source


def test_partition_worker_reuses_one_preflight_and_publishes_only_verified_bytes() -> None:
    source = SCRIPT.read_text()
    assert source.count("kpack_global_build_preflight.py\" create") == 1
    assert "KPACK_GLOBAL_PREFLIGHT_RECEIPT=\"$preflight\"" in source
    assert source.count("kpack_discovery_build_partitions.py\" verify") >= 3
    assert 'cp -a "$out" "$stage"' in source
    assert 'mv "$stage" "$publish_dir"' in source
    assert source.count("verify_published_partition_and_maybe_remove_local") == 4
    assert source.index('REUSED_PUBLISHED %s') < source.index(
        'bash "$root/tools/build_scalefirst_kpack_discovery_bundle.sh"')
    assert 'if [ -e "$out" ] || [ -L "$out" ]; then' in source
    assert 'rm -r --one-file-system -- "${resolved_out:?}"' in source
    assert 'find "$resolved_out" -xdev -type l -print -quit' in source


def test_partition_worker_requires_local_fast_disk_and_explicit_publish_root() -> None:
    source = SCRIPT.read_text()
    assert "KPACK_PARTITION_LOCAL_ROOT" in source
    assert "KPACK_LOCAL_SCRATCH_ROOT:-/root/autodl-tmp" in source
    assert "strict configured scratch child" in source
    assert "KPACK_LOCAL_SCRATCH_ROOT is too broad" in source
    assert "regular non-symlink directory" in source
    assert "KPACK_PARTITION_PUBLISH_ROOT" in source
    assert 'refusing broad publish root' in source


def test_partition_worker_can_collect_partition_failures_for_exact_resume() -> None:
    source = SCRIPT.read_text()
    full_runner = (
        ROOT / "tools/run_kpack_postfix_full_campaign_box.sh").read_text()
    assert 'KPACK_CONTINUE_ON_PARTITION_ERROR:-0' in source
    assert 'KPACK_CONTINUE_ON_PARTITION_ERROR must be 0 or 1' in source
    assert 'PARTITION_FAIL worker=%s/%s route=%s partition=%s/%s' in source
    assert 'failures.attempt-$$.tsv' in source
    assert 'PARTIAL worker=%s/%s failure_ledger=%s' in source
    assert 'return 3' in source
    assert 'KPACK_CONTINUE_ON_PARTITION_ERROR=1' in full_runner


def test_q4_runner_binds_one_disjoint_configurable_scratch_root() -> None:
    source = RUNNER.read_text()
    assert "KPACK_LOCAL_SCRATCH_ROOT:-/root/autodl-tmp" in source
    assert 'KPACK_LOCAL_SCRATCH_ROOT="$local_parent"' in source
    assert "KPACK_LOCAL_SCRATCH_ROOT is too broad" in source
    assert "OUT may not be inside KPACK_LOCAL_SCRATCH_ROOT" in source
    assert "KPACK_LOCAL_SCRATCH_ROOT may not be inside OUT" in source
    assert 'sha256sum | cut -c1-12' in source


def _tree(root: Path, manifest: bytes = b'{"authority": 1}\n') -> None:
    root.mkdir(parents=True)
    (root / "partition-bundle.json").write_bytes(manifest)
    (root / "payload.bin").write_bytes(b"payload")


def _cleanup_command(tmp_path: Path, out: Path, publish: Path, remove: int,
                     *, fail_verify: bool = False) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$VERIFY_LOG\"\n"
        "[ \"${FAIL_VERIFY:-0}\" = 0 ]\n")
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "SCRIPT_UNDER_TEST": str(SCRIPT),
        "REPO_UNDER_TEST": str(ROOT),
        "SDK_UNDER_TEST": str(tmp_path / "sdk"),
        "LOCAL_UNDER_TEST": str(out.parent),
        "OUT_UNDER_TEST": str(out),
        "PUBLISH_UNDER_TEST": str(publish),
        "REMOVE_UNDER_TEST": str(remove),
        "VERIFY_LOG": str(tmp_path / "verify.log"),
        "FAIL_VERIFY": "1" if fail_verify else "0",
    })
    return subprocess.run(
        ["bash", "-c", """
source "$SCRIPT_UNDER_TEST"
verify_published_partition_and_maybe_remove_local \
  "$REPO_UNDER_TEST" "$SDK_UNDER_TEST" "$LOCAL_UNDER_TEST" \
  "$OUT_UNDER_TEST" "$PUBLISH_UNDER_TEST" scalefirst 0 \
  "$REMOVE_UNDER_TEST"
"""], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, check=False)


def test_optional_cleanup_defaults_off_and_removes_only_after_both_verifications(
        tmp_path: Path) -> None:
    source = SCRIPT.read_text()
    assert 'remove_local="${KPACK_REMOVE_LOCAL_AFTER_PUBLISH:-0}"' in source
    local_root = tmp_path / "local"
    out = local_root / "scalefirst-p00"
    publish = tmp_path / "publish" / "scalefirst" / "p00"
    _tree(out)
    _tree(publish)

    retained = _cleanup_command(tmp_path, out, publish, 0)
    assert retained.returncode == 0, retained.stdout
    assert out.is_dir()

    invalid = _cleanup_command(tmp_path, out, publish, 2)
    assert invalid.returncode != 0
    assert "must be 0 or 1" in invalid.stdout
    assert out.is_dir()

    removed = _cleanup_command(tmp_path, out, publish, 1)
    assert removed.returncode == 0, removed.stdout
    assert not out.exists()
    assert f"REMOVED_LOCAL {out}" in removed.stdout
    verifies = (tmp_path / "verify.log").read_text().splitlines()
    assert len(verifies) == 4  # published + local for the two valid invocations


def test_cleanup_refuses_failed_publish_verify_or_manifest_difference(
        tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    out = local_root / "scalefirst-p00"
    publish = tmp_path / "publish" / "scalefirst" / "p00"
    _tree(out)
    _tree(publish)
    failed = _cleanup_command(tmp_path, out, publish, 1, fail_verify=True)
    assert failed.returncode != 0
    assert out.is_dir()

    (publish / "partition-bundle.json").write_bytes(b'{"authority": 2}\n')
    mismatched = _cleanup_command(tmp_path, out, publish, 1)
    assert mismatched.returncode != 0
    assert "published partition authority differs" in mismatched.stdout
    assert out.is_dir()


def test_cleanup_refuses_wrong_target_and_any_symlink_in_tree(
        tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    wrong = local_root / "not-the-assigned-output"
    publish = tmp_path / "publish" / "scalefirst" / "p00"
    _tree(wrong)
    _tree(publish)
    rejected = _cleanup_command(tmp_path, wrong, publish, 1)
    assert rejected.returncode != 0
    assert "not its exact expected child" in rejected.stdout
    assert wrong.is_dir()

    out = local_root / "scalefirst-p00"
    _tree(out)
    external = tmp_path / "must-survive"
    external.write_text("user data\n")
    (out / "link").symlink_to(external)
    rejected = _cleanup_command(tmp_path, out, publish, 1)
    assert rejected.returncode != 0
    assert "tree contains a symlink" in rejected.stdout
    assert out.is_dir()
    assert external.read_text() == "user data\n"
