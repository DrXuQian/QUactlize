from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/build_kpack_discovery_partition_worker.sh"


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
    assert "rm -" not in source


def test_partition_worker_requires_local_fast_disk_and_explicit_publish_root() -> None:
    source = SCRIPT.read_text()
    assert "KPACK_PARTITION_LOCAL_ROOT" in source
    assert "/root/autodl-tmp" in source
    assert "KPACK_PARTITION_PUBLISH_ROOT" in source
    assert 'refusing broad publish root' in source
