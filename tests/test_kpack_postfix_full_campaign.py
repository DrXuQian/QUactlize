from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import kpack_postfix_full_campaign as campaign  # noqa: E402


def test_live_full_denominator_is_exact() -> None:
    assert campaign.DEFAULT_PARTITIONS == 32
    _build, _workload, denominator = campaign.live_denominator()
    assert denominator["binary_shards"] == 2211
    assert denominator["parents"] == 70483
    assert denominator["workload_cells"] == 1381
    assert denominator["work_items"] == 339196
    assert denominator["work_items_by_qtype"] == {
        "10": 55556,
        "11": 19018,
        "12": 203018,
        "13": 17990,
        "14": 43614,
    }
    assert len(denominator["cross_product"]) == 20


def test_box_entry_is_one_full_authority_not_an_overlay() -> None:
    source = (ROOT / "tools/run_kpack_postfix_full_campaign_box.sh").read_text()
    assert "kpack_postfix_full_campaign.py\" emit-plan" in source
    assert "kpack_postfix_full_campaign.py\" check-source" in source
    assert "kpack_postfix_full_campaign.py\" finalize" in source
    assert "kpack_postfix_full_campaign.py\" bind-devices" in source
    assert "run_kpack_discovery_worker.py\" run" in source
    assert "aggregate_kpack_discovery_results.py\" aggregate" in source
    assert "stale sealed summary" in source
    assert 'if [ -f "$aggregate/summary.json"' not in source
    assert "--phase all" in source
    assert "--screen-iterations 5" in source
    assert "--confirm-iterations 11 --confirm-rounds 3" in source
    assert "--correctness-repeats 256" in source
    assert "--continue-on-atom-error" in source
    assert 'KPACK_BUILD_PARTITIONS:-32' in source
    assert 'KPACK_BUILD_WORKERS:-32' in source
    assert 'JOBS:-6' in source
    assert "nominal_build_slots" in source
    assert "are below 80% of" in source
    assert "LOW_UTILIZATION_WARNING" in source
    assert "KPACK_CONTINUE_ON_PARTITION_ERROR=1" in source
    assert "</proc/stat" in source
    assert "--qtype" not in source
    assert "old_bundle_overlay=0" in source
    assert "heuristic=NOT_YET_EMITTED" in source


def test_live_sdk_binding_rejects_changed_runtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = tmp_path / "sdk"
    paths = [
        "VERSION.txt", "bin/hgcc", "bin/hgobjdump", "lib/libhggc.so",
    ]
    for index, relative in enumerate(paths):
        path = sdk / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sdk-{index}\n".encode())
        if relative.startswith("bin/"):
            path.chmod(0o755)

    def record(relative: str) -> dict[str, object]:
        path = sdk / relative
        return {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "symlink_target": None,
        }

    catalog = {"sdk": {
        "receipt": record(paths[0]),
        "compiler": record(paths[1]),
        "inspector": record(paths[2]),
        "runtime_libraries": [record(paths[3])],
    }}
    monkeypatch.setenv("PPU_SDK", str(sdk))
    assert campaign._verify_live_sdk(catalog) == sdk
    (sdk / paths[3]).write_bytes(b"changed\n")
    with pytest.raises(campaign.CampaignError, match="live SDK file differs"):
        campaign._verify_live_sdk(catalog)
