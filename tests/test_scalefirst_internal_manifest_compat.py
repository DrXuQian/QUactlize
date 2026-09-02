"""ScaleFirst internal-sweep generated-manifest compatibility tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_scalefirst_internal_sweep as analyzer  # noqa: E402
import gen_scalefirst_internal_units as generator  # noqa: E402


def generate_v3(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "generated"
    shard = root / "q12-a64-bc0"
    document = generator.generate(
        12, 64, 0, shard, 100_000, False)
    return root, document


def as_historical_v2(document: dict) -> dict:
    result = copy.deepcopy(document)
    result["schema"] = analyzer.GENERATED_SHARD_V2
    result["denominator"].pop("authority_typed_rows")
    result.pop("parent_range")
    result.pop("compiled_parents")
    result.pop("non_typed_authority")
    result.pop("selection")
    for row in result["typed_rows"]:
        row.pop("parent_id")
    return result


def test_generator_v3_is_deterministic_and_accepted(tmp_path):
    root, first = generate_v3(tmp_path)
    manifest = root / "q12-a64-bc0/manifest.json"
    first_bytes = manifest.read_bytes()
    second = generator.generate(
        12, 64, 0, root / "q12-a64-bc0", 100_000, False)
    assert first == second
    assert manifest.read_bytes() == first_bytes
    loaded = analyzer.load_manifests(root, {"q12-a64-bc0"})
    assert loaded["q12-a64-bc0"]["schema"] == analyzer.GENERATED_SHARD_V3


def test_historical_v2_manifest_remains_accepted(tmp_path):
    root, current = generate_v3(tmp_path)
    historical = as_historical_v2(current)
    (root / "q12-a64-bc0/manifest.json").write_text(
        json.dumps(historical, indent=2, sort_keys=True) + "\n")
    loaded = analyzer.load_manifests(root, {"q12-a64-bc0"})
    assert loaded["q12-a64-bc0"]["schema"] == analyzer.GENERATED_SHARD_V2


@pytest.mark.parametrize("plant", [
    "compact-hash", "parent-range", "parent-id", "selection", "layout",
])
def test_v3_authority_plants_fail_closed(tmp_path, plant):
    root, document = generate_v3(tmp_path)
    broken = copy.deepcopy(document)
    if plant == "compact-hash":
        broken["non_typed_authority"]["sha256"] = "0" * 64
    elif plant == "parent-range":
        broken["parent_range"]["authority_count"] += 1
    elif plant == "parent-id":
        broken["typed_rows"][0]["parent_id"] += 1
    elif plant == "selection":
        broken["selection"]["mode"] = "parent-range"
    else:
        broken["identity"]["weight_layout"] = 1
    (root / "q12-a64-bc0/manifest.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        analyzer.load_manifests(root, {"q12-a64-bc0"})


def test_compact_range_v3_is_not_misread_as_exhaustive_internal_sweep(tmp_path):
    root = tmp_path / "generated"
    shard = root / "q12-a64-bc0"
    generator.generate(12, 64, 0, shard, 100_000, False,
                       parent_begin=0, parent_count=1)
    with pytest.raises(ValueError):
        analyzer.load_manifests(root, {"q12-a64-bc0"})
