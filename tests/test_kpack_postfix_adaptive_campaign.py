from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import kpack_postfix_adaptive_campaign as adaptive  # noqa: E402


def test_fixed_adaptive_contract_is_small_screen_then_strong_confirm() -> None:
    assert adaptive.WORKERS == 8
    assert adaptive.SCREEN_ITERATIONS == 2
    assert adaptive.SCREEN_WARMUPS == 1
    assert adaptive.CORRECTNESS_REPEATS == 1
    assert adaptive.CONFIRM_ITERATIONS == 11
    assert adaptive.CONFIRM_ROUNDS == 3
    assert adaptive.CONFIRM_WARMUPS == 3
    assert adaptive.PENDING_STAGE == "SCREEN_COMPLETE_CONFIRM_PENDING"


def test_box_entry_reuses_build_but_never_historical_timings() -> None:
    source = (ROOT / "tools/run_kpack_postfix_adaptive_campaign_box.sh").read_text()
    assert "REUSE_CAMPAIGN" in source
    assert "kpack_postfix_adaptive_campaign.py\" prepare" in source
    assert "kpack_postfix_adaptive_campaign.py\" seal-screen" in source
    assert "build_kpack_discovery_partition_worker.sh" not in source
    assert "--phase screen" in source
    assert "--phase all" not in source
    assert "--screen-iterations 2" in source
    assert "--correctness-repeats 1" in source
    assert "--screen-warmups 1 --confirm-warmups 3" in source
    assert "--confirm-iterations 11 --confirm-rounds 3" in source
    assert "SCREEN_COMPLETE_CONFIRM_PENDING" in source
    assert "full_confirm_fallback=0" in source
    assert "screen_logs=%s/%s" in source
    assert "KPACK_GLOBAL_SHORTLIST_COMMAND" not in source
    assert "eval " not in source


def test_reused_and_output_roots_must_be_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    adaptive._disjoint(source, output)
    nested = source / "nested"
    nested.mkdir()
    with pytest.raises(adaptive.AdaptiveError, match="inside"):
        adaptive._disjoint(source, nested)


def test_frozen_epoch_file_rejects_changed_resume(tmp_path: Path) -> None:
    path = tmp_path / "epoch.json"
    adaptive._write_frozen_bytes(path, b"first\n")
    adaptive._write_frozen_bytes(path, b"first\n")
    with pytest.raises(adaptive.AdaptiveError, match="differs"):
        adaptive._write_frozen_bytes(path, b"second\n")


def test_screen_log_digest_rejects_gap_extra_and_tamper(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    screen.mkdir()
    ids = [f"{index:064x}" for index in (1, 2)]
    for item in ids:
        (screen / f"{item}.log").write_text(f"screen {item}\n")
    before = adaptive._screen_digest(screen, ids, 0)
    (screen / f"{ids[0]}.log").write_text("changed\n")
    after = adaptive._screen_digest(screen, ids, 0)
    assert before != after
    (screen / "foreign.log").write_text("foreign\n")
    with pytest.raises(adaptive.AdaptiveError, match="union"):
        adaptive._screen_digest(screen, ids, 0)
