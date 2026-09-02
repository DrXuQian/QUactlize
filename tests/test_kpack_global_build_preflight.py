from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import kpack_global_build_preflight as preflight  # noqa: E402


def test_preflight_self_test() -> None:
    preflight.self_test()


def test_build_sh_only_caches_the_five_global_checks() -> None:
    text = (ROOT / "build.sh").read_text()
    assert "QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT" in text
    assert "kpack_global_build_preflight.py\" verify" in text
    for checker in preflight.CHECKERS:
        assert checker in text

    # These remain outside the receipt and continue to run at their original
    # per-target/per-build seams.
    for per_invocation in (
        "ci/check_dense_tactic_table.py",
        "ci/check_dense_splitk_sweep_contract.py",
        "ci/check_dense_streamk_sweep_target.py",
        "PPU_BUILD_RESUME",
        ".quactlize-source-head",
        "FQ_SWEEP_GENERATED_DIR",
        "SCALEFIRST_SWEEP_GENERATED_DIR",
        "PPU_SDK_ROOT/bin/hgcc",
    ):
        assert per_invocation in text


def test_both_bundle_builders_create_verify_bind_and_pass_receipt() -> None:
    for name in (
        "build_scalefirst_kpack_discovery_bundle.sh",
        "build_fully_quantized_kpack_discovery_bundle.sh",
    ):
        text = (ROOT / "tools" / name).read_text()
        assert "kpack_global_build_preflight.py\" create" in text
        assert "kpack_global_build_preflight.py\" verify" in text
        assert "KPACK_GLOBAL_PREFLIGHT_RECEIPT" in text
        assert 'install -m 0444 "$shared_global_preflight" "$global_preflight"' in text
        assert 'cmp -s "$shared_global_preflight" "$global_preflight"' in text
        assert '"global_preflight"' in text
        assert '"sha256":sha(global_preflight)' in text
        # Dense and grouped build.sh invocations each pass the same receipt.
        assert text.count(
            'QUACTLIZE_KPACK_GLOBAL_PREFLIGHT_RECEIPT="$global_preflight"'
        ) == 2


def test_receipt_scope_names_every_excluded_per_invocation_authority() -> None:
    source = (ROOT / "tools/kpack_global_build_preflight.py").read_text()
    for name in (
        "TARGET_AND_GENERATED_DIRECTORY",
        "CMAKE_AND_BUILD_SOURCE",
        "SOURCE_REVISION_AND_WORKTREE",
        "RECURSIVE_SUBMODULE_STATE",
        "SDK_AND_TOOLCHAIN",
    ):
        assert name in source
