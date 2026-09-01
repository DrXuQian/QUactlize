"""Installed GGUF packer entry-point and host-only planning contracts."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import types

from quactlize import formats
from quactlize import gguf_roles
from quactlize import pack_gguf


ROOT = pathlib.Path(__file__).parents[1]


def test_console_entry_targets_the_installed_module_and_tool_is_only_a_wrapper():
    project = (ROOT / "pyproject.toml").read_text()
    scripts = re.search(
        r'^\[project\.scripts\]\s*$\n(.*?)(?=^\[|\Z)', project,
        re.MULTILINE | re.DOTALL)
    assert scripts is not None
    entries = re.findall(
        r'^quactlize-pack-gguf\s*=\s*"([^"]+)"\s*$', scripts.group(1),
        re.MULTILINE)
    assert entries == ["quactlize.pack_gguf:main"]

    import tools.pack_gguf as compatibility

    assert compatibility.main is pack_gguf.main
    assert compatibility.restore_artifact is pack_gguf.restore_artifact
    source = (ROOT / "quactlize" / "pack_gguf.py").read_text()
    assert "from tools." not in source
    assert "import tools." not in source


def test_role_authority_is_shared_by_installed_packer_and_inventory_tool():
    import tools.gguf_internal_shape_inventory as inventory

    assert inventory.InventoryError is gguf_roles.InventoryError
    assert inventory.ROLE_RULES is gguf_roles.ROLE_RULES
    assert inventory.classify_role is gguf_roles.classify_role
    assert pack_gguf._route_role_authority(
        "blk.12.ffn_down_exps.weight", 3, "grouped") == (True, None)


def test_distribution_metadata_installs_entry_module_registry_and_packer_extra(tmp_path):
    result = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    info = next(tmp_path.glob("*.egg-info"))
    assert (info / "entry_points.txt").read_text().splitlines() == [
        "[console_scripts]",
        "quactlize-pack-gguf = quactlize.pack_gguf:main",
    ]
    sources = set((info / "SOURCES.txt").read_text().splitlines())
    assert {
        "quactlize/pack_gguf.py",
        "quactlize/gguf_roles.py",
        "quactlize/include/ppu_format_config.inc",
    } <= sources
    assert "[packer]\ngguf\n" in (info / "requires.txt").read_text()


def test_dry_run_plans_dense_and_authorised_grouped_without_device(
        monkeypatch, capsys, tmp_path):
    class FakeReader:
        def __init__(self, model):
            assert model == "model.gguf"
            self.tensors = [
                types.SimpleNamespace(
                    name="blk.0.attn_q.weight",
                    tensor_type=formats.QuantType.Q4_K,
                    shape=(512, 256)),
                types.SimpleNamespace(
                    name="blk.0.ffn_up_exps.weight",
                    tensor_type=formats.QuantType.Q3_K,
                    shape=(512, 256, 4)),
                types.SimpleNamespace(
                    name="blk.0.unknown_rank3.weight",
                    tensor_type=formats.QuantType.Q3_K,
                    shape=(512, 256, 4)),
                types.SimpleNamespace(
                    name="token_embd.weight",
                    tensor_type=formats.QuantType.Q4_K,
                    shape=(512, 256)),
                types.SimpleNamespace(
                    name="blk.0.unknown_rank2.weight",
                    tensor_type=formats.QuantType.Q4_K,
                    shape=(512, 256)),
                types.SimpleNamespace(
                    name="not_a_matrix",
                    tensor_type=formats.QuantType.Q4_K,
                    shape=(512,)),
            ]

    monkeypatch.setitem(sys.modules, "gguf", types.SimpleNamespace(GGUFReader=FakeReader))
    out = tmp_path / "artifacts"
    assert pack_gguf.main(["model.gguf", str(out), "--dry-run"]) == 0
    stdout = capsys.readouterr().out
    assert "6  ->  2 packable, 4 skipped" in stdout
    assert "routes     {'dense': 1, 'grouped': 1}" in stdout
    assert "layout     canonical K-pack" in stdout
    assert "--dry-run: nothing written, no device touched" in stdout
    assert not out.exists()
