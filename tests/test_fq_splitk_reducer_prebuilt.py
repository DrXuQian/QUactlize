from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fq_splitk_reducer_prebuilt as prebuilt
import plan_fq_splitk_reducer_lookup as reducer_plan


def _fixture(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"; source.mkdir()
    sdk = tmp_path / "sdk"; (sdk / "bin").mkdir(parents=True); (sdk / "lib").mkdir()
    bundle = tmp_path / "bundle"; bundle.mkdir()
    for relative in prebuilt.BUILD_INPUTS:
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n")
        if relative.endswith(".sh"): path.chmod(0o755)
    (sdk / "release.yaml").write_text("version: test-sdk\n")
    for relative in ("bin/hgcc", "bin/hgobjdump", *prebuilt.SDK_RUNTIME_FILES):
        path = sdk / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sdk:{relative}\n"); path.chmod(0o755)

    commit, tree = "a" * 40, "b" * 40
    original_run = prebuilt.run
    def fake_git(root, *arguments):
        if arguments == ("rev-parse", "HEAD"): return commit
        if arguments == ("rev-parse", "HEAD^{tree}"): return tree
        if arguments and arguments[0] in {"diff", "ls-files", "status"}: return ""
        raise AssertionError(arguments)
    def fake_run(argv, *, cwd):
        if argv == [str(sdk / "bin/hgcc"), "--version"]:
            return "hgcc test compiler"
        return original_run(argv, cwd=cwd)
    monkeypatch.setattr(prebuilt, "git", fake_git)
    monkeypatch.setattr(prebuilt, "run", fake_run)
    monkeypatch.setattr(prebuilt, "submodules", lambda root: [
        {"path": "third_party/actlize", "commit": "c" * 40}])

    plan = reducer_plan.materialize()
    files = {
        "test_fq_splitk_reducer_lookup": "binary\n",
        "box_identity_probe": "probe\n",
        "reducer-plan.json": json.dumps(plan, indent=2, sort_keys=True) + "\n",
        "fq_splitk_reducer_lookup_cases.inc": reducer_plan.cpp_include(plan).decode(),
        "ppu-elf-list.txt": (
            "initialize_partials_kernel\nvalidate_output_kernel\n"
            "PpuMixedInputSplitKParallelM1FastReductionKernelILi2ELi8\n"),
        "ppu-isa.txt": "isa\n",
        "build.log": "repository-global checks reused from receipt\nbuilt: test_fq_splitk_reducer_lookup\n",
        "cmake.log": "FQ Split-K reducer lookup: cases=1035\n",
        "build.make": "test_fq_splitk_reducer_lookup\n",
        "identity-probe-build.log": "probe command\n",
        "global-preflight.json": "{}\n",
    }
    source_authority = prebuilt.source_authority(source, require_clean=False)
    sdk_authority = prebuilt.sdk_authority(sdk)
    build_authority = {
        "schema": prebuilt.BUILD_AUTHORITY_SCHEMA,
        "source": source_authority, "sdk": sdk_authority,
        "build": {"target": prebuilt.TARGET, "arch": prebuilt.ARCH,
                  "cmake_enable": "FQ_SPLITK_REDUCER_LOOKUP_ENABLE=ON",
                  "plan_sha256": reducer_plan.digest(plan)},
    }
    files["build-authority.json"] = json.dumps(
        build_authority, indent=2, sort_keys=True) + "\n"
    for name, text in files.items():
        path = bundle / name; path.write_text(text)
        path.chmod(0o755 if name in {
            "test_fq_splitk_reducer_lookup", "box_identity_probe"} else 0o444)
    artifacts = {}
    key_for = {
        "test_fq_splitk_reducer_lookup": "binary",
        "box_identity_probe": "identity_probe",
        "build-authority.json": "build_authority",
        "global-preflight.json": "global_preflight",
        "reducer-plan.json": "plan",
        "fq_splitk_reducer_lookup_cases.inc": "include",
        "ppu-elf-list.txt": "elf_list", "ppu-isa.txt": "isa",
        "build.log": "build_log", "cmake.log": "cmake_log",
        "build.make": "build_make",
        "identity-probe-build.log": "identity_probe_build_log",
    }
    for name, key in key_for.items():
        path = bundle / name
        artifacts[key] = prebuilt.file_record(
            path, name, executable=name in {
                "test_fq_splitk_reducer_lookup", "box_identity_probe"})
    manifest = {
        "schema": prebuilt.SCHEMA,
        "source": source_authority, "sdk": sdk_authority,
        "build": {
            **build_authority["build"],
            "build_authority_sha256": prebuilt.sha256_file(
                bundle / "build-authority.json"),
            "measurement": {
                "rounds": 3, "warmups": 3, "samples": 11,
                "schedule_seed_schema": reducer_plan.SCHEDULE_SEED_SCHEMA,
                "round_seeds": [f"0x{seed:016x}" for seed in reducer_plan.ROUND_SEEDS],
                "raw_bit_correctness": "EVERY_OUTPUT_ELEMENT_BEFORE_AND_AFTER_TIMING",
                "top_n": None, "point_estimate_pruning": False,
            },
        },
        "artifacts": artifacts,
    }
    (bundle / "manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")
    return source, sdk, bundle, manifest


def test_portable_bundle_verifies_without_build_tree(tmp_path, monkeypatch):
    source, sdk, bundle, manifest = _fixture(tmp_path, monkeypatch)
    build_tree = tmp_path / "local-build-tree"; build_tree.mkdir()
    shutil.rmtree(build_tree)
    assert prebuilt.verify(bundle, source, sdk) == manifest
    assert "build_root" not in (bundle / "manifest.json").read_text()


@pytest.mark.parametrize("plant", [
    "binary", "plan", "sdk", "source", "build-authority", "extra",
])
def test_prebuilt_tampering_fails_closed(tmp_path, monkeypatch, plant):
    source, sdk, bundle, manifest = _fixture(tmp_path, monkeypatch)
    if plant == "binary":
        (bundle / "test_fq_splitk_reducer_lookup").write_text("changed\n")
    elif plant == "plan":
        value = json.loads((bundle / "reducer-plan.json").read_text())
        value["measurement"]["top_n"] = 1
        (bundle / "reducer-plan.json").write_text(json.dumps(value))
    elif plant == "sdk":
        (sdk / "lib/libhggc.so").write_text("changed sdk\n")
    elif plant == "source":
        first = source / prebuilt.BUILD_INPUTS[0]
        first.write_text("changed source\n")
    elif plant == "build-authority":
        value = json.loads((bundle / "build-authority.json").read_text())
        value["build"]["arch"] = "ppu9999"
        (bundle / "build-authority.json").write_text(json.dumps(value))
    else:
        (bundle / "unbound.bin").write_bytes(b"x")
    with pytest.raises((prebuilt.ManifestError, reducer_plan.PlanError)):
        prebuilt.verify(bundle, source, sdk)


def test_runner_is_execute_only_and_contract_is_frozen():
    runner = (ROOT / "tools/run_fq_splitk_reducer_lookup_box.sh").read_text()
    builder = (ROOT / "tools/build_fq_splitk_reducer_lookup_prebuilt.sh").read_text()
    assert "build.sh" not in runner and "PPU_BUILD_DIR" not in runner
    assert "--warmups=3" in runner and "--samples=11" in runner
    assert "--case-begin=0" in runner and "--case-end=1035" in runner
    assert "--plant-output-fault=0" in runner
    assert "KPACK_DEVICE_HOMOGENEITY" in runner and "KPACK_DEVICE_IDENTITY" in runner
    assert "TARGET=test_fq_splitk_reducer_lookup" in builder
    subprocess.run(["bash", "-n",
                    str(ROOT / "tools/build_fq_splitk_reducer_lookup_prebuilt.sh"),
                    str(ROOT / "tools/run_fq_splitk_reducer_lookup_box.sh")], check=True)
