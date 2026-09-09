from pathlib import Path
import subprocess

import pytest

from tools import build_kpack_grouped_postops as builder
from tools import run_kpack_grouped_postops as runner
from quactlize.runtime.compiler import ROOT, FLAGS


def test_five_format_plan_has_exact_same_parent_controls_and_no_sf():
    groups = builder.plan()
    assert len(groups) == 9
    assert {g["parent"]["qtype"] for g in groups} == set(range(10, 15))
    assert all(g["parent"]["route"] == "fq-grouped" for g in groups)
    assert [g["parent"]["qtype"] for g in groups if g["schedule"] == "persistent"] == [
        12,
        13,
    ]
    assert sum(len(runner.expected_keys(g, 4)) for g in groups) == 384
    assert all(g["parent"]["tm"] == 8 for g in groups if g["model"])


@pytest.mark.parametrize(
    "plant", [None, "missing", "duplicate", "status", "source", "job",
              "no-samples", "sample-count", "nan", "zero-time", "missing-checks"]
)
def test_resume_never_admits_partial_or_stale_job(plant):
    group = builder.plan()[0]
    authority = {"identity": "abc", "counts": {"samples": 11, "correctness_repeats": 7}}
    cells = [
        dict(case=c, split=s, round=r, variant=a, status="PASS",
             graph_elapsed_samples_us=[[16.0] * 11],
             correctness_checks=21 if c == "ragged" else 7)
        for c, s, r, a in runner.expected_keys(group, 4)
    ]
    result = dict(status="PASS", job=group["job"], authority=authority, cells=cells)
    if plant == "missing":
        cells.pop()
    if plant == "duplicate":
        cells.append(cells[0])
    if plant == "status":
        cells[0]["status"] = "FAIL"
    if plant == "source":
        result["authority"] = {}
    if plant == "job":
        result["job"] = "wrong"
    if plant == "no-samples":
        del cells[0]["graph_elapsed_samples_us"]
    if plant == "sample-count":
        cells[0]["graph_elapsed_samples_us"][0].pop()
    if plant == "nan":
        cells[0]["graph_elapsed_samples_us"][0][0] = float("nan")
    if plant == "zero-time":
        cells[0]["graph_elapsed_samples_us"][0][0] = 0.0
    if plant == "missing-checks":
        cells[0]["correctness_checks"] -= 1
    assert runner.result_complete(result, group, 4, authority) == (plant is None)


def test_summary_divides_graph_repeats_once():
    cells = [
        dict(
            case="model",
            split=4,
            round=r,
            variant=a,
            status="PASS",
            graph_elapsed_samples_us=[[value] * 11],
        )
        for r in range(4)
        for a, value in [("baseline", 320.0), ("candidate", 160.0)]
    ]
    rows = runner.summarize([dict(status="PASS", job="q4", cells=cells)])
    assert rows == [
        dict(
            job="q4",
            case="model",
            split=4,
            status="PASS",
            baseline_us=20.0,
            candidate_us=10.0,
            delta_pct=-50.0,
        )
    ]


def test_failed_job_selection_keeps_original_parents_and_denominator():
    names = ["fq-q14-tm16-ordinary", "fq-q12-tm8-ordinary"]
    selected = runner.select_jobs(builder.plan(), names)
    assert [g["job"] for g in selected] == names
    assert sum(len(runner.expected_keys(g, 4)) for g in selected) == 96
    assert runner.select_jobs(builder.plan(), None) == builder.plan()
    for wrong in (["missing"], names + names[:1]):
        with pytest.raises(ValueError, match="unknown/duplicate"):
            runner.select_jobs(builder.plan(), wrong)


def test_acu_uses_asight_install_and_checks_explicit_override(tmp_path):
    sdk = tmp_path / "sdk"
    paths = [sdk / "asight/bin/acu", sdk / "bin/acu", tmp_path / "explicit-acu"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    assert runner.resolve_acu(sdk) == paths[0]
    assert runner.resolve_acu(sdk, paths[2]) == paths[2]
    paths[0].chmod(0o644)
    assert runner.resolve_acu(sdk) == paths[1]
    paths[1].chmod(0o644)
    with pytest.raises(ValueError, match="ACU executable not found"):
        runner.resolve_acu(sdk)
    # An explicit typo must not silently use another profiler installation.
    paths[0].chmod(0o755)
    with pytest.raises(ValueError, match="ACU executable not found"):
        runner.resolve_acu(sdk, tmp_path / "missing")


def test_cuda_projection_matches_actual_hgcc_five_format_types(tmp_path):
    sdk = Path("/root/ppu-sdk/2.1.1")
    if not (sdk / "bin/hgcc").is_file():
        pytest.skip("requires real PPU SDK compiler")
    include = [
        ROOT / "quactlize/include",
        ROOT / "quactlize/runtime",
        ROOT / "third_party/actlize/include",
        ROOT / "third_party/actlize/tools/util/include",
        ROOT / "third_party/actlize/examples/common",
    ]
    command = [
        str(sdk / "bin/hgcc"),
        *FLAGS,
        "-DPPU_PACKED_SCALE=1",
        "-DPPU_PACKED_FORMAT=0",
        *[f"-I{x}" for x in include],
        "-c",
        str(ROOT / "tests/kpack_grouped_postops_types.cu"),
        "-o",
        str(tmp_path / "proof.o"),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_s1_and_partial_policy_source_seams():
    types = (ROOT / "quactlize/runtime/kernel_types.cuh").read_text()
    module = (ROOT / "quactlize/runtime/module.cuh").read_text()
    direct = (
        ROOT
        / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_grouped_splitk_direct_epilogue.hpp"
    ).read_text()
    assert "std::is_same_v<Output,float>" in types
    assert "GroupedSplitKDirectEpilogue<OutputEpilogue>" in types
    assert "PpuMixedInputSplitKParallelCompactReduction<2>" in module
    assert (
        "store_splitk_accumulators_direct(dst,local_shape,tile,local_coord," in direct
    )
    assert "params_.ptr_D[entry], params_.dD[entry]" in direct
    assert "accumulators,mma,residue,0,thread" in direct
    assert "__syncthreads" not in direct and "atomic" not in direct
    # No arbitrary alpha/beta can silently be ignored on the fixed partial edge.
    assert "struct ThreadArguments {};" in direct


def test_no_nvidia_stubs_or_silent_config_changes_in_product():
    product = (
        ROOT
        / "quactlize/include/actlize_extensions/cutlass/gemm/kernel/detail/ppu_grouped_splitk_direct_epilogue.hpp"
    ).read_text()
    assert "gemv_cuda" not in product and "stub_inc" not in product
    script = (ROOT / "tools/run_kpack_grouped_postops_box.sh").read_text()
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "build.sh" not in script and "source " not in script
