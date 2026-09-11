"""Compile/source/receipt checks; not a claim of PPU numeric admission."""
import json
from pathlib import Path
import re
import subprocess

import pytest

from dev.gemv_cuda.build_h800_candidates import source
from dev.gemv_ppu.build import ppu_api
from dev.gemv_ppu.h800_port import IMPLEMENTATIONS, POLICY, REFERENCE_RECIPES, candidate_source, reference_source, selection, verify
from dev.gemv_ppu.run_h800_port import VARIANTS, ROUNDS, parse_result, summarize
from dev.gemv_ppu.review_h800_port import receipt_entry
from dev.gemv_cuda.build import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", IMPLEMENTATIONS)
def test_native_port_preserves_the_frozen_kernel_body(family):
    original, _ = source(IMPLEMENTATIONS[family])
    translated = candidate_source(family)
    name = "q4_cooperative_metadata" if family == "small" else "q4_group_affine"
    start = original.index("__global__ void " + name + "(")
    end = original.index("\n}\n", start) + 3
    assert ppu_api(original[start:end]) in translated
    assert "cuda_runtime.h" not in translated and "cuda_fp16.h" not in translated
    assert "/compat/" not in translated and "<hggc_runtime.h>" in translated
    wanted = [r for r in POLICY.values() if r[0] == IMPLEMENTATIONS[family]]
    launch_start = translated.index('extern "C" int qkg_pair_launch_12')
    assert translated[launch_start:].count(">>>(") == len(wanted)
    assert "f.split!=1" in translated and "c.rows!=1" in translated and "c.experts!=1" in translated


def test_raw_control_has_all_sixty_intra_cta_configs_and_no_separate_reducer():
    body = reference_source()
    assert len(REFERENCE_RECIPES) == len(set(REFERENCE_RECIPES)) == 60
    parsed = [tuple(map(int, v)) for v in re.findall(r"launch_q4k_gemv<(\d+),(\d+),(\d+)>", body)]
    assert parsed == REFERENCE_RECIPES
    assert "reduce" not in body and "hggcGetLastError" in body
    assert (1, 1, 2) in parsed and (1, 2, 2) in parsed  # improved small H800 controls


def record(variant="kpack", n=512, k=2048):
    family, recipe = selection(n, k)
    implementation = IMPLEMENTATIONS[family] if variant == "kpack" else variant
    return dict(status="PASS", arm=variant, implementation=implementation, shape=[1,n,k], mode="warm",
                recipe=recipe if variant == "kpack" else [1, 1, 2] if variant == "raw-reference" else [1, 8, 1],
                error=1e-5, zero_code_negative="PASS", zero_a_check="PASS", output_type="F32",
                launches_per_call=1, inter_cta_split=1,
                weight_arithmetic="FP32_GROUP_AFFINE" if implementation.startswith("affine") else "PER_WEIGHT_FP16",
                storage="CANONICAL_KPACK4" if variant == "kpack" else "RAW_GGUF" if variant == "raw-reference" else "XPLANE",
                timing_scope="RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT",
                copies=1, weight_bytes=n*k*9//16, calls_per_graph=32, device=dict(l2_bytes=64*1024**2),
                samples_us=[10., 11., 12.], median_us=11.)


@pytest.mark.parametrize("variant", VARIANTS)
def test_control_kw_does_not_mislabel_an_inter_cta_reducer(variant):
    row = record(variant)
    assert parse_result("Q4_PPU_CELL " + json.dumps(row), variant, row["recipe"], row["shape"], "warm", 3) == row


@pytest.mark.parametrize("key,value", [("implementation", "wrong"), ("weight_arithmetic", "FP16_DOT"),
    ("output_type", "F16"), ("zero_a_check", "SKIP"), ("launches_per_call", 2), ("inter_cta_split", 2),
    ("error", float("nan")), ("median_us", 9.), ("samples_us", [1.]), ("calls_per_graph", 33),
    ("storage", "BF16"), ("timing_scope", "PRODUCER_ONLY")])
def test_dirty_or_mislabeled_results_fail(key, value):
    row = record() | {key: value}
    with pytest.raises(ValueError):
        parse_result("Q4_PPU_CELL " + json.dumps(row), "kpack", [1,16,1], [1,512,2048], "warm", 3)


def test_missing_round_is_incomplete_not_a_win():
    data = {v: [[record(v)] for _ in range(ROUNDS)] for v in VARIANTS}
    assert summarize(512,2048,"warm",data)["verdict"] == "WITHIN_5_PERCENT"
    data["kpack"].pop()
    assert summarize(512,2048,"warm",data)["verdict"] == "INCOMPLETE"


def test_slow_is_complete_but_does_not_pass_parity():
    data = {v: [[record(v)] for _ in range(ROUNDS)] for v in VARIANTS}
    for batch in data["kpack"]:
        batch[0]["median_us"] = 15.
    assert summarize(512,2048,"warm",data)["verdict"] == "PARITY_OPEN"


def test_box_script_uses_prebuilt_and_preserves_parent_shell():
    path = ROOT / "tools/run_q4_h800_port_ppu_box.sh"
    subprocess.run(["bash", "-n", path], check=True)
    body = path.read_text()
    assert "\n(\n" in body and body.rstrip().endswith(")")
    assert "compile=NONE JIT=NONE" in body and "rounds=6 samples=15" in body
    assert "RESUME_RUN" in body and '"$RUN.results.tgz"' in body
    assert "build_h800_port.py" not in body


def test_native_payloads_are_compile_only_and_bound():
    data = verify(ROOT / "prebuilt/ppu0010/q4-h800-port-v1", ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    assert not data["device_validated"] and not data["production_changed"]
    assert set(data["payloads"]) == {"small", "medium", "large", "reference"}


def test_offline_review_requires_unchanged_log_and_exact_device(tmp_path):
    row = record()
    log = tmp_path / 'child.log'
    log.write_text('Q4_PPU_CELL ' + json.dumps(row) + '\n')
    entry = dict(row=row, log=log.name, log_sha256=digest(log))
    args = (tmp_path, entry, 'kpack', row['recipe'], row['shape'], 'warm', 3)
    assert receipt_entry(*args, row['device']) == row
    with pytest.raises(ValueError, match='device'):
        receipt_entry(*args, {'l2_bytes': 0})
    log.write_text('Q4_PPU_CELL {}\n')
    with pytest.raises(ValueError, match='hash'):
        receipt_entry(*args, row['device'])
