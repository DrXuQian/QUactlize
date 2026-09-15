"""Host invariants for the dedicated low-level device gate, not device admission."""
import ctypes as C
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_bounded_capability_plan_has_required_axes():
    from dev.bf16_compute.plan import QTYPES, cases, modules, grouped_rows
    rows = cases()
    assert len(rows) == len({r["id"] for r in rows}) == 746
    assert len(modules()) == 40
    for q in QTYPES:
        group = [r for r in rows if r["q"] == q and r["family"] == "grouped" and r["compute"] == "bf16"]
        assert {r["profile"] for r in group} == {"small", "large", "ordinary"}
        assert {r["split"] for r in group} == {1, 2, 4, 8}
        if q != 8:
            fq = [r for r in group if r["quant"] == "fq"]
            assert {r["parent"].persistent for r in fq} == {0, 1}
            assert all(r["algorithm"] == r["parent"].persistent for r in fq)
        assert any(r["q"] == q and r["compute"] == "f16" and r["family"] == "grouped" for r in rows)
        simt = [r for r in rows if r["q"] == q and r["family"] == "simt" and r["compute"] == "bf16"]
        assert {r["tokens"] for r in simt} == set(range(1, 9))
        assert {r["storage"] for r in simt} == {1, 2}
        assert {r["mode"] for r in simt} == {0, 1, 2}
        moe = [r for r in rows if r["q"] == q and r["family"] == "moe"]
        assert {r["kind"] for r in moe} == {"tc", "simt", "mixed"}
        assert {r["merged"] for r in moe} == {False, True}
        assert {r["tokens"] for r in moe} == {1, 4, 8}
    assert len(grouped_rows("ordinary")) == 1025  # v3's real noncompact branch
    assert grouped_rows("large").max() == 257
    assert grouped_rows("ordinary")[-1] == 257
    assert not np.array_equal(grouped_rows("large", 0), grouped_rows("large", 1))
    assert {r["parent"].tk for r in rows if r["family"] == "outlier"} == {128}
    assert {r["down_q"] for r in rows if "down_q" in r} == {13, 14}


@pytest.mark.parametrize("q", (8, 10, 11, 12, 13, 14))
def test_dyadic_raw_oracle_and_half_metadata_are_exact(q):
    from dev.bf16_compute.fixture import raw_weight, planes, round_compute
    from reference import gguf_kpack as ref
    raw, official = raw_weight(q, 256, 512, 701)
    placed = planes(raw, q)
    delta = {8: 0, 10: 0, 11: 4, 12: -8, 13: -8, 14: 24}[q]
    block = 32 if q == 8 else 256
    offset = 0 if q == 8 else ref.SPECS[q].d_offset
    d = raw[..., offset:offset + 2].copy().view("<f2").reshape(256, -1)
    d = np.repeat(d.astype("f4"), block, axis=1)
    has_min = q in (10, 12, 13)
    code = official / d + (0.5 if has_min else 0)
    assert np.array_equal(code, np.rint(code))
    group = 32 if q in (8, 12, 13) else 16
    scale = np.repeat(placed["scale"].T.astype("f4"), group, axis=1)
    zero = np.repeat(placed["zero"].T.astype("f4"), group, axis=1) if q != 8 else 0
    for compute in ("bf16", "f16"):
        reconstructed = round_compute(round_compute((code + delta) * scale, compute) + zero, compute)
        assert np.array_equal(reconstructed, official)
    assert raw.nbytes == sum(placed[name].nbytes for name in ("low", "high", "units"))


def test_exact_bf16_range_and_oracle_negative():
    from dev.bf16_compute.fixture import bf16_bits, bf16_float, round_compute, compare
    value = np.array([243383.484375], "f4")
    assert bf16_bits(value).item() == 0x486e
    assert bf16_float(bf16_bits(value)).item() == 243712
    assert np.isinf(round_compute(value, "f16")).all()
    assert compare(np.ones(8), np.ones(8), np.ones(8))["bad"] == 0
    with pytest.raises(ValueError, match="typed oracle mismatch"):
        compare(np.zeros(8), np.ones(8), np.ones(8))


def test_private_moe_ctypes_exact_c_abi(tmp_path):
    from dev.bf16_compute.native import Projection, ProjectionV2, MoePlan, MoePlanV2, MixedPlanV2
    cpp = tmp_path / "sizes.cpp"
    cpp.write_text('#include "quactlize/execution/moe.h"\n#include <cstdio>\nint main(){'
        'printf("%zu %zu %zu %zu %zu\\n", sizeof(qk_moe_projection_v1),sizeof(qk_moe_projection_v2),'
        'sizeof(qk_moe_plan_v1),sizeof(qk_moe_plan_v2),sizeof(qkg_moe_compute_v2));}\n')
    exe = tmp_path / "sizes"
    subprocess.run(["g++", "-std=c++17", f"-I{ROOT}", str(cpp), "-o", str(exe)], check=True)
    actual = list(map(int, subprocess.check_output([exe], text=True).split()))
    assert actual == [C.sizeof(t) for t in (Projection, ProjectionV2, MoePlan, MoePlanV2, MixedPlanV2)]


def test_indexed_inputs_change_routing_and_preserve_storage_strides():
    from dev.bf16_compute.cases import simt_inputs
    from types import SimpleNamespace
    w = SimpleNamespace(k=512, n=256, experts=16)
    for tokens in range(1, 9):
        for channels in (1, 8):
            point = dict(tokens=tokens, mode=2, channels=channels)
            a, owners, from_rows, ids, offsets = simt_inputs(w, point, 0)
            changed = simt_inputs(w, point, 1)
            assert len(owners) == tokens * 8 and a.shape == (tokens * channels, 520)
            assert all(len(set(row[:8])) == 8 for row in ids)
            assert np.all(a[:, 512:] == 19) and offsets is None
            assert from_rows.max() < len(a)
            assert not np.array_equal(ids, changed[3])


def test_package_rejects_source_payload_and_coverage_drift(tmp_path, monkeypatch):
    from dev.bf16_compute import run
    from dev.bf16_compute.plan import cases
    from quactlize.runtime.compiler import sha
    so = tmp_path / "fake.so"
    so.write_bytes(b"host-only-package-test")
    record = dict(path=so.name, sha256=sha(so))
    manifest = dict(schema="quactlize.bf16-device-gate.v1", modules={"x": record},
                    simt=record, moe=record, source={}, cases=[dict(id=c["id"]) for c in cases()])
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    assert run.validate_package(tmp_path)["modules"] == {"x": record}
    manifest["cases"].pop()
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="denominator"):
        run.validate_package(tmp_path)
    manifest["cases"] = [dict(id=c["id"]) for c in cases()]
    manifest["source"] = {"dev/bf16_compute/run.py": "wrong"}
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source differs"):
        run.validate_package(tmp_path)
    manifest["source"] = {}
    target.write_text(json.dumps(manifest))
    so.write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload"):
        run.validate_package(tmp_path)


def test_execution_reuse_requires_exact_receipt_and_exports(tmp_path, monkeypatch):
    from dev.bf16_compute import package
    from quactlize.runtime.compiler import sha
    library = tmp_path / "execution.so"
    library.write_bytes(b"fake-elf-not-loaded")
    names = ("simt_query_v2", "simt_run_v2", "moe_simt_query_v1", "moe_simt_bind_v1",
             "moe_mixed_stage_v2", "moe_weighted_finish_v2")
    text = "\n".join("T quactlize_kpack_" + name for name in names)
    monkeypatch.setattr(package.subprocess, "check_output", lambda *a, **kw: text)
    manifest = dict(schema="quactlize.kpack-execution-build.v1", library=library.name,
        sha256=sha(library), source_hashes={}, simt_configs={str(q): [dict(
            variant=1 if q == 8 else 3, columns=4, warps=4, values=4)] for q in (8, 10, 11, 12, 13, 14)})
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    assert package.execution_receipt(tmp_path)[0] == library
    monkeypatch.setattr(package.subprocess, "check_output", lambda *a, **kw: "T quactlize_kpack_simt_query_v2")
    with pytest.raises(ValueError, match="endpoints"):
        package.execution_receipt(tmp_path)
    monkeypatch.setattr(package.subprocess, "check_output", lambda *a, **kw: text)
    manifest["simt_configs"].pop("14")
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recipe"):
        package.execution_receipt(tmp_path)
