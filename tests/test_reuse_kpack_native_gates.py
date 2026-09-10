import copy
import json
import subprocess

import pytest

from test_kpack_prefill_policy import fixture as prefill_fixture
from test_kpack_gemv_policy import fixture as gemv_fixture
from tools.reuse_kpack_native_gates import validate, reuse, ROOT
from tools.run_kpack_gemv_gate import subprocess_source
from quactlize.runtime.compiler import sha

DEVICE = dict(ordinal=0, pci="0000:08:00.0", visible_devices="0")


def summaries():
    native, gemv = prefill_fixture(), gemv_fixture()
    template = gemv["results"][0]["records"][0]
    gemv["plan"], gemv["results"] = [], []
    for q, n, k, e in (
        (12, 512, 2048, 256),
        (13, 2048, 512, 256),
        (14, 248320, 2048, 1),
    ):
        cases = (
            [
                dict(mode=2, rows=t * 8, channels=ch, topk=8)
                for t in (1, 4, 16)
                for ch in (1, 8)
            ]
            if e == 256
            else [dict(mode=0, rows=m, channels=1, topk=1) for m in (1, 4)]
        )
        gemv["plan"].append(dict(q=q, n=n, k=k, e=e, cases=cases))
        rows = []
        for case in cases:
            row = copy.deepcopy(template)
            row.update(q=q, n=n, k=k, experts=e, case=case)
            rows.append(row)
        gemv["results"].append(dict(records=rows))
    for s in (native, gemv):
        s["device"] = DEVICE.copy()
    native["manifest_sha256"] = gemv["native_manifest_sha256"] = "native"
    gemv["execution_sha256"] = "execution"
    return native, gemv


def test_exact_gates_reusable():
    validate(*summaries(), DEVICE, "native", "execution")


def test_native_only_does_not_require_parked_gemv():
    native, _ = summaries()
    validate(native, None, DEVICE, "native", "execution")
    for bad in (native | {"device": {}}, native | {"manifest_sha256": "different"},
                native | {"results": native["results"][:-1]}):
        with pytest.raises(ValueError):
            validate(bad, None, DEVICE, "native", "execution")


@pytest.mark.parametrize(
    "fault",
    [
        "device",
        "package",
        "execution",
        "partial-native",
        "partial-gemv",
        "wrong-context",
        "nan",
        "negative",
    ],
)
def test_bad_measurements_cannot_resume(fault):
    n, g = summaries()
    if fault == "device":
        g["device"]["pci"] = "0000:7e:00.0"
    elif fault == "package":
        n["manifest_sha256"] = "other"
    elif fault == "execution":
        g["execution_sha256"] = "other"
    elif fault == "partial-native":
        n["results"].pop()
    elif fault == "partial-gemv":
        g["results"].pop()
    elif fault == "wrong-context":
        g["plan"][-1]["n"] = 256
        for row in g["results"][-1]["records"]:
            row["n"] = 256
    elif fault == "nan":
        g["results"][0]["records"][0]["errors"][0] = float("nan")
    else:
        g["results"][0]["records"][0]["zero_low_negative"] = "MISSING"
    with pytest.raises(ValueError):
        validate(n, g, DEVICE, "native", "execution")


@pytest.mark.parametrize("include_gemv", [True, False])
def test_copy_preserves_old_run_and_does_not_reuse_model(tmp_path, include_gemv):
    source, output, bundle, legacy, execution = [
        tmp_path / name for name in ("old", "new", "native", "legacy", "execution")
    ]
    for p in (source, output, bundle, legacy, execution):
        p.mkdir()
    for p in (
        bundle / "manifest.json",
        bundle / "libquactlize_ppu_execution.so",
        execution / "manifest.json",
    ):
        p.write_text("identity fixture\n")
    for i in range(5):
        (legacy / f"libquactlize_ppu_fmt{i}.so").write_text(f"library {i}")
    n, g = summaries()
    n["manifest_sha256"] = g["native_manifest_sha256"] = sha(bundle / "manifest.json")
    g["execution_sha256"] = sha(bundle / "libquactlize_ppu_execution.so")
    g["manifest_sha256"] = sha(execution / "manifest.json")
    g["baseline_libraries"] = {p.name: sha(p) for p in legacy.iterdir()}
    g["source"] = subprocess_source()
    for name, data in (("native-gate", n), ("gemv-gate", g)):
        if name == "gemv-gate" and not include_gemv:
            continue
        (source / name).mkdir()
        (source / name / "summary.json").write_text(json.dumps(data))
        (source / f"{name}.log").write_text("raw log fixture\n")
    (source / "model").mkdir()
    (source / "model/summary.json").write_text("must not reuse")
    (source / "quactlize-dirty.patch").write_text("")
    (source / "quactlize-source.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    )
    hashes = {p: sha(p) for p in source.rglob("*") if p.is_file()}
    reuse(source, output, bundle, legacy, execution, DEVICE, include_gemv=include_gemv)
    assert hashes == {p: sha(p) for p in hashes}
    assert not (output / "model").exists()
    receipt = json.loads((output / "reused-gates.json").read_text())
    assert (receipt["native_contexts"], receipt["gemv_contexts"]) == (28, 14 if include_gemv else 0)
    assert (output / "gemv-gate").exists() == include_gemv
    for name, value in receipt["files"].items():
        assert sha(output / name) == value
    with pytest.raises(ValueError, match="already contains"):
        reuse(source, output, bundle, legacy, execution, DEVICE, include_gemv=include_gemv)
    empty = tmp_path / "retry"
    empty.mkdir()
    (source / "quactlize-dirty.patch").write_text("untracked oracle edit")
    with pytest.raises(ValueError, match="unversioned"):
        reuse(source, empty, bundle, legacy, execution, DEVICE, include_gemv=include_gemv)
    assert not list(empty.iterdir())
