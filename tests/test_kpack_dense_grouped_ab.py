import ctypes as C
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from tests.test_kpack_execution import raw_fixture
from tools import run_kpack_dense_grouped_ab as ab
from tools import run_kpack_decode_sweep as sweep
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values, prepare_expert


def test_n_concatenation_matches_independent_dense_packer_not_flat_bytes():
    n, k, e = 256, 512, 3
    raw = raw_fixture(12, n, k, e).reshape(e, n * (k // 256), -1)
    placed = [prepare_expert(raw[i], 12, n, k) for i in range(e)]
    w = SimpleNamespace(
        q=12,
        n=n,
        k=k,
        experts=e,
        planes={key: np.stack([p[key] for p in placed]) for key in placed[0]},
        categories=np.zeros((e, k), dtype="u1"),
        sums=np.arange(e * 4 * n).reshape(e, 4, n),
        abs_sums=np.ones((e, 4, n)),
        partial_sums={},
    )
    ids = (2, 0)
    dense = ab.concatenate_q4_experts(w, ids)
    expected = ref.prepare_dense(
        torch.from_numpy(raw[list(ids)].reshape(-1, raw.shape[-1])), 2 * n, k, 12
    )
    for name in ("low", "units"):
        assert np.array_equal(
            dense.planes[name].view("u1").reshape(-1),
            getattr(expected, name).numpy().view("u1").reshape(-1),
        )
        wrong = np.concatenate([w.planes[name][i].view("u1").reshape(-1) for i in ids])
        assert not np.array_equal(wrong, dense.planes[name].view("u1").reshape(-1))
    assert dense.n == 2 * n and dense.experts == 1
    assert not dense.planes["high"].size
    for bad in ((0, 0), (-1,), (3,), ()):
        with pytest.raises(ValueError):
            ab.concatenate_q4_experts(w, bad)
    w.categories[2, 0] = 1
    with pytest.raises(ValueError, match="activation"):
        ab.concatenate_q4_experts(w, ids)


def test_concatenated_oracle_and_all_split_planes_agree():
    w = IndexedWeights(
        12,
        256,
        2048,
        3,
        partial_specs=[(256, s) for s in (2, 4, 8)],
        partial_experts=(2, 0),
    )
    d = ab.concatenate_q4_experts(w, (2, 0))
    values = activation_values([0])
    expected = np.concatenate([values @ w.sums[e] for e in (2, 0)], axis=1)
    assert np.array_equal(values @ d.sums[0], expected)
    for key, parts in d.partial_sums.items():
        assert np.allclose(parts[0][0].sum(0), d.sums[0], rtol=1e-14, atol=1e-12)


def test_existing_payload_selection_and_five_arm_plan():
    records = ab.selected_modules()
    assert len(records) == 3 and len(ab.ARMS) == 5
    assert records[ab.DENSE_BEST]["parent"]["ap"] == 1
    assert records[ab.DENSE_MATCHED]["parent"]["ap"] == 0
    assert (
        records[ab.GROUPED]["key"]
        == "c7909f43fb689829d4065a89aaf9d85f9d60a13ec768be7e17d21289c1f8ca02"
    )
    assert all(r["parent"]["route"].startswith("fq-") for r in records.values())


@pytest.mark.parametrize(
    "plant", [None, "output", "partial", "guard", "workspace-size"]
)
def test_dense_shared_driver_keeps_alignment_and_checks_partials(monkeypatch, plant):
    w = IndexedWeights(12, 256, 512, 1, partial_specs=[(256, 2)], partial_experts=(0,))
    data = sweep.fixture(w, dict(mode=0, rows=1, topk=1, channels=1))
    data["values"] = activation_values([0])
    memory, calls = [], {}

    class SDK:
        def synchronize(self, stream):
            pass

        def fill(self, p, value, count):
            C.memset(p, value, count)

        def download(self, p, count):
            return C.string_at(p, count)

    class Resources:
        def __init__(self, sdk):
            self.stream = C.c_void_p(1)

        def alloc(self, count):
            buf = C.create_string_buffer(count + 256)
            memory.append(buf)
            return (C.addressof(buf) + 255) & ~255

        def upload(self, a):
            p = self.alloc(a.nbytes)
            C.memmove(p, a.ctypes.data, a.nbytes)
            return p

        def samples(self, fn, count):
            for _ in range(count):
                assert fn() == 0
            return [160.0] * count

        def close(self):
            pass

    class Module:
        record = dict(
            parent=dict(route="fq-dense", symbol="dense-test", tm=8, tn=64, tk=256)
        )

        def device_identity(self):
            return dict(ordinal=0, compute_units=72)

        def query(self, cp, rp, qp):
            c = C.cast(cp, C.POINTER(sweep.Call)).contents
            assert c.experts == c.m == 1
            assert not c.offsets_device and not c.rows_host and not c.rows_device
            q = C.cast(qp, C.POINTER(sweep.Query)).contents
            q.workspace_bytes = 2 * c.m * c.n * 4 + (
                16 if plant == "workspace-size" else 0
            )
            q.shared_bytes, q.occupancy = 38912, 6
            return 0

        def prepare(self, cp, rp, hp):
            calls["call"] = sweep.Call.from_buffer_copy(
                C.cast(cp, C.POINTER(sweep.Call)).contents
            )
            assert calls["call"].workspace % 128 == 0
            assert calls["call"].output % 16 == 0
            C.cast(hp, C.POINTER(C.c_void_p))[0] = C.c_void_p(1)
            return 0

        def run(self, handle, stream):
            c = calls["call"]
            partials = sweep.partial_gold(w, data, 256, 2)["golden"].astype("<f4")
            out = np.add(partials[0], partials[1], dtype="f4").astype("<f2")
            if plant == "output":
                out.fill(0)
            if plant == "partial":
                partials[1].fill(0)
            C.memmove(c.workspace, partials.ctypes.data, partials.nbytes)
            C.memmove(c.output, out.ctypes.data, out.nbytes)
            if plant == "guard":
                C.memset(c.workspace - 1, 0, 1)
            return 0

        def destroy(self, handle):
            pass

    class Replay:
        def __init__(self, sdk, stream, fn, repeats):
            self.fn = fn

        def __call__(self):
            return self.fn()

        def close(self):
            pass

    monkeypatch.setattr(sweep, "Resources", Resources)
    monkeypatch.setattr(sweep, "Replay", Replay)
    monkeypatch.setattr(sweep, "upload_into", lambda sdk, p, a: None)
    args = SimpleNamespace(
        correctness_repeats=2, warmups=1, rounds=1, samples=2, graph_repeats=16
    )

    def run():
        return sweep.gemm_cell(
            args, SDK(), Module(), w, [data], 2, "dense", capture_output=True
        )

    if plant:
        with pytest.raises(ValueError):
            run()
    else:
        result = run()
        assert result["median_us"] == 10.0
        assert result["timing_scope"] == "GEMM_PLUS_REDUCER"
        assert len(result["output_fp16_bits"]) == 256
        assert result["profiled_partial_error"] < 0.005


@pytest.mark.parametrize("plant", [False, True])
def test_round_order_failure_retention_and_unprofiled_summary(
    tmp_path, monkeypatch, plant
):
    calls = []

    def sample(args, sdk, records, weights, profiles, name):
        calls.append(name)
        if plant and len(calls) == 2:
            raise ValueError("planted arm failure")
        return dict(
            arm=name,
            arm_name=name,
            status="PASS",
            output_fp16_bits=[0x3C00],
            calls_per_graph=16,
            graph_elapsed_samples_us=[[160.0] * 3],
            timing_scope="GEMM_PLUS_REDUCER",
        )

    monkeypatch.setattr(ab, "sample_arm", sample)
    args = SimpleNamespace(output=tmp_path, rounds=2)
    result = ab.benchmark(args, None, {}, {}, {}, {})
    assert calls == list(ab.ARMS) + list(reversed(ab.ARMS))
    assert result["status"] == ("FAIL" if plant else "PASS")
    assert len(result["samples"]) == 10 - int(plant)
    assert result["summary"]["dense-historical-s8"]["median_us"] == 10.0
    assert result["summary"]["dense-historical-s8"][
        "effective_weight_MBU_pct"
    ] == pytest.approx(471.8592 / 2766 * 100)
    assert len((tmp_path / "summary.tsv").read_text().splitlines()) == 6
    assert (
        json.loads((tmp_path / "timing.json").read_text())["status"] == result["status"]
    )


@pytest.mark.parametrize("plant", [None, "report", "receipt", "identity", "exit"])
def test_acu_capture_requires_three_real_receipts_and_keeps_other_arms(
    tmp_path, monkeypatch, plant
):
    calls = []
    records = {
        symbol: dict(key=symbol, sha256="hash") for symbol, _, _ in ab.ARMS.values()
    }

    class Process:
        def __init__(self, cmd, stdout, stderr):
            name = cmd[cmd.index("--profile-arm") + 1]
            self.bad = not calls
            calls.append(name)
            output = ab.Path(cmd[cmd.index("--output") + 1])
            report = ab.Path(cmd[cmd.index("--export") + 1]).with_suffix(".acurep")
            if not (self.bad and plant == "report"):
                report.write_bytes(b"native report")
            if not (self.bad and plant == "receipt"):
                output.write_text(
                    json.dumps(
                        dict(
                            status="PASS",
                            arm_name=name,
                            result=dict(
                                build_key=(
                                    "wrong"
                                    if self.bad and plant == "identity"
                                    else ab.ARMS[name][0]
                                ),
                                module_sha256="hash",
                            ),
                        )
                    )
                )

        def wait(self, timeout):
            return 1 if self.bad and plant == "exit" else 0

    monkeypatch.setattr(ab.subprocess, "Popen", Process)
    args = SimpleNamespace(
        output=tmp_path,
        acu=ab.Path("acu"),
        sdk=tmp_path,
        native=tmp_path,
        compact=tmp_path,
    )
    assert ab.capture(args, records) == (plant is None)
    assert calls == list(ab.PROFILE_ARMS)
    rows = json.loads((tmp_path / "acu-index.json").read_text())
    assert sum(r["status"] == "CAPTURED" for r in rows) == 3 - int(plant is not None)
