import ctypes as C
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_kpack_grouped_decode_probe as probe


def test_timings_are_per_call_not_per_graph_or_dram_counters():
    r = probe.timing_summary([[160.0, 176.0], [144.0, 160.0]], 16, 4718592)
    assert r["median_us"] == 10
    assert r["effective_weight_GB_per_s"] == pytest.approx(471.8592)
    assert "NOT_MEASURED_DRAM" in r["bandwidth_scope"]


@pytest.mark.parametrize(
    "values", [[], [[float("nan")]], [[float("inf")]], [[0]], [[-1]]]
)
def test_invalid_timings(values):
    with pytest.raises(ValueError):
        probe.timing_summary(values, 16, 4718592)


def test_grid_counts_empty_experts_but_not_their_weights():
    rows = [int(i in range(0, 136, 17)) for i in range(256)]
    assert probe.grid_model(dict(persistent=0, tn=64), rows, 512) == {
        "device-only": [1, 8, 256],
        "host-compact": [8, 8, 1],
    }
    with pytest.raises(ValueError):
        probe.grid_model(dict(persistent=1, tn=64), rows, 512)
    rows[0] = 2
    with pytest.raises(ValueError):
        probe.grid_model(dict(persistent=0, tn=64), rows, 512)


@pytest.mark.parametrize("arm", ["both", "device-only", "host-compact"])
def test_run_uses_same_parent_and_weights_with_distinct_metadata_contracts(
    monkeypatch, tmp_path, arm
):
    calls, freed, memory = {}, [], []
    golden = np.ones((8, 512), dtype="<f2")
    ids = np.arange(8, dtype="i4") * 17
    data = dict(
        a=np.ones((1, 2048), dtype="<f2"),
        expert=ids,
        arows=np.zeros(8, dtype="i4"),
        golden=golden.astype("f8"),
        denom=golden.astype("f8"),
    )
    weights = SimpleNamespace(
        planes={x: np.ones((256, 16), dtype="u1") for x in ("low", "units")}
    )
    weights.planes["high"] = np.empty((256, 0), dtype="u1")
    monkeypatch.setattr(probe, "IndexedWeights", lambda *a, **kw: weights)
    monkeypatch.setattr(probe, "fixture", lambda *a: data)
    parent = dict(symbol="same-parent", persistent=0, tm=16, tn=64)
    manifest = dict(
        modules=[dict(key="a" * 64, path="kernel.so", sha256="test", parent=parent)]
    )
    choice = SimpleNamespace(
        algorithm=0,
        split=1,
        grid=0,
        build_key=b"a" * 64,
        parent=b"same-parent",
        device=0,
        compute_units=72,
        shared_bytes=38912,
        workspace_bytes=256,
        policy=4,
    )

    def launch(which):
        C.memmove(calls[which].output, golden.ctypes.data, golden.nbytes)
        return 0

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
            buf = C.create_string_buffer(count)
            memory.append(buf)
            return C.addressof(buf)

        def upload(self, a):
            p = self.alloc(a.nbytes)
            C.memmove(p, a.ctypes.data, a.nbytes)
            return p

        def samples(self, fn, count):
            for _ in range(count):
                assert fn() == 0
            return [40.0] * count

        def close(self):
            freed.append("resources")

    class Dispatch:
        def __init__(self, bundle):
            pass

        def query(self, *args):
            assert args[:7] == (12, 2, 8, 512, 2048, 256, 1)
            return choice

        def prepare(self, selected, call):
            assert selected is choice and not call.rows_host and not call.rows_device
            calls["device-only"] = probe.Call.from_buffer_copy(call)
            return lambda: launch("device-only")

        def close(self):
            freed.append("dispatch")

    class Module:
        def __init__(self, record):
            assert record["key"] == choice.build_key.decode()

        def query(self, cp, rp, qp):
            q = C.cast(qp, C.POINTER(probe.QueryResources)).contents
            q.workspace_bytes, q.shared_bytes = 256, 38912
            return 0

        def prepare(self, cp, rp, hp):
            c = C.cast(cp, C.POINTER(probe.Call)).contents
            rows = np.ctypeslib.as_array(
                C.cast(c.rows_host, C.POINTER(C.c_int)), shape=(256,)
            )
            assert list(np.flatnonzero(rows)) == list(ids)
            assert rows.sum() == 8 and rows.max() == 1 and c.rows_device
            calls["host-compact"] = probe.Call.from_buffer_copy(c)
            C.cast(hp, C.POINTER(C.c_void_p))[0] = C.c_void_p(42)
            return 0

        def run(self, handle, stream):
            return launch("host-compact")

        def destroy(self, handle):
            freed.append("module")

    class Replay:
        def __init__(self, sdk, stream, fn, repeats):
            self.fn, self.repeats = fn, repeats

        def __call__(self):
            for _ in range(self.repeats):
                assert self.fn() == 0
            return 0

        def close(self):
            freed.append("graph")

    for name in ("Resources", "Dispatch", "Module", "Replay"):
        monkeypatch.setattr(probe, name, locals()[name])
    args = SimpleNamespace(
        case="q4-up",
        bundle=tmp_path,
        arm=arm,
        samples=2,
        rounds=3,
        warmups=1,
        graph_repeats=2,
    )
    result = probe.run(args, SDK(), manifest)
    assert result["status"] == "PASS"
    assert result["raw_pair_equal"] == (True if arm == "both" else None)
    assert result["zero_output_negative"] == "DETECTED"
    assert "NOT_GRID_ONLY_AB" in result["comparison_scope"]
    for row in result["arms"].values():
        assert row["median_us"] == 20 and row["error"] == 0
        assert len(row["graph_elapsed_samples_us"]) == 3
    if arm == "both":
        for p in ("a", "low", "high", "metadata", "offsets_device"):
            assert getattr(calls["device-only"], p) == getattr(calls["host-compact"], p)
    assert freed[0] == "graph" and freed[-1] == "resources"
