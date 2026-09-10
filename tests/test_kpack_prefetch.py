import ctypes as C
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_kpack_prefetch as probe
from tools.build_kpack_prefetch import subjects


def test_exact_measured_subjects():
    rows = subjects()
    assert [(r["q"], r["n"], r["k"], r["split"]) for r in rows] == [
        (12, 512, 2048, 4),
        (13, 2048, 512, 1),
    ]
    for row in rows:
        p = row["module"]["parent"]
        assert (
            p["route"],
            p["tm"],
            p["tn"],
            p["tk"],
            p["wm"],
            p["wn"],
            p["stages"],
            p["persistent"],
        ) == ("fq-grouped", 8, 64, 256, 8, 16, 2, 0)


def test_ranges_only_active_experts_and_all_planes():
    planes = dict(low=4096, high=8192, units=16384)
    sizes = dict(low=512, high=256, units=256)
    ranges = probe.weight_ranges(planes, sizes, 4, [0, 3])
    assert ranges == [
        (4096, 128),
        (4480, 128),
        (8192, 64),
        (8384, 64),
        (16384, 64),
        (16576, 64),
    ]
    assert sum(n for _, n in ranges) == sum(sizes.values()) // 2
    sizes["high"] = 0
    assert len(probe.weight_ranges(planes, sizes, 4, [0, 3])) == 4
    for active in ([0, 0], [-1], [4], []):
        with pytest.raises(ValueError):
            probe.weight_ranges(planes, sizes, 4, active)
    with pytest.raises(ValueError):
        probe.weight_ranges(planes, sizes | {"units": 255}, 4, [0])
    with pytest.raises(ValueError):
        probe.weight_ranges(planes | {"low": 4097}, sizes, 4, [0])


def test_disjoint_full_allocations_not_just_selected_ranges():
    probe.disjoint([(4096, 1024)], [(5120, 1024)])
    with pytest.raises(ValueError):
        probe.disjoint([(4096, 1024)], [(5088, 128)])


def test_event_intervals_exclude_prefetch_and_measure_interference():
    times = dict(
        current_start=0.0,
        current_end=20.0,
        prefetch_start=2.0,
        prefetch_end=12.0,
        next_start=20.0,
        next_end=32.0,
    )
    assert probe.intervals(times, True, True) == dict(
        current_us=20.0,
        prefetch_us=10.0,
        target_us=12.0,
        total_us=32.0,
        envelope_overlap_us=10.0,
    )
    for changes in (
        {"next_start": 10.0},
        {"current_end": 0.0},
        {"prefetch_end": float("nan")},
    ):
        with pytest.raises(ValueError):
            probe.intervals(times | changes, True, True)
    late = times | dict(
        prefetch_start=22.0, prefetch_end=30.0, next_start=30.0, next_end=42.0
    )
    assert probe.intervals(late, True, True)["envelope_overlap_us"] == 0.0
    assert (
        probe.intervals(
            dict(prefetch_start=0.0, prefetch_end=8.0, next_start=8.0, next_end=20.0),
            False,
            True,
        )["target_us"]
        == 12.0
    )


def test_receipt_detects_dropped_load_or_wrong_range():
    words = np.zeros(128, dtype="<u8")
    words[0] = (8 << 32) | 0x1234
    sdk = SimpleNamespace(download=lambda *_: words.tobytes())
    target = SimpleNamespace(ranges=[(4096, 256)], checksum=0x1234)
    probe.check_receipt(sdk, 4096, 1, target, "load")
    with pytest.raises(ValueError):
        probe.check_receipt(sdk, 4096, 1, target, "hint")
    words[0] = 8 << 32
    probe.check_receipt(sdk, 4096, 1, target, "hint")
    with pytest.raises(ValueError):
        probe.check_receipt(sdk, 4096, 1, target, "load")
    words[0] = 7 << 32
    with pytest.raises(ValueError):
        probe.check_receipt(sdk, 4096, 1, target, "hint")


def test_graph_has_internal_fork_and_join_after_current_end(monkeypatch):
    log = []

    def create(ptr):
        ptr._obj.value = 100 + len(log)
        log.append(("create", ptr._obj.value))
        return 0

    def call(tag, *args):
        log.append((tag, *args))
        return 0

    lib = SimpleNamespace(
        hggcEventCreate=create,
        hggcEventRecord=lambda e, s: call("record", e.value, s),
        hggcStreamWaitEvent=lambda s, e, f: call("wait", s, e.value, f),
        hggcEventDestroy=lambda e: 0,
    )

    class Replay:
        def __init__(self, sdk, stream, fn, repeats):
            assert repeats == 1
            log.append(("capture_begin", stream))
            assert fn() == 0
            log.append(("capture_end", stream))

        def close(self):
            pass

    monkeypatch.setattr(probe, "Replay", Replay)
    api = SimpleNamespace(
        pressure=lambda *a: call("pressure"),
        prefetch=lambda *a: call("prefetch", a[-1]),
    )
    current = SimpleNamespace(launch=lambda: call("current"))
    target = SimpleNamespace(launch=lambda: call("target"), ranges=[(4096, 256)])
    graph = probe.Experiment(
        SimpleNamespace(lib=lib),
        SimpleNamespace(stream=1),
        SimpleNamespace(stream=2),
        api,
        current,
        target,
        0,
        256,
        0,
        0,
        0,
        "pair",
        "load",
        4,
    )
    origin = graph.events["origin"].value
    end = graph.events["prefetch_end"].value
    assert (
        log.index(("record", origin, 1))
        < log.index(("wait", 2, origin, 0))
        < log.index(("prefetch", 2))
    )
    assert (
        log.index(("current",))
        < log.index(("record", graph.events["current_end"].value, 1))
        < log.index(("wait", 1, end, 0))
        < log.index(("target",))
    )
    assert (
        log.index(("capture_begin", 1))
        < log.index(("pressure",))
        < log.index(("capture_end", 1))
    )
    graph.close()


def test_summary_does_not_mix_pair_and_sequential_baselines():
    cells = [
        dict(kind=k, mode=m, blocks=b, samples=[values])
        for k, m, b, values in (
            ("cold", "none", 0, dict(target_us=20.0, total_us=20.0)),
            ("pair", "none", 0, dict(current_us=20.0, target_us=18.0, total_us=38.0)),
            (
                "sequential",
                "load",
                4,
                dict(target_us=10.0, prefetch_us=15.0, total_us=25.0),
            ),
            (
                "pair",
                "load",
                4,
                dict(
                    current_us=22.0,
                    target_us=9.0,
                    prefetch_us=15.0,
                    total_us=31.0,
                    envelope_overlap_us=10.0,
                ),
            ),
        )
    ]
    rows = probe.summary(cells)
    assert rows[2]["target_delta_pct"] == -50.0
    assert rows[2]["total_delta_pct"] == 25.0
    assert rows[3]["target_delta_pct"] == -50.0
    assert rows[3]["current_delta_pct"] == pytest.approx(10.0)
    assert "NOT_PRODUCER" in rows[3]["overlap_scope"]
    assert len(probe.configurations([4, 16, 36])) == 15
