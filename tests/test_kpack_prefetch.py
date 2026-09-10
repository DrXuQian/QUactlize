import ctypes as C
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_kpack_prefetch as probe
from tools.build_kpack_prefetch import (
    SDK_FILES,
    SDK_TOOLS,
    SDK_RUNTIME,
    digest,
    subjects,
    runtime_sdk_report,
    validate_sdk_files,
)


@pytest.fixture
def sdk_manifest():
    files = {p: f"{i + 1:064x}" for i, p in enumerate(SDK_FILES)}
    return dict(sdk=digest(list(files.items())), sdk_files=files)


def test_sdk_exact_match_preserves_original_identity(sdk_manifest):
    report = runtime_sdk_report(sdk_manifest, sdk_manifest["sdk_files"])
    assert report["allowed"] and report["runtime_matches"]
    assert report["status"] == "EXACT_SDK"
    assert report["build_sdk"] == report["actual_sdk"] == sdk_manifest["sdk"]
    assert not report["differences"]
    assert not report["override_requested"]


@pytest.mark.parametrize("tool", SDK_TOOLS)
@pytest.mark.parametrize("value", [None, "a" * 64])
def test_prebuilt_execution_allows_changed_or_absent_tools(sdk_manifest, tool, value):
    actual = sdk_manifest["sdk_files"] | {tool: value}
    report = runtime_sdk_report(sdk_manifest, actual)
    assert report["allowed"] and report["runtime_matches"]
    assert report["status"] == "RUNTIME_MATCH_TOOLS_DIFFER"
    assert report["actual_sdk"] != report["build_sdk"]
    assert report["differences"] == [
        dict(
            path=tool, build_sha256=sdk_manifest["sdk_files"][tool], actual_sha256=value
        )
    ]


@pytest.mark.parametrize("library", SDK_RUNTIME)
def test_changed_runtime_requires_explicit_unverified_experiment(sdk_manifest, library):
    actual = sdk_manifest["sdk_files"] | {library: "a" * 64}
    strict = runtime_sdk_report(sdk_manifest, actual)
    assert strict["status"] == "REJECTED_RUNTIME_MISMATCH"
    assert not strict["allowed"] and not strict["runtime_matches"]
    opted_in = runtime_sdk_report(sdk_manifest, actual, allow_unverified=True)
    assert opted_in["status"] == "UNVERIFIED_RUNTIME"
    assert opted_in["allowed"] and opted_in["override_requested"]
    assert not opted_in["runtime_matches"]
    assert opted_in["actual_files"] == actual
    assert opted_in["build_files"] == sdk_manifest["sdk_files"]
    assert opted_in["device_validation"] == "REQUIRED_NOT_IMPLIED_BY_HASHES"
    with pytest.raises(ValueError, match="missing/invalid runtime"):
        runtime_sdk_report(
            sdk_manifest, actual | {library: None}, allow_unverified=True
        )


def test_sdk_override_does_not_change_build_authority(sdk_manifest):
    changed = sdk_manifest | {
        "sdk_files": sdk_manifest["sdk_files"] | {SDK_TOOLS[0]: "a" * 64}
    }
    with pytest.raises(ValueError, match="original combined identity"):
        runtime_sdk_report(changed, changed["sdk_files"], allow_unverified=True)
    with pytest.raises(ValueError, match="per-file build SDK receipt"):
        validate_sdk_files({"sdk": sdk_manifest["sdk"]})


@pytest.mark.parametrize("allow", [False, True])
def test_sdk_receipt_written_before_launch_or_rejection(
    sdk_manifest, tmp_path, monkeypatch, capsys, allow
):
    actual = sdk_manifest["sdk_files"] | {SDK_RUNTIME[0]: "a" * 64}
    monkeypatch.setattr(probe, "sdk_files", lambda *a, **kw: actual)
    args = SimpleNamespace(
        sdk=tmp_path, output=tmp_path, case="q12", allow_unverified_sdk=allow
    )
    if allow:
        assert probe.admit_runtime_sdk(args, sdk_manifest)["allowed"]
    else:
        with pytest.raises(ValueError, match="--allow-unverified-sdk"):
            probe.admit_runtime_sdk(args, sdk_manifest)
    receipt = json.loads((tmp_path / "q12.sdk.json").read_text())
    assert receipt["allowed"] is allow
    assert receipt["actual_files"] == actual
    assert "KPACK_PREFETCH_SDK " in capsys.readouterr().out


def test_published_sdk_receipt_matches_unchanged_prebuilt():
    manifest, _ = probe.verify(probe.ROOT / "prebuilt/ppu0010/kpack-prefetch-v1")
    assert (
        manifest["sha256"]
        == "c73e01204eaf520bb3e05056930d8aff1f22f83b85b5bb100be7b9b1b0ac7679"
    )
    for subject in manifest["subjects"]:
        assert subject["module"]["identity"]["sdk"] == manifest["sdk"]


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


@pytest.mark.parametrize("record_nodes", [True, False])
def test_graph_has_internal_fork_and_explicit_timestamps(monkeypatch, record_nodes):
    log, nodes = [], []

    def create(ptr):
        ptr._obj.value = 100 + len(log)
        log.append(("create", ptr._obj.value))
        return 0

    def call(tag, *args):
        log.append((tag, *args))
        return 0

    def record(event, stream, flags):
        assert flags == probe.EVENT_RECORD_EXTERNAL
        if record_nodes:
            nodes.append(event.value)
        return call("timestamp", event.value, stream, flags)

    def get_nodes(graph, out, count):
        count._obj.value = len(nodes)
        if out is not None:
            for i in range(len(nodes)):
                out[i] = i + 1
        return 0

    def get_type(node, out):
        out._obj.value = probe.GRAPH_EVENT_RECORD
        return 0

    def get_event(node, out):
        out._obj.value = nodes[node - 1]
        return 0

    lib = SimpleNamespace(
        hggcEventCreate=create,
        hggcEventRecord=lambda e, s: call("record", e.value, s),
        hggcEventRecordWithFlags=record,
        hggcGraphGetNodes=get_nodes,
        hggcGraphNodeGetType=get_type,
        hggcGraphEventRecordNodeGetEvent=get_event,
        hggcStreamWaitEvent=lambda s, e, f: call("wait", s, e.value, f),
        hggcEventDestroy=lambda e: call("destroy", e.value),
    )

    class Replay:
        def __init__(self, sdk, stream, fn, repeats):
            assert repeats == 1
            self.graph = C.c_void_p(777)
            log.append(("capture_begin", stream))
            assert fn() == 0
            log.append(("capture_end", stream))

        def close(self):
            log.append(("destroy_graph",))

    monkeypatch.setattr(probe, "Replay", Replay)
    api = SimpleNamespace(
        pressure=lambda *a: call("pressure"),
        prefetch=lambda *a: call("prefetch", a[-1]),
    )
    current = SimpleNamespace(launch=lambda: call("current"))
    target = SimpleNamespace(launch=lambda: call("target"), ranges=[(4096, 256)])
    args = (
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
    if not record_nodes:
        # Model the original bug: dependencies capture successfully, but no
        # event RECORD nodes exist. Reject before running the numerical fixture.
        with pytest.raises(ValueError, match="exactly one record node: origin"):
            probe.Experiment(*args)
        assert len([v for v in log if v[0] == "destroy"]) == 9
        assert ("destroy_graph",) in log
        return
    graph = probe.Experiment(*args)
    assert graph.timing_nodes == 7
    origin = graph.events["origin"].value
    fork, join = graph.fences["fork"].value, graph.fences["join"].value
    assert (
        log.index(("timestamp", origin, 1, 1))
        < log.index(("record", fork, 1))
        < log.index(("wait", 2, fork, 0))
        < log.index(("prefetch", 2))
        < log.index(("record", join, 2))
    )
    assert (
        log.index(("current",))
        < log.index(("timestamp", graph.events["current_end"].value, 1, 1))
        < log.index(("wait", 1, join, 0))
        < log.index(("target",))
    )
    assert (
        log.index(("capture_begin", 1))
        < log.index(("pressure",))
        < log.index(("capture_end", 1))
    )
    assert not ({fork, join} & set(nodes))
    assert [entry for entry in log if entry[0] == "wait"] == [
        ("wait", 2, fork, 0),
        ("wait", 1, join, 0),
    ]
    # A duplicate node for a timer is not a valid single timestamp boundary.
    nodes.append(origin)
    with pytest.raises(ValueError, match="exactly one record node: origin"):
        probe.validate_timing_nodes(
            SimpleNamespace(lib=lib), graph.graph.graph, graph.events
        )
    graph.close()
    graph.close()  # cleanup is idempotent


@pytest.mark.parametrize("status", [0, 1])
def test_graph_timestamp_read_success_or_named_failure(status):
    calls = []

    def elapsed(ms, begin, end):
        ms._obj.value = {2: 0.002, 3: 0.022}[end.value]
        calls.append("elapsed")
        return status

    graph = object.__new__(probe.Experiment)
    graph.kind, graph.prefetching = "cold", False
    graph.label, graph.timing_nodes = "kind=cold mode=none blocks=0", 3
    graph.events = dict(
        origin=C.c_void_p(1), next_start=C.c_void_p(2), next_end=C.c_void_p(3)
    )
    graph.graph = lambda: 0
    graph.r = SimpleNamespace(stream=9)
    graph.sdk = SimpleNamespace(
        synchronize=lambda stream: calls.append("sync"),
        lib=SimpleNamespace(hggcEventElapsedTime=elapsed, hggcEventQuery=lambda e: 0),
    )
    if status:
        with pytest.raises(
            RuntimeError,
            match=r"kind=cold.*origin->next_start.*status=1.*event_query=\[0,0\]",
        ):
            graph.sample()
    else:
        assert graph.sample() == pytest.approx(dict(target_us=20, total_us=22))
    assert calls[0] == "sync"


@pytest.mark.parametrize("drop_prefetch", [False, True])
def test_timing_self_test_covers_timelines_and_detects_missing_write(
    tmp_path, monkeypatch, drop_prefetch
):
    memory, replays = {}, []

    def fill(pointer, value, size, stream):
        memory[pointer] = value
        return 0

    class Resources:
        def __init__(self, sdk):
            self.stream = 1

        def alloc(self, size):
            pointer = len(memory) + 1
            memory[pointer] = 0
            return pointer

        def fill(self, pointer, value, size):
            fill(pointer, value, size, self.stream)

        def close(self):
            pass

    class Experiment:
        def __init__(self, sdk, r, pf, api, current, target, *rest):
            self.kind, self.mode, _ = rest[-3:]
            self.api, self.current, self.target = api, current, target
            self.timing_nodes = 3

        def sample(self):
            self.api.pressure()
            if self.kind == "pair":
                self.current.launch()
            if self.mode != "none" and not drop_prefetch:
                self.api.prefetch(2)
            self.target.launch()
            replays.append((self.kind, self.mode))
            return dict(target_us=10.0)

        def close(self):
            pass

    monkeypatch.setattr(probe, "Resources", Resources)
    monkeypatch.setattr(probe, "Experiment", Experiment)
    sdk = SimpleNamespace(
        lib=SimpleNamespace(hggcMemsetAsync=fill),
        synchronize=lambda stream: None,
        download=lambda pointer, size: bytes([memory[pointer]]) * size,
    )
    path = tmp_path / "timing.json"
    if drop_prefetch:
        with pytest.raises(ValueError, match="fork/join write differs"):
            probe.timing_self_test(sdk, path)
        assert json.loads(path.read_text())["status"] == "FAIL"
    else:
        result = probe.timing_self_test(sdk, path)
        assert result == json.loads(path.read_text())
        assert result["status"] == "PASS"
        assert len(replays) == 15 and len(set(replays)) == len(result["arms"]) == 5


def test_timing_admission_precedes_full_weight_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "SDK", lambda _: None)
    monkeypatch.setattr(probe, "graph_bind", lambda _: None)
    monkeypatch.setattr(probe, "timing_bind", lambda _: None)

    def reject(*args):
        raise RuntimeError("timer unavailable")

    def unexpected(*args):
        pytest.fail("fixture must not be constructed when timing admission fails")

    monkeypatch.setattr(probe, "timing_self_test", reject)
    monkeypatch.setattr(probe, "IndexedWeights", unexpected)
    with pytest.raises(RuntimeError, match="timer unavailable"):
        probe.run(
            SimpleNamespace(sdk=tmp_path, output=tmp_path, case="q12"), {}, {}, None, {}
        )


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
