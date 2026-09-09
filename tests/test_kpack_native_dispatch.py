import ctypes as C
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Request(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [
            (f, C.c_int32)
            for f in ("qtype", "route", "m", "n", "k", "experts", "max_rows")
        ]
        + [("mapping_id", C.c_uint64)]
    )


class Choice(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [(f, C.c_uint64) for f in ("ticket", "workspace_bytes", "shared_bytes")]
        + [
            (f, C.c_int32)
            for f in ("policy", "algorithm", "split", "grid", "device", "compute_units")
        ]
        + [("parent", C.c_char * 192), ("build_key", C.c_char * 65)]
    )


class Call(C.Structure):
    _fields_ = (
        [("version", C.c_uint32), ("size", C.c_uint32)]
        + [
            (f, C.c_int32)
            for f in ("m", "n", "k", "experts", "group_size", "device", "compute_units")
        ]
        + [("mapping_id", C.c_uint64)]
        + [
            (f, C.c_void_p)
            for f in (
                "a",
                "low",
                "high",
                "metadata",
                "zero",
                "output",
                "rows_host",
                "rows_device",
                "offsets_device",
                "workspace",
            )
        ]
        + [("workspace_bytes", C.c_uint64), ("stream", C.c_void_p)]
    )


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("native-policy")
    exe = tmp / "policy"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{ROOT}",
            str(ROOT / "tools/kpack_native_policy.cpp"),
            "-o",
            str(exe),
        ],
        check=True,
    )
    return exe


def query(probe, rows):
    return subprocess.check_output(
        [str(probe)],
        text=True,
        input="".join(" ".join(map(str, r)) + "\n" for r in rows),
    ).splitlines()


def test_all_formats_bounds_and_dense(probe):
    rows = [
        (q, r, 1024 if r >= 2 else 128, 512, 3072, 256 if r >= 2 else 1, 128)
        for q in range(10, 15)
        for r in range(4)
    ]
    results = query(probe, rows)
    assert len(results) == 20
    for row, line in zip(rows, results):
        if line == "MISS":
            assert row[1] < 2  # Unknown dense family is not a fabricated hit.
            continue
        p = line.split()
        assert int(p[1]) == row[0] and int(p[2]) == row[1]
        if row[1] >= 2:
            assert int(p[-1]) == 4 and int(p[12]) == 1


@pytest.mark.parametrize(
    "field,value", [(0, 9), (1, 4), (2, 0), (3, 513), (4, 513), (5, 0), (6, 0)]
)
def test_invalid_query(probe, field, value):
    row = [12, 2, 1024, 512, 3072, 256, 128]
    row[field] = value
    assert query(probe, [row]) == ["MISS"]


@pytest.mark.parametrize("q,n,k,split", [(12, 512, 2048, 4), (13, 2048, 512, 1)])
def test_measured_grouped_decode(probe, q, n, k, split):
    row = (q, 2, 8, n, k, 256, 1)
    parts = query(probe, [row])[0].split()
    layout = 1 if q == 12 else 2
    assert (
        parts[0]
        == f"fqg_q{q}_l{layout}_tm8_tn64_tk256_wm8_wn16_s2_ap0_dn64_nonpersistent"
    )
    assert int(parts[12]) == split
    assert int(parts[13]) == 0  # Ordinary compact, not persistent.
    assert int(parts[-1]) == 5  # Measured full GPU path, not a router-bound prediction.


@pytest.mark.parametrize("q,n,k", [(12, 512, 2048), (13, 2048, 512)])
def test_grouped_decode_does_not_expand_scope(probe, q, n, k):
    anchor = [q, 2, 8, n, k, 256, 1]
    rows = []
    for field, value in [
        (0, 14),
        (1, 3),
        (2, 7),
        (2, 16),
        (3, n * 2),
        (4, k * 2),
        (5, 128),
        (6, 2),
    ]:
        changed = anchor.copy()
        changed[field] = value
        rows.append(changed)
    for line in query(probe, rows):
        assert line == "MISS" or int(line.split()[-1]) != 5


def build_stub(
    tmp, probe, *, wrong_key=False, values=(12, 2, 528, 3072, 512, 256, 129)
):
    parts = query(probe, [values])[0].split()
    assert parts[0] != "MISS"
    symbol = parts[0]
    q, route, tm, tn, tk, wm, wn, stages, ap, dn = map(int, parts[1:11])
    mapping = 0x51344B5034540001 if q == 12 else 0x514B504B54000001
    key = "a" * 64
    module = tmp / "modules" / key
    module.mkdir(parents=True)
    (tmp / "catalog.inc").write_text(
        "static Image const kImages[] = {{"
        + ",".join(
            [json.dumps(symbol), json.dumps(key), json.dumps("b" * 64)]
            + list(map(str, [q, route, tm, tn, tk, wm, wn, stages, ap, dn]))
        )
        + "}};\n"
    )
    actual_key = "c" * 64 if wrong_key else key
    (tmp / "stub_identity.inc").write_text(
        "static qk_identity_v1 const stub_identity{1,sizeof(qk_identity_v1),"
        + ",".join(map(str, [q, route, tm, tn, tk, wm, wn, stages, ap, dn]))
        + f",{mapping}ULL,"
        + json.dumps(symbol)
        + ","
        + json.dumps(actual_key)
        + "};\n"
        + f"static constexpr int stub_expected_split = {int(parts[12])};\n"
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{ROOT}",
            f"-I{tmp}",
            str(ROOT / "tests/kpack_dispatch_stub.cpp"),
            "-o",
            str(module / "kernel.so"),
        ],
        check=True,
    )
    library = tmp / "dispatch.so"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O1",
            "-pthread",
            f"-I{tmp}",
            str(ROOT / "quactlize/dispatch/binding.cpp"),
            "-ldl",
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = C.CDLL(str(library), mode=C.RTLD_LOCAL)
    specifications = {
        "open": ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int),
        "close": ([C.c_void_p], None),
        "query": ([C.c_void_p, C.POINTER(Request), C.POINTER(Choice)], C.c_int),
        "prepare": (
            [C.c_void_p, C.POINTER(Choice), C.POINTER(Call), C.POINTER(C.c_void_p)],
            C.c_int,
        ),
        "run": ([C.c_void_p, C.c_void_p], C.c_int),
        "destroy": ([C.c_void_p], None),
    }
    functions = {}
    for name, (args, ret) in specifications.items():
        f = getattr(lib, "quactlize_kpack_dispatch_" + name + "_v1")
        f.argtypes = args
        f.restype = ret
        functions[name] = f
    runtime = C.c_void_p()
    assert functions["open"](str(tmp).encode(), C.byref(runtime)) == 0
    req = Request(1, C.sizeof(Request), *values, mapping)
    return functions, runtime, req


@pytest.mark.parametrize(
    "values,split",
    [((12, 2, 8, 512, 2048, 256, 1), 4), ((13, 2, 8, 2048, 512, 256, 1), 1)],
)
def test_measured_decode_binding_keeps_selected_split(tmp_path, probe, values, split):
    f, r, req = build_stub(tmp_path, probe, values=values)
    choice = Choice()
    assert f["query"](r, C.byref(req), C.byref(choice)) == 0
    assert (
        choice.policy == 5
        and choice.split == split
        and choice.algorithm == 0
        and choice.grid == 0
    )
    call = Call(
        version=1,
        size=C.sizeof(Call),
        m=8,
        n=values[3],
        k=values[4],
        experts=256,
        group_size=32,
        device=0,
        compute_units=72,
        mapping_id=req.mapping_id,
        a=0x1000,
        low=0x2000,
        metadata=0x3000,
        output=0x4000,
        offsets_device=0x5000,
        workspace=0x6000,
        workspace_bytes=256,
    )
    handle = C.c_void_p()
    assert f["prepare"](r, C.byref(choice), C.byref(call), C.byref(handle)) == 0
    assert f["run"](handle, None) == 0
    f["destroy"](handle)
    f["close"](r)


def test_full_binding_and_reuse(tmp_path, probe):
    f, r, req = build_stub(tmp_path, probe)
    choice = Choice()
    assert f["query"](r, C.byref(req), C.byref(choice)) == 0
    assert choice.policy == 4 and choice.workspace_bytes == 256
    again = Choice()
    assert f["query"](r, C.byref(req), C.byref(again)) == 0
    assert bytes(choice) == bytes(again)
    c = Call()
    c.version = 1
    c.size = C.sizeof(c)
    c.m, c.n, c.k, c.experts, c.group_size, c.device, c.compute_units = (
        528,
        3072,
        512,
        256,
        32,
        0,
        72,
    )
    c.mapping_id = req.mapping_id
    c.a, c.low, c.metadata, c.output, c.offsets_device, c.workspace = (
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        0x5000,
        0x6000,
    )
    c.workspace_bytes = 256
    h = C.c_void_p()
    c.rows_host = 0x7000
    assert f["prepare"](r, C.byref(choice), C.byref(c), C.byref(h)) == 2 and not h.value
    c.rows_host = None
    bad = Choice.from_buffer_copy(choice)
    bad.split = 4
    assert f["prepare"](r, C.byref(bad), C.byref(c), C.byref(h)) == 2
    assert f["prepare"](r, C.byref(choice), C.byref(c), C.byref(h)) == 0
    for _ in range(4):
        assert f["run"](h, None) == 0
    f["close"](r)
    assert f["run"](h, None) == 0  # Handle pins its exact module past runtime lifetime.
    f["destroy"](h)


def test_wrong_build_declines(tmp_path, probe):
    f, r, req = build_stub(tmp_path, probe, wrong_key=True)
    choice = Choice()
    assert f["query"](r, C.byref(req), C.byref(choice)) == 3
    f["close"](r)


def test_missing_parent_is_not_another_recipe(tmp_path, probe):
    f, r, req = build_stub(tmp_path, probe)
    req.qtype = 11
    req.mapping_id = 0x514B504B54000001
    choice = Choice()
    assert f["query"](r, C.byref(req), C.byref(choice)) == 1 and choice.ticket == 0
    f["close"](r)
