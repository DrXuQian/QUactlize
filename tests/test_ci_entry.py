"""CI must fail on missing coverage; no PPU or packaged binary is used here."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ci import run
from ci.suites import HOST, inventory
from test_kpack_device_pack import library


def test_inventory_has_one_owner_and_separates_legacy_and_device():
    groups = inventory()
    paths = [p for values in groups.values() for p in values]
    assert len(paths) == len(set(paths))
    assert set(paths) == {str(p.relative_to(run.ROOT / "tests"))
                          for p in (run.ROOT / "tests").rglob("test_*.py")}
    assert "test_ci_entry.py" in HOST
    assert "test_q8_kpack2.py" in HOST
    assert "test_dense_provider.py" in HOST
    assert "test_gguf_routes.py" in groups["separate"]


@pytest.mark.parametrize("body,expected", [
    ("<testcase name='a'/>", dict(tests=1, passed=1, failures=0, errors=0, skipped=0)),
    ("<testcase name='a'><skipped/></testcase>", dict(tests=1, passed=0, failures=0, errors=0, skipped=1)),
    ("<testcase name='a'><error/></testcase>", dict(tests=1, passed=0, failures=0, errors=1, skipped=0)),
])
def test_junit_counts_actual_cases_not_summary_claims(tmp_path, body, expected):
    xml = tmp_path / "a.xml"
    xml.write_text('<testsuites tests="999"><testsuite>' + body + '</testsuite></testsuites>')
    assert run.pytest_counts(xml) == expected
    xml.write_text("<testsuites/>")
    with pytest.raises(ValueError, match="empty suite"):
        run.pytest_counts(xml)


@pytest.mark.parametrize("fault", [None, "skip", "failure", "exit", "no-xml", "no-tests"])
def test_entry_denominator_and_fail_fast(tmp_path, monkeypatch, fault):
    calls = []
    monkeypatch.setattr(run, "preflight", lambda *args: None)
    def command(argv, log, env, timeout):
        calls.append(argv)
        assert "PYTEST_ADDOPTS" not in env
        assert "PYTEST_PLUGINS" not in env and "PYTHONOPTIMIZE" not in env
        assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        xml = next(Path(x.split("=", 1)[1]) for x in argv if x.startswith("--junitxml="))
        body = "<skipped/>" if fault == "skip" else "<failure/>" if fault == "failure" else ""
        if fault != "no-xml":
            xml.write_text("<testsuite>" + ("" if fault == "no-tests" else f"<testcase>{body}</testcase>") + "</testsuite>")
        return 3 if fault == "exit" else 0
    monkeypatch.setattr(run, "run_command", command)
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing")
    out = tmp_path / "result"
    assert run.main(["--suite", "host", "--output", str(out)]) == (0 if fault is None else 1)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == ("PASS" if fault is None else "FAIL")
    assert not summary["performance_admitted"]
    assert len(calls) == 1


def test_missing_dependencies_and_existing_output_are_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "preflight", lambda *a: (_ for _ in ()).throw(ValueError("no SDK")))
    out = tmp_path / "result"
    assert run.main(["--suite", "all", "--output", str(out)]) == 1
    data = json.loads((out / "summary.json").read_text())
    assert len(data["phases"]) == 1 and data["phases"][0]["error"] == "no SDK"
    assert (out / "host.xml").is_file()
    with pytest.raises(SystemExit):
        run.main(["--output", str(out)])
    assert json.loads((out / "summary.json").read_text()) == data


@pytest.mark.parametrize("suite,phases", [
    ("host", ["host"]),
    ("test", ["host", "accuracy"]),
    ("build", ["build"]),
    ("accuracy", ["accuracy"]),
    ("all", ["host", "build", "accuracy"]),
    ("tooling", ["tooling"]),
])
def test_shell_entry_works_outside_checkout_and_lists_without_dependencies(tmp_path, suite, phases):
    env = dict(os.environ, PYTHON=sys.executable, PPU_SDK=str(tmp_path / "missing"))
    r = subprocess.run(["bash", str(run.ROOT / "run_tests.sh"), "--list", "--suite", suite],
                       env=env, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["phases"] == phases


@pytest.mark.parametrize("suite", ["test", "accuracy"])
def test_test_only_requires_explicit_package_before_creating_output(tmp_path, capsys, suite):
    out = tmp_path / "result"
    with pytest.raises(SystemExit) as exc:
        run.main(["--suite", suite, "--output", str(out)])
    assert exc.value.code == 2
    assert "requires --package" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("failed_phase", [None, "host", "accuracy"])
def test_existing_package_checks_both_suites_without_rebuilding(tmp_path, monkeypatch, failed_phase):
    package = tmp_path / "installed package"
    package.mkdir()
    manifest = package / "manifest.json"
    manifest.write_bytes(b"host runner fixture, not an admitted device package\n")
    original = manifest.read_bytes()
    checks, calls = [], []

    def preflight(phase, sdk, path):
        checks.append(phase)
        assert path == package

    def command(argv, log, env, timeout):
        phase = env["QUACTLIZE_CI_SUITE"]
        calls.append(phase)
        assert phase in ("host", "accuracy")
        assert argv[:3] == [sys.executable, "-m", "pytest"]
        assert env["QUACTLIZE_CI_PACKAGE"] == str(package)
        assert env["PYTHON"] == env["QUACTLIZE_PPU_PYTHON"] == sys.executable
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        selected = [p for p in argv if p.endswith(".py")]
        assert selected == [str(run.ROOT / "tests" / p) for p in inventory()[phase]]
        xml = next(Path(x.split("=", 1)[1]) for x in argv if x.startswith("--junitxml="))
        # Even exit 0 cannot admit a skipped test in the combined test-only suite.
        body = "<skipped/>" if phase == failed_phase else ""
        xml.write_text(f"<testsuite><testcase>{body}</testcase></testsuite>")
        return 0

    monkeypatch.setattr(run, "preflight", preflight)
    monkeypatch.setattr(run, "run_command", command)
    monkeypatch.setenv("PYTHON", "/wrong/interpreter")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    out = tmp_path / "result"
    assert run.main(["--suite", "test", "--package", str(package), "--output", str(out)]) == int(
        failed_phase is not None)
    assert checks == calls == (["host"] if failed_phase == "host" else ["host", "accuracy"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["suite"] == "test"
    assert summary["status"] == ("PASS" if failed_phase is None else "FAIL")
    assert [row["phase"] for row in summary["phases"]] == calls
    assert manifest.read_bytes() == original
    assert sorted(package.iterdir()) == [manifest]
    assert not (out / "package").exists() and not (out / "build.command.json").exists()


def test_build_entry_honors_python_arguments_and_exit_code(tmp_path):
    # Intercept only the interpreter: no compiler, SDK, output package or device.
    interpreter = tmp_path / "python stub"
    interpreter.write_text(f"#!{sys.executable}\nimport json, sys\n"
                           "print(json.dumps(sys.argv[1:]))\nraise SystemExit(17)\n")
    interpreter.chmod(0o755)
    destination = tmp_path / "new package"
    args = ["--output", str(destination), "--jobs", "3"]
    result = subprocess.run(["bash", str(run.ROOT / "build.sh"), "--library", *args],
        cwd=tmp_path, env=dict(os.environ, PYTHON=str(interpreter)),
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 17, result.stderr
    assert json.loads(result.stdout) == [str(run.ROOT / "tools/build_ppu_library.py"), *args]
    assert not destination.exists()


def test_build_suite_uses_public_entry_and_same_python(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "preflight", lambda *args: None)
    calls = []
    def command(argv, log, env, timeout):
        calls.append(argv)
        assert argv[:3] == ["bash", str(run.ROOT / "build.sh"), "--library"]
        assert env["PYTHON"] == sys.executable
        assert env["QUACTLIZE_CI_SUITE"] == "build"
        return 7
    monkeypatch.setattr(run, "run_command", command)
    monkeypatch.setenv("PYTHON", "/wrong/interpreter")
    out = tmp_path / "result"
    assert run.main(["--suite", "build", "--output", str(out)]) == 1
    assert len(calls) == 1
    assert run.pytest_counts(out / "build.xml")["failures"] == 1


def test_command_nonzero_and_timeout_are_retained(tmp_path):
    env = dict(os.environ)
    assert run.run_command([sys.executable, "-c", "print('error'); raise SystemExit(7)"],
                           tmp_path / "fail.log", env, 30) == 7
    assert "error" in (tmp_path / "fail.log").read_text()
    with pytest.raises(TimeoutError):
        run.run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                        tmp_path / "timeout.log", env, 1)


def test_accuracy_inventory_and_independent_oracle():
    import ctypes as C
    import numpy as np
    from ppu.support import QTYPES, DECODE_M, DENSE_M, Params, Info, check_output
    from dev.bf16_compute.fixture import raw_weight, round_compute
    assert QTYPES == (8, 10, 11, 12, 13, 14)
    assert DECODE_M == tuple(range(1, 9))
    assert DENSE_M == tuple(range(1, 10)) + (16, 128, 256)
    assert C.sizeof(Params) == 112 and C.sizeof(Info) == 224
    assert Params.arrangement.offset == 40 and Params.sizes.offset == 80
    for q in QTYPES:
        _, w = raw_weight(q, 256, 512, 913 + q)
        a = np.random.default_rng(q).uniform(-.13, .13, (2, 512)).astype('f4')
        for compute in (0, 1):
            gold = round_compute(a, 'bf16' if compute else 'f16').astype('f8') @ w.astype('f8').T
            assert check_output(gold, a, w, compute)['bad'] == 0
            for bad in (np.zeros_like(gold), np.full_like(gold, np.nan), gold * 100):
                with pytest.raises(ValueError):
                    check_output(bad, a, w, compute)


def test_device_accuracy_never_requests_profilers_or_sweeps():
    from ci.suites import ACCURACY
    for name in ACCURACY:
        source = (run.ROOT / 'tests' / name).read_text()
        assert 'subprocess.run(' not in source
        assert '--profile' not in source and '--sweep' not in source
    source = (run.ROOT / 'tests/ppu/test_public_library.py').read_text()
    assert 'info.route != 3' in source  # cuBLAS remains ordinary calls.


@pytest.mark.parametrize('q', [8, 10, 11, 12, 13, 14])
@pytest.mark.parametrize('pair', [False, True])
def test_device_harness_packing_arguments_on_host(library, q, pair):
    """Test the harness's pointers/byte oracle, NOT GPU execution/scheduling."""
    import ctypes as C
    from types import SimpleNamespace
    import numpy as np
    from ppu.support import Arena, packed, Arrangement
    from quactlize.execution.native import arrangement

    allocations = {}
    def allocate(size):
        storage = C.create_string_buffer(size + 255)
        ptr = (C.addressof(storage) + 255) // 256 * 256
        allocations[ptr] = storage
        return ptr
    def stream(out, flags):
        out._obj.value = 1
        return 0
    def copy(dst, src, size, kind):
        C.memmove(dst, src, size)
        return 0
    sdk = SimpleNamespace(allocate=allocate, free=lambda p: allocations.pop(p),
        fill=lambda p, b, size: C.memset(p, b, size), download=C.string_at, synchronize=lambda s: None,
        lib=SimpleNamespace(hggcStreamCreateWithFlags=stream, hggcStreamDestroy=lambda s: 0, hggcMemcpy=copy))
    def canonical(q, out):
        C.memmove(out, C.byref(arrangement(q)), C.sizeof(Arrangement))
        return 0
    sizes_fn = library.quactlize_ppu_kpack_sizes_for_arrangement_v1
    # The same C struct has different Python class names in the two harnesses.
    old_args = sizes_fn.argtypes
    sizes_fn.argtypes = [C.c_int] * 4 + [C.c_void_p, C.c_void_p]
    def single(raw, low, high, units, n, k, experts, q, arr, stream):
        if raw == low:
            return 30
        if C.string_at(arr, C.sizeof(Arrangement)) != bytes(arrangement(q)):
            return 38
        return library.host_pack(q, raw, low, high, units, n, k, experts)
    def paired(gate, up, low, high, units, n, k, experts, q, arr, stream):
        from reference.gguf_kpack import SPECS
        width = k // 32 * 34 if q == 8 else k // 256 * SPECS[q].raw_bytes
        shape = (experts, n, width)
        g = np.frombuffer(C.string_at(gate, experts * n * width), 'u1').reshape(shape)
        u = np.frombuffer(C.string_at(up, experts * n * width), 'u1').reshape(shape)
        raw = np.concatenate((g, u), axis=1)
        return single(raw.ctypes.data, low, high, units, n * 2, k, experts, q, arr, stream)
    public = SimpleNamespace(canonical=canonical, sizes=sizes_fn, pack=single, pack_pair=paired)
    try:
        with Arena(sdk) as arena:
            assert packed(arena, public, q, experts=3, pair=pair, bit_patterns=True)[3] is None
        assert not allocations, 'harness leaked its owned buffers'
    finally:
        sizes_fn.argtypes = old_args
