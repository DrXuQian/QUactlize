#!/usr/bin/env python3
"""Product CI: --suite test checks an existing package (host + PPU accuracy).

Build packages with build.sh --library. --suite all retains host/build/accuracy.
No downloads, submodule updates, sweeps, model runs, or performance claims.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ci.suites import inventory

SUITE_PHASES = {
    "host": ("host",),
    "test": ("host", "accuracy"),
    "build": ("build",),
    "accuracy": ("accuracy",),
    "all": ("host", "build", "accuracy"),
    "tooling": ("tooling",),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def pytest_counts(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = dict(tests=len(cases), passed=0, failures=0, errors=0, skipped=0)
    for case in cases:
        kind = next((k for k in ("failure", "error", "skipped") if case.find(k) is not None), None)
        counts[{"failure": "failures", "error": "errors", "skipped": "skipped", None: "passed"}[kind]] += 1
    if not cases:
        raise ValueError("no test cases reported; an empty suite is not a pass")
    return counts


def run_command(command, log, env, timeout):
    """Stream to a retained log; heartbeat and reap our child group on timeout."""
    start = time.monotonic()
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError(f"phase exceeded {timeout}s; see {log}")
                try:
                    return process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    print(f"QUACTLIZE_CI_WAIT seconds={time.monotonic()-start:.0f} log={log}", flush=True)
        except BaseException:
            # Only the process group created by this call, never a name/PID search.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def preflight(phase, sdk, package):
    missing = []
    if phase in ("host", "tooling", "accuracy"):
        missing += [f"Python module {m}" for m in ("pytest", "numpy", "torch", "gguf")
                    if importlib.util.find_spec(m) is None]
    if phase in ("host", "tooling", "build"):
        missing += [name for name in ("git", "gcc", "g++") if shutil.which(name) is None]
        if not (sdk / "include/hggc_runtime.h").is_file():
            missing.append(f"SDK headers: {sdk}/include/hggc_runtime.h (set PPU_SDK/--sdk)")
        if not (ROOT / "third_party/actlize/include/cutlass/cutlass.h").is_file():
            missing.append("pinned third_party/actlize checkout; CI must checkout it using its own credentials")
    if phase == "accuracy" and not (package / "manifest.json").is_file():
        missing.append(f"built public package: {package}/manifest.json (run build.sh --library first)")
    if missing:
        raise ValueError("missing prerequisites: " + "; ".join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITE_PHASES), default="host",
                        help="test: host + accuracy, no package rebuild; all: also build; default: host")
    parser.add_argument("--sdk", type=Path, default=Path(os.environ.get("PPU_SDK") or "/usr/local/PPU_SDK"))
    parser.add_argument("--package", type=Path, help="fresh build destination or installed accuracy-test package")
    parser.add_argument("--output", type=Path, help="new directory for logs, JUnit and summary; default mkdtemp")
    parser.add_argument("--jobs", type=int, default=4, help="build parallelism; does not change test coverage")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per phase, including cold JIT")
    parser.add_argument("--list", action="store_true", help="print suites/files/requirements without building or importing tests")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    if sys.version_info < (3, 10):
        parser.error("product CI requires Python >= 3.10")
    catalog = inventory()
    phases = SUITE_PHASES[args.suite]
    if args.list:
        print(json.dumps(dict(phases=phases, inventory=catalog,
            all_scope="PRODUCT_HOST_BUILD_ACCURACY; excludes tooling and separate experiments"), indent=2))
        return 0
    if args.suite in ("test", "accuracy") and args.package is None:
        parser.error(f"--suite {args.suite} requires --package; no implicit search for an old library")
    if args.output is None:
        args.output = Path(tempfile.mkdtemp(prefix="quactlize-ci-"))
    else:
        args.output = args.output.absolute()
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"--output must be new: {args.output}")
    package = args.package.absolute() if args.package else args.output / "package"
    sdk = args.sdk.resolve()
    env = dict(os.environ, PPU_SDK=str(sdk), QUACTLIZE_CI_PACKAGE=str(package), PYTHON=sys.executable,
               QUACTLIZE_PPU_PYTHON=sys.executable, PYTHONUNBUFFERED="1",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    # CPU golden generation must not initialise Torch's GPU backend. Prevent
    # BLAS thread oversubscription; preserve the caller's device visibility.
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", COLUMNS="120")
    env.setdefault("QUACTLIZE_PPU_CACHE_DIR", str(args.output / "jit-cache"))
    # Ambient pytest options/plugins may silently filter tests or change exits.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    summary = dict(schema="quactlize.ci.v1", suite=args.suite, status="INCOMPLETE",
                   phases=[], sdk=str(sdk), package=str(package),
                   scope="PRODUCT_CI_NOT_PERFORMANCE_OR_EXHAUSTIVE_CONFIG_SWEEP",
                   performance_admitted=False, inventory=catalog)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "inventory.json", catalog)
    for phase in phases:
        row = dict(phase=phase, status="FAIL")
        summary["phases"].append(row)
        started = time.monotonic()
        log = args.output / f"{phase}.log"
        xml = args.output / f"{phase}.xml"
        print(f"QUACTLIZE_CI_START phase={phase} log={log}", flush=True)
        try:
            preflight(phase, sdk, package)
            if phase == "build":
                if package.exists():
                    raise ValueError(f"build package must be new (will not overwrite): {package}")
                command = ["bash", str(ROOT / "build.sh"), "--library", "--sdk", str(sdk),
                           "--output", str(package), "--jobs", str(args.jobs)]
            else:
                command = [sys.executable, "-m", "pytest", "-q", "-ra", "--tb=short",
                           "--strict-markers", "-o", "addopts=", "-o", "xfail_strict=true", "-p", "no:cacheprovider",
                           "--junitxml=" + str(xml),
                           *(["-s"] if phase == "accuracy" else []),
                           *[str(ROOT / "tests" / name) for name in catalog[phase]]]
            row["command"] = command
            write_json(args.output / f"{phase}.command.json", command)
            row["returncode"] = run_command(command, log, dict(env, QUACTLIZE_CI_SUITE=phase), args.timeout)
            if xml.is_file():
                row["counts"] = pytest_counts(xml)
            if row["returncode"] != 0:
                raise ValueError(f"{phase} exited {row['returncode']}; see {log}")
            if phase != "build":
                counts = row.get("counts")
                if not counts or counts["failures"] or counts["errors"] or counts["skipped"]:
                    raise ValueError(f"incomplete pytest coverage: {counts}; skips/xfails are not passes")
            row["status"] = "PASS"
        except Exception as exc:
            row["error"] = str(exc)
            print(f"QUACTLIZE_CI_FAIL phase={phase}: {exc}", flush=True)
        finally:
            row["seconds"] = time.monotonic() - started
            if not xml.is_file():
                suite = ET.Element("testsuite", name=phase, tests="1", failures=str(int(row["status"] != "PASS")))
                case = ET.SubElement(suite, "testcase", name=phase, time=str(row["seconds"]))
                if row["status"] != "PASS":
                    ET.SubElement(case, "failure", message=row.get("error", "interrupted"))
                ET.ElementTree(suite).write(xml, encoding="utf-8", xml_declaration=True)
            write_json(args.output / "summary.json", summary)
        print(f"QUACTLIZE_CI_PART phase={phase} status={row['status']}", flush=True)
        if row["status"] != "PASS":
            break
    summary["status"] = "PASS" if len(summary["phases"]) == len(phases) and all(
        r["status"] == "PASS" for r in summary["phases"]) else "FAIL"
    write_json(args.output / "summary.json", summary)
    print(f"QUACTLIZE_CI_DONE status={summary['status']} results={args.output}", flush=True)
    return int(summary["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
