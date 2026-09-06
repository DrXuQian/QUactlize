#!/usr/bin/env python3
"""Cached budgeted profiler: small kernel DSOs and persistent drivers.

Uses the existing generated unit macros and benchmark fixtures, not a second
kernel implementation. Each module has <=32 parents. Source/SDK/flags are
part of cache identity. No GPU or PyTorch is needed to build.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import threading
import time
from kpack_tuning_plan import ROOT, SCHEMA, Candidate, digest, candidates

# Driver, registry, unit, types namespace, functions namespace, macro, defines.
SPEC = {
    "fq-dense": (
        "test_fully_quantized_internal_sweep.cu",
        "fq_tc_registry.inc",
        "fully_quantized_splitk_producer_unit.inc",
        "fq_internal_sweep",
        "fq_internal_sweep_generated",
        "FQ_TC",
        "FQ_SWEEP",
    ),
    "sf-dense": (
        "test_scalefirst_internal_sweep.cu",
        "scalefirst_registry.inc",
        "scalefirst_internal_sweep_unit.inc",
        "scalefirst_internal_sweep",
        "scalefirst_internal_sweep_generated",
        "SCALEFIRST",
        "SCALEFIRST_SWEEP",
    ),
    "fq-grouped": (
        "test_fully_quantized_grouped_kpack_discovery.cu",
        "fq_grouped_kpack_registry.inc",
        "fully_quantized_grouped_kpack_discovery_unit.inc",
        "fully_quantized_grouped_kpack",
        "fully_quantized_grouped_kpack_generated",
        "FQ_GROUPED_KPACK",
        "FQ_GROUPED_KPACK",
    ),
    "sf-grouped": (
        "test_scalefirst_grouped_kpack_discovery.cu",
        "scalefirst_grouped_registry.inc",
        "scalefirst_grouped_kpack_discovery_unit.inc",
        "scalefirst_grouped_kpack",
        "scalefirst_grouped_kpack_generated",
        "SCALEFIRST_GROUPED",
        "SCALEFIRST_GROUPED_KPACK",
    ),
}
FLAGS = [
    "--forward-unknown-to-host-compiler",
    "--forward-unknown-to-host-linker",
    "-arch=ppu_10",
    "-x",
    "hg",
    "-DSWITCH_TO_HGGCRT",
    "-std=c++17",
    "-O3",
    "-Xcompiler",
    "-ftemplate-depth=8192",
    "-Xllvm",
    "-wno-loop-miss-transform",
    "-Xllvm",
    "-ppu-simt-branch=false",
    "-Xllvm",
    "-ppu-patch-fence-ppu=false",
    "-Xllvm",
    "-ppu-cg-to-kp1=true",
    "-Xllvm",
    "-ppu-fix-uninit=true",
    "--expt-relaxed-constexpr",
    "-DUSE_CLANG",
    "-DCUTLASS_USE_PACKED_TUPLE=1",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-fPIC",
    "-DKPACK_TUNER_E2E=1",
]
LIBRARIES = ("libhggc_wrapper.so", "libhggcrt1.so", "libhggc.so", "libhg_wrapper.so")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_identity() -> str:
    paths = set()
    for directory in (
        "quactlize/include",
        "quactlize/csrc/device",
        "benchmarks",
        "third_party/actlize/include",
        "third_party/actlize/tools/util/include",
        "third_party/actlize/examples/common",
    ):
        paths.update(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".cu", ".inc")
        )
    paths.add(Path(__file__).resolve())
    return digest([(str(p.relative_to(ROOT)), sha(p)) for p in sorted(paths)])


def sdk_identity(sdk: Path) -> dict:
    paths = [sdk / "bin/hgcc", sdk / "bin/hgobjdump"] + [
        sdk / "lib" / n for n in LIBRARIES
    ]
    if any(not p.is_file() for p in paths):
        raise ValueError(
            "SDK must contain bin/hgcc, bin/hgobjdump and lib/hggc runtimes"
        )
    return {str(p.relative_to(sdk)): sha(p) for p in paths}


def emit_rows(rows: list[Candidate], macro: str) -> str:
    if not rows:
        return f"#define {macro}(X)\n"
    result = [f"#define {macro}(X) " + chr(92)]
    for i, c in enumerate(rows):
        args = [
            c.symbol,
            c.qtype,
            0 if c.route.endswith("dense") else (1 if c.qtype == 12 else 2),
            c.tm,
            c.tn,
            c.tk,
            c.wm,
            c.wn,
            c.stages,
        ]
        if c.route.endswith("dense"):
            args += [0, c.ap, c.dn]
        else:
            args += [c.dn]
            if c.route == "fq-grouped":
                args += [c.persistent]
        result.append(
            "  X("
            + ",".join(map(str, args))
            + ")"
            + (" " + chr(92) if i + 1 < len(rows) else "")
        )
    return "\n".join(result) + "\n"


def registry_preamble(q: int, route: str) -> str:
    prefix = SPEC[route][5]
    values = {
        "QTYPE": q,
        "WEIGHT_LAYOUT": 1 if q == 12 else 2,
        "ARTIFACT_TK": 0,
        "BCHUNK": 0,
        "TYPED_ROWS": 0,
        "TYPE_ROWS": 0,
    }
    return (
        "".join(f"#define {prefix}_GENERATED_{k} {v}\n" for k, v in values.items())
        + f"#define {prefix}_REGISTRY_ROWS(X)\n"
    )


def module_source(rows: list[Candidate], contract: str) -> str:
    c = rows[0]
    _, _, unit, ns, generated, prefix, _ = SPEC[c.route]
    if c.route.endswith("dense"):
        args = "FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN"
        fields = "#FN,Q,A,TM,TN,TK,WM,WN,ST,BC,AP,DN"
    else:
        args = "FN,Q,L,TM,TN,TK,WM,WN,ST,DN"
        fields = "#FN,Q,L,TM,TN,TK,WM,WN,ST,DN"
        if c.route == "fq-grouped":
            args += ",P"
            fields += ",(P!=0)"
    return (
        emit_rows(rows, f"{prefix}_UNIT_ROWS")
        + f'#include "{unit}"\n#include "kpack_tuner_registry.hpp"\n'
        + f"#define REGISTER({args}) {{{fields},&{generated}::FN}},\n"
        + f"static {ns}::RegistryRow const rows[] = {{ {prefix}_UNIT_ROWS(REGISTER) }};\n"
        + '#undef REGISTER\nextern "C" kpack_tuner::Module const* kpack_tuner_module_v1() {\n'
        + f'  static kpack_tuner::Module const value{{1,sizeof(rows[0]),sizeof(rows)/sizeof(rows[0]),"{contract}",rows}};\n'
        + "  return &value;\n}\n"
    )


def definitions(q: int, route: str) -> list[str]:
    prefix = SPEC[route][6]
    layout = 1 if q == 12 else 2
    values = {
        f"{prefix}_QTYPE": q,
        f"{prefix}_ARTIFACT_TK": 0,
        f"{prefix}_BCHUNK": 0,
        f"{prefix}_WEIGHT_LAYOUT": layout,
        "PPU_PACKED_SCALE": int(route.startswith("fq")),
        "PPU_PACKED_FORMAT": {10: 2, 11: 3, 12: 0, 13: 1, 14: 4}[q],
    }
    if route == "fq-dense":
        values["FQ_TC_WEIGHT_LAYOUT"] = layout
    return [f"-D{k}={v}" for k, v in values.items()]


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _build_unlocked(
    plan: dict, output: Path, sdk: Path, jobs: int, per_module: int
) -> dict:
    if plan.get("schema") != SCHEMA or not 1 <= per_module <= 32 or jobs < 1:
        raise ValueError("invalid plan/build controls")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "source": source_identity(),
        "sdk": sdk_identity(sdk),
        "flags": FLAGS,
        "host_compiler": subprocess.check_output(
            ["c++", "--version"], text=True
        ).splitlines()[0],
    }
    previous_bundle = output / "bundle.json"
    if (
        previous_bundle.exists()
        and json.loads(previous_bundle.read_text())["identity"] != identity
    ):
        raise ValueError(
            "source/SDK changed; use a new build output and preserve the previous bundle"
        )
    cache = output / "cache" / digest(identity)[:20]
    cache.mkdir(parents=True, exist_ok=True)
    include = [
        ROOT / p
        for p in (
            "third_party/actlize/include",
            "third_party/actlize/tools/util/include",
            "third_party/actlize/examples/common",
            "quactlize/include",
            "quactlize/include/gemv_lowbit",
            "quactlize/csrc/device",
            "quactlize/csrc",
            "benchmarks",
        )
    ]
    include += [sdk / "include", sdk / "targets/x86_64-linux/include"]
    base = [str(sdk / "bin/hgcc"), *FLAGS, *[f"-I{p}" for p in include]]
    link = [
        "c++",
        "-Wl,--enable-new-dtags",
        "-Wl,--allow-shlib-undefined",
        f"-Wl,-rpath,{sdk/'lib'}",
    ]
    link_tail = ["-ldl", *[str(sdk / "lib" / n) for n in LIBRARIES]]
    groups = {}
    for row in plan["candidates"]:
        c = Candidate(**row)
        groups.setdefault((c.qtype, c.route), []).append(c)
    for pair, rows in groups.items():
        authority = set(candidates(*pair))
        if len(set(rows)) != len(rows) or not set(rows) <= authority:
            raise ValueError("planned candidate differs from generator authority")
    tasks = [
        (
            "layout",
            base
            + [
                "-c",
                str(ROOT / "quactlize/csrc/device/ppu_dense_layout.cu"),
                "-o",
                str(cache / "layout.o"),
            ],
            cache / "layout.o",
            None,
        )
    ]
    pairs = {}
    for (q, route), rows in sorted(groups.items()):
        pair = f"q{q}-{route}"
        folder = cache / pair
        folder.mkdir(exist_ok=True)
        contract = digest([identity, q, route])
        driver, reg, *_ = SPEC[route]
        write_if_changed(folder / reg, registry_preamble(q, route))
        write_if_changed(
            folder / "driver.cu",
            "#define main kpack_benchmark_main\n"
            + f'#include "{driver}"\n#undef main\n#include "kpack_tuner_driver.inc"\n',
        )
        flags = [*definitions(q, route), f"-I{folder}"]
        obj, binary = folder / "driver.o", folder / "driver"
        tasks.append(
            (
                pair + "-driver",
                base
                + flags
                + [
                    f'-DKPACK_TUNER_CONTRACT="{contract}"',
                    "-c",
                    str(folder / "driver.cu"),
                    "-o",
                    str(obj),
                ],
                obj,
                (
                    link
                    + [str(obj), str(cache / "layout.o"), "-o", str(binary)]
                    + link_tail,
                    binary,
                ),
            )
        )
        modules = []
        for begin in range(0, len(rows), per_module):
            batch = rows[begin : begin + per_module]
            key = digest([c.symbol for c in batch])[:20]
            source, obj, binary = (
                folder / f"{key}.cu",
                folder / f"{key}.o",
                folder / f"{key}.so",
            )
            write_if_changed(source, module_source(batch, contract))
            tasks.append(
                (
                    pair + "-" + key,
                    base + flags + ["-c", str(source), "-o", str(obj)],
                    obj,
                    (
                        link
                        + ["-shared", "-Wl,-Bsymbolic", str(obj), "-o", str(binary)]
                        + link_tail,
                        binary,
                    ),
                )
            )
            modules.append({"path": str(binary), "symbols": [c.symbol for c in batch]})
        pairs[pair] = {
            "qtype": q,
            "route": route,
            "driver": str(folder / "driver"),
            "modules": modules,
        }
    stopped = threading.Event()
    processes = set()
    processes_lock = threading.RLock()

    def compile_task(task):
        label, cmd, obj, _ = task
        if stopped.is_set():
            return label, False
        receipt = obj.with_suffix(".receipt.json")
        if (
            receipt.exists()
            and obj.exists()
            and json.loads(receipt.read_text()).get("sha256") == sha(obj)
        ):
            return label, True
        start = time.monotonic()
        with obj.with_suffix(".build.log").open("w") as log:
            log.write("ARGV " + json.dumps(cmd) + "\n")
            log.flush()
            with processes_lock:
                if stopped.is_set():
                    return label, False
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                processes.add(proc)
            try:
                proc.wait()
            finally:
                with processes_lock:
                    processes.discard(proc)
        if proc.returncode:
            return label, False
        receipt.write_text(
            json.dumps({"sha256": sha(obj), "seconds": time.monotonic() - start}) + "\n"
        )
        return label, True

    failures = []
    handlers = {}
    if threading.current_thread() is threading.main_thread():

        def interrupt(signum, _frame):
            stopped.set()
            with processes_lock:
                for proc in processes:
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            print(
                f"KPACK_TUNER_BUILD_INTERRUPTED signal={signum} completed_objects_preserved=1",
                flush=True,
            )

        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupt)
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = {pool.submit(compile_task, t) for t in tasks}
            count = 0
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    with processes_lock:
                        alive = len(processes)
                    print(
                        f"KPACK_TUNER_BUILD_PROGRESS completed={count}/{len(tasks)} active_compilers={alive} jobs={jobs}",
                        flush=True,
                    )
                for future in done:
                    count += 1
                    label, ok = future.result()
                    if not ok:
                        failures.append(label)
                    if not stopped.is_set():
                        print(
                            f"KPACK_TUNER_BUILD completed={count}/{len(tasks)} status={'PASS' if ok else 'FAIL'} unit={label}",
                            flush=True,
                        )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if stopped.is_set():
        raise ValueError("build interrupted; successful objects remain cached")
    if failures:
        raise ValueError(
            f"{len(failures)} compile units failed; successful objects cached; logs={cache}; first={failures[0]}"
        )
    for label, _, obj, task in tasks:
        if not task:
            continue
        cmd, binary = task
        receipt = binary.with_suffix(binary.suffix + ".receipt.json")
        deps = [sha(obj)] + (
            [sha(cache / "layout.o")] if label.endswith("driver") else []
        )
        if receipt.exists() and binary.exists():
            old = json.loads(receipt.read_text())
            if old.get("deps") == deps and old.get("sha256") == sha(binary):
                continue
        with binary.with_suffix(binary.suffix + ".link.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt.write_text(json.dumps({"deps": deps, "sha256": sha(binary)}) + "\n")
    paths = [Path(v["driver"]) for v in pairs.values()] + [
        Path(m["path"]) for v in pairs.values() for m in v["modules"]
    ]
    if source_identity() != identity["source"]:
        raise ValueError("source changed during compilation; no bundle published")
    bundle = {
        "schema": "quactlize.kpack-tuner-build.v1",
        "identity": identity,
        "plan_sha256": digest(plan),
        "pairs": pairs,
        "payloads": {str(p): sha(p) for p in paths},
    }
    (output / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"KPACK_TUNER_BUILD_DONE parents={len(plan['candidates'])} payloads={len(paths)} bundle={output/'bundle.json'}",
        flush=True,
    )
    return bundle


def build(plan: dict, output: Path, sdk: Path, jobs: int, per_module: int = 8) -> dict:
    if output.is_symlink():
        raise ValueError("build output may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build_unlocked(plan, output, sdk, jobs, per_module)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--parents-per-module", type=int, default=8)
    a = p.parse_args()
    build(
        json.loads(a.plan.read_text()),
        a.output.resolve(),
        a.sdk.resolve(),
        a.jobs,
        a.parents_per_module,
    )


if __name__ == "__main__":
    main()
