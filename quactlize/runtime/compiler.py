"""Content-addressed, compile-only PPU parent cache. No device is needed.

Only declared parent tuples are generated. There is no shell interpolation,
unchecked source-code field, runtime eval or compilation on an inference miss.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .tuning import ROUTES, digest

ROOT = Path(__file__).resolve().parents[2]
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
]
LIBRARIES = ("hggc_wrapper", "hggcrt1", "hggc", "hg_wrapper")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_parent(parent):
    if (
        parent.get("route") not in ROUTES
        or parent.get("qtype") not in range(10, 15)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", parent.get("symbol", ""))
    ):
        raise ValueError("invalid parent identity")
    for field in ("tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn", "persistent"):
        if type(parent.get(field)) is not int:
            raise ValueError("parent fields must be integers")
    if (
        parent["tm"] not in (8, 16, 32, 64, 128, 256)
        or parent["tn"] not in (16, 32, 64, 128, 256)
        or parent["tk"] not in (64, 128, 256)
        or parent["wm"] not in (8, 16, 32, 64, 128)
        or parent["wn"] not in (16, 32, 64, 128)
        or parent["stages"] not in (2, 3, 4, 6, 8, 12)
        or parent["dn"] not in (16, 32, 64)
        or parent["ap"] not in (0, 1)
        or parent["persistent"] not in (-1, 0, 1)
        or parent["tm"] % parent["wm"]
        or parent["tn"] % parent["wn"]
        or parent["tn"] % parent["dn"]
    ):
        raise ValueError("invalid parent geometry")
    if parent["ap"] and (
        parent["qtype"] not in (10, 12)
        or parent["tm"] != 8
        or parent["wm"] != 8
        or parent["route"].endswith("grouped")
    ):
        raise ValueError("invalid packed-A parent")
    if (parent["route"] == "fq-grouped") != (parent["persistent"] in (0, 1)):
        raise ValueError("persistent parent axis differs from route")
    minimum_k = {10: 128, 11: 256, 12: 64, 13: 256, 14: 128}[parent["qtype"]]
    if parent["tk"] % minimum_k:
        raise ValueError("parent does not contain whole K-pack transport tiles")


class Compiler:
    def __init__(self, sdk, cache, jobs=1):
        self.sdk = Path(sdk).resolve()
        self.cache = Path(cache).resolve()
        if jobs < 1:
            raise ValueError("jobs must be positive")
        self.jobs = jobs
        sdk_files = [self.sdk / "bin/hgcc", self.sdk / "bin/hgobjdump"] + [
            self.sdk / "lib" / f"lib{name}.so" for name in LIBRARIES
        ]
        self.sdk_identity = digest(
            [(str(p.relative_to(self.sdk)), sha(p)) for p in sdk_files]
        )
        self.includes = [
            ROOT / "quactlize/include",
            ROOT / "quactlize/runtime",
            ROOT / "third_party/actlize/include",
            ROOT / "third_party/actlize/tools/util/include",
            ROOT / "third_party/actlize/examples/common",
        ]
        paths = sorted(
            {
                p
                for directory in self.includes
                for p in directory.rglob("*")
                if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".inc")
            }
        )
        self.kernel_identity = digest(
            [(str(p.relative_to(ROOT)), sha(p)) for p in paths]
        )
        self.input_stats = {
            p: (p.stat().st_mtime_ns, p.stat().st_size)
            for p in paths + sdk_files + [Path(__file__)]
        }
        self.identity = dict(
            sdk=self.sdk_identity,
            kernel=self.kernel_identity,
            flags=FLAGS,
            generator=sha(__file__),
            host=subprocess.check_output(["g++", "--version"], text=True),
        )

    def source(self, parent, key):
        validate_parent(parent)
        fields = {
            "QTYPE": parent["qtype"],
            "ROUTE": ROUTES.index(parent["route"]),
            **{
                f.upper(): parent[f]
                for f in ("tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn")
            },
            "PERSISTENT_PARENT": parent["persistent"],
        }
        return (
            "".join(f"#define QK_{k} {v}\n" for k, v in fields.items())
            + f'#define QK_PARENT "{parent["symbol"]}"\n#define QK_BUILD_KEY "{key}"\n'
            + '#include "module.cuh"\n'
        )

    def build(self, parent):
        validate_parent(parent)
        self.check_inputs()
        key = digest(
            dict(identity=self.identity, parent=parent, source=self.source(parent, ""))
        )
        path = self.cache / key
        path.mkdir(parents=True, exist_ok=True)
        with (path / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = path / "manifest.json"
            if receipt.exists():
                value = json.loads(receipt.read_text())
                if (
                    value.get("key") != key
                    or value.get("parent") != parent
                    or value.get("identity") != self.identity
                    or value.get("sha256") != sha(path / "kernel.so")
                ):
                    raise ValueError("compiled cache identity/payload differs")
                return value | {"path": str(path / "kernel.so"), "cache_hit": True}
            with tempfile.TemporaryDirectory(prefix="build-", dir=path) as tmp:
                stage = Path(tmp)
                source = stage / "kernel.cu"
                source.write_text(self.source(parent, key))
                obj, so = stage / "kernel.o", stage / "kernel.so"
                q = parent["qtype"]
                defines = [
                    f'-DPPU_PACKED_SCALE={int(parent["route"].startswith("fq"))}',
                    f"-DPPU_PACKED_FORMAT={ {10:2,11:3,12:0,13:1,14:4}[q] }",
                ]
                commands = [
                    [
                        str(self.sdk / "bin/hgcc"),
                        *FLAGS,
                        *defines,
                        *[f"-I{p}" for p in self.includes],
                        "-c",
                        str(source),
                        "-o",
                        str(obj),
                    ],
                    [
                        "g++",
                        "-shared",
                        "-Wl,-Bsymbolic",
                        str(obj),
                        "-o",
                        str(so),
                        f"-L{self.sdk / 'lib'}",
                        *[f"-l{name}" for name in LIBRARIES],
                        f"-Wl,-rpath,{self.sdk / 'lib'}",
                    ],
                ]
                with (path / "build.log").open("w") as log:
                    for command in commands:
                        log.write(json.dumps(command) + "\n")
                        log.flush()
                        status = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ),
                        ).returncode
                        if status:
                            raise RuntimeError(
                                f"parent compile/link failed rc={status}; log={path / 'build.log'}"
                            )
                self.check_inputs()
                value = dict(
                    key=key, parent=parent, identity=self.identity, sha256=sha(so)
                )
                os.replace(so, path / "kernel.so")
                temporary = stage / "manifest.json"
                temporary.write_text(json.dumps(value, sort_keys=True))
                os.replace(temporary, receipt)
            return value | {"path": str(path / "kernel.so"), "cache_hit": False}

    def check_inputs(self):
        for path, expected in self.input_stats.items():
            stat = path.stat()
            if (stat.st_mtime_ns, stat.st_size) != expected:
                raise RuntimeError(
                    f"source/SDK changed during compilation: {path}; restart compile-only"
                )

    def compile_only(self, parents, progress=None):
        unique = {p["symbol"]: p for p in parents}
        if any(unique[p["symbol"]] != p for p in parents):
            raise ValueError("same parent name has different geometry")
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            jobs = {pool.submit(self.build, p): p["symbol"] for p in unique.values()}
            records = {}
            for job in as_completed(jobs):
                records[jobs[job]] = job.result()
                if progress:
                    progress(len(records), len(jobs))
            return [records[name] for name in unique]
