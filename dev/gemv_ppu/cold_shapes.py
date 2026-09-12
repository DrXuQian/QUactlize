"""Extend the admitted Q4 C8 body to the six existing M1 comparison shapes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.cold_geometry import source as single_shape_source
from dev.gemv_ppu.h800_port import verify as verify_controls
from dev.gemv_ppu.run import SHAPES
from quactlize.runtime.compiler import FLAGS, LIBRARIES

SCHEMA = "quactlize.q4-cold-shapes-ppu.v1"
PAYLOAD = "libq4_ppu_cold_shapes.so"
RECIPE = (8, 8, 1)


def geometry(n, k):
    if (n, k) not in SHAPES:
        raise ValueError("shape outside the six-shape Q4 comparison")
    groups, workers = k // 32, 32
    return dict(columns=8, warps=8, p=4, tile_n=32, grid=n//32, threads=256,
                k_workers=workers, k_passes=(groups+workers-1)//workers,
                last_pass_workers=(groups-1)%workers+1)


def source():
    original = single_shape_source()
    body = original[:original.index('extern "C" int q4_cold_c8_run(')]
    body += '''extern "C" int q4_cold_shapes_run(int n,int k,void const* a,void const* low,
        void const* units,void* out,void* stream) {
    if(!a || !low || !units || !out || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(out)&3)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for n, k in SHAPES:
        body += f'''    if(n=={n} && k=={k}) {{
        q4_group_affine<8,8,4,{n},{k},true><<<n/32,256,0,static_cast<hggcStream_t>(stream)>>>(
            a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return body + "    return QKG_SHAPE;\n}\n"


def verify(candidate, controls, baseline, *, sources=True):
    control = verify_controls(controls, baseline, sources=sources)
    m = json.loads((candidate / "manifest.json").read_text())
    if (m.get("schema") != SCHEMA or m.get("shapes") != [list(s) for s in SHAPES]
            or m.get("recipe") != list(RECIPE)
            or m.get("control_manifest_sha256") != digest(controls / "manifest.json")
            or m.get("compiler_sha256") != control["compiler_sha256"]
            or m.get("payload", {}).get("file") != PAYLOAD):
        raise ValueError("cold-shapes package/geometry/control identity differs")
    path = candidate / PAYLOAD
    if digest(path) != m["payload"]["sha256"]:
        raise ValueError("cold-shapes payload missing, changed, or an LFS pointer")
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("cold-shapes payload is not ELF")
    for name, expected in m["source_hashes"].items() if sources else []:
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("cold-shapes source differs: " + name)
    return m


def build(sdk, output, controls, baseline):
    sdk, controls, baseline = (p.resolve(strict=True) for p in (sdk, controls, baseline))
    control = verify_controls(controls, baseline)
    if digest(sdk / "bin/hgcc") != control["compiler_sha256"]:
        raise ValueError("use the existing control compiler")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = dict(control["source_hashes"])
    for name in ("dev/gemv_ppu/cold_geometry.py", "dev/gemv_ppu/cold_shapes.py"):
        hashes[name] = digest(ROOT / name)
    (output / "c8.cu").write_text(source())
    env = dict(os.environ)
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    includes = [output, ROOT / "dev/gemv_ppu", ROOT / "quactlize/execution", ROOT / "quactlize/include",
                ROOT / "benchmarks", ROOT / "third_party/actlize/include", ROOT / "third_party/actlize/tools/util/include"]
    commands = []
    def run(label, command):
        command = list(map(str, command))
        commands.append(dict(label=label, argv=command))
        with (output / f"{label}.log").open("w") as stream:
            rc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc:
            raise ValueError(f"{label}: rc={rc}; log={output / (label+'.log')}")
        print(f"Q4_COLD_SHAPES_BUILD phase={label} status=PASS", flush=True)
    started = time.monotonic()
    def compile_one(pair):
        name, src = pair
        run(name, [sdk / "bin/hgcc", *FLAGS, "-DQKG_QTYPE=12", *[f"-I{p}" for p in includes],
                   "-c", src, "-o", output / f"{name}.o"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compile_one, [("c8", output / "c8.cu"), ("probe", ROOT / "dev/gemv_ppu/probe.cu")]))
    run("link", ["g++", "-shared", "-Wl,-Bsymbolic", output / "c8.o", output / "probe.o",
                 "-o", output / PAYLOAD, f"-L{sdk / 'lib'}", *[f"-l{name}" for name in LIBRARIES]])
    run("isa", [sdk / "bin/hgobjdump", "--dump-isa", output / PAYLOAD])
    isa = (output / "isa.log").read_text()
    if "q4_ppu_marker" not in isa or any(
            f"q4_group_affineILi8ELi8ELi4ELi{n}ELi{k}ELb1E" not in isa for n, k in SHAPES):
        raise ValueError("native marker or one of the six exact C8 specializations is missing")
    if any(digest(ROOT / name) != expected for name, expected in hashes.items()):
        raise ValueError("source changed during the build")
    manifest = dict(schema=SCHEMA, shapes=SHAPES, recipe=RECIPE, production_changed=False,
        device_validated=False, control_manifest_sha256=digest(controls / "manifest.json"),
        compiler_sha256=digest(sdk / "bin/hgcc"), inspector_sha256=digest(sdk / "bin/hgobjdump"),
        runtime={f"lib{name}.so": digest(sdk / "lib" / f"lib{name}.so") for name in LIBRARIES},
        source_hashes=hashes, generated_sha256=digest(output / "c8.cu"),
        payload=dict(file=PAYLOAD, sha256=digest(output / PAYLOAD), isa_sha256=digest(output / "isa.log")),
        commands=sorted(commands, key=lambda c: c["label"]), seconds=time.monotonic()-started)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    verify(output, controls, baseline)
    print(f"Q4_COLD_SHAPES_BUILD status=COMPILED device_validated=0 seconds={manifest['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--controls", type=Path, default=ROOT / "prebuilt/ppu0010/q4-h800-port-v1")
    p.add_argument("--baseline", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    a = p.parse_args()
    build(a.sdk, a.output, a.controls, a.baseline)
