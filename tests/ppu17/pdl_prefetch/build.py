"""Build the explicit first-B-tile experiment against an immutable PDL library."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "quactlize/backends/ppu17"
sys.path.insert(0, str(BACKEND))
from build import sha  # backend builder; direct execution below uses __main__
from dependency import require_cutlass


def inspect_prefetch(text, sass):
    native = {c.splitlines()[0].strip(): c for c in re.split(r"Function\s*:\s*", sass)[1:]}
    results = []
    # Anonymous-namespace helper kernels use bare .entry, not .visible.
    # They must terminate the preceding body's scope too.
    for chunk in re.split(r"(?m)^\s*(?:\.(?:visible|weak)\s+)?\.entry\s+", text)[1:]:
        symbol = chunk.split("(", 1)[0].strip()
        match = re.search(r"tma_wgmmaILi(\d+)E", symbol)
        if not match or "Lb1ELb1EEEv" not in symbol:
            continue
        q = int(match[1])
        if chunk.count("griddepcontrol.wait;") != 1:
            raise ValueError("one grid dependency wait required")
        prefix, suffix = chunk.split("griddepcontrol.wait;", 1)
        if "ld.global" in prefix or "st.global" in prefix or "ret;" in prefix:
            raise ValueError("dependent access/retirement precedes grid wait")
        registers, descriptors = {}, []
        for line in prefix.splitlines():
            move = re.search(r"mov\.[bu]64\s+(%rd\d+),\s*\S+_param_([0-3]);", line)
            if move:
                registers[move[1]] = int(move[2])
            cast = re.search(r"cvta\.param\.u64\s+(%rd\d+),\s*(%rd\d+);", line)
            if cast:
                registers[cast[1]] = registers.get(cast[2], -1)
            if "cp.async.bulk.tensor" in line:
                pointer = re.search(r",\s*\[(%rd\d+),", line)
                descriptors.append(registers.get(pointer[1], -1) if pointer else -1)
        expected = [2, 3] if q in (11, 13, 14) else [2]
        if descriptors != expected:
            raise ValueError(f"early TMA must reference only low/high descriptors: q={q} {descriptors}")
        if "mbarrier.try_wait" not in suffix or suffix.index("mbarrier.try_wait") > suffix.index("ret;"):
            raise ValueError("early transaction is not drained before CTA retirement")
        if not all(op in suffix for op in ("wgmma.mma_async", "wgmma.wait_group", "st.global", "griddepcontrol.launch_dependents")):
            raise ValueError("missing math/store/lifetime/release body")
        body = native.get(symbol, "")
        if not all(op in body for op in ("UTMALDG", "ACQBULK", "PREEXIT", "HGMMA", "STG")):
            raise ValueError("missing assembled prefetch/protocol/math/output")
        if body.index("UTMALDG") > body.index("ACQBULK"):
            raise ValueError("assembled TMA was not moved ahead of dependency wait")
        results.append(dict(q=q, symbol=symbol, early_descriptors=descriptors))
    if len(results) != 24 or any(sum(r["q"] == q for r in results) != 4 for q in (8,10,11,12,13,14)):
        raise ValueError("incomplete prefetch specialization inventory")
    return results


def inspect_sf(text, sass):
    native = {c.splitlines()[0].strip(): c for c in re.split(r"Function\s*:\s*", sass)[1:]}
    result = []
    for chunk in re.split(r"(?m)^\s*(?:\.(?:visible|weak)\s+)?\.entry\s+", text)[1:]:
        symbol = chunk.split("(", 1)[0].strip()
        if "expand_scales" not in symbol:
            continue
        late = "expand_scales_late" in symbol
        enabled = late or "Lb1E" in symbol
        wait, release = "griddepcontrol.wait;", "griddepcontrol.launch_dependents;"
        if chunk.count(wait) != int(enabled) or chunk.count(release) != int(enabled):
            raise ValueError("SF dependency count differs")
        if "ld.global" not in chunk or "st.global" not in chunk:
            raise ValueError("SF arithmetic/read/store body missing")
        if enabled and chunk.index(wait) > chunk.index("ld.global"):
            raise ValueError("SF unit read precedes wait")
        if late and chunk.index(release) < chunk.rindex("st.global"):
            raise ValueError("late SF release precedes a metadata store")
        if enabled and not late and chunk.index(release) > chunk.index("ld.global"):
            raise ValueError("early SF control changed")
        body = native.get(symbol, "")
        if "STG" not in body or ("ACQBULK" in body) != enabled or ("PREEXIT" in body) != enabled:
            raise ValueError("SF assembled body/protocol differs")
        if late and body.index("PREEXIT") < body.rindex("STG"):
            raise ValueError("assembled late SF release precedes metadata store")
        result.append(dict(symbol=symbol, trigger="late" if late else "early" if enabled else "ordinary"))
    if len(result) != 36 or any(sum(r["trigger"] == t for r in result) != 12 for t in ("late", "early", "ordinary")):
        raise ValueError("incomplete SF policy specialization inventory")
    return result


def unchanged_controls(sass, baseline_sass):
    def entries(source):
        rows = {}
        for chunk in re.split(r"Function\s*:\s*", source)[1:]:
            symbol = chunk.splitlines()[0].strip()
            if "expand_scales_late" in symbol or "Lb1ELb1EEEv" in symbol:
                continue
            if "tma_wgmma" not in symbol and "expand_scales" not in symbol:
                continue
            words = re.findall(r"/\* 0x([0-9a-f]{16}) \*/", chunk)
            if not words or symbol in rows: raise ValueError("missing/duplicate control encoding")
            rows[symbol] = hashlib.sha256("".join(words).encode()).hexdigest()
        return rows
    candidate, parent = entries(sass), entries(baseline_sass)
    if len(candidate) != 72 or set(candidate) != set(parent):
        raise ValueError("incomplete same-body SF/TMA controls")
    changed = [s for s, digest in candidate.items() if parent[s] != digest]
    if changed: raise ValueError("original SF/TMA control encodings changed: " + str(changed))
    return dict(bodies=len(candidate), status="PASS", scope="EXACT_ENCODED_ORDINARY_AND_PDL_CONTROLS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--cutlass", type=Path, required=True)
    p.add_argument("--compiler", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    baseline, out = a.baseline.resolve(), a.output.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("output must be outside source checkout")
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((baseline / "manifest.json").read_text())
    library = baseline / "libquactlize_ppu17.so"
    compiler = a.compiler.resolve(strict=True)
    if sha(library) != original["library_sha256"] or not original["pdl_compiled"]:
        raise ValueError("baseline library identity/PDL capability differs")
    if sha(compiler) != original["compiler_sha256"]:
        raise ValueError("use the same compiler as the admitted baseline")
    dependency = require_cutlass(a.cutlass)
    files = sorted(set(BACKEND.glob("*.cuh")) | set(BACKEND.glob("*.h")) |
                   set((ROOT / "tests/ppu17").glob("*.cuh")) |
                   set(Path(__file__).parent.glob("*.py")) |
                   set(Path(__file__).parent.glob("*.cu")) |
                   set(Path(__file__).parent.glob("*.cuh")) |
                   set(Path(__file__).parent.glob("*.hpp")) |
                   set(Path(__file__).parent.glob("*.cpp")) |
                   {ROOT / "tests/ppu17/pdl_chain.cu"})
    for relative, digest in original["sources"].items():
        if relative.startswith("quactlize/include/"):
            if sha(ROOT/relative) != digest:
                raise ValueError("offline format owner differs from baseline: " + relative)
            files.append(ROOT/relative)
    sources = {str(f.relative_to(ROOT)): sha(f) for f in files}
    flags = ["-std=c++17", "-O3", "--expt-relaxed-constexpr", "--extended-lambda",
             "-gencode=arch=compute_90a,code=sm_90a", "-DQ17_ENABLE_PDL=1",
             "-DCUTLASS_ENABLE_GDC_FOR_SM90=1", "-lineinfo", "-I"+str(BACKEND),
             "-I"+str(ROOT/"quactlize/include"), "-I"+str(a.cutlass.resolve()/"include")]
    def run(name, command):
        with (out / (name + ".log")).open("x") as log:
            log.write(json.dumps(command)+"\n"); log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("PDL_PREFETCH_BUILD PASS " + name, flush=True)
    links = ["-L"+str(baseline), "-lquactlize_ppu17", "-Xlinker=-rpath="+str(baseline)]
    run("resource-policy-build", ["c++", "-std=c++17", "-O2", str(Path(__file__).parent/"resource_policy_test.cpp"), "-o", str(out/"resource-policy-test")])
    run("resource-policy-host", [str(out/"resource-policy-test")])
    chain = [str(compiler), *flags, "-DQ17_TEST_WEIGHT_PREFETCH=1", str(ROOT/"tests/ppu17/pdl_chain.cu")]
    jobs = [("chain", [*chain, *links, "-o", str(out/"chain")]),
            ("ptx", [*chain, "--ptx", "-o", str(out/"chain.ptx")]),
            ("benchmark", [str(compiler), *flags, str(Path(__file__).parent/"benchmark.cu"), *links, "-o", str(out/"benchmark")])]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: run(*item), jobs))
    inspector = compiler.parent / "cuobjdump"
    sass = subprocess.check_output([str(inspector), "--dump-sass", str(out/"chain")], text=True)
    (out/"chain.sass").write_text(sass)
    text = (out/"chain.ptx").read_text()
    codegen = inspect_prefetch(text, sass)
    sf_codegen = inspect_sf(text, sass)
    baseline_sass = subprocess.check_output([str(inspector), "--dump-sass", str(library)], text=True)
    controls = unchanged_controls(sass, baseline_sass)
    # Per-body regressions: missing wait/release, wrong (A) descriptor,
    # missing producer drain, missing native launch hint.
    plants = [(text.replace("griddepcontrol.wait;", "absent_wait;"), sass),
              (text.replace("griddepcontrol.launch_dependents;", "absent_release;"), sass),
              (re.sub(r"(mov\.[bu]64\s+%rd\d+,\s*\S+)_param_2;", r"\1_param_1;", text), sass),
              (text.replace("mbarrier.try_wait", "absent_barrier_wait"), sass),
              (text, sass.replace("PREEXIT", "NOP"))]
    for changed, native in plants:
        try: inspect_prefetch(changed, native)
        except ValueError: pass
        else: raise ValueError("codegen negative was not rejected")
    sf_plants = [(text.replace("griddepcontrol.wait;", "absent_wait;"), sass),
                 (text.replace("griddepcontrol.launch_dependents;", "absent_release;"), sass),
                 (text.replace("st.global", "absent_store"), sass),
                 (text, sass.replace("PREEXIT", "NOP"))]
    for changed, native in sf_plants:
        try: inspect_sf(changed, native)
        except ValueError: pass
        else: raise ValueError("SF codegen negative was not rejected")
    run("cpu-fixture", [str(out/"chain"), "--cpu-only", "--prefetch"])
    if any(sha(ROOT/f) != digest for f,digest in sources.items()):
        raise ValueError("sources changed during experiment build")
    receipt = dict(status="PASS", scope="EXPERIMENT_BUILD_NOT_DEVICE_ADMISSION",
                   parent=original["build_key"], library_sha256=sha(library), dependency=dependency,
                   compiler_sha256=sha(compiler), flags=flags, sources=sources,
                   applications={name:sha(out/name) for name in ("chain","benchmark")},
                   codegen=codegen, sf_codegen=sf_codegen, unchanged_controls=controls,
                   negatives=len(plants)+len(sf_plants), native_ppu17="NOT_RUN",
                   sf_controls_schema=1, device_admission="PENDING")
    (out/"manifest.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("PDL_PREFETCH_BUILD COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()
