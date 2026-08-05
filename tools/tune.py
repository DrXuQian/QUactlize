#!/usr/bin/env python3
"""OFFLINE TACTIC TUNING: sweep a model's real shapes once, write a table, never search again.

WHEN THIS RUNS, which is the whole question it answers: once per (model, machine), beside the offline pack. Not
at llama.cpp's warmup -- that lives behind `--no-warmup`, a flag whose purpose is being turned off, so tuning
there is tuning that is off by default in practice while the code still says it happens. Not at inference: a
3-5 s search mid-conversation pollutes the first measurement of whatever it interrupts, and under CUDA graphs a
cold search during capture is the exact undefined behaviour we already had to decline once.

The reason this costs no new friction is that our weights ALREADY require an offline pass -- tools/pack_gguf.py
shuffles them. TRT-LLM needs `trtllm-build` because it has a build; llama.cpp has none, which is why the answer
looked homeless. We have the step already; this rides along.

    python3 tools/tune.py --model Qwen3.5-35B-A3B --bin build_ppu/.../test_lowbit_dense_bench -o tactics.json

TWO LAYERS, AND THIS TOOL SERVES THE SECOND. Reading the complete TRT-LLM fpA_intB source settles what a
library actually does: it COMPILES IN a small curated set (five tile configs for SM80), enumerates them at run
time, and profiles among those. So there are two separate questions, asked at different rates by different
people:

    COVERAGE   which few configs should the library COMPILE?     227 candidates, all shapes at once,
                                                                 run rarely, by us -> analyse.py --coverage
    TACTIC     for THIS model's shapes, which shipped config?    a handful of candidates, run per model
                                                                 -> this file

Conflating them produces a specific, quiet failure: a tactic table naming a config the library does not contain.
Every lookup misses, every miss falls back to the compiled default, and the artifact looks fine. So --shipped
takes the list the library actually contains and restricts the search to it, and a table produced WITHOUT it is
stamped as a coverage sweep rather than a tactic -- because that is what it is, and the two are not
interchangeable however similar the file looks.

WHAT IT WILL NOT DO, and each refusal is a mistake this project has already made once:

  * it does not extend an observation beyond the bucket that was measured. M maps to its last positive power of
    two, as in TRT-LLM, and only bucket values with a resolved run are written. A request mapping to any other
    bucket misses and takes the compiled fallback; the last measured winner is never made open-ended.
  * it does not record a winner from an unresolved sweep. The bench declines to save one when candidates tie or
    when a single pass cannot rank; this reads that refusal and leaves the entry ABSENT. An absent tactic gets
    regenerated; a wrong one does not.
  * it does not write a table without provenance. A tactic chosen from 17 hand-written configs and one chosen
    from 227 generated ones are not comparable, and a table recording only the winner cannot tell them apart.
"""
import argparse
import ctypes
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

CACHE_SCHEMA = "quactlize-ppu-tactic-cache-v1"
BUCKET_POLICY = "last-positive-power-of-two(raw-total-tokens);store-only-profiled-buckets;v1"
ROUTE_SCHEMA = "route+normalized-op+qtype+n+k+group-size+m-bucket+experts+top-k;v2"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


class ConfigV1(ctypes.Structure):
    _fields_ = [("enable_cuda_kernel", ctypes.c_bool),
                ("name", ctypes.c_char_p),
                ("tile_m", ctypes.c_int32),
                ("tile_n", ctypes.c_int32),
                ("warp_m", ctypes.c_int32),
                ("warp_n", ctypes.c_int32),
                ("stages", ctypes.c_int32)]


def fnv1a64(data: bytes, value: int = FNV_OFFSET) -> int:
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & ((1 << 64) - 1)
    return value


def fnv_hex(data: bytes) -> str:
    return f"{fnv1a64(data):016x}"


def hash_field(value: int, text: str) -> int:
    return fnv1a64(text.encode() + b"\0", value)


def load_inventory(so_path: pathlib.Path):
    """Return (dense, grouped, canonical inventory hash) from the library the cache will name."""
    lib = ctypes.CDLL(str(so_path))

    def read(symbol: str):
        fn = getattr(lib, symbol)
        fn.argtypes = [ctypes.POINTER(ctypes.POINTER(ConfigV1))]
        fn.restype = ctypes.c_int32
        ptr = ctypes.POINTER(ConfigV1)()
        count = fn(ctypes.byref(ptr))
        if count < 0 or (count and not ptr):
            raise RuntimeError(f"{symbol} returned an invalid inventory")
        out = []
        for i in range(count):
            row = ptr[i]
            if not row.name:
                raise RuntimeError(f"{symbol} config {i} has no name")
            out.append(dict(enable_cuda_kernel=bool(row.enable_cuda_kernel), name=row.name.decode(),
                            tile_m=row.tile_m, tile_n=row.tile_n, warp_m=row.warp_m,
                            warp_n=row.warp_n, stages=row.stages))
        return out

    dense = read("quactlize_ppu_list_configs")
    grouped = read("quactlize_ppu_list_grouped_configs")
    value = FNV_OFFSET
    for family, rows in (("dense", dense), ("grouped", grouped)):
        value = hash_field(value, family)
        value = hash_field(value, str(len(rows)))
        for row in rows:
            value = hash_field(value, "cuda" if row["enable_cuda_kernel"] else "tensor")
            for field in ("name", "tile_m", "tile_n", "warp_m", "warp_n", "stages"):
                value = hash_field(value, str(row[field]))
    return dense, grouped, f"{value:016x}"


def projection_op(name: str) -> str:
    return {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_o",
            "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down",
            "expert_gate": "ffn_moe_gate", "expert_up": "ffn_moe_up",
            "expert_down": "ffn_moe_down"}[name]


def write_runtime_cache(path: pathlib.Path, so_path: pathlib.Path, device: str, model: str, cfg: dict,
                        route: str, qtype: int, gs: int, table: list, inventory: list, config_hash: str,
                        validity_lib=None):
    """Write only resolved winners. Raw attempts and unresolved rows remain in the separate JSON artifact."""
    if any(ch in device for ch in "|=\r\n") or any(ch in model for ch in "|=\r\n"):
        raise ValueError("device/model provenance contains a cache delimiter")
    by_shape = {}
    from benchmarks.workloads import projections
    for name, n, k, _ in projections(cfg):
        by_shape.setdefault((n, k), set()).add(projection_op(name))

    inventory_by_key = {}
    for row in inventory:
        inventory_by_key[canonical(row["name"]) or row["name"]] = row

    # Inventory membership says a family was compiled, not that it admits this route and problem. Ask the same
    # exported predicates inference uses before persisting a winner; this prevents (for example) the dense CUDA
    # record from leaking into a fully-quantized cache merely because both routes share the dense inventory.
    lib = validity_lib if validity_lib is not None else ctypes.CDLL(str(so_path))
    dense_args = [ctypes.c_int] * 5 + [ctypes.c_char_p]
    grouped_args = [ctypes.c_int] * 7 + [ctypes.c_char_p]

    def ask(name: str, argtypes: list, *args) -> bool:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int32
        return bool(fn(*args))

    def winner_valid(compiled: dict, measured_m: int, n: int, k: int) -> bool:
        name = compiled["name"].encode()
        if route == "dense_lowbit":
            symbol_name = ("quactlize_ppu_gemv_lowbit_config_valid_v1" if compiled["enable_cuda_kernel"]
                           else "quactlize_ppu_dense_lowbit_config_valid_v1")
            return ask(symbol_name, dense_args, measured_m, n, k, gs, qtype, name)
        if route == "dense_fully_quantized":
            return not compiled["enable_cuda_kernel"] and ask(
                "quactlize_ppu_dense_fully_quantized_config_valid_v1", dense_args,
                measured_m, n, k, gs, qtype, name)
        assignments = measured_m * cfg["topk"]
        if route == "grouped_lowbit":
            return not compiled["enable_cuda_kernel"] and ask(
                "quactlize_ppu_grouped_lowbit_config_valid_v1", grouped_args,
                assignments, n, k, gs, cfg["experts"], assignments, qtype, name)
        symbol_name = ("quactlize_ppu_vecdot_moe_config_valid_v1" if compiled["enable_cuda_kernel"]
                       else "quactlize_ppu_grouped_fully_quantized_config_valid_v1")
        # max_rows=total_rows is the conservative distribution: if the compiled family admits this, every
        # histogram with the same total token bucket does. Runtime still re-asks with its actual histogram.
        return ask(symbol_name, grouped_args, assignments, n, k, gs, cfg["experts"], assignments, qtype, name)

    entries = []
    for shape in table:
        for bucket in shape["buckets"]:
            key = canonical(bucket["config"]) or bucket["config"]
            compiled = inventory_by_key.get(key)
            if compiled is None:
                raise ValueError(f"winner {bucket['config']!r} is absent from the loaded library")
            if not winner_valid(compiled, bucket["measured_m"], shape["n"], shape["k"]):
                raise ValueError(
                    f"winner {compiled['name']!r} is not valid for route={route}, "
                    f"M={bucket['measured_m']} N={shape['n']} K={shape['k']} gs={gs} qtype={qtype}")
            for op in sorted(by_shape.get((shape["n"], shape["k"]), ())):
                grouped_route = route in ("grouped_lowbit", "grouped_fully_quantized")
                experts = cfg["experts"] if grouped_route else 0
                top_k = cfg["topk"] if grouped_route else 0
                entries.append(
                    f"entry|route={route}|op={op}|qtype={qtype}|n={shape['n']}|k={shape['k']}|gs={gs}|"
                    f"m_bucket={bucket['m_bucket']}|experts={experts}|top_k={top_k}|"
                    f"cuda={int(compiled['enable_cuda_kernel'])}|config={compiled['name']}")

    elf_hash = fnv_hex(so_path.read_bytes())
    lines = [f"schema={CACHE_SCHEMA}", f"bucket_policy={BUCKET_POLICY}",
             f"bucket_policy_fnv1a64={fnv_hex(BUCKET_POLICY.encode())}",
             f"route_schema_fnv1a64={fnv_hex(ROUTE_SCHEMA.encode())}",
             f"loaded_elf_fnv1a64={elf_hash}", f"config_space_fnv1a64={config_hash}",
             f"device={device}", f"workload={model}", *entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            out.write("\n".join(lines) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def canonical(name: str):
    """-> (tm, tn, wm, wn, st), or None if this is not a config name.

    ONE CONFIG, TWO SPELLINGS, IN ONE PIPELINE. The bench's stdout names it from an X-macro stringification --
    `64x128:64x64:s3`, no schema and no TileK. analyse.py names the same thing from the sample's fields --
    `i4 64x128x64:64x64:s3`, with both. So a shipped list produced from `analyse.py --coverage` and fed to this
    tool would match NOTHING, silently: every winner would be reported as unshipped and the table would come out
    empty but well-formed. Comparing tuples rather than strings makes the spelling irrelevant.

    TileK and schema are deliberately dropped rather than compared. They are build-time constants of the binary
    (QUANT= and BENCH_TSK=), already guarded by the .inc's static_assert, and a config differing only in them is
    not a different tactic -- it is a different binary."""
    m = re.match(r"^\s*(?:\w+\s+)?(\d+)x(\d+)(?:x\d+)?:(\d+)x(\d+):s(\d+)\s*$", name)
    return tuple(int(g) for g in m.groups()) if m else None


def bench_run(binary: str, n: int, k: int, gs: int, m: int, reps: int, jsonl: str):
    """-> (config or None, tflops, note). None means the sweep did not resolve, and the note says why."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "BENCH_JSONL": jsonl, "BENCH_REPS": str(reps),
           "HOME": str(pathlib.Path.home())}
    r = subprocess.run([binary, f"--m={m}", f"--n={n}", f"--k={k}", f"--g={gs}", "--search_configs"],
                       capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        return None, 0.0, f"bench exited {r.returncode}"
    # The bench states its own verdict; parse THAT rather than re-deciding from the table it printed. Two
    # decision procedures over one run is how they come to disagree.
    if "NOT saved" in out:
        why = next((l.strip() for l in out.splitlines() if "NOT saved" in l), "declined")
        return None, 0.0, why
    m_ = re.search(r"====\s+WINNER:\s+(\S+)\s+at\s+([0-9.]+)\s+TFLOP/s\s+\(separated\)", out)
    if not m_:
        lead = re.search(r"====\s+(UNRESOLVED|LOWEST):\s+(\S+)", out)
        return None, 0.0, f"no separated winner ({lead.group(1) if lead else 'no verdict line'})"
    return m_.group(1), float(m_.group(2)), "separated"


def m_bucket(m: int) -> int:
    """TRT-LLM's round rule: the last positive power of two, with no bucket for non-positive M."""
    return 0 if m <= 0 else 1 << (m.bit_length() - 1)


def measured_buckets(by_m: dict) -> list:
    """Turn resolved measurements into exact bucket winners without extrapolating past the evidence."""
    by_bucket = {}
    for m in sorted(by_m):
        bucket = m_bucket(m)
        previous = by_bucket.get(bucket)
        if previous and previous["config"] != by_m[m]:
            raise ValueError(f"M={previous['measured_m']} and M={m} map to bucket {bucket} but disagree")
        by_bucket[bucket] = {"m_bucket": bucket, "measured_m": m, "config": by_m[m]}
    return [by_bucket[b] for b in sorted(by_bucket)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a name from benchmarks/workloads.py MODELS")
    ap.add_argument("--bin", required=True, help="the built test_lowbit_dense_bench")
    ap.add_argument("-o", "--out", required=True, help="tactic table to write")
    ap.add_argument("--gs", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5, help="passes per candidate; 1 cannot rank and is refused")
    ap.add_argument("--jsonl", default="", help="keep the raw samples here too")
    ap.add_argument("--shipped", default="",
                    help="file listing the config names the LIBRARY contains, one per line. Without it this is "
                         "a coverage sweep over the bench's whole compiled set, not a tactic table, and the "
                         "output says so. Once quactlize_ppu_list_configs() exists this should read from the "
                         ".so directly rather than from a file somebody maintained by hand.")
    ap.add_argument("--runtime-cache", default="",
                    help="also write the strict winner-only cache consumed by llama's ggml backend")
    ap.add_argument("--so", default="", help="libquactlize_ppu.so used to derive runtime-cache provenance")
    ap.add_argument("--device-id", default="", help="stable target-device identity stamped into --runtime-cache")
    ap.add_argument("--route", choices=("dense_lowbit", "dense_fully_quantized", "grouped_lowbit",
                                        "grouped_fully_quantized"),
                    default="dense_lowbit")
    ap.add_argument("--qtype", type=int, default=12, help="GGUF qtype recorded in --runtime-cache")
    a = ap.parse_args()

    if a.reps < 2:
        print("--reps must be at least 2: one pass cannot separate candidates against the recorded 13% spread",
              file=sys.stderr)
        return 2

    from benchmarks.workloads import MODELS, N_TOKENS, projections
    if a.model not in MODELS:
        print(f"unknown model {a.model!r}; have {', '.join(MODELS)}", file=sys.stderr)
        return 2
    cfg = MODELS[a.model]

    binary = pathlib.Path(a.bin)
    if not binary.is_file():
        print(f"no such binary: {binary}", file=sys.stderr)
        return 2

    # PROVENANCE. Which config set the winners were chosen from -- read from the generated table's own header,
    # not restated -- plus what was tuned and when. Without this a table cannot be compared with a later one.
    inc = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "lowbit_dense_configs.inc"
    header = [l.strip("/ \n") for l in inc.read_text().splitlines()[:4] if "configs" in l or "stages" in l] \
        if inc.is_file() else ["config table not found"]

    # THE SHIPPED SET, and what its absence means. A tactic naming a config the library does not contain misses
    # on every lookup, falls back to the compiled default, and leaves an artifact that looks correct -- the same
    # failure shape as `a tactic the binary cannot select is worse than no tactic`, one level up. So the two
    # products are named differently in the output rather than distinguished by whoever reads it later.
    shipped = None
    runtime_inventory = None
    config_hash = ""
    so_path = pathlib.Path(a.so) if a.so else None
    if a.runtime_cache:
        if a.shipped:
            print("--runtime-cache derives the shipped set from --so; do not also pass --shipped", file=sys.stderr)
            return 2
        if not so_path or not so_path.is_file() or not a.device_id:
            print("--runtime-cache requires an existing --so and non-empty --device-id", file=sys.stderr)
            return 2
        try:
            dense_inventory, grouped_inventory, config_hash = load_inventory(so_path)
        except (AttributeError, OSError, RuntimeError, UnicodeError) as e:
            print(f"cannot read tactic inventory from {so_path}: {e}", file=sys.stderr)
            return 2
        runtime_inventory = (grouped_inventory
                             if a.route in ("grouped_lowbit", "grouped_fully_quantized")
                             else dense_inventory)
        shipped = [row["name"] for row in runtime_inventory]
        shipped_keys = {canonical(r) or r for r in shipped}
        print(f"tuning against the {len(shipped)} config(s) exported by {so_path}")
    elif a.shipped:
        p = pathlib.Path(a.shipped)
        if not p.is_file():
            print(f"no such shipped-config list: {p}", file=sys.stderr)
            return 2
        raw = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        # A NAME THAT IS NOT A TILE IS NOT AN ERROR. The library's config record leads with a family
        # discriminator, and an entry in the CUDA-core family has no tile geometry at all -- canonical() returns
        # None for it BY DESIGN, so that it can never be matched against a tile config. Rejecting the list on
        # that basis would refuse exactly the encoding 051 asked for. So: parseable names compare as tuples (which
        # bridges the bench's spelling and analyse.py's), and anything else compares as an opaque exact string.
        #
        # A TYPO IS STILL LOUD, without a check for it. A mistyped tile name becomes an opaque name that matches
        # nothing, so every shape comes back "not in the shipped set" and the table is empty with a reason on
        # every row. That is more visible than a warning nobody reads, and it needs no rule that could itself be
        # wrong about what a valid name looks like.
        shipped = raw
        shipped_keys = {canonical(r) or r for r in raw}
        opaque = [r for r in raw if canonical(r) is None]
        if opaque:
            print(f"  {len(opaque)} shipped entr(ies) carry no tile geometry (e.g. {opaque[0]!r}); they match by "
                  f"name only")
        if not shipped:
            print(f"{p} lists no configs. An empty shipped set cannot be tuned against -- it would rank nothing "
                  f"and write a table of defaults.", file=sys.stderr)
            return 2
        print(f"tuning against the {len(shipped)} config(s) the library ships")
    else:
        print("no --shipped list: this is a COVERAGE sweep over the bench's whole compiled set, not a tactic\n"
              "table. Feed the samples to `analyse.py --coverage` to choose what the library should compile;\n"
              "a winner here may name a config the library does not contain.")

    shapes = sorted({(n, k) for _, n, k, _ in projections(cfg)})
    jsonl = a.jsonl or "/dev/null"
    table, skipped = [], []
    t0 = time.time()
    for n, k in shapes:
        by_m = {}
        for m in N_TOKENS:
            conf, tf, note = bench_run(str(binary), n, k, a.gs, m, a.reps, jsonl)
            # A winner outside the shipped set is not a tactic. Recording it would produce a lookup that always
            # misses; recording nothing leaves an entry that gets regenerated. The second is recoverable.
            if conf and shipped is not None and (canonical(conf) or conf) not in shipped_keys:
                note = f"winner {conf!r} is not in the shipped set"
                conf = None
            state = conf if conf else f"-- {note}"
            print(f"  n={n:<6} k={k:<6} m={m:<5} {state}")
            if conf:
                by_m[m] = conf
            else:
                skipped.append(dict(n=n, k=k, m=m, why=note))
        if by_m:
            table.append(dict(n=n, k=k, gs=a.gs, buckets=measured_buckets(by_m)))

    doc = dict(model=a.model, gs=a.gs, reps=a.reps,
               # WHAT THIS FILE IS, stated in the file. Two artifacts with identical structure and incompatible
               # meanings is how one gets used as the other.
               kind=("tactic" if shipped is not None else "coverage-sweep"),
               shipped_configs=(shipped if shipped is not None else None),
               config_set=header, tuned_seconds=round(time.time() - t0, 1),
               shapes=table, unresolved=skipped)
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")

    if a.runtime_cache:
        try:
            write_runtime_cache(pathlib.Path(a.runtime_cache), so_path, a.device_id, a.model, cfg,
                                a.route, a.qtype, a.gs, table, runtime_inventory, config_hash)
        except (AttributeError, OSError, ValueError) as e:
            print(f"cannot write runtime cache: {e}", file=sys.stderr)
            return 2
        print(f"wrote runtime cache {a.runtime_cache}: {sum(len(s['buckets']) for s in table)} measured "
              "shape bucket(s), winners only")

    n_b = sum(len(s["buckets"]) for s in table)
    print(f"\nwrote {a.out}: {len(table)} shape(s), {n_b} bucket(s), {len(skipped)} unresolved")
    if n_b == len(table):
        print("  every shape has ONE measured bucket. Other M buckets remain absent and visibly fall back;\n"
              "  this file does not assert that the winner extends past the measurement.")
    if skipped:
        print(f"  {len(skipped)} (shape, M) did not resolve and are ABSENT rather than guessed; a lookup that\n"
              f"  misses them must fall back to the compiled default and say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
