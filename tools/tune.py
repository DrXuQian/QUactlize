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

  * it does not invent M buckets. TRT-LLM picks powers of two before profiling because it has no reason to
    prefer anything else; we measure every token count the user fixed and put a boundary ONLY where the winner
    actually changed. A bucket edge nobody measured is a number nobody revisits.
  * it does not record a winner from an unresolved sweep. The bench declines to save one when candidates tie or
    when a single pass cannot rank; this reads that refusal and leaves the entry ABSENT. An absent tactic gets
    regenerated; a wrong one does not.
  * it does not write a table without provenance. A tactic chosen from 17 hand-written configs and one chosen
    from 227 generated ones are not comparable, and a table recording only the winner cannot tell them apart.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


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


def collapse(by_m: dict) -> list:
    """Turn {m: config} into buckets, PUTTING A BOUNDARY ONLY WHERE THE WINNER CHANGED.

    [{'m_max': 4, 'config': X}, {'m_max': None, 'config': Y}] means X up to and including 4, Y above.
    A single entry with m_max None means M does not move the winner and a reader needs no bucket logic.
    """
    ms = sorted(by_m)
    out = []
    for m in ms:
        c = by_m[m]
        if out and out[-1]["config"] == c:
            out[-1]["m_max"] = m           # extend the run
        else:
            out.append({"m_max": m, "config": c})
    if out:
        out[-1]["m_max"] = None            # the last bucket is open-ended
    return out


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
    if a.shipped:
        p = pathlib.Path(a.shipped)
        if not p.is_file():
            print(f"no such shipped-config list: {p}", file=sys.stderr)
            return 2
        raw = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        bad = [r for r in raw if canonical(r) is None]
        if bad:
            print(f"{p}: {len(bad)} line(s) are not config names, e.g. {bad[0]!r}. A list that silently drops\n"
                  f"unparseable rows would tune against a SMALLER shipped set than the library has.",
                  file=sys.stderr)
            return 2
        shipped = raw
        shipped_keys = {canonical(r) for r in raw}
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
            if conf and shipped is not None and canonical(conf) not in shipped_keys:
                note = f"winner {conf!r} is not in the shipped set"
                conf = None
            state = conf if conf else f"-- {note}"
            print(f"  n={n:<6} k={k:<6} m={m:<5} {state}")
            if conf:
                by_m[m] = conf
            else:
                skipped.append(dict(n=n, k=k, m=m, why=note))
        if by_m:
            table.append(dict(n=n, k=k, gs=a.gs, buckets=collapse(by_m)))

    doc = dict(model=a.model, gs=a.gs, reps=a.reps,
               # WHAT THIS FILE IS, stated in the file. Two artifacts with identical structure and incompatible
               # meanings is how one gets used as the other.
               kind=("tactic" if shipped is not None else "coverage-sweep"),
               shipped_configs=(shipped if shipped is not None else None),
               config_set=header, tuned_seconds=round(time.time() - t0, 1),
               shapes=table, unresolved=skipped)
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=2) + "\n")

    n_b = sum(len(s["buckets"]) for s in table)
    print(f"\nwrote {a.out}: {len(table)} shape(s), {n_b} bucket(s), {len(skipped)} unresolved")
    if n_b == len(table):
        print("  every shape has ONE bucket -- M does not move the winner here, so a reader needs no bucket\n"
              "  logic at all. That is a smaller and stronger artifact than a bucketed one, and it is a\n"
              "  measurement rather than an assumption.")
    if skipped:
        print(f"  {len(skipped)} (shape, M) did not resolve and are ABSENT rather than guessed; a lookup that\n"
              f"  misses them must fall back to the compiled default and say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
