#!/usr/bin/env python3
"""DECIDE THE WINNER FROM SAMPLES -- outside the bench, where the decision can be fed planted data.

The benches emit one JSON line per (fixture, config, pass) via benchmarks/bench_samples.hpp; nothing in them
ranks anything. This file ranks. The split is the point (docs/BENCH_DESIGN.md): selection written into a .cu has
no unit test, needs a second copy for the second bench, and consumes the run-to-run spread instead of leaving it
on disk where a later question can be answered without a new run.

THE PROCEDURE, and each choice is load-bearing rather than conventional:

  * MEDIAN per candidate, not mean or min. One stall inflates a mean. A min is the best moment rather than the
    expected one, and taking a min over more passes makes every candidate look better without making the
    comparison better.
  * BAND = [min, max] over passes. With a handful of passes a quantile confidence interval is arithmetic
    theatre; min/max is conservative and cannot claim a separation the samples do not show.
  * A CANDIDATE WHOSE BAND REACHES INTO THE LEADER'S IS A TIE, not a loser. The recorded cross-run spread on
    this hardware is 13%; ordering two candidates inside that by their point estimates is ordering noise.
  * ONE PASS IS NOT A RANKING and is reported as such rather than as a winner.

    python3 benchmarks/analyse.py run.jsonl                  # per fixture: leader, band, ties
    python3 benchmarks/analyse.py run.jsonl --json           # machine-readable verdicts
    python3 benchmarks/analyse.py --self-test                # planted data; proves each rule can fire
"""
import argparse
import collections
import json
import pathlib
import statistics
import sys

CONFIG_KEYS = ("schema", "tm", "tn", "tk", "wm", "wn", "st")
FIXTURE_KEYS = ("fixture", "dist", "n", "k", "gs", "experts", "rows", "mmax")


def config_name(s: dict) -> str:
    return f"{s['schema']} {s['tm']}x{s['tn']}x{s['tk']}:{s['wm']}x{s['wn']}:s{s['st']}"


def load(text: str):
    """-> (runs, samples, complaints). A malformed line is a complaint and never a skip: an analyser that
    ignores what it cannot parse reports a verdict over a subset it never mentions."""
    runs, samples, bad = [], [], []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            bad.append(f"line {n}: not JSON ({e.msg})")
            continue
        kind = r.get("rec")
        if kind == "run":
            runs.append(r)
        elif kind == "s":
            missing = [k for k in CONFIG_KEYS + FIXTURE_KEYS + ("pass", "us") if k not in r]
            if missing:
                bad.append(f"line {n}: sample missing {','.join(missing)}")
            else:
                samples.append(r)
        else:
            bad.append(f"line {n}: unknown rec {kind!r}")
    return runs, samples, bad


def incompatible(runs) -> list:
    """Two runs with different build identities must not be merged. Averaging a PPU_PACKED_FORMAT=0 run with a
    =2 one produces a number for a library that does not exist."""
    builds = {r.get("build", "") for r in runs}
    return sorted(builds) if len(builds) > 1 else []


def verdicts(samples):
    """-> [ {fixture..., passes, leader, median, band, ties:[...]} ], one per fixture."""
    by_fixture = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for s in samples:
        fk = tuple(s[k] for k in FIXTURE_KEYS)
        meta[fk] = {k: s[k] for k in FIXTURE_KEYS}
        by_fixture[fk][config_name(s)].append(float(s["us"]))

    out = []
    for fk, cands in by_fixture.items():
        passes = max(len(v) for v in cands.values())
        stats = {c: (statistics.median(v), min(v), max(v), len(v)) for c, v in cands.items()}
        leader = min(stats, key=lambda c: stats[c][0])
        lo_l, hi_l = stats[leader][1], stats[leader][2]
        ties = sorted(
            (c for c in stats if c != leader and stats[c][1] <= hi_l),
            key=lambda c: stats[c][0])
        out.append(dict(
            meta[fk],
            passes=passes,
            candidates=len(cands),
            leader=leader,
            median=stats[leader][0],
            band=[lo_l, hi_l],
            ties=[dict(config=c, median=stats[c][0], band=[stats[c][1], stats[c][2]]) for c in ties],
            # A one-pass file has no band, so it cannot separate anything. Said in the verdict rather than left
            # for the reader to notice that band[0] == band[1].
            ranked=passes >= 2,
        ))
    return out


def report(v: dict) -> str:
    head = (f"{v['fixture']}  n={v['n']} k={v['k']} gs={v['gs']}"
            + (f" experts={v['experts']} rows={v['rows']} mmax={v['mmax']} dist={v['dist']}"
               if v["experts"] else ""))
    lines = [head, f"  {v['candidates']} candidates over {v['passes']} pass(es)"]
    if not v["ranked"]:
        lines.append(f"  LOWEST (NOT a ranking -- one pass): {v['leader']}  {v['median']:.2f} us")
        return "\n".join(lines)
    lines.append(f"  leader: {v['leader']}  median {v['median']:.2f} us  "
                 f"band [{v['band'][0]:.2f}, {v['band'][1]:.2f}]")
    if not v["ties"]:
        lines.append("  SEPARATED: no other candidate's band reaches the leader's.")
    else:
        lines.append(f"  UNRESOLVED: {len(v['ties'])} candidate(s) tie. Expand these strata before calling a winner:")
        for t in v["ties"]:
            lines.append(f"      {t['config']:<34} median {t['median']:8.2f}  band [{t['band'][0]:.2f}, {t['band'][1]:.2f}]")
    return "\n".join(lines)


SELF_TEST = """
{"rec":"run","bench":"planted","build":"PPU_PACKED_FORMAT=0","reps":3}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"pass":0,"us":100.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"pass":1,"us":101.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"pass":2,"us":102.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":32,"tn":64,"tk":64,"wm":32,"wn":32,"st":3,"pass":0,"us":101.5}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":32,"tn":64,"tk":64,"wm":32,"wn":32,"st":3,"pass":1,"us":103.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":32,"tn":64,"tk":64,"wm":32,"wn":32,"st":3,"pass":2,"us":104.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":16,"tn":32,"tk":64,"wm":16,"wn":16,"st":2,"pass":0,"us":300.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":16,"tn":32,"tk":64,"wm":16,"wn":16,"st":2,"pass":1,"us":301.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":16,"tn":32,"tk":64,"wm":16,"wn":16,"st":2,"pass":2,"us":302.0}
not json at all
{"rec":"s","fixture":"g","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":0,"rows":0,"mmax":0,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"pass":0,"us":50.0}
"""


def self_test() -> int:
    runs, samples, bad = load(SELF_TEST)
    vs = {v["fixture"]: v for v in verdicts(samples)}
    checks = []
    checks.append(("a malformed line is reported, not skipped", len(bad) == 1 and "not JSON" in bad[0]))
    f = vs["f"]
    # The 32x64 candidate's band starts at 101.5, inside the leader's [100,102] -> a tie, NOT a loss, even
    # though its median (103.0) is worse. This is the rule the old `if (u < b.us)` could not express.
    checks.append(("the overlapping candidate ties rather than losing",
                   len(f["ties"]) == 1 and f["ties"][0]["config"].startswith("i4 32x64")))
    checks.append(("the far-away candidate does not tie",
                   all("16x32" not in t["config"] for t in f["ties"])))
    checks.append(("leader is the lowest median", f["leader"].startswith("i4 64x128")))
    g = vs["g"]
    checks.append(("a one-pass fixture refuses to rank", g["ranked"] is False))
    checks.append(("fixtures are separated", len(vs) == 2))
    # Two builds in one file must be refused rather than merged.
    two = SELF_TEST + '{"rec":"run","bench":"planted","build":"PPU_PACKED_FORMAT=2","reps":3}\n'
    checks.append(("two build identities are refused", bool(incompatible(load(two)[0]))))

    ok = True
    for name, passed in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="a file written by BENCH_JSONL=")
    ap.add_argument("--json", action="store_true", help="emit the verdicts as JSON")
    ap.add_argument("--self-test", action="store_true", help="planted data; proves each rule can fire")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.jsonl:
        ap.error("give a .jsonl, or --self-test")
    p = pathlib.Path(a.jsonl)
    if not p.is_file():
        print(f"no such file: {p}", file=sys.stderr)
        return 2

    runs, samples, bad = load(p.read_text())
    for b in bad:
        print(f"  MALFORMED {b}", file=sys.stderr)
    if not samples:
        print("no samples in this file -- nothing to rank. (Was BENCH_JSONL set for the run?)", file=sys.stderr)
        return 1
    clash = incompatible(runs)
    if clash:
        print("REFUSING TO RANK: this file mixes builds, and a verdict over two libraries describes neither:",
              file=sys.stderr)
        for b in clash:
            print(f"    {b}", file=sys.stderr)
        return 1

    vs = verdicts(samples)
    if a.json:
        print(json.dumps(vs, indent=2))
    else:
        for v in vs:
            print(report(v))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
