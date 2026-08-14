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

# `bc` IS PART OF THE CONFIG KEY. PPU_B_CHUNK became a tactic row field on 2026-08-07, so bc=0 and bc=1 are two
# different candidates that a sweep runs separately. Leaving it out of the key collapses them onto one name and
# every pair gets read as two REPEATS of one candidate -- which is worse than losing the distinction, because the
# tie logic would then report a spread between two different kernels as measurement noise.
#
# `warp_k` is an OPTIONAL WIRE FIELD but a REQUIRED NORMALIZED KEY. Records written before standalone Marlin had
# no K-cohort axis, and load() normalizes those to zero. A positive value is a real tactic axis: omitting it from
# the key would collapse otherwise-identical WarpK rows into repeats of one candidate.
#
# `bc_eff` is deliberately NOT a key: it is derived (the collective grants chunking only for bits in {1,2} with a
# fragment that is exactly one delivery), so a refused request compiles to the same kernel as bc=0. It stays in the
# exported record for diagnostics, but must not leak into config_name either -- that name is also used for grouping.
CONFIG_KEYS = ("schema", "tm", "tn", "tk", "wm", "wn", "warp_k", "st", "bc")
FIXTURE_KEYS = ("fixture", "dist", "n", "k", "gs", "experts", "rows", "mmax")


def config_name(s: dict) -> str:
    warp = f"x{s['warp_k']}" if int(s.get("warp_k", 0)) > 0 else ""
    return f"{s['schema']} {s['tm']}x{s['tn']}x{s['tk']}:{s['wm']}x{s['wn']}{warp}:s{s['st']} bc{s['bc']}"


def load(text: str):
    """-> (runs, samples, complaints). A malformed line is a complaint and never a skip: an analyser that
    ignores what it cannot parse reports a verdict over a subset it never mentions."""
    runs, samples, attempts, excludeds, bad = [], [], [], [], []
    # WHICH BUILD EACH SAMPLE BELONGS TO. The association is positional -- a record belongs to the most recent
    # `run` header above it -- and this loop used to throw that away, keeping `runs` and `samples` in separate
    # lists with nothing joining them. The cost showed up the first time it mattered: BENCH_JSONL APPENDS, so a
    # path reused across days accumulates several builds, incompatible() correctly refuses to rank the mixture,
    # and there was no way to recover the run you actually wanted short of re-running it. Stamping it here makes
    # --build possible and costs one assignment per line.
    #
    # Records before any `run` header get "" -- that is a real state (a file whose header was lost, or written by
    # a bench too old to emit one) and it must be selectable rather than silently folded into the first build.
    current_build = ""
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
            current_build = r.get("build", "")
        elif kind == "a":
            # WHAT LAUNCHED. Written and flushed by the bench BEFORE the kernel, so that a device assert -- which
            # takes the whole process -- still leaves the failing candidate named. Carries no `us`: there is
            # nothing measured yet, and unfinished() below is what turns these into a verdict.
            #
            # VALIDATED LIKE A SAMPLE, minus `us`. An attempt missing an identity key cannot match its own sample,
            # so the candidate reads as having died when it completed -- a false alarm on a healthy sweep, which
            # is how a warning stops being read. Checked here so it is reported at the LINE rather than surfacing
            # later as an inexplicable count.
            r.setdefault("warp_k", 0)  # legacy records predate the standalone-Marlin WarpK axis
            missing = [k for k in CONFIG_KEYS + FIXTURE_KEYS + ("pass",) if k not in r]
            if missing:
                bad.append(f"line {n}: attempt missing {','.join(missing)}")
            else:
                r["_build"] = current_build
                attempts.append(r)
        elif kind == "x":
            # TRIED AND EXCLUDED. Carries `why`, and it is the only record that says an option was reachable and
            # rejected -- which is what a pruning decision needs and what absence cannot supply.
            r.setdefault("warp_k", 0)
            missing = [k for k in CONFIG_KEYS + FIXTURE_KEYS + ("pass", "why") if k not in r]
            if missing:
                bad.append(f"line {n}: exclusion missing {','.join(missing)}")
            else:
                r["_build"] = current_build
                excludeds.append(r)
        elif kind == "s":
            r.setdefault("warp_k", 0)
            missing = [k for k in CONFIG_KEYS + FIXTURE_KEYS + ("pass", "us") if k not in r]
            if missing:
                bad.append(f"line {n}: sample missing {','.join(missing)}")
            else:
                r["_build"] = current_build
                samples.append(r)
        else:
            bad.append(f"line {n}: unknown rec {kind!r}")
    return runs, samples, attempts, excludeds, bad


def unfinished(samples, attempts, excludeds=()) -> list:
    """-> the attempts that never produced a sample, i.e. where the sweep died.

    THE READER'S HALF OF THE ATTEMPT RECORD. Writing `a` lines and never checking them would be the same defect
    as a gate that is never registered: the durability exists in the file and not in anyone's hands. One line of
    rule -- AN ATTEMPT WITH NO MATCHING SAMPLE IS WHERE IT STOPPED -- and it is the only way to name the config
    that killed a run, because the bench's own report of which candidate ran happens after the launch returns.

    Matched on the full identity plus `pass`, so a candidate that dies on its third repetition is reported at
    that repetition rather than being masked by its two successes.
    """
    key = lambda r: tuple(r.get(k) for k in CONFIG_KEYS + FIXTURE_KEYS + ("pass",))
    # An EXCLUSION resolves an attempt just as a sample does. The candidate was tried and the bench said no --
    # that is an outcome, not a disappearance, and reporting it as "the run did not finish" would cry wolf on
    # every healthy sweep that contains one unsupported config.
    done = {key(s) for s in samples} | {key(x) for x in excludeds}
    return [a for a in attempts if key(a) not in done]


def prune_report(samples, excludeds, tol=0.05):
    """-> (axis_rows, heuristic_rows). What the sweep says can be DROPPED, and whether a rule could replace it.

    TWO QUESTIONS, AND THEY ARE NOT THE SAME ONE.

    (a) CAN THE COMPILED SET SHRINK? Each row is 293 template instantiations x 4 group sizes; every value on
        every axis costs build time forever. An axis value that never wins anywhere, and whose removal costs no
        fixture more than `tol`, is dead weight. REGRET, not win count, is the criterion: a value that wins
        nothing but is always within 1% is free to drop, while one that wins once by 30% is not.

    (b) IS A HEURISTIC VIABLE AT ALL? The alternative to a tactic cache is a rule -- "M<=4 takes this, M>=512
        takes that". That is only honest if, inside each M bucket, ONE config is within `tol` of the best for
        EVERY fixture in it. If no single config covers a bucket, a rule over M cannot reproduce the sweep and
        the cache is doing work a heuristic cannot.

    Buckets are floor-power-of-two over `rows`, matching the tactic cache's own bucketing -- comparing a
    heuristic against a differently-bucketed cache would be measuring the bucketing, not the heuristic.

    EXCLUSIONS COUNT SEPARATELY. A config that was tried and rejected everywhere is prunable for a different
    reason -- it is not slow, it is unbuildable for those shapes -- and merging the two would hide which.
    """
    vs = verdicts(samples)
    if not vs:
        return [], []
    # best[fixture][config] = median us, from the same statistics verdicts() uses
    per_fix = collections.defaultdict(dict)
    for s in samples:
        per_fix[tuple(s[k] for k in FIXTURE_KEYS)].setdefault(config_name(s), []).append(float(s["us"]))
    per_fix = {f: {c: statistics.median(v) for c, v in d.items()} for f, d in per_fix.items()}
    best = {f: min(d.values()) for f, d in per_fix.items()}

    def regret_without(pred):
        """The worst fixture's loss, as a ratio, if every config satisfying `pred` were dropped. inf when a
        fixture would be left with NO config at all -- which is not a large regret, it is no answer."""
        worst = 0.0
        for f, d in per_fix.items():
            kept = [us for c, us in d.items() if not pred(c)]
            if not kept:
                return float("inf")
            worst = max(worst, kept and (min(kept) / best[f] - 1.0))
        return worst

    cfg_fields = {}
    for s in samples:
        cfg_fields[config_name(s)] = {k: s[k] for k in ("tm", "tn", "tk", "wm", "wn", "warp_k", "st")}
    winners = {min(d, key=d.get) for d in per_fix.values()}

    axis_rows = []
    # Preserve the legacy prune report when no sample opted into WarpK. Once one does, it is a real sweep axis
    # and must appear in the keep/drop census just like WarpM/WarpN.
    axes = ("tm", "tn", "wm", "wn") + (("warp_k",) if any(f["warp_k"] > 0 for f in cfg_fields.values()) else ()) + ("st",)
    for axis in axes:
        for val in sorted({f[axis] for f in cfg_fields.values()}):
            hit = lambda c, a=axis, v=val: cfg_fields[c][a] == v
            wins = sum(1 for w in winners if hit(w))
            axis_rows.append({"axis": axis, "value": val, "wins": wins,
                              "configs": sum(1 for c in cfg_fields if hit(c)),
                              "regret_if_dropped": regret_without(hit)})

    # (b) one config per M bucket, within tol of the best for every fixture in the bucket
    buckets = collections.defaultdict(list)
    for f in per_fix:
        rows = dict(zip(FIXTURE_KEYS, f))["rows"]
        b = 1 << max(0, int(rows).bit_length() - 1) if rows else 0
        buckets[b].append(f)
    heur = []
    for b in sorted(buckets):
        fs = buckets[b]
        common = set.intersection(*(set(per_fix[f]) for f in fs))
        ok = [(c, max(per_fix[f][c] / best[f] - 1.0 for f in fs)) for c in common]
        ok.sort(key=lambda x: x[1])
        heur.append({"bucket": b, "fixtures": len(fs),
                     "best_single": ok[0][0] if ok else None,
                     "worst_regret": ok[0][1] if ok else float("inf"),
                     "viable": bool(ok) and ok[0][1] <= tol})
    return axis_rows, heur


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


# =============================================================================================================
# COVERAGE -- A DIFFERENT QUESTION FROM RANKING, AND THE ONE THE 227-CONFIG SWEEP ACTUALLY ANSWERS.
#
# Everything above ranks candidates WITHIN one fixture and names a winner. That is the per-model tune's question,
# asked often, against the handful of configs a shipped library contains.
#
# The big sweep asks something else. Reading TRT-LLM's fpA_intB (the complete tree, Kernels/general/w4a16_gemm/
# fpA_intB_standalone/) settles what a library does: it COMPILES IN a small curated set -- five tile configs for
# SM80 -- enumerates them at run time, and profiles among those. So the sweep over 227 candidates is not choosing
# a tactic. It is choosing WHICH FEW TO COMPILE, once, for everyone. That is a set question:
#
#     ranking:  for THIS shape, which config is fastest?              -> one fixture, many configs
#     coverage: which SET, shipped, is fast enough EVERYWHERE?        -> many fixtures, one set
#
# Ranking cannot answer it. The union of per-fixture winners is an upper bound on the set, not the set: a config
# that wins one fixture by 1% while another already in the set is 1% behind everywhere is not worth an
# instantiation, and a ranking report cannot see that because it never compares across fixtures.
#
# WHAT IS COMPUTED. For a set S and fixture f, the cost of shipping only S is the best S can do at f relative to
# the best ANY measured candidate does at f:
#
#     regret(f, S) = min_{c in S, c measured at f} median(f, c) / min_c median(f, c)  -  1
#
# and a set that contains nothing measured at f does not cover f at all -- reported as uncovered, never as zero.
# The greedy ladder then adds, at each step, the config that lowers the WORST regret the most, giving:
#
#     ship 1 config  -> worst case X% off the per-fixture best
#     ship 2         -> Y%
#     ...
#
# so the size/performance trade is read off measurements rather than argued. TRT-LLM chose five by hand and
# labelled one of its prunes "purely to improve compilation speed"; this makes the same decision answerable.
#
# TWO REFUSALS, both because a coverage claim is stronger than a ranking claim and outlives it:
#   * a fixture whose own sweep did not SEPARATE (ties inside the leader's band) has no trustworthy optimum, so
#     regrets measured against it are noise. Those fixtures are listed and excluded from the worst-case, not
#     quietly averaged in.
#   * a config measured at only some fixtures cannot be credited at the others. Absence is not a zero.

def coverage(samples, verdict_list):
    """-> {ladder: [...], unresolved: [...], never_wins: [...], fixtures: n, configs: n}

    Greedy set cover over fixtures, minimising worst-case regret. Deterministic: ties in the greedy step break on
    mean regret and then on config name, so two runs over one file give one answer."""
    med = collections.defaultdict(dict)            # fixture key -> config -> median us
    by_fx = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in samples:
        by_fx[tuple(s[k] for k in FIXTURE_KEYS)][config_name(s)].append(float(s["us"]))
    for fk, cands in by_fx.items():
        for c, v in cands.items():
            med[fk][c] = statistics.median(v)

    # A fixture that could not separate its own leader has no optimum to measure regret against.
    unresolved = {tuple(v[k] for k in FIXTURE_KEYS) for v in verdict_list if v["ties"] or not v["ranked"]}
    usable = [fk for fk in med if fk not in unresolved]
    if not usable:
        return dict(ladder=[], unresolved=sorted(str(u) for u in unresolved), never_wins=[],
                    fixtures=0, configs=len({c for f in med.values() for c in f}))

    best = {fk: min(med[fk].values()) for fk in usable}
    all_cfgs = sorted({c for fk in usable for c in med[fk]})

    def regrets(S):
        """-> {fixture: regret or None-if-uncovered}"""
        out = {}
        for fk in usable:
            have = [med[fk][c] for c in S if c in med[fk]]
            out[fk] = (min(have) / best[fk] - 1.0) if have else None
        return out

    def score(S):
        r = regrets(S)
        if any(v is None for v in r.values()):
            return (1, len([v for v in r.values() if v is None]), 0.0)   # uncovered dominates
        return (0, max(r.values()), statistics.fmean(r.values()))

    chosen, ladder = [], []
    remaining = list(all_cfgs)
    while remaining:
        pick = min(remaining, key=lambda c: (score(chosen + [c]), c))
        chosen.append(pick)
        remaining.remove(pick)
        r = regrets(chosen)
        unc = [fk for fk, v in r.items() if v is None]
        cov = [v for v in r.values() if v is not None]
        worst_fk = max((fk for fk in r if r[fk] is not None), key=lambda fk: r[fk], default=None)
        ladder.append(dict(
            k=len(chosen), added=pick, set=list(chosen),
            uncovered=len(unc),
            worst=(max(cov) if cov else None),
            mean=(statistics.fmean(cov) if cov else None),
            worst_fixture=(dict(zip(FIXTURE_KEYS, worst_fk)) if worst_fk else None),
        ))
        if not unc and max(cov) <= 1e-12:
            break                                   # every fixture has its own optimum in the set

    winners = {min(med[fk], key=lambda c: med[fk][c]) for fk in usable}
    never = [c for c in all_cfgs if c not in winners]
    return dict(ladder=ladder, unresolved=sorted(str(u) for u in unresolved), never_wins=never,
                fixtures=len(usable), configs=len(all_cfgs))


def invariant(samples):
    """DENSE MUST NOT BE SLOWER THAN GROUPED WITH ONE EXPERT. -> [ {shape, dense, grouped, ratio, ok} ]

    Grouped with a single expert IS dense: the same math through the same collective family, one group. But it
    additionally pays a scheduler, a pointer-array epilogue and a boundary decode. So at one shape

        best(dense)  <=  best(grouped, experts=1)

    and a violation cannot be explained away. Not by grouped overhead -- that is what makes the inequality hold.
    Not by the masked-row tax -- one group has no masked rows. Not by the ragged distribution -- there is none at
    L=1. It is the one comparison whose wrong answer is unambiguous, which is why it is worth a check rather than
    an eyeball over two logs.

    Each side contributes its OWN BEST, not a common config: the question is whether the dense path can be made
    to reach what the grouped path reaches, and holding the config fixed would answer a different one.
    """
    best = collections.defaultdict(dict)          # (n,k,gs,rows) -> {'dense'|'grouped': (config, median)}
    for s in samples:
        e = int(s["experts"])
        if e > 1:
            continue                              # a real MoE run is not a term in this inequality
        side = "dense" if e == 0 else "grouped"
        key = (s["n"], s["k"], s["gs"], s["rows"])
        best[key].setdefault(side, {}).setdefault(config_name(s), []).append(float(s["us"]))

    out = []
    for key, sides in sorted(best.items()):
        if len(sides) != 2:
            continue                              # only one side measured at this shape; not a comparison
        picked = {}
        for side, cands in sides.items():
            med = {c: statistics.median(v) for c, v in cands.items()}
            c = min(med, key=lambda x: med[x])
            picked[side] = (c, med[c])
        d_cfg, d_us = picked["dense"]
        g_cfg, g_us = picked["grouped"]
        out.append(dict(n=key[0], k=key[1], gs=key[2], rows=key[3],
                        dense_config=d_cfg, dense_us=d_us,
                        grouped_config=g_cfg, grouped_us=g_us,
                        ratio=d_us / g_us, ok=d_us <= g_us))
    return out


def invariant_report(rows: list) -> str:
    if not rows:
        return ("INVARIANT: no shape has BOTH a dense (experts=0) and a one-expert grouped sample. Nothing was\n"
                "compared. Run both benches at the same n/k/gs/rows into the same BENCH_JSONL, or into two files\n"
                "passed together -- a missing side reads as a pass otherwise, which is the failure this check\n"
                "exists to avoid.")
    L = ["INVARIANT  dense <= grouped(experts=1)",
         "  n      k      gs   rows    dense us   grouped us   ratio   ",
         "  -----  -----  ---  ------  ---------  -----------  --------"]
    for r in rows:
        mark = "ok" if r["ok"] else "VIOLATED"
        L.append(f"  {r['n']:<5}  {r['k']:<5}  {r['gs']:<3}  {r['rows']:<6}  {r['dense_us']:9.2f}  "
                 f"{r['grouped_us']:11.2f}  {r['ratio']:6.3f}  {mark}")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        r = bad[0]
        L.append(f"\n  {len(bad)} shape(s) VIOLATE the invariant. At n={r['n']} k={r['k']} gs={r['gs']} the dense")
        L.append(f"  path's best ({r['dense_config']}) is {(r['ratio']-1)*100:.1f}% slower than the grouped path's")
        L.append(f"  best ({r['grouped_config']}) doing strictly more work. That is a dense-path defect: the")
        L.append("  scheduler and pointer-array epilogue grouped pays cannot make it faster.")
    else:
        L.append(f"\n  all {len(rows)} shape(s) hold.")
    return "\n".join(L)


def coverage_report(cov: dict) -> str:
    if not cov["ladder"]:
        return ("COVERAGE: nothing to cover. Every fixture in this file failed to separate its own leader, so\n"
                "there is no optimum to measure a shipped set against. Widen the repeats before asking this.")
    L = [f"COVERAGE over {cov['fixtures']} separated fixture(s), {cov['configs']} measured config(s)",
         "  ship  worst-case  mean    added",
         "  ----  ----------  ------  -----"]
    for e in cov["ladder"]:
        # A SET THAT DOES NOT COVER EVERY FIXTURE HAS NO WORST CASE -- IT HAS A HOLE. Printing the worst regret
        # over the fixtures it does reach, next to a count of the ones it does not, reads as "0.00% -- no loss"
        # when the truth is "this set cannot run one of the shapes at all". So the number is withheld until the
        # hole is closed, and the reason takes its place.
        if e["uncovered"]:
            w, m = "no cover", "  --  "
            note = f"   <- {e['uncovered']} shape(s) have NO config in this set; it cannot run them at all"
        else:
            w = f"{e['worst']*100:6.2f}%"
            m = f"{e['mean']*100:5.2f}%"
            note = ""
        L.append(f"  {e['k']:>4}  {w:>10}  {m:>6}  {e['added']}{note}")
    last = cov["ladder"][-1]
    if last["worst"] is not None and last["worst"] <= 1e-12:
        L.append(f"\n  {last['k']} config(s) reach EVERY fixture's own optimum. Shipping more cannot help;")
        L.append("  shipping fewer costs the worst-case shown above.")
    else:
        L.append(f"\n  the ladder did not reach zero regret: worst remaining is at "
                 f"{last['worst_fixture']}.")
    if cov["unresolved"]:
        L.append(f"\n  EXCLUDED, {len(cov['unresolved'])} fixture(s) did not separate their own leader, so a")
        L.append("  regret measured against them would be noise:")
        for u in cov["unresolved"][:6]:
            L.append(f"      {u}")
    if cov["never_wins"]:
        L.append(f"\n  {len(cov['never_wins'])} config(s) win nowhere. That is a claim about THIS fixture set:")
        L.append("  a config absent from a model's shapes has not been shown useless, only untested here.")
    return "\n".join(L)


SELF_TEST = """
{"rec":"run","bench":"planted","build":"PPU_PACKED_FORMAT=0","reps":3}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"bc":0,"bc_eff":0,"pass":0,"us":100.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"bc":0,"bc_eff":0,"pass":1,"us":101.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"bc":0,"bc_eff":0,"pass":2,"us":102.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":32,"tn":64,"tk":64,"wm":32,"wn":32,"st":3,"bc":0,"bc_eff":0,"pass":0,"us":101.5}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":32,"tn":64,"tk":64,"wm":32,"wn":32,"st":3,"bc":0,"bc_eff":0,"pass":1,"us":103.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":32,"tn":64,"tk":64,"wm":32,"wn":32,"st":3,"bc":0,"bc_eff":0,"pass":2,"us":104.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":16,"tn":32,"tk":64,"wm":16,"wn":16,"st":2,"bc":0,"bc_eff":0,"pass":0,"us":300.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":16,"tn":32,"tk":64,"wm":16,"wn":16,"st":2,"bc":0,"bc_eff":0,"pass":1,"us":301.0}
{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":256,"rows":128,"mmax":420,"tm":16,"tn":32,"tk":64,"wm":16,"wn":16,"st":2,"bc":0,"bc_eff":0,"pass":2,"us":302.0}
not json at all
{"rec":"s","fixture":"g","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":0,"rows":0,"mmax":0,"tm":64,"tn":128,"tk":64,"wm":64,"wn":64,"st":3,"bc":0,"bc_eff":0,"pass":0,"us":50.0}
"""


# COVERAGE PLANTED DATA. Deliberately NOT reusing SELF_TEST: every fixture there either ties or has one pass, so
# coverage correctly refuses it -- which tests the refusal and nothing else. Here X wins at A, Y wins at B, and
# neither wins the other, so no single config can serve both. That is the whole point of asking coverage rather
# than reading a ranking: a per-fixture report would name two winners and say nothing about whether one suffices.
def _s(fx, tm, tn, wm, wn, st, us, p):
    return ('{"rec":"s","fixture":"%s","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":0,"rows":0,'
            '"mmax":0,"tm":%d,"tn":%d,"tk":64,"wm":%d,"wn":%d,"st":%d,"bc":0,"bc_eff":0,"pass":%d,"us":%.1f}' %
            (fx, tm, tn, wm, wn, st, p, us))


def _cov_data():
    L = ['{"rec":"run","bench":"planted","build":"B","reps":3}']
    # X = 64x128:64x64 s3     Y = 32x64:32x32 s3     Z = 16x32:16x16 s2 (measured at A only)
    for p, jitter in enumerate((0.0, 0.5, 1.0)):
        L += [_s("A", 64, 128, 64, 64, 3, 100 + jitter, p), _s("A", 32, 64, 32, 32, 3, 200 + jitter, p),
              _s("A", 16, 32, 16, 16, 2, 500 + jitter, p),
              _s("B", 64, 128, 64, 64, 3, 200 + jitter, p), _s("B", 32, 64, 32, 32, 3, 100 + jitter, p),
              _s("C", 64, 128, 64, 64, 3, 100 + jitter, p), _s("C", 32, 64, 32, 32, 3, 105 + jitter, p)]
    return "\n".join(L) + "\n"


def _uncovered_data():
    """One fixture measured ONLY by a config the other fixture never saw. Absence must read as uncovered."""
    L = ['{"rec":"run","bench":"planted","build":"B","reps":3}']
    for p, j in enumerate((0.0, 0.5, 1.0)):
        L += [_s("A", 64, 128, 64, 64, 3, 100 + j, p), _s("A", 32, 64, 32, 32, 3, 300 + j, p),
              _s("E", 16, 32, 16, 16, 2, 100 + j, p)]
    return "\n".join(L) + "\n"


def self_test() -> int:
    runs, samples, attempts, excludeds, bad = load(SELF_TEST)
    vs = {v["fixture"]: v for v in verdicts(samples)}
    checks = []
    checks.append(("a malformed line is reported, not skipped", len(bad) == 1 and "not JSON" in bad[0]))

    # THE ATTEMPT RECORD'S READER, PLANTED BOTH WAYS. An `a` with a matching `s` must NOT be reported (or every
    # healthy sweep cries wolf and the warning stops being read), and an `a` without one MUST be.
    _a = ('{"rec":"a","fixture":"f","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":0,"rows":128,'
          '"mmax":128,"tm":64,"tn":64,"tk":64,"wm":64,"wn":32,"st":3,"bc":0,"bc_eff":0,"pass":0}')
    _s_done = _a.replace('"rec":"a"', '"rec":"s"')[:-1] + ',"us":209.27}'
    _r, _sm, _at, _ex, _bd = load(_a + "\n" + _s_done + "\n")
    checks.append(("an attempt that produced a sample is not reported as unfinished",
                   _bd == [] and unfinished(_sm, _at) == []))
    _r, _sm, _at, _ex, _bd = load(_a + "\n")
    checks.append(("an attempt with no sample IS reported as unfinished",
                   _bd == [] and len(unfinished(_sm, _at)) == 1))
    # And a DIFFERENT pass must not satisfy it: a config that dies on its third repetition is still a death.
    _r, _sm, _at, _ex, _bd = load(_a + "\n" + _s_done.replace('"pass":0', '"pass":1') + "\n")
    # --prune, PLANTED BOTH WAYS. A report that can only ever say "prunable" is as useless as one that can only
    # say "keep": each direction gets a fixture set constructed to force it.
    def _p(fix, rows, tm, st, us):
        return ('{"rec":"s","fixture":"%s","dist":"d","schema":"i4","n":512,"k":2048,"gs":32,"experts":0,'
                '"rows":%d,"mmax":%d,"tm":%d,"tn":64,"tk":64,"wm":32,"wn":32,"st":%d,"bc":0,"bc_eff":0,"pass":0,"us":%.1f}'
                % (fix, rows, rows, tm, st, us))
    # st=2 is never a winner and never more than 1% behind -> DEAD, droppable.
    dead = "\n".join([_p("A", 8, 64, 3, 100.0), _p("A", 8, 64, 2, 100.5),
                      _p("B", 8, 64, 3, 200.0), _p("B", 8, 64, 2, 201.0)]) + "\n"
    ax, _h = prune_report(load(dead)[1], [], 0.05)
    st2 = next(r for r in ax if r["axis"] == "st" and r["value"] == 2)
    checks.append(("an axis value that never wins and costs nothing is reported droppable",
                   st2["wins"] == 0 and st2["regret_if_dropped"] <= 0.05))
    # AND THE OTHER DIRECTION, which needs DIFFERENT data. In `dead` the runner-up is 0.5% behind, so dropping
    # the WINNER also costs 0.5% and the report calls it droppable -- correctly: when two configs are within
    # half a percent, either one is expendable and "it wins" is not by itself a reason to keep it. Regret is the
    # criterion, and that is the point. To show the report can refuse, the winner has to actually be worth
    # something.
    lead = "\n".join([_p("A", 8, 64, 3, 100.0), _p("A", 8, 64, 2, 200.0),
                      _p("B", 8, 64, 3, 100.0), _p("B", 8, 64, 2, 200.0)]) + "\n"
    ax2, _h3 = prune_report(load(lead)[1], [], 0.05)
    lead3 = next(r for r in ax2 if r["axis"] == "st" and r["value"] == 3)
    checks.append(("an axis value whose removal costs the worst fixture dearly is NOT reported droppable",
                   lead3["wins"] > 0 and lead3["regret_if_dropped"] > 0.05))
    # Two fixtures in ONE bucket whose winners disagree by a lot -> no rule over M can cover it.
    split = "\n".join([_p("C", 8, 64, 3, 100.0), _p("C", 8, 16, 3, 300.0),
                       _p("D", 8, 64, 3, 300.0), _p("D", 8, 16, 3, 100.0)]) + "\n"
    _a2, h2 = prune_report(load(split)[1], [], 0.05)
    checks.append(("a bucket whose fixtures want different configs is reported NOT coverable by a rule",
                   len(h2) == 1 and not h2[0]["viable"]))

    checks.append(("a sample from another pass does not mask an unfinished attempt",
                   len(unfinished(_sm, _at)) == 1))
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

    # ---- invariant ---------------------------------------------------------------------------------------
    def _inv(dense_us, grouped_us):
        L = ['{"rec":"run","bench":"planted","build":"B","reps":3}']
        for p_, j in enumerate((0.0, 0.5, 1.0)):
            for e, us in ((0, dense_us), (1, grouped_us)):
                L.append('{"rec":"s","fixture":"f","dist":"d","schema":"i4","n":4096,"k":4096,"gs":32,'
                         '"experts":%d,"rows":2048,"mmax":2048,"tm":64,"tn":128,"tk":64,"wm":64,"wn":32,'
                         '"st":3,"bc":0,"bc_eff":0,"pass":%d,"us":%.1f}' % (e, p_, us + j))
        return "\n".join(L) + "\n"

    ok_rows = invariant(load(_inv(200.0, 220.0))[1])
    checks.append(("dense faster than grouped(L=1) holds", len(ok_rows) == 1 and ok_rows[0]["ok"]))
    # THE LOAD-BEARING ONE: the check must FIRE, not merely pass on good data.
    bad_rows = invariant(load(_inv(240.0, 220.0))[1])
    checks.append(("dense SLOWER than grouped(L=1) is reported as a violation",
                   len(bad_rows) == 1 and not bad_rows[0]["ok"] and bad_rows[0]["ratio"] > 1.0))
    # A missing side must not read as a pass.
    one_side = "\n".join(l for l in _inv(200.0, 220.0).splitlines() if '"experts":1' not in l) + "\n"
    checks.append(("a shape with only one side is not compared", invariant(load(one_side)[1]) == []))
    # A real MoE run (experts>1) is not a term in the inequality.
    moe = _inv(200.0, 220.0).replace('"experts":1', '"experts":256')
    checks.append(("a real MoE run is excluded from the invariant", invariant(load(moe)[1]) == []))

    # ---- coverage ----------------------------------------------------------------------------------------
    # Every fixture in SELF_TEST either ties or has one pass, so coverage must produce no ladder at all rather
    # than a ladder over fixtures whose optima are not separated.
    c0 = coverage(samples, verdicts(samples))
    checks.append(("coverage refuses fixtures that did not separate", c0["ladder"] == [] and c0["fixtures"] == 0))

    cs = load(_cov_data())[1]
    c = coverage(cs, verdicts(cs))
    checks.append(("coverage sees all three planted fixtures", c["fixtures"] == 3))
    # THE LOAD-BEARING CHECK. X and Y each win one fixture the other loses badly, so one config cannot be
    # enough -- and a per-fixture ranking, which is all this file could do before, cannot express that.
    checks.append(("one config does NOT suffice when winners differ across fixtures",
                   len(c["ladder"]) >= 2 and c["ladder"][0]["worst"] > 0.5))
    checks.append(("two configs reach every fixture's own optimum",
                   c["ladder"][1]["worst"] <= 1e-12 and len(c["ladder"]) == 2))
    checks.append(("the ladder stops once regret is zero", c["ladder"][-1]["k"] == 2))
    checks.append(("a config that wins nowhere is named", any("16x32" in n for n in c["never_wins"])))
    # 5% at fixture C is real and must not be rounded away by the mean.
    checks.append(("the first rung's worst case is the fixture it loses, not the average",
                   c["ladder"][0]["worst_fixture"]["fixture"] in ("A", "B")))

    us_ = load(_uncovered_data())[1]
    u = coverage(us_, verdicts(us_))
    checks.append(("a config absent from a fixture cannot cover it (absence is not zero regret)",
                   u["ladder"][0]["uncovered"] == 1 and u["ladder"][0]["worst"] is not None))

    ok = True
    for name, passed in checks:
        print(f"  [{'ok ' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="*", help="file(s) written by BENCH_JSONL=")
    ap.add_argument("--json", action="store_true", help="emit the verdicts as JSON")
    ap.add_argument("--coverage", action="store_true",
                    help="which SET to compile into the library, not which config wins one shape")
    ap.add_argument("--invariant", action="store_true",
                    help="check dense <= grouped(experts=1); exits non-zero on a violation")
    ap.add_argument("--prune", action="store_true",
                    help="which config axes never earn their place, and whether one config per M bucket could "
                         "replace the cache")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="how much regret counts as free, for --prune (default 0.05 = 5%%)")
    ap.add_argument("--self-test", action="store_true", help="planted data; proves each rule can fire")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.jsonl:
        ap.error("give a .jsonl, or --self-test")
    text = ""
    for one in a.jsonl:
        p = pathlib.Path(one)
        if not p.is_file():
            print(f"no such file: {p}", file=sys.stderr)
            return 2
        text += p.read_text()

    runs, samples, attempts, excludeds, bad = load(text)
    for b in bad:
        print(f"  MALFORMED {b}", file=sys.stderr)
    # BEFORE ANYTHING IS RANKED. A sweep that died leaves a ranking over its PREFIX, which reads as a complete
    # answer -- the leader of the configs that happened to run before the crash is not the leader of the space.
    # So this is loud, it is first, and it names the row so it can be reproduced.
    stopped = unfinished(samples, attempts, excludeds)
    if stopped:
        print(f"  ⚠ {len(stopped)} candidate(s) LAUNCHED AND PRODUCED NO SAMPLE -- the run did not finish.",
              file=sys.stderr)
        for s in stopped[:8]:
            print(f"      {s.get('schema')} {s.get('tm')}x{s.get('tn')}:{s.get('wm')}x{s.get('wn')}:s{s.get('st')}"
                  f" tk={s.get('tk')} gs={s.get('gs')} pass={s.get('pass')}  fixture={s.get('fixture')}",
                  file=sys.stderr)
        if len(stopped) > 8:
            print(f"      ... and {len(stopped) - 8} more", file=sys.stderr)
        print("    Any ranking below covers only the candidates that completed. Reproduce a named row with"
              " --config before treating the leader as the winner.", file=sys.stderr)

    if not samples:
        print("no samples in this file -- nothing to rank. (Was BENCH_JSONL set for the run?)", file=sys.stderr)
        return 1
    clash = [] if a.invariant else incompatible(runs)
    if clash:
        print("REFUSING TO RANK: this file mixes builds, and a verdict over two libraries describes neither:",
              file=sys.stderr)
        for b in clash:
            print(f"    {b}", file=sys.stderr)
        return 1

    vs = verdicts(samples)
    if a.invariant:
        # NOT gated on incompatible(): comparing the two benches is the point, and they are two binaries with
        # two `bench` names by construction. A build-define mismatch would still matter, so it is reported.
        rows = invariant(samples)
        print(json.dumps(rows, indent=2) if a.json else invariant_report(rows))
        return 1 if any(not r["ok"] for r in rows) else 0
    if a.prune:
        axis_rows, heur = prune_report(samples, excludeds, a.tol)
        if a.json:
            print(json.dumps({"axes": axis_rows, "buckets": heur}, indent=2))
            return 0
        print(f"AXIS VALUES, and what dropping each would cost the WORST fixture (tol {a.tol:.0%})")
        print(f"  {'axis':<5} {'value':>6} {'wins':>5} {'configs':>8} {'regret if dropped':>18}")
        for r in sorted(axis_rows, key=lambda r: (r["axis"], r["value"])):
            g = r["regret_if_dropped"]
            mark = "  <- DEAD, drop it" if r["wins"] == 0 and g <= a.tol else (
                   "  <- sole cover" if g == float("inf") else "")
            print(f"  {r['axis']:<5} {r['value']:>6} {r['wins']:>5} {r['configs']:>8} "
                  f"{('never viable' if g == float('inf') else f'{g:+.1%}'):>18}{mark}")
        if excludeds:
            byw = collections.Counter(x.get("why", "?") for x in excludeds)
            print(f"\n  {len(excludeds)} exclusion(s) -- tried and refused, prunable at the EMITTER not by speed:")
            for why, n in byw.most_common(5):
                print(f"    {n:>6}  {why}")
        print(f"\nCOULD ONE CONFIG PER M BUCKET REPLACE THE CACHE? (bucket = floor power of two of rows)")
        print(f"  {'bucket':>7} {'fixtures':>9} {'worst regret':>13}  best single config")
        for h in heur:
            g = h["worst_regret"]
            print(f"  {h['bucket']:>7} {h['fixtures']:>9} "
                  f"{('no common config' if g == float('inf') else f'{g:+.1%}'):>13}  "
                  f"{h['best_single'] or '-'}{'' if h['viable'] else '   <- a rule cannot cover this bucket'}")
        if all(h["viable"] for h in heur):
            print(f"\n  Every bucket has one config within {a.tol:.0%} of best everywhere in it: a heuristic over M")
            print("  reproduces this sweep, and the cache is not buying anything these shapes can show.")
        else:
            bad = [h["bucket"] for h in heur if not h["viable"]]
            print(f"\n  Bucket(s) {bad} have no single config within {a.tol:.0%} of best for every fixture in them.")
            print("  A rule over M alone cannot reproduce the sweep there -- the winner depends on N/K too.")
        return 0
    if a.coverage:
        cov = coverage(samples, vs)
        print(json.dumps(cov, indent=2) if a.json else coverage_report(cov))
        return 0
    if a.json:
        print(json.dumps(vs, indent=2))
    else:
        for v in vs:
            print(report(v))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
