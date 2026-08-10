#!/usr/bin/env python3
"""PER TENSOR OF A NAMED MODEL: the ONE offline artifact to build, and where that "one" breaks down.

THE QUESTION THIS EXISTS TO ANSWER PRECISELY -- is the offline format M-dependent? The two halves have different
answers and conflating them is the trap:

  * THE BYTES ARE NOT chosen per M. ci/local_gates.py's lint_tactic_cannot_change_offline_layout enforces that no
    config row carries TileK, and quactlize/include/ppu_format_config.inc fixes TileK per format. A sweep that
    moved the layout per M would invalidate every artifact on disk.
  * WHICH KERNEL READS THEM IS chosen per M. Three routes -- the CUDA-core GEMV at decode, the mixed-input GEMM in
    the middle, dequantise-then-dense at large M -- and two schemes (SCALE_FIRST, FULLY_QUANTIZED) whose TileK the
    registry deliberately makes DIFFERENT for four of the five k-quants.

So the deliverable is not "a layout per M". It is: for each tensor, the one artifact that serves every M the
deployment will see, the route table saying which kernel reads it at which M, and AN EXPLICIT STATEMENT WHEREVER
THOSE TWO CONFLICT -- because a conflict means the tensor needs two artifacts, and that is a storage decision for
the operator rather than one a script should quietly take.

WHAT MAKES ONE ARTIFACT ENOUGH, when it is. A placed plane's stored bytes depend on (bits, WON, TileK, F); see
quactlize/layouts.py:xplane, whose parameter list was measured on the stored bytes rather than assumed. When the
fold F is 1 the placement stops depending on TileK -- that is PlacedArrangement.layout_is_tile_free -- and the
registry's own rule ("the canonical minimum 32-byte run for the narrowest code plane: 256/bits") is exactly the
rule that forces F=1. This tool CHECKS that rather than restating it: if a registry row ever stops being
tile-free, the SCALE_FIRST and FULLY_QUANTIZED TileKs stop being servable by one file and the report says so.

THE EVIDENCE UNDER THAT, because the predicate's own docstring used to have none:
dev/fold_derivation/l115_artifact_tactic_code_slots.cu walks xplane::place_from_map's physical address layouts
and reports owner_diff, of which 0 is the "one resident artifact serves a larger tactic T" contract. Run at HEAD
2026-08-11 it is 0 on every cross-T row, INCLUDING the two-plane Q6_K A=128 -> T=256 F=1/1 row that this tool's
Q3/Q5/Q6 lines depend on. It builds in ~15 s and needs no device; prefer running it to trusting this paragraph.

WHAT THIS TOOL WILL NOT DO. It will not name an arrangement nobody can build. tools/pack_gguf.py refuses for the
same reason and the reason is the same sentence: a manifest naming an unbuildable arrangement reads as a
capability. Every cell here is taken from quactlize/schemes.py's status matrix, the shipping registry, or the
shipped config tables -- never from this file's own opinion.

    python3 tools/plan_offline.py                  every named model
    python3 tools/plan_offline.py --model A3B      one of them
    python3 tools/plan_offline.py --formats Q4_K,Q6_K
    python3 tools/plan_offline.py --check          exit non-zero if any invariant below fails (for CI)
"""
import argparse
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from quactlize import formats as F                      # noqa: E402
from quactlize import schemes as S                      # noqa: E402
from tools.pack_gguf import format_registry             # noqa: E402  -- PARSED, not mirrored

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------------------------------------
# THE NAMED SHAPES. From docs/BACKTEST.md section C and the dense band, cited per row so a number here can be
# traced to the measurement it came from rather than to this file. Prose is not an input: if a shape is not in
# BACKTEST it does not belong in this list.
class Tensor(collections.namedtuple("Tensor", "role n k experts rows_per_expert source")):
    @property
    def is_moe(self):
        return self.experts > 0


MODELS = {
    # qwen3-30B-A3B. FC1's N differs across the recorded configurations because they are different deployments of
    # the same model (expert count and top-k differ), not different measurements of one shape -- so all of them
    # are listed and each names its row.
    "A3B": [
        Tensor("FC1 (256 exp, top-8)",   512, 2048, 256, 128, "BACKTEST C1"),
        Tensor("FC1 (128 exp, top-8)",  1024, 2048, 128, 128, "BACKTEST C2"),
        Tensor("FC1 (dropless)",        1536, 2048, 128, 128, "BACKTEST C8"),
        Tensor("FC2 (dropless)",        2048,  768, 128, 128, "BACKTEST C9"),
    ],
    # The dense band this work has measured throughout.
    "dense-4096": [
        Tensor("dense W", 4096, 4096, 0, 0, "BACKTEST dense band, M=2048"),
    ],
}

#: The M values a deployment actually presents, per kind. Decode is one row per active expert for MoE and one row
#: for dense; prefill is the token count the band was measured at.
M_BANDS = {"decode": 1, "prefill": 2048}

#: The five GGUF k-quants are the formats with a shipping registry row. Anything else has no offline arrangement
#: to plan, which is a fact about the registry and not a judgement made here.
PLANNABLE = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")


# ---------------------------------------------------------------------------------------------------------------
def shipped_won() -> dict:
    """{table stem: {WON: count}} over the COMPILED-IN config tables.

    WON = TN / max(WN, 16), the warp-column count. quactlize/layouts.py:xplane records that the stored bytes of a
    placed plane change with this ratio -- measured on the stored bytes of place_derived, with (TN=64,WN=32) and
    (TN=128,WN=64) byte-identical and WON=2 vs 4 differing over about half the buffer. So it is a property of the
    ARTIFACT, and the number of distinct values reachable at run time is the number of artifacts a tensor needs.
    """
    out = {}
    for f in sorted((ROOT / "quactlize" / "include").glob("ppu_*_configs.inc")):
        seen = collections.Counter()
        for line in f.read_text().splitlines():
            m = re.match(r"\s*X\((.*?)\)\s*\\?\s*$", line)
            if not m:
                continue
            args = [a.strip().strip('"') for a in m.group(1).split(",")]
            if len(args) != 7:
                continue
            tn, wn = int(args[3]), int(args[5])
            seen[tn // max(wn, 16)] += 1
        if seen:
            out[f.stem] = dict(sorted(seen.items()))
    return out


def arrangement(row, tile_k):
    """The PlacedArrangement for one registry row at one TileK, or the reason there is none."""
    try:
        return F.PlacedArrangement(row["low_bits"], tile_k, row["high_bits"]), None
    except Exception as exc:                                        # pragma: no cover -- constructor is total
        return None, str(exc)


def folds(arr):
    """(F_low, F_high, tile_free) or an explanation. fold_for RAISES on an illegal (bits, TileK); that refusal is
    the answer, not something to swallow -- it means no artifact exists for that pairing."""
    try:
        return (arr.fold, arr.high_fold, arr.layout_is_tile_free()), None
    except Exception as exc:
        return None, str(exc)


def plan_tensor(t: Tensor, row: dict) -> dict:
    """Everything decided for one (tensor, format), with each conflict named."""
    q = F.QuantType[row["name"]]
    shape_big = S.Shape.GROUPED if t.is_moe else S.Shape.DENSE
    shape_dec = S.Shape.GEMV_MOE if t.is_moe else S.Shape.GEMV

    sf, fq = row["scale_first_tile_k"], row["fully_quantized_tile_k"]
    a_sf, e_sf = arrangement(row, sf)
    a_fq, e_fq = arrangement(row, fq)
    f_sf, ef_sf = folds(a_sf) if a_sf else (None, e_sf)
    f_fq, ef_fq = folds(a_fq) if a_fq else (None, e_fq)

    conflicts = []

    # CONFLICT 1 -- the two schemes want different TileK and the bytes are not TileK-free.
    if sf != fq:
        if f_sf and f_fq and f_sf[2] and f_fq[2]:
            pass  # both F=1: TileK does not reach the stored bytes, so one file serves both
        else:
            bad = [f"SCALE_FIRST TK={sf} F={f_sf[:2] if f_sf else ef_sf}",
                   f"FULLY_QUANTIZED TK={fq} F={f_fq[:2] if f_fq else ef_fq}"]
            conflicts.append(
                "TWO ARTIFACTS: the schemes want different TileK and at least one arrangement is not "
                "TileK-free, so the same bytes cannot serve both -- " + "; ".join(bad))

    # CONFLICT 2 -- WON is a config-row field, it changes the stored bytes, and no manifest field carries it.
    won = shipped_won()
    tbl = "ppu_grouped_configs" if t.is_moe else "ppu_dense_configs"
    reach = won.get(tbl, {})
    if len(reach) > 1:
        conflicts.append(
            f"WON IS UNRECORDED AND NOT UNIQUE: {tbl} ships {len(reach)} distinct warp-column counts "
            f"{reach} (WON = TN/max(WN,16)). layouts.xplane measured the stored bytes as WON-dependent, and "
            f"PlacedArrangement carries only (bits, tile_k, high_bits), so an artifact cannot say which WON it "
            f"was placed for and a reader cannot check. See the INVARIANTS section.")

    # The status of each cell, taken from the matrix rather than assumed.
    cells = {}
    for scheme in (S.Scheme.SCALE_FIRST, S.Scheme.FULLY_QUANTIZED, S.Scheme.DEQUANT_FIRST):
        for shape in (shape_dec, shape_big):
            try:
                cells[(scheme.name, shape.name)] = S.format_of(scheme, shape, q)
            except Exception as exc:
                cells[(scheme.name, shape.name)] = f"<{exc.__class__.__name__}>"

    routes = {}
    for band, m in M_BANDS.items():
        try:
            routes[band] = F.select_path(q, m)
        except Exception as exc:
            routes[band] = f"NO PATH ({exc})"

    # BYTES: elements / block.weights * block.block_bytes, times the expert count for MoE. NOT / group_size --
    # group_size is the SCALE group (32 for Q4_K, 16 for Q3/Q6) while a block is 256 weights, so dividing by it
    # inflated the answer 8x. Caught by checking one number by hand: Q4_K at N=512 K=2048 is 1,048,576 elements
    # = 4096 blocks x 144 B = 589,824 B per expert, and the first version printed exactly 8x that.
    blk = F.BLOCKS.get(q)
    growth = F.storage_growth(q)
    per_expert = (t.n * t.k // blk.weights * blk.block_bytes) if blk else None
    total = per_expert * max(t.experts, 1) if per_expert is not None else None
    return dict(tensor=t, qtype=q, row=row, sf=sf, fq=fq, f_sf=f_sf, f_fq=f_fq,
                ef_sf=ef_sf, ef_fq=ef_fq, conflicts=conflicts, cells=cells, routes=routes,
                block=blk, growth=growth, bytes_per_expert=per_expert, bytes_total=total)


# ---------------------------------------------------------------------------------------------------------------
def invariants() -> list:
    """The claims this tool's headline rests on, each as a check that can FAIL rather than a sentence.

    1. Every registry row is TileK-free at BOTH of its TileKs. This is what makes "one artifact per tensor" true;
       the registry's 256/bits rule is what forces it. If it ever fails, the report's conflict 1 fires and the
       operator has a storage decision to make -- so it must be checked, not asserted.
    2. The shipped placement pins ONE WON. unfused_weight_dequantize.hpp calls place_derived with TN=64, WN=32,
       i.e. WON=2. If that ever becomes two call sites with different ratios, an artifact serves neither.
    """
    out = []
    reg = format_registry()
    for row in reg.values():
        for label, tk in (("SCALE_FIRST", row["scale_first_tile_k"]),
                          ("FULLY_QUANTIZED", row["fully_quantized_tile_k"])):
            arr, err = arrangement(row, tk)
            if err:
                out.append((False, f"{row['name']} {label} TK={tk}: no arrangement ({err})"))
                continue
            fl, ferr = folds(arr)
            if ferr:
                out.append((False, f"{row['name']} {label} TK={tk}: {ferr}"))
            else:
                out.append((fl[2], f"{row['name']:<5} {label:<16} TK={tk:<3} F={fl[0]} F_hi={fl[1]} "
                                   f"tile-free={fl[2]}"))

    src = (ROOT / "quactlize" / "include" / "unfused_weight_dequantize.hpp").read_text()
    calls = re.findall(r"place_derived<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,",
                       src)
    wons = {int(tn) // max(int(wn), 16) for _b, _tm, tn, _tk, _wm, wn in calls}
    out.append((len(wons) == 1,
                f"shipping placement pins WON={sorted(wons)} over {len(calls)} place_derived call sites "
                f"in unfused_weight_dequantize.hpp"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None, help="one of " + ", ".join(MODELS))
    ap.add_argument("--formats", default=",".join(PLANNABLE))
    ap.add_argument("--check", action="store_true", help="exit non-zero if an invariant fails")
    args = ap.parse_args()

    reg = {r["name"]: r for r in format_registry().values()}
    want = [f.strip() for f in args.formats.split(",") if f.strip()]
    missing = [f for f in want if f not in reg]
    if missing:
        print(f"no shipping registry row for {missing}; the registry decides what is plannable, not this tool",
              file=sys.stderr)
        return 2

    print("INVARIANTS  (each is a check, not a claim)")
    inv = invariants()
    for ok, line in inv:
        print(f"  {'ok  ' if ok else 'FAIL'} {line}")
    bad = [l for ok, l in inv if not ok]
    print()

    if args.check:
        if bad:
            print(f"{len(bad)} invariant(s) failed", file=sys.stderr)
            return 1
        print("all invariants hold")
        return 0

    models = {args.model: MODELS[args.model]} if args.model else MODELS
    if args.model and args.model not in MODELS:
        print(f"unknown model {args.model!r}; have {list(MODELS)}", file=sys.stderr)
        return 2

    all_conflicts = []
    for mname, tensors in models.items():
        print(f"================ {mname}")
        for t in tensors:
            kind = f"MoE {t.experts} experts x ~{t.rows_per_expert} rows" if t.is_moe else "dense"
            print(f"\n  {t.role}   N={t.n} K={t.k}   {kind}   [{t.source}]")
            for fname in want:
                p = plan_tensor(t, reg[fname])
                f_sf = f"F={p['f_sf'][0]}/{p['f_sf'][1]}" if p["f_sf"] else f"({p['ef_sf']})"
                f_fq = f"F={p['f_fq'][0]}/{p['f_fq'][1]}" if p["f_fq"] else f"({p['ef_fq']})"
                free = "one artifact serves both TileK" if (p["f_sf"] and p["f_fq"]
                                                            and p["f_sf"][2] and p["f_fq"][2]) else "SEE CONFLICT"
                nb = p["bytes_total"]
                print(f"    {fname}  planes {p['row']['low_bits']}"
                      f"{'+' + str(p['row']['high_bits']) if p['row']['high_bits'] else ''} bit"
                      f"   gs={p['row']['group_size']}"
                      f"   SCALE_FIRST TK={p['sf']} {f_sf}"
                      f"   FULLY_QUANTIZED TK={p['fq']} {f_fq}"
                      f"   -> {free}")
                if nb:
                    per = f" ({p['bytes_per_expert']:,} x {t.experts} experts)" if t.is_moe else ""
                    blk = p["block"]
                    # TWO DIFFERENT RATIOS, and they were briefly conflated here. `compress` is native bytes over
                    # fp16 -- what the tensor costs on disk. storage_growth is the fractional INCREASE incurred by
                    # running the fp16-SCALE path instead of the native channel, which is the number the storage
                    # constraint is about. 0.281 and +11.1% for Q4_K; printing the second as "0.111x vs fp16"
                    # would have understated the file by 2.5x.
                    comp = blk.block_bytes / (blk.weights * 2)
                    g = f"   fp16-scale path would add {p['growth'] * 100:.1f}%" if p["growth"] else ""
                    print(f"           native bytes {nb:,}{per}   = {comp:.3f}x fp16{g}")
                else:
                    print("           native bytes: unknown")
                print(f"           route  " + "   ".join(f"{b}(M={M_BANDS[b]}): {r}"
                                                         for b, r in p["routes"].items()))
                for c in p["conflicts"]:
                    all_conflicts.append((mname, t.role, fname, c))

    print("\n================ CONFLICTS  (each one is a storage decision, not a bug report)")
    if not all_conflicts:
        print("  none: every planned tensor is served by exactly one artifact")
    else:
        seen = set()
        for mname, role, fname, c in all_conflicts:
            key = (c.split(":")[0], fname if "TWO ARTIFACTS" in c else "")
            if key in seen:
                continue
            seen.add(key)
            where = f"{mname}/{role}/{fname}" if "TWO ARTIFACTS" in c else "every tensor below"
            print(f"\n  [{where}]\n    {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
