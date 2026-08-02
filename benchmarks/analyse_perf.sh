#!/usr/bin/env bash
# PAIRED CONTRASTS BETWEEN ANY TWO VARIANTS, with intervals, from an existing perf_samples.txt.
#
# WHY THIS IS SEPARATE FROM do_perf. The table do_perf prints compares every variant against base, which answers
# "is there a tax". The questions that decide what to DO next are contrasts between two non-base variants:
#
#     pack vs packnop                              the decode ARITHMETIC alone
#     (packsplit/pack) / (splitnop/packnop)        the placement effect with the duplicated-read cost subtracted
#
# Those cannot be read off the vs-base column. Dividing two point estimates gives a number with no interval, and the
# interval is the whole point -- the first analysis of this run computed +3.6% and +3.3% by hand from the medians
# and could say nothing about whether either was established. A contrast has to be formed WITHIN each block and
# aggregated across blocks, exactly like the vs-base ratio.
#
# Reading a file rather than re-running is deliberate: 140 observations already exist and cost twenty minutes of box
# time. An analysis that requires re-measurement to ask a second question is an analysis that will not be asked.
#
#   ./analyse_perf.sh [samples-file]        default ~/ab/perf_samples.txt
#
# Input format, one observation per line, as do_perf writes it:   <block> <variant> <us>
set -uo pipefail
SAMPLES="${1:-${OUT:-$HOME/ab}/perf_samples.txt}"
[ -f "$SAMPLES" ] || { echo "no samples at $SAMPLES -- run './run_batch.sh perf' first"; exit 1; }

echo "== paired contrasts from $(wc -l < "$SAMPLES") observations in $SAMPLES =="
echo

awk '
  # Mean of the per-block log ratios, with a normal-approximation interval. Multiplicative quantities are symmetric
  # in log space; a linear interval on a ratio is wrong on the small side.
  function contrast(numA, denA, numB, denB, label,    b, n, lr, mu, sq, sd, se, lo, hi, flag) {
    n = 0; mu = 0; sq = 0
    for (b in blk) {
      if (!((b" "numA) in t) || !((b" "denA) in t)) continue
      lr = log(t[b" "numA] / t[b" "denA])
      if (numB != "") {                       # a difference of two ratios: subtract in log space
        if (!((b" "numB) in t) || !((b" "denB) in t)) continue
        lr -= log(t[b" "numB] / t[b" "denB])
      }
      n++; mu += lr; sq += lr*lr
    }
    if (n < 2) { printf "   %-42s  (n=%d, no interval)\n", label, n; return }
    sd = sqrt((sq - n*(mu/n)*(mu/n)) / (n-1)); mu = mu/n; se = sd/sqrt(n)
    lo = exp(mu - 1.96*se); hi = exp(mu + 1.96*se)
    flag = (lo <= 1.0 && hi >= 1.0) ? "  <-- includes 1.0: not established" : ""
    printf "   %-42s %+7.1f%%   %.3f .. %.3f  (n=%d)%s\n", label, (exp(mu)-1)*100, lo, hi, n, flag
  }
  { t[$1" "$2] = $3; blk[$1] = 1; v[$2] = 1 }
  END {
    printf "   %-42s %7s   %s\n", "contrast", "effect", "95% CI"
    print "   " substr("--------------------------------------------------------------------------------", 1, 76)
    if (("pack" in v) && ("base" in v))     contrast("pack","base","","","pack / base  (the native-format tax)")
    if (("packnop" in v) && ("base" in v))  contrast("packnop","base","","","packnop / base  (transport + stores + barrier)")
    if (("pack" in v) && ("packnop" in v))  contrast("pack","packnop","","","pack / packnop  (the decode ARITHMETIC alone)")
    if (("packfuse" in v) && ("pack" in v)) contrast("packfuse","pack","","","packfuse / pack  (the STORE-conflict fix)")
    if (("swz" in v) && ("base" in v))      contrast("swz","base","","","swz / base  (swizzle: removed 0 conflicts)")
    if (("bdqnop" in v) && ("base" in v))   contrast("bdqnop","base","","","bdqnop / base  (baseline int4 dequant, upper bound)")
    if (("packsplit" in v) && ("splitnop" in v))
      contrast("packsplit","pack","splitnop","packnop",
               "(packsplit/pack)/(splitnop/packnop)  placement")
  }' "$SAMPLES"

cat <<'EOT'

   HOW TO READ THE DECOMPOSITION
     pack/base      is the tax. If its interval excludes 1.0, a tax exists.
     packnop/base   is the part that is NOT arithmetic: the raw 16-byte staging read, the two explicit fp16 plane
                    stores, and the slot they occupy between cp_async_wait and the publishing barrier.
     pack/packnop   is the arithmetic. If this is the small half, optimising the decoder cannot reach parity and
                    the publication shape is what costs.

   packfuse/pack  IS THE ONLY CONTRAST THAT PRICES A BANK CONFLICT HERE, and it must be read against pack rather
                  than base: base has no decoder stores at all, so packfuse/base would fold the whole native-format
                  tax into a number about store conflicts. The interval to beat is 1.0 -- pack's +73,728 store
                  conflicts should go to ~0 while nothing else moves. Shared bytes are unchanged, so an occupancy
                  explanation is unavailable in advance, which is what makes this contrast clean.

   THE DECOMPOSITION IS PROVISIONAL WITHOUT acu. packnop consumes only u[0], so the loads of the other three words
   may be eliminated entirely. If its raw shared-load and store counts do not match pack's, pack/packnop is the
   difference between two different kernels and not an ablation. Capture acu once for pack and once for packnop --
   for the counters, not for timing -- and check that only decoder ALU disappears.

   THE PLACEMENT CONTRAST is a difference of differences on purpose. packsplit/pack alone cannot separate "splitting
   buys nothing" from "it buys less than the duplicated 16-byte read costs"; splitnop exists to price that read
   without the decode, so dividing removes it. Above 1.0 means splitting is actively worse.
EOT
