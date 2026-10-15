# Small-M retry and rule-coverage review

Scope: offline ZW810 component selection, not production promotion or model TPOT.
Large-M work is paused at the user's request. The original receipts remain
immutable; this review does not rewrite an old `INCOMPLETE` as a device PASS.

## Returned evidence

- Original campaign: `kpack-rule-campaign.OXZL69AV`, source `f38a53fb`.
- Compact review SHA256:
  `4d78e5c015b790e96cc6a204eb86c542437a44b5e959823d5f5605f36b928dab`.
- Six-point retry archive: `kpack-rule-campaign.OXZL69AV.retry6.1792027993167069813.tgz`.
  SHA256: `0fce9389c80ebb4412e714e5e4f8f75dd12d5dc90ced59fe99eebdf2b6c6d782`.
- Both use authority SHA256
  `e4ab2c7875749cf82546063eee0abe42ed7625074f928ca2241fcde880e0b145`:
  aiswu52, ordinal 0, PCI `0000:08:00.0`, 72 CUs, verified 64 MiB L2.

All six retries pass independent correctness and recomputed 6x5 alternating
component confirmation. They were blocked by the finalist-count limit, not a
numerical error. The retry retained 25–34 competitive arms per point and took
149.94 seconds. Journal completion is now 1114/1114; seven selection flags remain
in the raw summary.

Six flags are inconclusive, near-tied dense selections. Retain their original
recipes: largest median gap to the fastest measured finalist is 0.078814%;
largest paired-round gap is 0.285805%. This admits no unstable challenger and
does not change the original verdict function or correctness tolerance.

The remaining flag is Q8 indexed BF16, M8/N1024/K2048, E256/top8/ch1/repeat4.
Original TM16/DN64/S4: 96.052502 us; measured TM8/DN32/S4 control: 87.293889 us
(-9.118568%). These are sums of isolated prepare, producer and finish timings,
not measured model latency. The producer accounts for almost all the gap.

## Rule correction and bounded follow-up

R24 withheld the entire target N/K family. That removed every Q8 indexed TC
donor, so its blind rule pool contained only SIMT. R25 fills an entirely absent
TC family using same-format SF geometry from another operator and another N/K.
No target winner or old latency is passed to generation. FQ, packed-A and
grouped-only geometries do not cross this boundary. The target endpoint,
metadata precision, resources, correctness and timings remain independently
validated. Already-covered TC families are unchanged.

Host replay over all 1114 registered points removes no previous candidate and
changes only 12 Q8 indexed points at N1024/K2048: M1–7 plus five M8 router profiles.
The formerly missed control is generated from a dense N8192/K2048 donor. This
is post-hoc coverage repair, not independent proof of R25 generalization.

Use a fresh result directory with `POINT_FILTER` set to
`{"qtype":8,"operation":"indexed","n":1024,"k":2048}`. Reuse the existing
resource/vector caches, preserve all old timing receipts, and confirm on the
same idle physical card. `COMPLETE_SMALLM_SUBSET` is deliberately distinct from
full-cohort completion. Do not resume the source-sealed R24 campaign using R25.
Production tables and original-wiring model A/B remain separate work.
