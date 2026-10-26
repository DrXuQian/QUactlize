# Q8 load windows: M and geometry can reverse a gain

Use this case when transferring a register-liveness/load-window optimization
across M or thread mappings. Scope: PPU-ZW810, Q8 K-pack2, dense S1, F32
endpoints, separate F16/BF16 activation rounding, unchanged FP32 dot order.
Measured producer source `4bda7f6`, Box compiler SHA256
`0ecb9407820ccb7852bb0b2fdce6f50f22a189eb2f38e64e92d4cfd5423b2983`.

Evidence: `review-lite-1792469872546474625.tar.xz`, SHA256
`a008a15b4850e8e94d15fed7bd9af78d7cb4f76ee9172dabd24a888ef0c74d93`;
details in `docs/measurements/q8-address-window-20261020.txt`.
280 numerical pairs, 160 rotating event pairs, 288 single-launch ACU raw
CSVs. All captured core clocks are 1.7GHz. The event phase did not
independently record clocks; its microseconds are not ACU cycles.
The lightweight upload includes hashed exports/receipts, not full acurep
or DSOs. Do not describe reviewing those exports as reimporting the reports.

The candidate loads half a Q8 group before its corresponding arithmetic,
instead of hoisting both halves. No fewer weight bytes or different format.

| Workload/config | Half-window cycle change vs matched original |
|---|---:|
| M1, N4096, K2048, C8/W4/P4 | +13.95% |
| M8, N4096, K2048, C8/W4/P4 | −25.64% |
| M1, N4096, K2048, C4/W8/P4 | −14.83% |
| M1, N4096, K4096, C8/W4/P4 | +27.79% |
| M1, N4096, K4096, C4/W8/P4 | −11.81% |

Each listed result has three matching-frequency pairs with the same sign.
M1 gains cannot be inferred from M8, nor can one M1 loss reject the
optimization at another mapping. Preserve the best original config;
matched-control improvement alone does not prove a better selected recipe.

For M1/N4096/K4096/C4, registers fall 74→64, but occupancy stays about43.3%.
DRAM is essentially unchanged at17.87MB, L1/L2 traffic about73MB, and vector
loads remain77,824. VMEM-busy per-issue ratio falls3.086→1.347 while
load-dependency ratio rises0.079→1.636. This supports a scheduling/liveness
tradeoff, not improved coalescing, reduced HBM volume, or an occupancy gain.
By contrast, the M8/N4096/K2048/C8 case raises occupancy63.17→84.67%;
do not reuse that explanation for the M1 case. Ratios are not time shares.

Counterexample to using only profiler scores: M1/N512/K2048/C4 half-window
reduces ACU cycles4.07% but increases rotating-event latency12.40%.
Different measurement protocols must not substitute for one another.
Do not admit a cold regression or invent its cause from the disagreement.
For close finalists, the next discriminating check is the intended cold
selection protocol with contemporaneous controls and clock evidence.
Keep both candidates if event and cycle rankings disagree; do not repeat
already-complete ACU captures solely to search for a preferred result.

The admission boundary is the measured shape/precision/M/configuration.
This case does not establish performance for MoE, other Split-K factors,
all M2..7, a global selector, or model TPOT.

Implementation-binding follow-up: the later full small-M refresh admitted
27 exact dense/F32/F16/S1 keys, using four different bodies across seven
body/geometry instantiations. See `docs/measurements/q8_refresh_binding_20261020.txt`.
Geometry alone is not the identity of these winners. Keep the body in host
explain, package capability checks and the next sweep's incumbent recipe;
otherwise a matching tile can silently run or benchmark the old body.
These27 exact updates also exposed shared bucket donors: preserve the
previous donor explicitly when the new body has only exact-shape evidence.
The pre/post host selector audit checked12,294requests and changed only27.
All seven new production native bodies matched the corresponding measured
source under the same local compiler, with zero reported stack. This proves
the local port, not binary equality to another SDK or model-level speedup.

## CTA-local tokens: change reuse, then re-evaluate the ranking

Q6 dense/F32/F16, N1024/K5120, PPU-ZW810 SDK2.1.1, candidate `724b1ed2`:
moving M2–8 into one CTA shared packed loads, metadata exchange and code
reconstruction without changing each row's FP32 dot order. Asys isolated
producer-plus-reducer means for M5/6/7 fell from the best old TC/SIMT
18.984/19.013/19.043us to13.128/13.833/14.437us. Changing the selector alone
could not have produced that gain: the old per-row SIMT implementation lost.
M8 selected C8/P4, not C4/P8. Local C4/M8 codegen reported28B stack versus
zero for C8; this is a liveness hypothesis, not a Box ACU causal proof.
Details, source-bound archives, completed M4 and native comparison are in
`docs/PPU_Q6_TOKEN_CTA.md`.

Derive traffic per CTA AND across tokens. Sharing across M removes repeated
requests/unpack; it does not repair existing within-warp plane duplication or
guarantee a proportionate HBM-byte reduction. Useful FMA and A traffic still
scale with M. Keep C4/C8, actual reducers and current TC controls. Generic
Q2–Q5 instantiations retain their different group widths/high maps; Q8 needs
its scale-only fast-code path. These extensions were only host/codegen-checked
at this checkpoint (`docs/PPU_TOKEN_CTA_FORMATS.md`), not device-admitted.
Compare new implementations under the requested timing protocol, and never
carry Q6's speedup or old source identity into unmeasured formats or MoE.
