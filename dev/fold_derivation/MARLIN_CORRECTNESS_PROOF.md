# Dense Marlin scheduler correctness boundary

This note separates the three claims that are easy to blur together:

1. the scheduler algebra for every positive, representable integer problem;
2. exhaustive closure of the repository's finite deployment space;
3. the device-only named-barrier ordering/lifecycle check.

The first two are local proofs.  They do not need a PPU.  The third is the
only Marlin correctness question left to the box.

## Proposition A: exact-once composition

Let

```text
Q  = ceil(M/TM) * ceil(N/TN) * L
Kt = ceil(K/TK)
T  = Q * Kt
B  = blocks_per_cu (default 1)
G  = max(Q, CU*B)
I  = ceil(T/G)
```

CTA `b` owns the half-open stripe `[b*I, min((b+1)*I,T))` in the
flattened `(q,k_tile)` space, with K as the fast coordinate.  Adjacent CTA
intervals do not overlap and their union is `[0,T)`.  Production
`fetch_next_work` only clips one such interval at the next q boundary; its
cursor advances to the clipped end.  Splitting an interval this way neither
adds nor removes a cell.  Consequently every `(q,k_tile)` appears exactly
once.

The production q decode is N-fast:

```text
n = q % Nt
q_m = q / Nt
m = q_m % Mt
l = q_m / Mt
q = (l*Mt + m)*Nt + n
```

The lock ID is this same global q, so two distinct output tiles cannot alias a
lock.  The proof includes continuation across N, M and L; a stripe is not
required to end at any of those boundaries.

The exhaustive deployment proof below fixes the shipping/default `B=1` lane.
In that lane the scheduler-owned launch protection follows directly.  If `Q >= CU`, then
`G=Q`, `I=ceil(Q*Kt/Q)=Kt`, and CTA q owns exactly the complete K range of
output q.  Thus every `slice_count` is one and the handoff count is zero.
When `Q < CU`, the scheduler is in the stripe regime, but that does not imply
an actual split.  Since `G=CU` and `Q<CU`, `I<=Kt`; therefore

```text
no split
<=> I = Kt
<=> ceil(Q*Kt/CU) = Kt
<=> Q*Kt > CU*(Kt-1)
<=> (CU-Q)*Kt < CU
```

The final inequality is strict.  At `Q=64,Kt=8,CU=72`, the left side is 64
and the class is unsplit; at `Q=63` it is 72 and the class splits.  In the
unsplit case `active_blocks=Q`: each active CTA owns one complete output tile
and the remaining `CU-Q` launch slots are idle.  This is ceil quantization of
the uniform stripe length, not a hole in the default-B1 `G=max(Q,CU)` policy.

The complete current default-B1 census is:

| Mt | Nt | L | Kt | CU | Q | G | I | active CTA | raw multiplicity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 1 | 8 | 72 | 64 | 72 | 8 | 64 | 2,034 |
| 2 | 32 | 1 | 8 | 72 | 64 | 72 | 8 | 64 | 204 |
| 4 | 16 | 1 | 8 | 72 | 64 | 72 | 8 | 64 | 123 |
| 8 | 8 | 1 | 8 | 72 | 64 | 72 | 8 | 64 | 90 |
| 16 | 4 | 1 | 8 | 72 | 64 | 72 | 8 | 64 | 72 |
| 32 | 2 | 1 | 8 | 72 | 64 | 72 | 8 | 64 | 142 |

These are six `(Mt,Nt)` factorizations of the same scheduler geometry, not six
different mechanisms.  Their raw multiplicities sum to 2,665.  L133 checks
the iff for every raw `Q<CU` tuple, prints every unique class, and requires the
printed multiplicities to match the independently generated manifest.

On the L133 exact fixture each logical cell contributes one of `{-1,0,1}`.
Every partial sum has at most 400 terms and is exactly representable in FP32;
the periodic full result is in `{-1,0,1}` and is exactly representable in
FP16.  The pass criterion is fixed before execution: raw integer equality of
the scheduled sum and the DP sum.  No ULP or tolerance is selected after the
result.  Therefore, for this fixture, exact-once coverage implies numerical
identity with DP.

### Explicit blocks-per-CU experiment

The host scheduler argument may explicitly request `B>1`, after the exact
kernel's runtime occupancy has bounded it.  The lowered device `Params`,
workspace layout, global-q lock IDs, and cooperative ABI do not change; only
`G/I/active` and the resulting peer chains do.  Omitting the argument is
identical to explicit `B=1`.

L126 exhausts all 1,024 `(q,k_tile)` cells for the decode anchor
`Q=32,Kt=32,CU=72` at `B={1,2,4,6}`.  It requires exact-once ownership,
reverse peer IDs, globally unique q locks, and the exact launch/decomposition
ladder.  Device runs additionally require eight stable bit-exact launches on
the same workspace for every rung; that is the memory-order portion the host
proof cannot establish.

### Exhaustive finite deployment closure

The runtime integer interface is unbounded, so “exhaustive” must name a finite
authority.  `ci/check_dense_marlin_exhaustive.py` derives, rather than copies,
the repository deployment domain:

- all 4,790 committed i4/i2/i1 dense tactics;
- all 66 dense shapes exported by `benchmarks/fixtures.py` and
  `benchmarks/workloads.py`, plus A0 and the decode anchor parsed from their
  executable sources (68 shapes total);
- both observed PPU CU counts, 32 and 72;
- one additional `Mt=2,Nt=2,L=2,Kt=31,CU=9` construction per tactic to force
  continuation across N, M and L.

The raw Cartesian domain and its closure are:

```text
deployment tuples                         651,440
cross-N/M/L construction tuples             4,790
raw tuples scanned / remaining            656,230 / 0
distinct production Params checked          2,815 / 2,815
protected / stripe-regime classes           2,465 / 350
actual-split / Q<CU-ceil-unsplit classes       344 / 6
production work segments                42,231,743
logical (q,k_tile) cells             2,632,768,288
output sums checked                    42,215,890
handoffs                                   15,853
observed cross N / M / L               4,717 / 767 / 1
```

The 4,790 cross-L raw tuples intentionally collapse to one scheduler-visible
Params class after tile-count lowering.  The raw count proves every committed
tactic was admitted; the unique count prevents repeating an identical walk
from being mistaken for broader algebraic coverage.

Eight independently compiled mutations must turn the proof red: dropping the
output-tile floor from G, flooring I, omitting q-boundary clipping, losing K-fast order,
swapping the M/N decode, stopping after the first segment, using an N-local
lock, and reversing the peer-order convention.

## Proposition B: production code consumes the same algebra

An independent host reimplementation would not close the expert-pitch class
of bug.  L134 therefore starts from the shipping dense type:

```text
Cfg<128,16,128,128,16,32,3>::MarlinGemm::GemmKernel
```

The named kernel's raw `M/N/K/L` shape passes through its actual
`scheduler_problem_shape` and the scheduler's actual
`make_params_for_problem_shape`.  The same production helpers then supply the
kernel's output coordinate, K-tile coordinate, FP32 workspace element offset,
and global-q lock.  The concrete compile-time witnesses include:

- classic `16x2048x2048`, yielding `Q=16,Kt=16,CU=20,G=20,I=13`;
- decode `1x4096x4096`, yielding `Mt=1,Nt=32,Kt=32,CU=72,G=72,I=15`;
- residue/batch `17x129x3841xL2`, proving ceil-div lowering and continuation
  across N, M and L.

The following units are part of the checked seam:

| Value | Unit at production consumer |
|---|---|
| `M_idx/N_idx/L_idx` | output-tile ordinal |
| `K_idx`, `k_tile_count` | K-tile ordinal/count, never scalar K, code or byte |
| `output_tile_idx`, `lock_idx` | global output q |
| reduction workspace offset | `q * TM * TN` FP32 elements, not bytes |
| flattened core cursor | `(q,k_tile)` cells only; never a sub-byte pointer |

The broader sub-byte unit inventory is in
`dev/fold_derivation/SUBBYTE_UNIT_AUDIT.md`.

L134 emits a nonempty NVCC PTX entry with runtime raw M/N/K/L, CU and
`blockIdx.x`.  The entry follows the real scheduler object's fetch loop and
retains the quotient/remainder decomposition.  A 35-field device constant
pins concrete values from the same production type.  A deliberately wrong
`I=12` assertion must expose the actual 13, and a raw-shape bypass that treats
scalar M/N/K as tile ordinals must fail.

This PTX is deliberately limited to the pure integer/helper seam that local
NVCC can compile.  It is not PPU MMA ISA and does not claim to inspect the
mixed-input mainloop body.  The source contract separately requires the real
PPU named kernel to call those same output/K/workspace/lock helpers; replacing
any helper with duplicated arithmetic is a contract failure.

## Device-only boundary: ordered handoff

The algebra cannot prove PPU named-barrier memory ordering or that the final
peer resets a lock before a later launch reuses the same workspace.  The
Marlin box arm therefore uses its own invocation's
`ORDER-INDEPENDENT+FP16-EXACT` fixture and runs eight additional launches in
one process with the same initialized Gemm and workspace.  It poisons only D;
there is no external lock reset or workspace initialization between launches.
Every launch must have raw bitdiff zero against that invocation's golden, and
the position/value fingerprints must be identical across all eight launches.

That criterion is fixed in advance.  Exactness evidence is carried by the
same `initialize()` return value as the arm being judged; a log line from a
gate, A0, DP, Stream-K or another shape cannot be substituted.  Timing remains
outside this correctness loop and still uses the existing 20 independent
event pairs.
