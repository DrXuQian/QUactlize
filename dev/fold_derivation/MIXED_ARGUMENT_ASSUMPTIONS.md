# Mixed-input argument assumptions

This audit records places where the mixed-input collective API declares an
input `X`, while the implementation either uses a different value `Y` or only
supports a narrower format than the type suggests.  Line references and
statuses are anchored at commit `8e4196c`; later remediation should update the
status rather than erase the original mismatch.

The commit that introduces L128 fixes the two rows marked **L128 FIXED** below:
outer `dA` base selection and logical-N metadata residue. Their cells retain
the `8e4196c` expression so the discovery and planted negative remain auditable;
production now consumes the shared helpers named in L128.

L129 closes the remaining admission gaps.  The three collectives expose one
shared `can_implement(problem, arguments)` contract, and every dense/grouped,
DP/persistent/Stream-K/Marlin kernel wrapper delegates to it.  A public query
and the corresponding launch therefore cannot disagree about these format
restrictions.  L129's aligned arms and deliberately perturbed arguments are
pure-host and constructive; no device result is inferred.

The three shipping collective families are abbreviated below as:

- **ordinary** — `quactlize_mma_mixed_input.hpp`
- **fold** — `ppu_mma_aiu_fold.hpp`
- **2-plane** — `ppu_mma_aiu_mixed_input_2plane.hpp`

Status terms are deliberately distinct:

- **FIXED**: the declared value reaches the implementation and has a
  constructive regression test.
- **LATENT BUG (public guard)**: the lower layer is wrong for part of its
  declared domain, but every audited public adapter currently rejects or
  avoids that part.  Relaxing the public guard would activate the bug.
- **FORMAT RESTRICTION / MISSING FAIL-CLOSE**: the canonical artifact is
  intentionally narrower than the generic argument type.  The supported case
  is not wrong; accepting a noncanonical value and silently ignoring it is.
- **INTENTIONAL ABI RESTRICTION**: the shared representation is explicit and
  there is no separately declared value to ignore.

## Audit table

| Parameter | Declaration | Actual use (or none) | `Y` actually used | When `X != Y` | Current impact | Constructive falsifier + independent anchor | Status at `8e4196c` |
|---|---|---|---|---|---|---|---|
| `dS` | `Arguments::dS` in ordinary `:776`, fold `:415`, 2-plane `:524` | `Params::dS` is populated and `detail::make_metadata_tile` consumes it for both S and Z (ordinary `:795,919,942`; fold `:432,531,542`; 2-plane `:553,681,704`) | Caller-provided MN-major metadata stride | Before `8e4196c`, any padding or non-tight expert pitch differed from the reconstructed tight `(1, N, N*scale_k)` layout | Fixed by `8e4196c` (`Honor caller metadata strides`). G5 itself passes a tight stride; its changing high-expert symptom was later reproduced as neighboring bytes reached through an independent B OOB, so it is not evidence of a remaining `dS` remap | `l127_metadata_stride.cu`: hold pointer, shape, and payload fixed; change tight `dS` to a padded `dS` and require addresses/read values to change. Independent anchor is the explicit 64-bit formula `n + g*dS_g + e*dS_l` with unique per-coordinate tags | **FIXED** |
| Zero-plane stride (`dZ`) | No independent `dZ`; `ptr_Z` is declared beside `dS` | Z is tiled with the same `Params::dS` as S | S and Z share one metadata layout | A caller needs independent S and Z pitches | No ignored field exists, but the ABI cannot express independently padded planes. Current artifacts use a shared layout | Exercise a shared padded `dS` and require both S and Z addresses to follow it. There is intentionally no “independent `dZ`” positive test until the ABI gains that field | **INTENTIONAL ABI RESTRICTION**; add `dZ` only with an ABI change |
| Outer A base versus `dA` | `Arguments::dA` / `Params::dA` in all three families | At `8e4196c`, the inner tensor used `dA`, but the expert/batch base was `ptr_A + a_row_off*K` (ordinary `:878-880`; fold `:501-503`; 2-plane `:637-639`). L128 now routes all three through `mixed_a_expert_base` | Historical Y: tight row pitch `K`; ragged base `row_offsets[e]*K`, uniform base `e*M*K` | `stride_M != K`, `stride_L != M*stride_M`, or a future K-slice carries a full leading dimension different from local `K` | The latent pointer-semantics defect is fixed without changing tight shipping addresses | L128 gives rows and experts distinct padding/tag regions. Expected base is `row_offsets[e]*stride_M` (ragged) or `e*stride_L` (uniform); planting `offset*K` reds on 4/5 experts in both arms | **L128 FIXED** (was latent at `8e4196c`) |
| Runtime `group_size` versus static schedule group size | Runtime `group_size` is accepted and copied into Params | Runtime value derives `scale_k`/reload quantities, while static schedules choose the scale group with `DispatchPolicy::StaticGroupSize` (ordinary around `:854,1281`; fold `:477,899`; 2-plane `:613,1170`) | Two different group sizes can govern allocation/address extent and per-K group selection | A static schedule is instantiated for one group size but receives another runtime value | Public dense/MoE dispatch already selected or guarded the matching schedule; L129 now rejects the mismatch at the collective boundary before either direct or wrapped launch | Exhaust `StaticGroupSize={-1,0,16,32,64,128}` against runtime values. `0` is runtime, `-1` is per-column and therefore requires `g=K`, positive values require equality; `ScaleTileK=ceil(TileK/g)` is checked independently | **L129 FIXED** (was a latent validation bug) |
| Logical-N residue for metadata in fold / 2-plane | Logical problem `N`, `TileN`, and `Fold` are all known | At `8e4196c`, fold `:536` and 2-plane `:686` used `N - size<0>(gB)*n_coord`; `size<0>(gB)` is physical `TileN/Fold`. L128 now routes all three families through `mixed_logical_n_residue` | Historical Y: physical `TileN/Fold` advanced a predicate that lives in logical-N coordinates | `Fold > 1` and the last N tile is partial. Example: `TileN=64`, `Fold=2`, `N=96`, `n_coord=1`: expected residue is 32, old result is 64 | The latent OOB predicate is fixed; the public 256-aligned domain remains unchanged | L128 exhausts `Fold={1,2,4}`, N=1..129. Independent anchor is `N - n_coord*TileN`; the old physical-width formula reds on 132/585 cases | **L128 FIXED** (was hidden by public guard) |
| Interleaved `dB` | Generic B stride `dB` is accepted | The interleaved branches synthesize canonical `make_stride(...)` layouts (ordinary `:1224`; fold `:852`; 2-plane plane 1 `:1054`) | Canonical interleaved byte layout and expert pitch `N*K*bits/8` | Caller supplies row padding, a different expert pitch, or any otherwise noncanonical `dB` | Shipping adapters were already canonical. L129 rejects every noncanonical marker in the collective instead of accepting and ignoring it | Hold pointer/shape fixed and perturb each `dB` component; all three arms reject. Independent anchors are the explicit `N*K*bits/8` byte pitch and the existing xplane placement roundtrip | **L129 FIXED** (format remains intentionally canonical) |
| Noninterleaved sub-byte `dB` outer pitch | `dBL` is in logical B elements and `ptr_B` is a typed int4/int2 pointer | Historical ordinary/fold/2-plane code called generic `make_gmem_ptr(ptr_B)` and sliced L before entering the AIU mix tensor | Raw C++ `sizeof(Element)` pointer arithmetic: one byte per logical sub-byte element | `bits < 8` and `L > 1`; G5 int4 advanced 8192 bytes for an 8192-code pitch whose physical size is 4096 bytes | This was an active grouped correctness bug: low experts read `2e`, and expert 128 began OOB. All four noninterleaved plane seams now call `mixed_subbyte_l_slice`, which slices with `subbyte_iterator` and only then converts the byte-aligned base back to the AIU raw pointer. Interleaved byte-pitch branches remain unchanged | L130 instantiates the exact CuTe overload: historical raw delta 8192 B versus explicit subbyte delta 4096 B. Its retained-observation replay produces B 1→2 / 3→6 and zero 129→85 / 190→126 before the fix; source plants reject restoring the generic overload or bypassing any collective seam | **FIXED** (active incident, not a format restriction) |
| Plane-2 `dB2` / `dB2_valid` | 2-plane arguments expose `dB2` and a validity flag | Noninterleaved code uses `dB2_valid ? dB2 : dB` (`:1130`); interleaved plane 2 synthesizes its layout (`:1115`) | Canonical plane-2 interleaving; fallback to `dB` only when plane layouts coincide | Interleaved plane 2 has a noncanonical pitch, or `dB2_valid=false` while the two planes have different folds/layouts | L129 rejects both contradictions. Noninterleaved `dB2` remains arbitrary and is demonstrably consumed: perturbing it changes the independently calculated source address | Vary all three `dB2` components; interleaved rejects. With interleave off, the same perturbation remains legal and changes `n*dB20+k*dB21+e*dB2L` | **L129 FIXED**; noninterleaved path verified consumed |
| Packed `ptr_S`, `dS`, and `ptr_Z` semantics | The generic argument names describe half scale/zero tensors | Packed-scale paths reinterpret `ptr_S` as packed raw units; `ptr_Z` is not a separate input and generic `dS` is only a logical tight-layout marker | Format-defined packed-unit byte layout | A caller treats the generic fields as ordinary half S/Z metadata, supplies `ptr_Z`, or expects arbitrary `dS` to control packed-unit addressing | Production already passed raw units/null. L129 rejects non-null `ptr_Z`, non-tight `dS`, incomplete group/tile/unit extents; the packed Q4 test now supplies the same null contract | Perturb each `dS` component or provide `ptr_Z`; all reject. The independent format anchor remains the GGUF pack/decode roundtrip | **L129 FIXED** (intentional ABI overload, now fail-closed) |
| Divisibility of interleave/fold/packed extents | Runtime `N`, `K`, folds, interleave factor, packed-unit geometry, and sub-byte outer strides are declared | Several layouts use integer division: `K/kCon`, `N/Fold`, ordinary packed `scale_k/ScaleTileK`, 2-plane packed tile/unit counts, and the noninterleaved AIU seam converts a sliced sub-byte base back to a raw byte pointer | A canonical artifact whose extents divide every format quantum exactly and whose per-expert bit offset is byte-aligned | `K % kCon != 0`, `N % Fold != 0`, metadata groups do not divide packed groups-per-unit, or `dBL*bits` / `dB2L*bits` lands between bytes | Public adapters already guarded their shipped shape domain. L129 moves the same property to the collective boundary so direct callers cannot enter a truncated layout or trip `raw_pointer_cast` after a fractional-byte L slice | Every aligned arm accepts; `+1` fold/interleave residues, packed group/tile/unit tails, and low/high fractional-byte outer pitches reject. Expected answers are explicit modulo equalities | **L129 FIXED** (lower-level fail-close) |

## G5 scope and non-conclusions

The G5 ID probe is deliberately narrower than this table, and its first
interpretation was wrong:

- `q == 8` zeros the B contribution only if B stays in bounds.  The raw-pointer
  outer pitch made expert 128 start one-past the 1 MiB B allocation, so the
  high-expert zero arm was also reading B codes from neighboring allocations.
- Its S/Z layout is tight, its static and runtime group sizes match, it uses the
  noninterleaved path, and all relevant extents divide their format quanta.
- Consequently, the observed high-half values were not a stable metadata
  mapping and cannot be used to infer a remaining `dS` defect.

L130 now includes the production seam that its first version skipped.  The
first model manually converted logical codes to physical bytes, so it proved
the intended byte map while never instantiating the real
`make_gmem_ptr(typed_int4_pointer)` overload.  The corrected oracle does both:

- the historical raw CuTe slice advances 8192 bytes, while an explicitly
  subbyte-aware slice advances the 4096-byte artifact pitch;
- below expert 128 that reads payload expert `2e`, reproducing B 1→2 and 3→6;
- expert 128 begins exactly at B's end.  The next 128 KiB scale allocation
  spans sixteen bad strides (128..143); fp16(1/32) has bytes `00 28`, which as
  int4 codes contributes exactly −44 after scaling.  This reproduces 129→85
  and the now-uniform 128→84;
- a later zero-filled region contributes exactly −64, reproducing 190→126,
  201→137, and 255→191.

Thus one wrong B expert pitch explains the apparently unrelated exact ×2 and
the two zero-arm constants.  The latter are OOB allocation contents, not two
piecewise expert transforms.  The old `e >= 128 -> e - 64` arm remains only as
a planted historical red proving the probe can see that shape; it is not the
diagnosis.  No additional device run is needed to establish the integer
addressing fix.

The host layout oracle must also be scoped carefully.  Internal agreement is
not an anchor: a wrong `Copy_Traits` and a model built from the same traits can
agree.  A trustworthy result needs both the kernel/model isomorphism checks
(same template arguments and `L/scale_k/N/K`, same `load_init` path) and an
independent anchor such as `place_derived -> recover_derived == identity` plus
zero byte-map difference against a shipping artifact.  If those checks pass
for all experts, the justified conclusion is only that G5's defect is outside
that modeled chain—not that the device result is correct or that B shares the
same behavior.

## Maintenance rule

For each new mixed-input argument, add a constructive “change X” arm before
shipping it.  If the format intentionally fixes X, reject noncanonical X at
the lowest shared boundary.  A parameter that is accepted, copied through an
API, and then replaced by a legal default layout is the dangerous case: all
layout algebra can remain internally consistent while modeling a problem the
caller did not request.
