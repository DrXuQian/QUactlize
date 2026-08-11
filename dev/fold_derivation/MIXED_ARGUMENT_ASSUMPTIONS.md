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
| `dS` | `Arguments::dS` in ordinary `:776`, fold `:415`, 2-plane `:524` | `Params::dS` is populated and `detail::make_metadata_tile` consumes it for both S and Z (ordinary `:795,919,942`; fold `:432,531,542`; 2-plane `:553,681,704`) | Caller-provided MN-major metadata stride | Before `8e4196c`, any padding or non-tight expert pitch differed from the reconstructed tight `(1, N, N*scale_k)` layout | Fixed by `8e4196c` (`Honor caller metadata strides`). Audited production adapters currently pass the tight value, so this hardening does **not** explain G5: that fixture is tight and its `e >= 128` failure survives independently | `l127_metadata_stride.cu`: hold pointer, shape, and payload fixed; change tight `dS` to a padded `dS` and require addresses/read values to change. Independent anchor is the explicit 64-bit formula `n + g*dS_g + e*dS_l` with unique per-coordinate tags | **FIXED** |
| Zero-plane stride (`dZ`) | No independent `dZ`; `ptr_Z` is declared beside `dS` | Z is tiled with the same `Params::dS` as S | S and Z share one metadata layout | A caller needs independent S and Z pitches | No ignored field exists, but the ABI cannot express independently padded planes. Current artifacts use a shared layout | Exercise a shared padded `dS` and require both S and Z addresses to follow it. There is intentionally no “independent `dZ`” positive test until the ABI gains that field | **INTENTIONAL ABI RESTRICTION**; add `dZ` only with an ABI change |
| Outer A base versus `dA` | `Arguments::dA` / `Params::dA` in all three families | At `8e4196c`, the inner tensor used `dA`, but the expert/batch base was `ptr_A + a_row_off*K` (ordinary `:878-880`; fold `:501-503`; 2-plane `:637-639`). L128 now routes all three through `mixed_a_expert_base` | Historical Y: tight row pitch `K`; ragged base `row_offsets[e]*K`, uniform base `e*M*K` | `stride_M != K`, `stride_L != M*stride_M`, or a future K-slice carries a full leading dimension different from local `K` | The latent pointer-semantics defect is fixed without changing tight shipping addresses | L128 gives rows and experts distinct padding/tag regions. Expected base is `row_offsets[e]*stride_M` (ragged) or `e*stride_L` (uniform); planting `offset*K` reds on 4/5 experts in both arms | **L128 FIXED** (was latent at `8e4196c`) |
| Runtime `group_size` versus static schedule group size | Runtime `group_size` is accepted and copied into Params | Runtime value derives `scale_k`/reload quantities, while static schedules choose the scale group with `DispatchPolicy::StaticGroupSize` (ordinary around `:854,1281`; fold `:477,899`; 2-plane `:613,1170`) | Two different group sizes can govern allocation/address extent and per-K group selection | A static schedule is instantiated for one group size but receives another runtime value | Public dense/MoE dispatch currently selects or guards the matching schedule, so the mismatch is hidden. Direct collective use can silently combine inconsistent meanings | Exhaust static/runtime pairs in `{16,32,64,128}`: a static schedule must accept equality only; a dynamic schedule (`-1`) must accept supported runtime values. Independent anchor is `floor(k/group_size)` with a unique tag per scale group | **LATENT validation bug (public guard)** |
| Logical-N residue for metadata in fold / 2-plane | Logical problem `N`, `TileN`, and `Fold` are all known | At `8e4196c`, fold `:536` and 2-plane `:686` used `N - size<0>(gB)*n_coord`; `size<0>(gB)` is physical `TileN/Fold`. L128 now routes all three families through `mixed_logical_n_residue` | Historical Y: physical `TileN/Fold` advanced a predicate that lives in logical-N coordinates | `Fold > 1` and the last N tile is partial. Example: `TileN=64`, `Fold=2`, `N=96`, `n_coord=1`: expected residue is 32, old result is 64 | The latent OOB predicate is fixed; the public 256-aligned domain remains unchanged | L128 exhausts `Fold={1,2,4}`, N=1..129. Independent anchor is `N - n_coord*TileN`; the old physical-width formula reds on 132/585 cases | **L128 FIXED** (was hidden by public guard) |
| Interleaved `dB` | Generic B stride `dB` is accepted | The interleaved branches synthesize canonical `make_stride(...)` layouts (ordinary `:1224`; fold `:852`; 2-plane plane 1 `:1054`) | Canonical interleaved byte layout and expert pitch `N*K*bits/8` | Caller supplies row padding, a different expert pitch, or any otherwise noncanonical `dB` | Shipping adapters select interleaving only for canonical placed artifacts and aligned domains. The defect is accepting a stride that this format cannot honor | With the same pointer/shape, perturb `dB`: the call must reject rather than silently produce the same output. Independent anchors are the explicit byte-pitch formula and xplane `place_derived -> recover_derived == identity` | **FORMAT RESTRICTION / MISSING FAIL-CLOSE** |
| Plane-2 `dB2` / `dB2_valid` | 2-plane arguments expose `dB2` and a validity flag | Noninterleaved code uses `dB2_valid ? dB2 : dB` (`:1130`); interleaved plane 2 synthesizes its layout (`:1115`) and ignores both | Canonical plane-2 interleaving; fallback to `dB` only when plane layouts coincide | Interleaved plane 2 has a noncanonical pitch, or `dB2_valid=false` while the two planes have different folds/layouts | Current launchers provide `dB2` when plane folds differ. Unsupported values are still representable and can be silently ignored at the collective boundary | Vary plane-2 stride independently. Reject altered interleaved stride and reject `dB2_valid=false` when folds differ. Independent anchor is plane-specific `place_hi`/inverse plus its explicit `N*K*bits/8` pitch | **FORMAT RESTRICTION / MISSING FAIL-CLOSE** |
| Packed `ptr_S`, `dS`, and `ptr_Z` semantics | The generic argument names describe half scale/zero tensors | Packed-scale paths reinterpret `ptr_S` as packed raw units; `ptr_Z` is not a separate input and generic `dS` is not the packed-unit pitch | Format-defined packed-unit byte layout | A caller treats the generic fields as ordinary half S/Z metadata, supplies `ptr_Z`, or expects arbitrary `dS` to control packed-unit addressing | This is an intentional ABI overload used by the packed public API, not an arithmetic bug in its supported format. The generic surface currently does not fail closed on contradictory arguments | A non-null `ptr_Z` or altered noncanonical `dS` must be rejected, not accepted with unchanged output. Independent anchor is a format-specific pack/decode round trip against the GGUF packed-unit layout | **FORMAT RESTRICTION / MISSING FAIL-CLOSE** |
| Divisibility of interleave/fold/packed extents | Runtime `N`, `K`, folds, interleave factor, and packed-unit geometry are declared | Several layouts use integer division: `K/kCon`, `N/Fold`, ordinary packed `scale_k/ScaleTileK`, and 2-plane packed tile/unit counts | A canonical artifact whose extents divide every format quantum exactly | `K % kCon != 0`, `N % Fold != 0`, or metadata groups do not divide packed groups-per-unit | Public adapters currently guard 256-aligned N/K and, where required, 512-aligned packed K. Lower collective entry points do not uniformly reject all unsupported residues | For every quantum, test an aligned case and `+/-1` residues; aligned must accept and residue must reject. Independent anchor is the explicit set of modular equalities, not the implementation's truncated quotient | **FORMAT RESTRICTION / MISSING lower-level FAIL-CLOSE** |

## G5 scope and non-conclusions

The G5 ID probe is deliberately narrower than this table:

- `q == 8` zeros the B contribution, so it observes the zero-metadata plane;
  it does not establish B's expert indexing.
- Its S/Z layout is tight, its static and runtime group sizes match, it uses the
  noninterleaved path, and all relevant extents divide their format quanta.
- Consequently, honoring `dS` in `8e4196c` is independent correctness
  hardening but cannot explain the observed `e >= 128 -> e - 64` pattern.

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
