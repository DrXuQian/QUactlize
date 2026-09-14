# Component-based prefill selection

The September 14 import validates 2,385/2,385 fresh components and joins the
44 previously measured Q4/Q5 points: 502 workloads, represented by 430 shape/M
knots. The generated table is `policies/kpack_zw810_cost_v1.hpp`; production
lookup is `quactlize/dispatch/cost.hpp`. There is no online timing or tuning.

Costs are:

- FQ: complete measured GEMM, including its actual reducer.
- SF: independently measured scale/zero expansion plus complete SF GEMM.
- Full BF16: independently measured active-weight expansion plus installed
  cuBLAS (dense) or DeepGEMM Python-JIT (grouped) GEMM.

These sums exclude external adapters and producer-consumer cache interaction.
They are selection costs, not measured end-to-end model speedups. SF expands
metadata only, on every call. Full expansion writes BF16 weights on every call;
grouped uses a GPU list of the actual active experts and preserves E/N/K strides.
The native composition and its caller adapters still require PPU admission.

## Bounds and fallback

Small-M decode selection is unchanged. Its complete TC measurements already
include reduction; do not add the separately measured reducer a second time.
The sole noisy standalone point, dense M1/N4096/S4, has been rechecked at
1.83625 us with a round range below 5%.

Cross-route selection covers measured dense families at M128..4096 and grouped
top8/E256 families at tokens128..4096. Outside a family's measured M range or
for an unknown family, retain the existing K-pack selector. Interior nearest-knot
transfer is explicitly labelled `predicted`, not an experimentally guaranteed
5% bound. No full-dequant option is introduced for small M or Q8_0.

Within a 5% tie, prefer FQ, then SF. If the measured expansion runtime is absent,
only FQ is eligible; if the BF16 provider is absent, compare FQ/SF. No Xplane
fallback is introduced. A present but broken explicitly configured provider
reports an actionable error instead of reinterpreting weight bytes.

Grouped selection does not read the router back to the CPU. For a measured
shape/token pair, it minimizes worst measured regret over available routing
profiles, considering only tactics actually measured on every such profile.
There are **13 knots with regret above 5%** under this restriction (maximum
47.26%). These are retained as visible router-sensitive performance debt, not
declared parity. The source report lists all profiles, candidates and regrets.
Unmeasured routers are not covered by a performance guarantee.

With all providers available, the current knots choose dense full/SF/FQ at
194/80/6 knots and grouped full/SF/FQ at 2/65/83 knots. This is not a global
full-BF16 switch, nor a claim of optimality over unmeasured configurations.

## Reproduce locally

```bash
python3 tools/import_kpack_cost_policy.py \
  --supplement /root/kpack-cost-multi.i8l9xA.results.tgz \
  --prior-gemm /root/kpack-prefill-gemm.KlD17K.results.tgz \
  --output docs/measurements/kpack_component_policy_20260914.json \
  --header policies/kpack_zw810_cost_v1.hpp
```

The importer reads archives without extracting them, checks component hashes,
device/runtime identities, numerical controls, fixture bytes and routing IDs,
and rejects unstable candidate timings. No peak-bandwidth estimate supplies a
missing latency. Valid same-model cross-card comparison is authorized without
renaming PCI identities or normalizing measurements.
