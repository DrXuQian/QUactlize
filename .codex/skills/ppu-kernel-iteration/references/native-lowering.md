# Native PPU lowering and execution structure

## A compilable PTX path can still be an expensive compatibility path

Some PTX operations unsupported natively on the selected PPU can compile
through compatibility lowering. That can expand one source operation into
many native instructions, register exchanges, shared-memory round trips or
synchronization. Other operations simply fail compilation. Treat compiler
acceptance, numerical correctness and efficient native execution as three
different facts.

For the exact shipping specialization, inspect actual native code and
resources before attributing a performance gap to hardware or the algorithm:

- Bind compiler/SDK, target architecture, compatibility flags, source and
  binary. Instantiate the device body; a parsed header or unused template is
  not coverage of a target-specific branch.
- Check native operation families, operand forms, load/store widths, register
  transfers, barriers, spills and local-memory accesses. A source intrinsic
  or inline-PTX mnemonic is not proof that a single equivalent native opcode
  was emitted. Renaming/removing a PTX prefix is not an implementation proof.
- Normalize by useful work (matrix fragment, MMA, token, or output element),
  including any emulation/transpose/exchange sequence. Keep static body size
  separate from dynamically executed per-PC counts. Check the enclosing
  loops: equal static counts can hide very different repetition.
- Prefer a documented native SDK operation or a proven actlize atom when a
  compatibility expansion is costly. Prove its lane/register/shared-memory
  mapping and rounding first; then test the exact body and the full call.
  Compile-only access to a native opcode does not certify that mapping or
  predict a latency win.

Do not equate every large instruction count with compatibility emulation.
Generic signed layout arithmetic, scalar publication, repeated conversions
and a poor register layout also expand code. Attribute the extra sequence to
the actual source/IR/native lowering before naming its cause.

## Reference implementations should guide structure, not only tile sizes

When Triton is faster, compare its **selected/generated** kernel with ours:
CTA/warp ownership, resident accumulators, tile layouts, global-to-shared and
shared-to-register paths, intermediate materialization, pipelining and
synchronization. Source-level `tl.dot` is not that implementation description.
Inspect the compiled primitive sequence as well as the tensor formula.

Equal MMA counts but roughly twice the delivery/address/control instructions
justify inspecting or replacing that execution layer. They do not prove that
the recurrence is mathematically wrong. Changing shared layout, fragment
ownership and load atom together may be necessary; replacing only the load
opcode is unsafe. Keep the admitted old path as a counterfactual while a
separate implementation is proved. Fusion is not intrinsically faster: a
large fused CTA may force unlike stages to share an unfavorable resource
budget or materialize intermediates inefficiently.

Observed GDN example (SDK2.1.1, PPU0010, B1/S2048/Hk16/Hv32/D128/C64):
WY and FLA state/output had equal BF16 MMA counts, but WY executed about
2.17x/2.35x as many native instructions. WY used SWZL loads with scalar
descriptors; FLA used NCOM lane-address loads. The pinned actlize header
disabled plain PPU0010 LDSM because its old PTX grammar was unproved, while
the installed public `hggc_mma.h` exposed `awmma::ldmatrix`. A local SDK probe
compiled its normal/16x16-transpose forms to one native NCOM load each,
without vector-to-scalar transfers or stack. That was compile-only reachability,
not a lane/register numerical gate. This identifies an adapter/codegen seam
to test, not evidence that SWZL is always slow or
that swapping it alone closes the entire gap. Detailed evidence:
GDN-QSA-sm80 `docs/PPU_WY_SHARED_ACU_20260922.md`, commit4267b28.

## An API win and slower GPU kernels can coexist

A full-public-API event span can include GPU idle gaps while the host
allocates/dispatches the next operation. Summed ACU kernel durations omit
those gaps and come from a different instrumented/replayed protocol. A C++
multi-launch entry and Python/Triton launch path can have different dispatch
costs even when their arithmetic scopes match.

Report full-call and kernel-only results separately. Do not subtract timings
from different runs/protocols and label the difference measured Python,
allocation or launch overhead. Establish that attribution with one complete
call timeline or an explicitly paired prepared-workspace/graph experiment;
verify that the runtime supports the latter. Do not let host overhead hide
a slower kernel or remove required work from only one timed arm.
