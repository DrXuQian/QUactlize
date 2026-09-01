"""THE HOST SIDE OF EACH ROUTE, so a route can be RUN and TIMED rather than described.

WHY THIS FILE EXISTS. schemes.py had six cells reading "a piece exists, the path does not", with notes saying the
remainder was host-side wiring rather than arithmetic. That note is true and it was also treated as a blocker for
longer than it deserved: for DEQUANT_FIRST the remaining step is `a @ w.T` against torch's own cuBLAS, and no part
of it needs the PPU toolchain. A path nobody can call is a path nobody can time, and the project's measurement gap
was never in the kernels -- it was that no two routes had ever produced the same number in the same process.

WHAT EACH ROUTE ACTUALLY COSTS, since that is the only reason to have more than one:

    dequant_first   materialise the whole weight as fp16, then one library GEMM. 3.6x the native block in DRAM
                    traffic and a full extra write+read, but the GEMM itself is cuBLAS-grade. Pays when M is large
                    enough that reuse amortises the materialisation.
    scale_first     materialise only the scale planes. ~1/16 of the traffic above, and the GEMM is mixed-input.
                    Pays in the middle band. At decode the planes would be rebuilt every token, which is why the
                    pre-pass does not answer the M=1 case.
    fully_quantized materialise nothing. The only route whose DRAM traffic is the checkpoint's own bytes.

THE SHAPE CONVENTION, which is the thing most likely to be got wrong silently. A GGUF weight of logical shape
(n, k) is stored as n*(k/256) blocks in ROW-MAJOR order: block b of row i is at flat index i*(k/256) + b. So
`gguf_dequantize` returning [n*k/256, 256] reshapes to (n, k) directly. Getting this backwards produces a weight
that is a valid permutation of the right one, and a test with a symmetric fixture will not notice -- which is why
the golden test drives these through an ASYMMETRIC activation.
"""
from typing import Optional

import torch

from . import (gguf_dequantize, gguf_vecdot_dense, gguf_vecdot_moe, gguf_prepare_gemv, gguf_prepare_dense,
               gguf_gemv_artifact_dequantize, gguf_gemv_artifact_dequantize_scale,
               gguf_dense_artifact_dequantize, gguf_dense_artifact_dequantize_scale,
               gguf_gemv_scale_first, gguf_gemv_scale_first_moe, gguf_dense_scale_first)
from .formats import (PLACED_ARRANGEMENT_VERSION_V1, PLACED_ARRANGEMENT_VERSION_V2,
                      PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1,
                      PLACED_LAYOUT_Q4_N16K64_DIRECT_V1,
                      PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1, PlacedArrangement,
                      PlacedArrangementV2, QuantType,
                      canonical_fully_quantized_layout, placed_arrangement,
                      kquant_kpack_arrangement, placed_code_planes,
                      q4_kpack4_arrangement, q4_n16k64_direct_arrangement,
                      validate_fully_quantized_resident_geometry)


def _check_shape(blocks: torch.Tensor, n: int, k: int, qtype: int) -> None:
    if k % 256:
        raise ValueError(f"k={k} is not a multiple of the k-quant superblock (256)")
    want = n * (k // 256)
    if blocks.dim() != 2 or blocks.shape[0] != want:
        raise ValueError(
            f"blocks should be [{want}, type_size] for an ({n}, {k}) weight in {QuantType(qtype).name}, "
            f"got {tuple(blocks.shape)}. Row-major: block b of row i sits at i*(k/256)+b")


def dequantize_weight(blocks: torch.Tensor, n: int, k: int, qtype: int) -> torch.Tensor:
    """RAW GGUF blocks -> the fp16 weight (n, k). The materialisation dequant_first pays for."""
    _check_shape(blocks, n, k, qtype)
    return gguf_dequantize(blocks, int(qtype)).view(n, k)


def matmul_dequant_first(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int,
                         weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """DEQUANT_FIRST, dense: expand to fp16, then torch's cuBLAS. a is (m, k); returns (m, n).

    `weight` lets a caller hoist the materialisation out of a timing loop, which is the honest way to measure the
    two halves separately -- amortised over many GEMMs this route is exactly cuBLAS, and the whole argument for the
    other routes is the part that is NOT amortised."""
    if weight is None:
        weight = dequantize_weight(blocks, n, k, qtype)
    if a.shape[-1] != k:
        raise ValueError(f"activation has k={a.shape[-1]}, weight has k={k}")
    return a.to(weight.dtype) @ weight.t()


def matmul_dequant_first_grouped(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int,
                                 num_experts: int, rows_per_expert: torch.Tensor) -> torch.Tensor:
    """DEQUANT_FIRST, grouped (MoE): the same route with a per-expert weight and ragged rows.

    `blocks` holds num_experts weights back to back, each (n, k). `rows_per_expert` is int64 [num_experts] and must
    sum to a.shape[0]; rows are assumed already gathered into expert order, which is what the grouped kernels also
    assume. The loop is deliberately a loop: this is the FALLBACK, and its cost model is one library GEMM per
    expert. Replacing it with DeepGemm changes the GEMM, not the route."""
    if rows_per_expert.numel() != num_experts:
        raise ValueError(f"rows_per_expert has {rows_per_expert.numel()} entries, expected {num_experts}")
    total = int(rows_per_expert.sum().item())
    if total != a.shape[0]:
        raise ValueError(f"rows_per_expert sums to {total}, activation has {a.shape[0]} rows")
    per_expert = n * (k // 256)
    _check_shape(blocks, n * num_experts, k, qtype)
    out = a.new_empty((a.shape[0], n), dtype=torch.float16)
    start = 0
    for e in range(num_experts):
        rows = int(rows_per_expert[e].item())
        if rows == 0:
            continue
        w = dequantize_weight(blocks[e * per_expert:(e + 1) * per_expert], n, k, qtype)
        out[start:start + rows] = a[start:start + rows].to(torch.float16) @ w.t()
        start += rows
    return out


def matmul_native_gemv(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int) -> torch.Tensor:
    """FULLY_QUANTIZED, GEMV: nothing materialised. a is (1, k) fp16/fp32; returns (1, n) fp32.

    The dedicated dense host op forwards one complete activation to vecdot_rows_kernel, which shares it across n
    output columns and accumulates k/256 raw GGUF blocks inside one launch. Without a device library it retains a
    bounded scalar witness and refuses large rows rather than silently impersonating an inference path."""
    _check_shape(blocks, n, k, qtype)
    if a.shape[0] != 1:
        raise ValueError(f"the GEMV band is m=1; got m={a.shape[0]}")
    # The device ABI is fp16. Apply that contract before the backend branch so the CPU witness and production launch
    # differ only in accumulation order, not in which activation values they received.
    return gguf_vecdot_dense(blocks, a.to(torch.float16).contiguous(), n, k, int(qtype))


def _row_offsets(rows_per_expert: torch.Tensor, experts: int, total_rows: int) -> torch.Tensor:
    if rows_per_expert.numel() != experts:
        raise ValueError(f"rows_per_expert has {rows_per_expert.numel()} entries, expected {experts}")
    rows = rows_per_expert.to(dtype=torch.int64, device="cpu")
    if bool((rows < 0).any()):
        raise ValueError("rows_per_expert must be nonnegative")
    if int(rows.sum().item()) != total_rows:
        raise ValueError(f"rows_per_expert sums to {int(rows.sum().item())}, activation has {total_rows} rows")
    return torch.cat((torch.zeros(1, dtype=torch.int64), rows.cumsum(0))).to(torch.int32).contiguous()


def matmul_native_gemv_moe(a: torch.Tensor, blocks: torch.Tensor, n: int, k: int, qtype: int,
                            num_experts: int, rows_per_expert: torch.Tensor) -> torch.Tensor:
    """FULLY_QUANTIZED/GEMV_MOE: native GGUF bytes, gathered rows, one CUDA-core launch."""
    if a.dim() != 2 or a.shape[1] != k:
        raise ValueError(f"activation must be [total_rows,{k}], got {tuple(a.shape)}")
    bpr = k // 256
    want = n * bpr
    if blocks.dim() == 2:
        if blocks.shape[0] != num_experts * want:
            raise ValueError(f"flat blocks need {num_experts * want} rows, got {blocks.shape[0]}")
        blocks = blocks.view(num_experts, want, blocks.shape[1])
    if blocks.dim() != 3 or blocks.shape[:2] != (num_experts, want):
        raise ValueError(f"blocks must be [{num_experts},{want},type_size], got {tuple(blocks.shape)}")
    offsets = _row_offsets(rows_per_expert, num_experts, a.shape[0])
    return gguf_vecdot_moe(blocks.contiguous(), a, offsets, int(qtype))


def prepare_scale_first(blocks: torch.Tensor, n: int, k: int, qtype: int):
    """Offline artifact for both SCALE_FIRST decode shapes: (low, high, scale, zero).

    "RESIDENT" IS NO LONGER THE RIGHT WORD and that is the whole point of the B/C merge. This produces the fp16
    scale/zero planes FROM RAW, which is why SCALE_FIRST used to need a stored arrangement of its own -- a third
    thing in HBM beside raw GGUF and the packed artifact. Since weights exist once, three arrangements is what
    made runtime kernel switching impossible.

    dequantize_scale_from_units derives the same planes from FULLY_QUANTIZED's packed units instead, so they can
    live in a workspace. This function stays because the switch has not been made: the code planes and the derived
    scales are both verified identical between the two producers
    (test_packed_unit_scale_derivation_matches_the_scale_first_planes), but the last commit -- repointing this at
    BC -- is held until that test has passed on ppu001 rather than only locally."""
    return gguf_prepare_gemv(blocks, n, k, int(qtype))


def dequantize_scale_first(artifact, qtype: int) -> torch.Tensor:
    """GEMV scale-first artifact -> fp16 `[experts,n,k]`, without running GEMV."""
    return gguf_gemv_artifact_dequantize(*artifact, int(qtype))


def dequantize_scale_first_scales(artifact, qtype: int):
    """GEMV scale-first artifact -> consumer-ready `(scale, zero)` fp16 planes."""
    return gguf_gemv_artifact_dequantize_scale(artifact[2], artifact[3], int(qtype))


def matmul_scale_first_gemv(a: torch.Tensor, artifact, qtype: int) -> torch.Tensor:
    """SCALE_FIRST/GEMV over prebuilt `(low, high, scale, zero)` planes."""
    low, high, scale, zero = artifact
    return gguf_gemv_scale_first(a, low, high, scale, zero, int(qtype))


def matmul_scale_first_gemv_moe(a: torch.Tensor, artifact, qtype: int,
                                 rows_per_expert: torch.Tensor) -> torch.Tensor:
    """SCALE_FIRST/GEMV_MOE over prebuilt planes and gathered/ragged rows."""
    low, high, scale, zero = artifact
    experts = int(low.shape[0])
    offsets = _row_offsets(rows_per_expert, experts, a.shape[0])
    return gguf_gemv_scale_first_moe(a, low, high, scale, zero, offsets, int(qtype))


def prepare_scale_first_dense(blocks: torch.Tensor, n: int, k: int, qtype: int,
                              derive_from_bc: bool = True):
    """Offline artifact for SCALE_FIRST/DENSE's fixed fpA tactic: (low, high, scale, zero).

    DERIVED FROM BC BY DEFAULT since the merge. derive_from_bc=False keeps the original raw producer reachable --
    not as a fallback but as the INDEPENDENT ARM the oracle compares against: once the derivation is the only
    path, a test that compares it to itself proves nothing, and this cell's whole history is of a green
    self-comparison (test_fpA_kquant_dense) that could not see a wrong shared constant."""
    _check_shape(blocks, n, k, qtype)
    if not derive_from_bc:
        return gguf_prepare_dense(blocks, n, k, int(qtype))
    # THE MERGE, MADE STRUCTURAL. One producer, one placement, one scale channel: the fp16 planes come out of the
    # PACKED UNITS rather than out of a second pass over raw. Both halves are verified bit-exact against the old
    # path on ppu001 -- the code planes, and the scale planes after the accessor's documented transpose is taken
    # out of the comparison -- so this is the same artifact by a shorter route, not a new one.
    #
    # WHY IT MATTERS BEYOND TIDINESS. While this produced from raw, C was a THIRD arrangement that a deployment
    # had to keep beside raw GGUF and the packed artifact, and weights exist once in HBM. Deriving instead is what
    # lets the fp16 planes live in a workspace, which is the whole reason the merge was worth doing.
    low, high, units = _op("gguf_prepare_fully_quantized_dense")(blocks, n, k, int(qtype))
    scale, zero = dequantize_scale_from_units(units, qtype)
    return low, high, scale, zero


def dequantize_scale_first_dense(artifact, qtype: int) -> torch.Tensor:
    """Dense fpA scale-first artifact -> fp16 `[1,n,k]`, without running GEMM."""
    return gguf_dense_artifact_dequantize(*artifact, int(qtype))


def dequantize_scale_first_dense_scales(artifact, qtype: int):
    """Dense fpA scale-first artifact -> consumer-ready `(scale, zero)` fp16 planes."""
    return gguf_dense_artifact_dequantize_scale(artifact[2], artifact[3], int(qtype))



def _op(name: str):
    """Resolve a torch op that may postdate this build, with a message instead of a NameError.

    The module-level `from . import (...)` list cannot carry these: an op added after the installed extension was
    built would make importing routes.py fail outright, taking the whole suite with it for a reason unrelated to
    what anyone is running. Resolving at call time turns that into one clear error at one call site.
    """
    import quactlize as _q
    fn = getattr(_q._ops(), name, None)
    if fn is None:
        raise RuntimeError(
            f"quactlize::{name} is not registered in this build. Either the extension predates the op "
            f"(python3 setup.py build_ext --inplace) or the device library was built without the feature.")
    return fn


def has_op(name: str) -> bool:
    """Whether an op is in THIS build. Tests use it to tell 'not built yet' (a legitimate skip) from 'built and
    wrong' (a failure); those are different states and a skip for the second one would hide a defect.

    IT ASKS THE OPERATOR REGISTRY, NOT THE MODULE NAMESPACE, and that distinction is the whole point. Every op
    reaches quactlize/__init__.py through a HAND-WRITTEN wrapper, so a newly registered op is invisible as a
    module attribute until somebody adds one. Keying on that would have made this test skip forever on a box
    where the op was built, registered and working -- a green run that checked nothing. This happened when
    gguf_dense_fully_quantized was registered in C++ before the Python module namespace exposed it.

    torch.ops.quactlize is the source of truth: the op is there iff RegisterOperators ran."""
    import quactlize as _q
    try:
        return getattr(_q._ops(), name, None) is not None
    except ImportError:
        return False                     # extension not built at all -- absent, not stale


PLACED_ARTIFACT_VERSION = PLACED_ARRANGEMENT_VERSION_V1
PLACED_ARTIFACT_VERSION_V2 = PLACED_ARRANGEMENT_VERSION_V2


class PlacedArtifact(tuple):
    """(low, high, units) AND the versioned placement it was produced for.

    WHY IT IS A tuple SUBCLASS. Every existing consumer treats the artifact as a 3-tuple -- tools/pack_gguf.py
    does `low, high, units = prepare_fully_quantized_dense(...)`, _unpack_fq does `list(artifact)`, and the
    tests index it. Subclassing tuple keeps all of that byte-identical while giving the object somewhere to
    carry what the tensors cannot say about themselves.

    WHAT IT CARRIES AND WHY. `arrangement` is formats.PlacedArrangement(bits, tile_k, high_bits), not just fold:
    l105 showed that F is derived but is not enough to identify every physical class. The descriptor is attached
    at the producer, where the actual *_for_tile argument is known, and every reader below obtains it from this
    object. There is intentionally no reader-side tile_k argument for a caller to forget or contradict.

    THE VERSION IS PART OF THE CONTRACT. A future descriptor may add WK or another placement axis. Treating an
    old tuple as the new version would restore the exact silent-guessing failure this class removes, so unknown
    versions and tuple-stripped artifacts fail before an op is called.
    """
    # NO __slots__: a tuple subclass cannot have a nonempty one (TypeError at class creation).

    def __new__(cls, tensors, arrangement, version: Optional[int] = None):
        return super().__new__(cls, tuple(tensors))

    def __init__(self, tensors, arrangement, version: Optional[int] = None):
        super().__init__()
        if not isinstance(arrangement, (PlacedArrangement, PlacedArrangementV2)):
            raise TypeError(
                "PlacedArtifact arrangement must be formats.PlacedArrangement or PlacedArrangementV2; "
                f"got {type(arrangement).__name__}")
        self.arrangement = arrangement
        if version is None:
            version = (PLACED_ARTIFACT_VERSION_V2 if isinstance(arrangement, PlacedArrangementV2)
                       else PLACED_ARTIFACT_VERSION)
        self.arrangement_version = int(version)

    def __reduce__(self):
        """Persist tensors and descriptor as one identity.

        tuple's default reducer serialises only its elements, which either loses the placement or cannot call this
        class's required constructor on unpickle/deepcopy.  Carry both descriptor fields explicitly so a copied or
        torch-saved folded artifact cannot silently become an ordinary 3-tuple.
        """
        return (type(self), (tuple(self), self.arrangement, self.arrangement_version))

    # Tensor-valued tuple equality is not a useful scalar predicate, and inheriting tuple.__eq__ would ignore the
    # descriptor entirely.  Make identity comparison explicit and boolean while retaining tuple indexing/unpacking.
    def __eq__(self, other):
        if not isinstance(other, PlacedArtifact):
            return False
        if (self.arrangement_version != other.arrangement_version or
                self.arrangement != other.arrangement or len(self) != len(other)):
            return False
        return all(torch.equal(a, b) for a, b in zip(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    @property
    def requested_tile_k(self):
        """Compatibility VIEW, not the ABI: old diagnostics named this field before the full descriptor existed."""
        return (self.arrangement.tile_k if isinstance(self.arrangement, PlacedArrangement)
                else self.arrangement.artifact_tile_k)


def _arrangement_v2_wire(arrangement: PlacedArrangementV2):
    """Exact torch-op field order for ``quactlize_ppu_placed_arrangement_v2``."""
    return (PLACED_ARTIFACT_VERSION_V2, arrangement.layout, arrangement.bits, arrangement.high_bits,
            arrangement.artifact_tile_k, arrangement.transport_tile_k, arrangement.group_size,
            arrangement.reserved, arrangement.mapping_id)


def _require_placed_artifact(artifact, qtype: int, where: str):
    """Validate descriptor version, qtype and plane widths before any reader op sees the bytes."""
    if not isinstance(artifact, PlacedArtifact):
        raise TypeError(
            f"{where}: expected a PlacedArtifact carrying its placement descriptor; got "
            f"{type(artifact).__name__}. Converting one to tuple/list strips (bits,tile_k,high_bits), so the "
            f"reader must refuse rather than guess a fold")
    arrangement = artifact.arrangement
    expected_version = (PLACED_ARTIFACT_VERSION_V2 if isinstance(arrangement, PlacedArrangementV2)
                        else PLACED_ARTIFACT_VERSION)
    if artifact.arrangement_version != expected_version:
        raise ValueError(
            f"{where}: unsupported placed-artifact descriptor version {artifact.arrangement_version}; "
            f"this {type(arrangement).__name__} requires version {expected_version}. Guessing another descriptor "
            f"would silently ignore a placement axis")
    expected_bits = placed_code_planes(qtype)
    got_bits = (arrangement.bits, arrangement.high_bits)
    if got_bits != expected_bits:
        raise ValueError(
            f"{where}: artifact code planes are {got_bits[0]}+{got_bits[1]} bits but "
            f"{QuantType(qtype).name} requires {expected_bits[0]}+{expected_bits[1]}; this is a qtype/artifact "
            f"mismatch, not a layout the reader can reinterpret")
    # Force validation of the derived arrangement before dispatch. This catches non-integral delivery runs at the
    # Python ABI instead of relying on whichever reader happens to instantiate first.
    if isinstance(arrangement, PlacedArrangement):
        _ = arrangement.fold, arrangement.high_fold
    else:
        arrangement.validate()
        if arrangement.layout == PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 and QuantType(qtype) != QuantType.Q4_K:
            raise ValueError(f"{where}: Q4 K-pack4 bytes cannot be consumed as {QuantType(qtype).name}")
        if arrangement.layout == PLACED_LAYOUT_Q4_N16K64_DIRECT_V1 and QuantType(qtype) != QuantType.Q4_K:
            raise ValueError(
                f"{where}: Q4 N16xK64 direct bytes cannot be consumed as {QuantType(qtype).name}")
        if arrangement.layout == PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1:
            expected = kquant_kpack_arrangement(qtype)
            if arrangement != expected:
                raise ValueError(
                    f"{where}: {QuantType(qtype).name} requires K-pack descriptor {expected}, got {arrangement}; "
                    "the plane pack factors are part of the bytes and cannot be inferred from layout=2 alone")
    low, high, units = _unpack_fq(artifact, where)
    if low.dtype != torch.uint8 or high.dtype != torch.uint8:
        raise ValueError(f"{where}: placed code planes must be uint8, got low={low.dtype} high={high.dtype}")
    if low.dim() != 3:
        raise ValueError(f"{where}: low plane must be [experts,n,packed_k], got {tuple(low.shape)}")
    if expected_bits[1] == 0:
        if high.numel() != 0:
            raise ValueError(f"{where}: {QuantType(qtype).name} is single-plane but high has {high.numel()} bytes")
    else:
        if high.dim() != 3 or high.shape[:2] != low.shape[:2]:
            raise ValueError(
                f"{where}: high plane must share [experts,n] with low; got low={tuple(low.shape)} "
                f"high={tuple(high.shape)}")
        # Both dimensions encode the same logical K. Cross multiplication avoids truncating malformed sub-byte
        # extents into the same integer quotient (for example, two odd byte counts which each floor to K=0).
        if low.shape[2] * expected_bits[1] != high.shape[2] * expected_bits[0]:
            raise ValueError(
                f"{where}: low/high planes encode different K extents: {tuple(low.shape)} vs {tuple(high.shape)}")
    return low, high, units, arrangement


def _require_shipping_kpack_artifact(artifact, qtype: int, where: str):
    """Admit only the two versioned K-pack byte classes used by product compute.

    Legacy Xplane descriptors and the experimental direct layout remain
    available to development producers and inverse tests, but a shipping GEMM
    must never infer or reinterpret them.  Exact mapping validation happens in
    ``_require_placed_artifact`` before this layout/qtype admission check.
    """
    low, high, units, arrangement = _require_placed_artifact(
        artifact, qtype, where)
    if not isinstance(arrangement, PlacedArrangementV2):
        raise ValueError(
            f"{where}: shipping compute requires a version-2 K-pack artifact; "
            "Xplane-v1 is development compatibility only")
    q = QuantType(qtype)
    expected = (q4_kpack4_arrangement() if q == QuantType.Q4_K
                else kquant_kpack_arrangement(q))
    if arrangement != expected:
        raise ValueError(
            f"{where}: {q.name} shipping compute requires K-pack descriptor "
            f"{expected}, got {arrangement}")
    return low, high, units, arrangement


def _unpack_fq(artifact, where: str):
    """(low, high, units), with a message about the CONTRACT instead of a bare unpacking error.

    The tuple grew from (low, units) to (low, high, units) on 2026-08-03 so that `units` could sit at a fixed
    index while the plane count varies. A producer built before that change returns two tensors, and plain
    tuple unpacking then says "not enough values to unpack" -- true, and silent about which side is old.
    """
    xs = list(artifact)
    if len(xs) != 3:
        raise ValueError(
            f"{where}: the artifact has {len(xs)} tensor(s); the contract is (low, high, units) -- three, with "
            f"`high` an EMPTY tensor for a single-plane format. Two means the device library or the host "
            f"extension predates the (low, high, units) change: rebuild both, and check that the producer emits "
            f"the empty high plane for Q4_K/Q2_K rather than omitting it.\n"
            f"  shapes seen: {[tuple(x.shape) for x in xs]}")
    return xs[0], xs[1], xs[2]


def dequantize_scale_from_units(units: torch.Tensor, qtype: int, zmul: Optional[int] = None):
    """THE DERIVATION THAT MAKES B AND C ONE FORMAT: packed scale units -> fp16 (scale, zero) planes.

    Today's prepass goes raw GGUF -> fp16 planes, which is why SCALE_FIRST needed a stored arrangement of its
    own. This one starts from the PACKED UNIT, so the fp16 planes become a workspace derivation of the
    FULLY_QUANTIZED artifact rather than a third thing to keep resident. That is the whole content of the merge:
    the two schemes already share their weight bytes and differ only in the scale channel.

    It is also this format's dequant-scale kernel in the sense of the standing entry rule -- a stored arrangement
    without an inverse can only be checked end to end, which mixes a packing mistake with a computation mistake.

    Accepts dense units [k_unit, n, unit_bytes] or grouped [E, k_unit, n, unit_bytes], and ALWAYS returns
    [E, k/group_size, n] for both -- a dense weight is one expert, so the consumer needs no second code path.
    """
    # zmul DERIVED, not defaulted. It was `= 0` for one commit, which is right for Q2_K and silently wrong for
    # the other four -- they would have returned plausible finite scales. An explicit value is still accepted,
    # for probing a format against a correction it does not own.
    if zmul is None:
        from .formats import placed_code_zmul
        zmul = placed_code_zmul(qtype)
    return _op("gguf_packed_scale_prepass")(units, int(qtype), int(zmul))


def matmul_bc_gemv(a: torch.Tensor, artifact, qtype: int) -> torch.Tensor:
    """DECODE OFF THE MERGED FORM: SIMT reads placed code planes plus packed units.

    Dense ``a`` is ``[M,K]`` for ``1 <= M < 8``.  Batch rows are native
    grid-y work in one launch and share one resident weight artifact; this is
    not a Python loop over M independent GEMV launches.

    This is what removes raw GGUF from residency. Until it existed the decode path read raw blocks while the
    tensor-core paths read the placed planes, so a deployment that wanted both had to keep two arrangements of
    the same weight -- and weights exist once in HBM, which is the constraint the whole merge is about.

    NOT YET A REPLACEMENT FOR matmul_native_gemv. The admission bar is parity with the raw path at the real layer
    shape, not "no worse under one condition", and it has not been met: at N=K=2048 the measurement is a cold tie
    and warm +11%. Both paths stay callable until it is.
    """
    low, high, units, arrangement = _require_placed_artifact(artifact, qtype, "matmul_bc_gemv")
    if isinstance(arrangement, PlacedArrangementV2):
        raise NotImplementedError(
            "matmul_bc_gemv: K-pack4 has no CUDA-core/BC reader. Use the tensor-core dense route; refusing to "
            "send a v2 byte map through the v1 Xplane BC ABI")
    return _op("gguf_gemv_bc_for_arrangement")(
        a, low, high, units, int(qtype), PLACED_ARTIFACT_VERSION,
        arrangement.bits, arrangement.tile_k, arrangement.high_bits)


def matmul_bc_gemv_moe(a: torch.Tensor, artifact, qtype: int, num_experts: int,
                       rows_per_expert: torch.Tensor) -> torch.Tensor:
    """The MoE decode arm of the same thing. Rows are ragged and an expert may have none.

    The route takes rows_per_expert and converts to offsets here, matching matmul_native_gemv_moe rather than the
    op's row_offsets argument. One convention per layer: the alternative is a caller that has to remember which
    side of the seam it is on, and the cumulative-offset form is the one that has an empty expert as a silent
    special case.
    """
    if isinstance(artifact, PlacedArtifact):
        if isinstance(artifact.arrangement, PlacedArrangementV2):
            raise NotImplementedError(
                "matmul_bc_gemv_moe: K-pack v2 has no grouped CUDA-core/BC reader; use the tensor-core grouped "
                "route. Refusing to erase its physical descriptor and send those bytes through the Xplane-v1 ABI")
        raise ValueError(
            "matmul_bc_gemv_moe: the grouped legacy op cannot carry a placed-artifact descriptor; refusing to "
            "strip it and guess the registry map. A grouped arrangement-aware ABI must land with its producer")
    low, high, units = _unpack_fq(artifact, "matmul_bc_gemv_moe")
    offsets = _row_offsets(rows_per_expert, num_experts, a.shape[0])
    return _op("gguf_gemv_bc_moe")(a, low, high, units, offsets, int(qtype))


def dequantize_fully_quantized(artifact, qtype: int, grouped: bool = False) -> torch.Tensor:
    """BC's DEQUANT-ALL: (low, high, units) -> the fp16 weight. The other half of this format's inverse.

    THE ENTRY RULE THIS SATISFIES. A stored arrangement needs dequant-all AND dequant-scale, because without them
    the only way to check it is end to end -- and that mixes a PACKING mistake with a COMPUTATION mistake.
    dequantize_scale_from_units gave BC the second; this is the first, and until it existed BC was the one format
    in the tree carrying only half its inverse while the artifact contract requires both.

    IT IS A COMPOSITION, NOT A REIMPLEMENTATION, and that is the point rather than a shortcut:

        (low, high, units) --dequantize_scale_from_units--> (low, high, scale, zero) --existing dequant-all--> w

    The merge's whole claim is that B and C share their weight bytes and differ only in the scale channel. If
    that holds, BC's dequant-all IS the scale-first dequant-all with the scale derived instead of stored, and
    writing a second decoder would create a third thing to keep correct. If it does not hold, this composition
    gives the wrong answer against official gguf -- which is a better outcome than a private decoder that agrees
    with itself.
    """
    arrangement = None
    if isinstance(artifact, PlacedArtifact) or not grouped:
        low, high, units, arrangement = _require_placed_artifact(
            artifact, qtype, "dequantize_fully_quantized")
    else:
        # This is only the legacy descriptor-less grouped Xplane tuple.  The
        # Q4 and generic K-pack grouped producers return PlacedArtifact above
        # and therefore always select their exact v2 inverse.
        low, high, units = _unpack_fq(artifact, "dequantize_fully_quantized(grouped=True)")
    scale, zero = dequantize_scale_from_units(units, qtype)
    if arrangement is not None:
        if isinstance(arrangement, PlacedArrangementV2):
            op = ("gguf_grouped_artifact_dequantize_for_arrangement_v2" if grouped
                  else "gguf_dense_artifact_dequantize_for_arrangement_v2")
        else:
            op = ("gguf_grouped_artifact_dequantize_for_tile" if grouped
                  else "gguf_dense_artifact_dequantize_for_tile")
    else:
        op = "gguf_grouped_artifact_dequantize" if grouped else "gguf_dense_artifact_dequantize"
    if grouped and not has_op(op):
        raise NotImplementedError(
            "the grouped artifact dequant-all op is not in this build; the dense one composes because the dense "
            "artifact is one expert. A grouped weight needs the per-expert form -- request it rather than "
            "looping here, since a python loop over experts would not exercise the same addressing.")
    args = (low, high, scale, zero, int(qtype))
    if isinstance(arrangement, PlacedArrangementV2):
        return _op(op)(*args, *_arrangement_v2_wire(arrangement))
    return _op(op)(*args, arrangement.tile_k) if arrangement is not None else _op(op)(*args)


def prepare_fully_quantized_dense(blocks: torch.Tensor, n: int, k: int, qtype: int,
                                  tile_k: Optional[int] = None, layout: str = "auto"):
    """Offline artifact for FULLY_QUANTIZED/DENSE: the code plane plus the PACKED SCALE UNIT.

    Unlike prepare_scale_first_dense, the scale is NOT expanded to fp16 planes -- it stays in the format's own
    packed bytes, reordered into 16-byte units the collective can bulk-copy. That is the whole difference between
    the two cells, and it is why the storage does not grow.

    Returns (low, high, units). `high` is the second weight plane -- EMPTY for a single-plane format and
    uint8[1, n, k/8] for Q5_K's 1-bit plane -- so the tuple's shape does not change with the format and `units`
    is always the LAST element. That is the property both oracles plant their fault on.

    ``layout="auto"`` selects the canonical K-pack bytes for every supported format.  Explicit
    ``layout="q4-n16k64-direct"`` builds the non-default Q4 direct-reader ABI, while
    ``layout="kquant-kpack"`` selects the per-plane b16 map for Q2/Q3/Q5/Q6; each plane uses Pack=16/bits and the
    exact pack factors travel in the canonical descriptor.  ``tile_k`` is accepted only together with an explicit
    ``layout="xplane"`` development request; automatic production selection never falls back to Xplane.

    tile_k SELECTS AN XPLANE PLACEMENT, and F follows from it -- see formats.fold_for. Passing None keeps the
    Xplane format's default placement when Xplane was explicitly selected.
    An explicit tile_k routes to the *_for_tile op, which is the only way to obtain a folded artifact: the
    default producer pins the fold at 1, so before this existed, `PlacedArrangement` could record arrangements
    nothing could build.
    """
    validate_fully_quantized_resident_geometry(qtype, n, k)
    _check_shape(blocks, n, k, int(qtype))
    if layout == "auto":
        if tile_k is not None:
            raise ValueError(
                "tile_k is an explicit Xplane compatibility setting; use layout='xplane' for a development "
                "artifact instead of changing the canonical automatic layout")
        layout = canonical_fully_quantized_layout(qtype)
    if layout == "q4-kpack4":
        if QuantType(qtype) != QuantType.Q4_K:
            raise ValueError(f"q4-kpack4 is defined only for Q4_K, got {QuantType(qtype).name}")
        if tile_k is not None:
            raise ValueError("q4-kpack4 has no artifact TileK axis; tile_k must be None")
        arrangement = q4_kpack4_arrangement()
        arrangement.validate()
        tensors = _op("gguf_prepare_fully_quantized_dense_for_arrangement_v2")(
            blocks, n, k, int(qtype), *_arrangement_v2_wire(arrangement))
        return PlacedArtifact(tensors, arrangement, PLACED_ARTIFACT_VERSION_V2)
    if layout == "q4-n16k64-direct":
        if QuantType(qtype) != QuantType.Q4_K:
            raise ValueError(
                f"q4-n16k64-direct is defined only for Q4_K, got {QuantType(qtype).name}")
        if tile_k is not None:
            raise ValueError("q4-n16k64-direct has no artifact TileK axis; tile_k must be None")
        # The byte map is N16-atomic, while the existing fully-quantized
        # torch producer has a stricter public tensor ABI: N/K are multiples
        # of 256.  Do not advertise a shape the real producer rejects before
        # it reaches the layout transform.
        if n % 256:
            raise ValueError(f"q4-n16k64-direct producer requires n divisible by 256, got n={n}")
        arrangement = q4_n16k64_direct_arrangement()
        arrangement.validate()
        tensors = _op("gguf_prepare_fully_quantized_dense_for_arrangement_v2")(
            blocks, n, k, int(qtype), *_arrangement_v2_wire(arrangement))
        return PlacedArtifact(tensors, arrangement, PLACED_ARTIFACT_VERSION_V2)
    if layout == "kquant-kpack":
        if QuantType(qtype) == QuantType.Q4_K:
            raise ValueError("Q4_K retains the shipping q4-kpack4 layout; use layout='q4-kpack4'")
        if tile_k is not None:
            raise ValueError("kquant-kpack has no artifact TileK axis; tile_k must be None")
        arrangement = kquant_kpack_arrangement(qtype)
        arrangement.validate()
        tensors = _op("gguf_prepare_fully_quantized_dense_for_arrangement_v2")(
            blocks, n, k, int(qtype), *_arrangement_v2_wire(arrangement))
        return PlacedArtifact(tensors, arrangement, PLACED_ARTIFACT_VERSION_V2)
    if layout != "xplane":
        raise ValueError(
            f"unknown fully-quantized dense layout {layout!r}; expected 'auto', 'xplane', 'q4-kpack4', "
            "'q4-n16k64-direct' or 'kquant-kpack'")
    arrangement = placed_arrangement(qtype, tile_k)
    if tile_k is None:
        tensors = _op("gguf_prepare_fully_quantized_dense")(blocks, n, k, int(qtype))
    else:
        tensors = _op("gguf_prepare_fully_quantized_dense_for_tile")(blocks, n, k, int(qtype), int(tile_k))
    return PlacedArtifact(tensors, arrangement)


def matmul_fully_quantized_dense(a: torch.Tensor, artifact, qtype: int):
    """a @ w.T with NOTHING expanded: the GEMM reads the format's own packed scale bytes.

    Raises rather than falling back when the library was built without PPU_PACKED_SCALE=1 -- the device symbol
    always exists and the launch returns rc=34, which surfaces here as an error naming the macro. A silent
    fallback would make a build that cannot run this cell indistinguishable from one that can.
    """
    low, high, units, arrangement = _require_shipping_kpack_artifact(
        artifact, qtype, "matmul_fully_quantized_dense")
    return _op("gguf_dense_fully_quantized_for_arrangement_v2")(
        a, low, high, units, int(qtype), *_arrangement_v2_wire(arrangement))


def prepare_fully_quantized_grouped(blocks: torch.Tensor, n: int, k: int,
                                    qtype: int, experts: int,
                                    layout: str = "auto"):
    """The MoE artifact: one (low, units) pair per expert, nothing expanded.

    blocks is [E*n*(k/256), type_size] -- expert-major, so expert e's weight is a contiguous slice. ``auto``
    selects the canonical K-pack artifact for every supported format.  Use
    ``layout="q4-n16k64-direct"`` to build the explicit non-default Q4 direct-reader ABI, or
    ``layout="kquant-kpack"`` to build the descriptor-carrying Q2/Q3/Q5/Q6 artifact.
    Returns (low, high, units), with `high` empty for single-plane formats and uint8[E, n, k/8] for Q5_K.
    """
    validate_fully_quantized_resident_geometry(qtype, n, k)
    want = experts * n * (k // 256)
    if blocks.dim() != 2 or blocks.shape[0] != want:
        raise ValueError(f"blocks should be [{want}, type_size] for {experts} experts of an ({n}, {k}) weight, "
                         f"got {tuple(blocks.shape)}")
    if layout == "auto":
        layout = canonical_fully_quantized_layout(qtype)
    if layout == "q4-kpack4":
        if QuantType(qtype) != QuantType.Q4_K:
            raise ValueError(f"q4-kpack4 is defined only for Q4_K, got {QuantType(qtype).name}")
        arrangement = q4_kpack4_arrangement()
        arrangement.validate()
        tensors = _op("gguf_prepare_fully_quantized_grouped_for_arrangement_v2")(
            blocks, n, k, int(qtype), int(experts), *_arrangement_v2_wire(arrangement))
        return PlacedArtifact(tensors, arrangement, PLACED_ARTIFACT_VERSION_V2)
    if layout == "q4-n16k64-direct":
        if QuantType(qtype) != QuantType.Q4_K:
            raise ValueError(
                f"q4-n16k64-direct is defined only for Q4_K, got {QuantType(qtype).name}")
        if n % 256:
            raise ValueError(f"q4-n16k64-direct producer requires n divisible by 256, got n={n}")
        arrangement = q4_n16k64_direct_arrangement()
        arrangement.validate()
        tensors = _op("gguf_prepare_fully_quantized_grouped_for_arrangement_v2")(
            blocks, n, k, int(qtype), int(experts), *_arrangement_v2_wire(arrangement))
        return PlacedArtifact(tensors, arrangement, PLACED_ARTIFACT_VERSION_V2)
    if layout == "kquant-kpack":
        if QuantType(qtype) == QuantType.Q4_K:
            raise ValueError("Q4_K retains the shipping q4-kpack4 layout; use layout='q4-kpack4'")
        arrangement = kquant_kpack_arrangement(qtype)
        arrangement.validate()
        tensors = _op("gguf_prepare_fully_quantized_grouped_for_arrangement_v2")(
            blocks, n, k, int(qtype), int(experts), *_arrangement_v2_wire(arrangement))
        return PlacedArtifact(tensors, arrangement, PLACED_ARTIFACT_VERSION_V2)
    if layout != "xplane":
        raise ValueError(
            f"unknown fully-quantized grouped layout {layout!r}; expected 'auto', 'xplane', 'q4-kpack4', "
            "'q4-n16k64-direct' or 'kquant-kpack'")
    return _op("gguf_prepare_fully_quantized_grouped")(
        blocks, n, k, int(qtype), int(experts))


def matmul_fully_quantized_grouped(a: torch.Tensor, artifact, qtype: int, rows_per_expert: torch.Tensor):
    """The grouped GEMM with the format's own packed scale bytes. Rows are RAGGED and an expert may have none.

    The argument order here is (a, artifact, qtype, rows) to match every other grouped route on this surface;
    the C++ op takes (a, low, units, rows, qtype) and the reorder happens here rather than in the kernel. One
    convention per layer -- the alternative is a caller that has to remember which side it is on.
    """
    low, high, units, arrangement = _require_shipping_kpack_artifact(
        artifact, qtype, "matmul_fully_quantized_grouped")
    return _op("gguf_grouped_fully_quantized_for_arrangement_v2")(
        a, low, high, units, rows_per_expert.to(torch.int32), int(qtype),
        *_arrangement_v2_wire(arrangement))


def matmul_scale_first_dense(a: torch.Tensor, artifact, qtype: int, scale_zero=None) -> torch.Tensor:
    """SCALE_FIRST/DENSE through fpA_intB_ppu.cuh.

    A legacy four-tensor artifact keeps the established Xplane route.  Passing the canonical K-pack4
    ``PlacedArtifact`` reuses its resident code bytes; scale/zero are derived from its packed units (or supplied as
    a hoisted ``scale_zero=(scale, zero)`` workspace) and the persistent v2 reader is selected.  Thus prefill and
    fully-quantized decode share one offline weight format without forcing metadata expansion into the checkpoint.
    """
    if isinstance(artifact, PlacedArtifact):
        low, high, units, arrangement = _require_shipping_kpack_artifact(
            artifact, qtype, "matmul_scale_first_dense")
        if QuantType(qtype) != QuantType.Q4_K:
            raise NotImplementedError(
                "matmul_scale_first_dense: non-Q4 K-pack artifacts have no ScaleFirst reader; use "
                "matmul_kpack_dense, which routes them through fully-quantized compute")
        if scale_zero is None:
            scale_zero = dequantize_scale_from_units(units, qtype)
        if not isinstance(scale_zero, (tuple, list)) or len(scale_zero) != 2:
            raise ValueError("scale_zero must be the (scale, zero) pair derived from this artifact's units")
        scale, zero = scale_zero
        return _op("gguf_dense_scale_first_for_arrangement_v2")(
            a, low, high, scale, zero, int(qtype), *_arrangement_v2_wire(arrangement))
    low, high, scale, zero = artifact
    return gguf_dense_scale_first(a, low, high, scale, zero, int(qtype))


KPACK4_SCALEFIRST_MIN_ROWS = 64


def prepare_q4_kpack4_scale_workspace(artifact, qtype: int = QuantType.Q4_K):
    """Hoist the fp16 affine workspace used by K-pack4 prefill.

    The returned tensors are runtime workspace, not a second checkpoint artifact. Keeping this operation explicit
    prevents the dispatcher below from rebuilding scale/zero inside every timed prefill call.
    """
    _low, _high, units, _arrangement = _require_shipping_kpack_artifact(
        artifact, qtype, "prepare_q4_kpack4_scale_workspace")
    if QuantType(qtype) != QuantType.Q4_K:
        raise NotImplementedError(
            "prepare_q4_kpack4_scale_workspace is Q4_K-only; non-Q4 K-pack compute does not use a "
            "ScaleFirst workspace")
    return dequantize_scale_from_units(units, qtype)


def matmul_kpack_dense(a: torch.Tensor, artifact, qtype: int,
                       scale_workspace=None) -> torch.Tensor:
    """Production dense dispatcher over the canonical K-pack artifact.

    Q4 uses fully-quantized K-pack4 below the persistent kernel's admitted
    ``M>=64`` boundary.  The measured decode band is ``M<=8``; ``M=9..63``
    stays on the same fully-quantized reader instead of being sent to a
    ScaleFirst kernel that explicitly rejects it.  At ``M>=64`` Q4 uses
    persistent ScaleFirst over the same code bytes and requires a workspace
    returned by :func:`prepare_q4_kpack4_scale_workspace`. Q2/Q3/Q5/Q6 use
    their fully-quantized K-pack readers for every M.
    """
    if a.dim() != 2:
        raise ValueError(f"activation must be [M,K], got {tuple(a.shape)}")
    _require_shipping_kpack_artifact(artifact, qtype, "matmul_kpack_dense")
    if QuantType(qtype) != QuantType.Q4_K:
        if scale_workspace is not None:
            raise ValueError(
                "non-Q4 K-pack compute is fully quantized for every M and does not accept a ScaleFirst workspace")
        return matmul_fully_quantized_dense(a, artifact, qtype)
    if a.shape[0] < KPACK4_SCALEFIRST_MIN_ROWS:
        if scale_workspace is not None:
            raise ValueError(
                "Q4 scale_workspace is accepted only by the M>=64 persistent ScaleFirst route")
        return matmul_fully_quantized_dense(a, artifact, qtype)
    if scale_workspace is None:
        raise ValueError(
            "K-pack4 prefill requires a hoisted scale_workspace; call "
            "prepare_q4_kpack4_scale_workspace(artifact) once outside the hot path")
    return matmul_scale_first_dense(a, artifact, qtype, scale_zero=scale_workspace)


def matmul_q4_kpack4_dense(a: torch.Tensor, artifact, qtype: int = QuantType.Q4_K,
                           scale_workspace=None) -> torch.Tensor:
    """Backward-compatible Q4 name for :func:`matmul_kpack_dense`."""
    if QuantType(qtype) != QuantType.Q4_K:
        _require_shipping_kpack_artifact(
            artifact, qtype, "matmul_q4_kpack4_dense")
        raise ValueError("matmul_q4_kpack4_dense accepts only Q4_K; use matmul_kpack_dense for other K-quants")
    return matmul_kpack_dense(
        a, artifact, qtype, scale_workspace=scale_workspace)


ROUTES = {
    "dequant_first": matmul_dequant_first,
    "native_gemv": matmul_native_gemv,
    "native_gemv_moe": matmul_native_gemv_moe,
    "scale_first_gemv": matmul_scale_first_gemv,
    "scale_first_gemv_moe": matmul_scale_first_gemv_moe,
    "scale_first_dense": matmul_scale_first_dense,
    "kpack_dense": matmul_kpack_dense,
    "q4_kpack4_dense": matmul_q4_kpack4_dense,
}
