#!/usr/bin/env python3
"""GGUF -> BC artifacts, one per tensor, with the arrangement recorded beside them.

WHY THIS DOES NOT WAIT FOR THE ON-DISK FORMAT. Producing (low, high, units) has been possible and
device-validated for a while; what is still undecided is the CONTAINER llama.cpp will read. Those are separate,
and coupling them would have held a working tool behind a decision. So the writer is one small function at the
bottom and everything above it is container-agnostic: when the GGUF-native form lands, that function changes and
nothing else does.

IT IS ALSO USEFUL BEFORE llama.cpp EXISTS AS A CONSUMER. Every artifact this repo has measured so far came from a
SYNTHESISED fixture -- random code bytes with sane fp16 headers, chosen because the official gguf package has no
k-quant quantiser to ask for the bytes of a given weight. That fixture is deliberate and it covers the code space
better than a real checkpoint would, but it cannot answer questions about a real model's shapes, its mix of
formats, or how long packing one takes.

WHAT IT REFUSES TO DO, and each refusal is a mistake this project has made or nearly made:
  * it does not GUESS an arrangement. The (fold, tile_k) a tensor is packed for is recorded per tensor, because
    dense and MoE want different folds and a header that omits them leaves the reader to assume.
  * it does not silently skip a tensor. Anything unsupported is listed with its type, so "packed the model" and
    "packed the tensors we happened to handle" are distinguishable.
  * it does not write a partial artifact on failure. A directory that exists is a directory that finished.

    python3 tools/pack_gguf.py MODEL.gguf OUT_DIR \
        [--layout-policy canonical|all-kpack] [--limit N] [--dry-run]

Needs the device library (the placement lives there), so it runs on the box. The default Q4 K-pack4 path uses
the format-selected FMT0 handle:
    QUACTLIZE_PPU_LIB_FMT0=<...>/libquactlize_ppu.so python3 tools/pack_gguf.py ...

``--layout-policy all-kpack`` is the format-unification producer:

  Q2_K -> low2 Pack8
  Q3_K -> low2 Pack8 + high1 Pack16
  Q4_K -> low4 Pack4 (the shipping q4-kpack4 descriptor)
  Q5_K -> low4 Pack4 + high1 Pack16 (including its proved high-plane transpose)
  Q6_K -> low4 Pack4 + high2 Pack8

Every plane is stored as converter-native little-endian b16 words and metadata
stays in the byte-neutral packed-unit channel. ``canonical`` preserves the
currently adjudicated automatic policy while the five-format performance run
is in flight; selecting ``all-kpack`` is explicit and recorded in the manifest.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .gguf file")
    ap.add_argument("out", help="output directory; created, and only finished artifacts land in it")
    ap.add_argument("--limit", type=int, default=0, help="stop after N packable tensors (0 = all)")
    ap.add_argument(
        "--layout-policy", choices=("canonical", "all-kpack"), default="canonical",
        help=("canonical keeps the current automatic shipping policy; all-kpack emits q4-kpack4 for Q4_K and "
              "the per-plane K-pack map for Q2/Q3/Q5/Q6"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be packed, touching no device and writing nothing")
    a = ap.parse_args()
    if a.limit < 0:
        ap.error("--limit must be nonnegative (0 means all packable tensors)")

    try:
        from gguf import GGUFReader
    except ImportError:
        print("needs the official gguf package: pip install gguf", file=sys.stderr)
        return 2

    from quactlize import formats as F

    reader = GGUFReader(a.model)
    # THE SUPPORTED SET COMES FROM THE FORMAT TABLE, not from a list here. A second list is a second place to
    # forget a format, and this file would forget it silently -- the tensor would land in "skipped" looking like
    # an unsupported type rather than like an omission.
    supported = {int(q): q for q in (F.QuantType.Q2_K, F.QuantType.Q3_K, F.QuantType.Q4_K,
                                     F.QuantType.Q5_K, F.QuantType.Q6_K)}

    seen, packable, skipped = Counter(), [], []
    route_mix = Counter()
    for t in reader.tensors:
        tt = int(t.tensor_type)
        seen[t.tensor_type.name] += 1
        ok, route, why = _packability(
            tt, len(t.shape), supported, a.layout_policy)
        if ok and route == "grouped":
            ok, why = _grouped_role_authority(t.name)
        if ok:
            try:
                _tensor_geometry(t.shape)
            except ValueError as exc:
                skipped.append((t.name, t.tensor_type.name, str(exc)))
            else:
                packable.append((t, route))
                route_mix[route] += 1
        else:
            skipped.append((t.name, t.tensor_type.name, why))

    print(f"model      {a.model}")
    print(f"tensors    {sum(seen.values())}  ->  {len(packable)} packable, {len(skipped)} skipped")
    print(f"type mix   {dict(seen)}")
    print(f"routes     {dict(route_mix)}")
    print(f"layout     {a.layout_policy}")
    # THE MIX IS THE POINT OF PRINTING IT. A _K_M checkpoint is named for its dominant format and carries others,
    # and one device library serves ONE PPU_PACKED_FORMAT -- so this line says how many libraries a deployment of
    # this model needs, which is a fact about our build rather than about the file.
    kq = {n: c for n, c in seen.items() if n in {q.name for q in supported.values()}}
    if len(kq) > 1:
        print(f"           MIXED: {len(kq)} k-quant formats in one file {kq} -- one library per format")

    if skipped:
        print(f"\nskipped ({len(skipped)}), first 10:")
        for n, ty, why in skipped[:10]:
            print(f"  {n:<48} {ty:<8} {why}")

    if a.dry_run:
        print("\n--dry-run: nothing written, no device touched")
        return 0

    import torch
    from quactlize import routes
    import quactlize

    todo = packable[:a.limit] if a.limit else packable
    # Check exactly the handles this plan will call. Every layout-v2 producer is
    # owned by its format-selected library; canonical non-Q4 Xplane remains on
    # the legacy/default handle. Asking only one makes a mixed-format model fail
    # late at its first tensor of another physical layout.
    v2_qtypes = {
        int(t.tensor_type) for t, _route in todo
        if _target_layout(int(t.tensor_type), a.layout_policy)
        in ("q4-kpack4", "kquant-kpack")
    }
    uses_default = any(
        _target_layout(int(t.tensor_type), a.layout_policy) == "xplane"
        for t, _route in todo)
    required_backends = {}
    for qtype in sorted(v2_qtypes):
        name = F.QuantType(qtype).name
        required_backends[f"{name} layout-v2"] = quactlize.gguf_backend_for_qtype(qtype)
    if uses_default:
        required_backends["canonical non-Q4 Xplane"] = quactlize.gguf_backend()
    unavailable = {name: value for name, value in required_backends.items() if not value.startswith("ppu")}
    if unavailable:
        details = "\n".join(f"  {name}: {value}" for name, value in unavailable.items())
        print(f"\nrefusing to pack because required device placement backend(s) are unavailable:\n{details}\n"
              "  Set the matching QUACTLIZE_PPU_LIB_FMT* handle for each layout-v2 qtype, and "
              "QUACTLIZE_PPU_LIB only when canonical Xplane tensors are requested.",
              file=sys.stderr)
        return 3

    if not todo:
        print("\nrefusing to create an empty artifact bundle: this plan contains no packable tensors", file=sys.stderr)
        return 4

    final_out = pathlib.Path(a.out)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if final_out.exists():
        print(f"refusing to overwrite existing output {final_out}", file=sys.stderr)
        return 4
    # Root-level atomic publication. A failed run leaves an explicitly named diagnostic directory and never makes
    # the requested final path exist; individual tensor atomic renames alone were insufficient because the old
    # root directory appeared complete after the first tensor.
    out = final_out.with_name(final_out.name + f".partial.{os.getpid()}")
    if out.exists():
        print(f"refusing to reuse partial output {out}", file=sys.stderr)
        return 4
    out.mkdir()
    manifest, t0 = [], time.time()
    for i, (t, route) in enumerate(todo):
        n, k, experts = _tensor_geometry(t.shape)
        qtype = int(t.tensor_type)
        # GGUF dimensions are fast-first [K,N,(E)].  Flattening therefore leaves one expert's N rows
        # contiguous and experts adjacent, exactly the grouped producer's [E*N*(K/256), type_size] ABI.
        block_rows = (experts or 1) * n * (k // 256)
        blocks = torch.from_numpy(t.data.reshape(block_rows, -1).copy())
        layout, artifact = _prepare_artifact(
            routes, blocks, n, k, qtype, experts, route, a.layout_policy)
        low, high, units = artifact

        stem = t.name.replace("/", "_").replace(".", "_")
        tmp = out / (stem + ".partial")
        tmp.mkdir(exist_ok=True)
        _write(tmp, low, high, units)
        tmp.rename(out / stem)                          # a directory that exists is a directory that finished

        # THE ARRANGEMENT IS NOT INFERRED FROM THE OUTPUT. The selected policy
        # names one producer, that producer returns the descriptor for the bytes
        # it built, and this exact expected value prevents a manifest from
        # relabelling Xplane as K-pack (or one K-pack mapping as another).
        arr = artifact.arrangement
        expected = (
            F.q4_kpack4_arrangement() if layout == "q4-kpack4" else
            F.kquant_kpack_arrangement(qtype) if layout == "kquant-kpack" else
            F.PlacedArrangement(
                bits=_low_bits(qtype), tile_k=_tile_k(qtype),
                high_bits=_high_bits(qtype)))
        if arr != expected:
            raise RuntimeError(
                f"{t.name}: producer returned arrangement {arr}, pack plan expected {expected}; refusing to write "
                f"a manifest that describes different bytes from the ones just produced")
        low_bits, high_bits = F.placed_code_planes(qtype)
        manifest.append({"name": t.name, "dir": stem, "ggml_type": qtype, "type_name": t.tensor_type.name,
                         "route_class": route, "layout_name": layout,
                         "plane_packs": {"low": 16 // low_bits,
                                         "high": 16 // high_bits if high_bits else 0},
                         "rank": len(t.shape), "n": n, "k": k, "experts": experts,
                         "arrangement_version": artifact.arrangement_version,
                         "arrangement": arr._asdict(),
                         "fold": ([arr.fold, arr.high_fold]
                                  if isinstance(arr, F.PlacedArrangement) else None),
                         "shapes": {"low": list(low.shape), "high": list(high.shape), "units": list(units.shape)}})
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  packed {i+1}/{len(todo)}  ({time.time()-t0:.1f}s)")

    held_back = [
        {"name": t.name, "type_name": t.tensor_type.name, "reason": "held back by --limit"}
        for t, _route in packable[len(todo):]
    ] if a.limit else []
    skipped_manifest = [
        {"name": name, "type_name": type_name, "reason": reason}
        for name, type_name, reason in skipped
    ]
    (out / "manifest.json").write_text(json.dumps({
        "artifact_schema_version": 2,
        "model": a.model,
        "selection": {"layout_policy": a.layout_policy, "limit": a.limit,
                      "packable_total": len(packable), "packed": len(manifest),
                      "skipped": len(skipped_manifest), "held_back_by_limit": len(held_back)},
        "tensors": manifest,
        "skipped": skipped_manifest,
        "held_back_by_limit": held_back,
    }, indent=2) + "\n")
    out.rename(final_out)
    print(f"\nwrote {len(manifest)} artifact(s) + manifest.json to {final_out}")
    print("The manifest carries the ARRANGEMENT and route class per tensor, not per format: dense and grouped\n"
          "readers must consume the exact bytes their producer built; a reader that infers either will infer wrongly.")
    return 0


def _target_layout(qtype: int, layout_policy: str) -> str:
    """Resolve one explicit whole-model layout policy without duplicating plane widths.

    The arrangement constructors derive Pack=16/bits from the shared format
    registry. Keeping this function at the layout-name level prevents the GGUF
    tool from growing a second Q2/Q3/Q5/Q6 plane table.
    """
    from quactlize import formats as F
    q = F.QuantType(int(qtype))
    if layout_policy == "canonical":
        return F.canonical_fully_quantized_layout(q)
    if layout_policy == "all-kpack":
        return "q4-kpack4" if q == F.QuantType.Q4_K else "kquant-kpack"
    raise ValueError(
        f"unknown layout policy {layout_policy!r}; expected 'canonical' or 'all-kpack'")


def _prepare_artifact(routes, blocks, n: int, k: int, qtype: int,
                      experts, route: str, layout_policy: str):
    """Raw GGUF block tensor -> exact resident (low, high, units) artifact."""
    layout = _target_layout(qtype, layout_policy)
    if route == "grouped":
        if experts is None:
            raise ValueError("grouped K-pack preparation requires an expert extent")
        artifact = routes.prepare_fully_quantized_grouped(
            blocks, n, k, qtype, experts, layout=layout)
    elif route == "dense":
        if layout in ("q4-kpack4", "kquant-kpack"):
            artifact = routes.prepare_fully_quantized_dense(
                blocks, n, k, qtype, layout=layout)
        else:
            artifact = routes.prepare_fully_quantized_dense(
                blocks, n, k, qtype, tile_k=_tile_k(qtype))
    else:
        raise ValueError(f"unknown GGUF pack route {route!r}")
    return layout, artifact


def _packability(qtype: int, rank: int, supported,
                 layout_policy: str = "canonical"):
    """Return ``(packable, route, reason)`` without consulting tensor names.

    Rank three is GGUF's grouped ``[K,N,E]`` storage, not a dense matrix with
    an ignorable axis. The all-Kpack policy has descriptor-aware grouped
    producers/readers for all five formats. Canonical Xplane rank-three rows
    remain visible but held back rather than losing their descriptor.
    """
    if int(qtype) not in supported:
        return False, None, "not a k-quant this build packs"
    if rank == 2:
        return True, "dense", None
    if rank == 3:
        if _target_layout(qtype, layout_policy) not in (
                "q4-kpack4", "kquant-kpack"):
            return False, None, (
                "3-D grouped artifact lacks a descriptor-aware reader under "
                f"layout policy {layout_policy}")
        return True, "grouped", None
    return False, None, f"{rank}-D, expected dense rank 2 or grouped rank 3"


def _tensor_geometry(shape):
    """Translate GGUF fast-first dimensions to the route ABI's ``(N,K,E-or-None)``."""
    dims = tuple(int(x) for x in shape)
    if len(dims) not in (2, 3):
        raise ValueError(f"GGUF tensor rank must be 2 (dense) or 3 (grouped), got shape={dims}")
    k, n = dims[:2]
    experts = dims[2] if len(dims) == 3 else None
    if k <= 0 or n <= 0 or (experts is not None and experts <= 0):
        raise ValueError(f"GGUF tensor dimensions must be positive, got shape={dims}")
    if k % 256 or n % 256:
        raise ValueError(f"resident tensor-core artifact requires N/K multiples of 256, got N={n} K={k}")
    return n, k, experts


def _grouped_role_authority(name: str):
    """Prove that a rank-3 tensor is a recognised GGML ``MUL_MAT_ID`` weight.

    Rank alone is not an operation. Reuse the immutable inventory's exact llama.cpp tensor-symbol rules instead of
    maintaining a second regex list here; unknown rank-3 tensors remain visible in the skipped manifest.
    """
    from tools.gguf_internal_shape_inventory import InventoryError, classify_role
    try:
        role, source = classify_role(name, 3)
    except InventoryError as exc:
        return False, f"rank-3 tensor has no grouped role authority: {exc}"
    if role.route_class != "grouped" or role.operation != "MUL_MAT_ID":
        return False, (f"rank-3 tensor role {role.name} is {role.route_class}/{role.operation}, "
                       "not grouped/MUL_MAT_ID")
    return True, None


# THE PLANE WIDTHS COME FROM THE REGISTRY TOO. These used to decode schemes.CODE_PLANE's string tags through a
# dict written here -- correct, and a third spelling of "which planes does this format have" alongside the .inc
# and PPU_PACKED_FORMAT. The high plane folds independently of the low one (Q3's int1 plane at TK=64 needs F2=4
# while its int2 low plane needs 2), which is why it is a stored column rather than something derived from the
# low width.
def _low_bits(qtype: int) -> int:
    return format_registry()[int(qtype)]["low_bits"]


def _high_bits(qtype: int) -> int:
    return format_registry()[int(qtype)]["high_bits"]


def format_registry() -> dict:
    """-> {qtype: {name, low_bits, high_bits, group_size, scale_first_tile_k, fully_quantized_tile_k,
    packed_format}} parsed from quactlize/include/ppu_format_config.inc.

    PARSED, NOT MIRRORED. The .inc is an X-macro precisely so that C++ can include it and Python can read the
    same bytes; a Python dict repeating the rows would be the fifth copy of the decision this file exists to
    stop making. A row whose arity changed raises rather than silently yielding fewer fields -- a partial parse
    that still produces a plausible TileK is the exact failure mode that started this.
    """
    inc = pathlib.Path(__file__).resolve().parent.parent / "quactlize" / "include" / "ppu_format_config.inc"
    if not inc.is_file():
        raise FileNotFoundError(f"the shipping format registry is missing: {inc}")
    fields = ("name", "qtype", "low_bits", "high_bits", "group_size",
              "scale_first_tile_k", "fully_quantized_tile_k", "packed_format")
    out = {}
    for m in re.finditer(r"^\s*X\((.*?)\)\s*\\?\s*$", inc.read_text(), re.M):
        args = [a.strip().strip('"') for a in m.group(1).split(",")]
        if len(args) != 1 + len(fields):
            raise ValueError(f"{inc.name}: row has {len(args)} args, expected {1+len(fields)}: {m.group(1)!r}")
        row = dict(zip(fields, args[1:]))
        for k in fields[1:]:
            row[k] = int(row[k])
        out[row["qtype"]] = row
    if not out:
        raise ValueError(f"{inc.name}: no X(...) rows parsed; the parser is wrong, not the registry")
    return out


def _tile_k(qtype: int) -> int:
    """The FULLY_QUANTIZED TileK for this format, READ FROM THE SHIPPING REGISTRY rather than restated here.

    This used to be `128 if Q6_K else 256`, written out in Python. The value was right and the fork was not:
    TileK was being decided in four places -- this function, the library's dispatch, the bench's per-width
    default, and the emitter's argument -- and two of them agreeing is what let a wrong value pass a
    consistency check. quactlize/include/ppu_format_config.inc is now the one source, and its own header says
    the offline packer should parse it; this is that.

    THE REASON THE VALUE IS WHAT IT IS still belongs somewhere, and it is in the .inc: the packed metadata unit
    covers a whole 256-code superblock and one k-tile consuming it measured best, while Q6_K keeps 128 because
    its 256-K high-plane inverse is incomplete -- which produced conditioned error 8.76e-1 before it was caught.
    """
    return format_registry()[int(qtype)]["fully_quantized_tile_k"]


def _write(d: pathlib.Path, low, high, units) -> None:
    """THE CONTAINER, and the only part that changes when the on-disk format is decided.

    Raw .npy today: it needs no schema, any reader has it, and nothing above this function knows about it. When
    the GGUF-native representation lands this becomes a writer into that, and the rest of the tool is unchanged.
    """
    import numpy as np
    for name, t in (("low", low), ("high", high), ("units", units)):
        np.save(d / f"{name}.npy", t.numpy())


def restore_artifact(root: pathlib.Path, record: dict):
    """Restore one manifest entry without dropping the descriptor that makes its code planes readable.

    This is intentionally the inverse of _write plus the manifest record, not a convenience that returns three
    tensors. Returning a plain tuple here would make every correct reader reject it -- and permitting that tuple
    would put guessing back into the ABI. Unknown/missing versions fail closed before any bytes are loaded.
    """
    import numpy as np
    import torch
    from quactlize import formats as F
    from quactlize import routes

    version = record.get("arrangement_version")
    raw = record.get("arrangement")
    if version == routes.PLACED_ARTIFACT_VERSION:
        if not isinstance(raw, dict) or set(raw) != {"bits", "tile_k", "high_bits"}:
            raise ValueError(
                f"artifact {record.get('name', '<unnamed>')}: v1 arrangement must contain exactly "
                f"bits/tile_k/high_bits, got {raw!r}")
        arrangement = F.PlacedArrangement(int(raw["bits"]), int(raw["tile_k"]), int(raw["high_bits"]))
        _ = arrangement.fold, arrangement.high_fold
        expected = F.placed_arrangement(int(record["ggml_type"]), arrangement.tile_k)
    elif version == routes.PLACED_ARTIFACT_VERSION_V2:
        fields = set(F.PlacedArrangementV2._fields)
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError(
                f"artifact {record.get('name', '<unnamed>')}: arrangement_version=2 requires exactly "
                f"{sorted(fields)}, got {raw!r}")
        arrangement = F.PlacedArrangementV2(*(int(raw[name]) for name in F.PlacedArrangementV2._fields))
        arrangement.validate()
        qtype = F.QuantType(int(record["ggml_type"]))
        expected = (F.q4_kpack4_arrangement()
                    if qtype == F.QuantType.Q4_K
                    else F.kquant_kpack_arrangement(qtype))
    else:
        raise ValueError(
            f"artifact {record.get('name', '<unnamed>')}: arrangement_version={version!r}; this build reads "
            f"versions {routes.PLACED_ARTIFACT_VERSION} and {routes.PLACED_ARTIFACT_VERSION_V2}. Missing is "
            "legacy/ambiguous, not version 1")
    if arrangement != expected:
        raise ValueError(
            f"artifact {record.get('name', '<unnamed>')}: manifest arrangement {arrangement} disagrees with "
            f"ggml_type={record['ggml_type']} ({expected})")
    d = pathlib.Path(root) / record["dir"]
    tensors = tuple(torch.from_numpy(np.load(d / f"{name}.npy")) for name in ("low", "high", "units"))
    return routes.PlacedArtifact(tensors, arrangement, version)


if __name__ == "__main__":
    raise SystemExit(main())
