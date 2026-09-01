#!/usr/bin/env python3
"""GGUF -> production K-pack bundle, with one placed artifact per tensor.

The directory container is a named, versioned interchange boundary. It is not
a rewritten GGUF file: each tensor directory contains the three resident arrays
``low.npy``, ``high.npy`` and ``units.npy``, while ``manifest.json`` binds those
bytes to their exact arrangement descriptor. ``load_kpack_bundle`` validates
the complete directory before returning any artifact.

IT IS ALSO USEFUL BEFORE llama.cpp EXISTS AS A CONSUMER. Every artifact this repo has measured so far came from a
SYNTHESISED fixture -- random code bytes with sane fp16 headers, chosen because the official gguf package has no
k-quant quantiser to ask for the bytes of a given weight. That fixture is deliberate and it covers the code space
better than a real checkpoint would, but it cannot answer questions about a real model's shapes, its mix of
formats, or how long packing one takes.

WHAT IT REFUSES TO DO, and each refusal is a mistake this project has made or nearly made:
  * it does not GUESS an arrangement. The versioned K-pack descriptor is recorded per tensor so a reader cannot
    reinterpret bytes through another layout or per-plane pack factor.
  * it does not silently skip a tensor. Anything unsupported is listed with its type, so "packed the model" and
    "packed the tensors we happened to handle" are distinguishable.
  * it does not write a partial artifact on failure. A directory that exists is a directory that finished.

    quactlize-pack-gguf MODEL.gguf OUT_DIR [--dry-run]

Placement is host code exported by the format-selected PPU library. Conversion
does not launch a PPU kernel, but the matching library and its runtime
dependencies must be loadable. A complete install is selected with
``QUACTLIZE_PPU_BUNDLE``; an individual Q4 K-pack4 override uses FMT0:
    QUACTLIZE_PPU_BUNDLE=<...>/ppu0010 quactlize-pack-gguf ...
    QUACTLIZE_PPU_LIB_FMT0=<...>/libquactlize_ppu.so quactlize-pack-gguf ...

The sole canonical format-unification policy is:

  Q2_K -> low2 Pack8
  Q3_K -> low2 Pack8 + high1 Pack16
  Q4_K -> low4 Pack4 (the shipping q4-kpack4 descriptor)
  Q5_K -> low4 Pack4 + high1 Pack16 (including its proved high-plane transpose)
  Q6_K -> low4 Pack4 + high2 Pack8

Every plane is stored as converter-native little-endian b16 words and metadata
stays in the byte-neutral packed-unit channel. Xplane is not an automatic
fallback and cannot be selected by this product packer.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter
from typing import NamedTuple


KPACK_BUNDLE_SCHEMA = "quactlize.kquant-kpack.bundle"
KPACK_BUNDLE_VERSION = 1
KPACK_BUNDLE_MANIFEST = "manifest.json"
_BUNDLE_TOP_LEVEL_FIELDS = {
    "schema", "schema_version", "arrangement_version", "model", "selection",
    "tensors", "skipped",
}
_BUNDLE_TENSOR_FIELDS = {
    "name", "dir", "ggml_type", "type_name", "route_class", "layout_name",
    "plane_packs", "rank", "n", "k", "experts", "arrangement_version",
    "arrangement", "shapes", "sha256",
}
_BUNDLE_ARRAYS = ("low", "high", "units")


class KPackBundle(NamedTuple):
    """A validated manifest and its descriptor-carrying resident artifacts."""

    manifest: dict
    artifacts: dict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .gguf file")
    ap.add_argument("out", help="output directory; created, and only finished artifacts land in it")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be packed, touching no device and writing nothing")
    a = ap.parse_args(argv)

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
        rank = len(t.shape)
        ok, route, why = _packability(tt, rank, supported)
        if ok:
            ok, why = _route_role_authority(t.name, rank, route)
        if ok:
            try:
                _tensor_geometry(t.shape, tt)
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
    print("layout     canonical K-pack")
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

    todo = packable
    # Check exactly the format-selected handles this plan will call. Asking only
    # one makes a mixed-format model fail late at its first tensor of another
    # physical layout.
    v2_qtypes = {
        int(t.tensor_type) for t, _route in todo
        if _target_layout(int(t.tensor_type)) in ("q4-kpack4", "kquant-kpack")
    }
    required_backends = {}
    for qtype in sorted(v2_qtypes):
        name = F.QuantType(qtype).name
        required_backends[f"{name} layout-v2"] = quactlize.gguf_backend_for_qtype(qtype)
    unavailable = {name: value for name, value in required_backends.items() if not value.startswith("ppu")}
    if unavailable:
        details = "\n".join(f"  {name}: {value}" for name, value in unavailable.items())
        print(f"\nrefusing to pack because required device placement backend(s) are unavailable:\n{details}\n"
              "  Set QUACTLIZE_PPU_BUNDLE to a verified six-library bundle, or set the matching "
              "QUACTLIZE_PPU_LIB_FMT* handle for each qtype.",
              file=sys.stderr)
        return 3

    if not todo:
        print("\nrefusing to create an empty artifact bundle: this plan contains no packable tensors", file=sys.stderr)
        return 4

    final_out = pathlib.Path(a.out)
    try:
        out = _create_bundle_staging_root(final_out)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    manifest, t0 = [], time.time()
    for i, (t, route) in enumerate(todo):
        qtype = int(t.tensor_type)
        n, k, experts = _tensor_geometry(t.shape, qtype)
        # GGUF dimensions are fast-first [K,N,(E)].  Flattening therefore leaves one expert's N rows
        # contiguous and experts adjacent, exactly the grouped producer's [E*N*(K/256), type_size] ABI.
        block_rows = (experts or 1) * n * (k // 256)
        blocks = torch.from_numpy(t.data.reshape(block_rows, -1).copy())
        layout, artifact = _prepare_artifact(
            routes, blocks, n, k, qtype, experts, route)
        low, high, units = artifact

        stem = _tensor_dir_name(i, t.name)
        tmp = out / (stem + ".partial")
        tmp.mkdir(exist_ok=True)
        _write(tmp, low, high, units)
        tmp.rename(out / stem)                          # a directory that exists is a directory that finished

        # THE ARRANGEMENT IS NOT INFERRED FROM THE OUTPUT. The selected policy
        # names one producer, that producer returns the descriptor for the bytes
        # it built, and this exact expected value prevents a manifest from
        # relabelling Xplane as K-pack (or one K-pack mapping as another).
        arr = artifact.arrangement
        expected = (F.q4_kpack4_arrangement() if layout == "q4-kpack4"
                    else F.kquant_kpack_arrangement(qtype))
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
                         "shapes": {"low": list(low.shape), "high": list(high.shape), "units": list(units.shape)},
                         "sha256": _bundle_file_hashes(out / stem)})
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  packed {i+1}/{len(todo)}  ({time.time()-t0:.1f}s)")

    skipped_manifest = [
        {"name": name, "type_name": type_name, "reason": reason}
        for name, type_name, reason in skipped
    ]
    (out / KPACK_BUNDLE_MANIFEST).write_text(json.dumps({
        "schema": KPACK_BUNDLE_SCHEMA,
        "schema_version": KPACK_BUNDLE_VERSION,
        "arrangement_version": 2,
        "model": a.model,
        "selection": {"layout_policy": "production-kpack-only",
                      "packable_total": len(packable), "packed": len(manifest),
                      "skipped": len(skipped_manifest)},
        "tensors": manifest,
        "skipped": skipped_manifest,
    }, indent=2) + "\n")
    out.rename(final_out)
    print(f"\nwrote {len(manifest)} artifact(s) + manifest.json to {final_out}")
    print("The manifest carries the ARRANGEMENT and route class per tensor, not per format: dense and grouped\n"
          "readers must consume the exact bytes their producer built; a reader that infers either will infer wrongly.")
    return 0


def _target_layout(qtype: int) -> str:
    """Resolve the sole whole-model layout policy without duplicating plane widths.

    The arrangement constructors derive Pack=16/bits from the shared format
    registry. Keeping this function at the layout-name level prevents the GGUF
    tool from growing a second Q2/Q3/Q5/Q6 plane table.
    """
    from quactlize import formats as F
    return F.canonical_fully_quantized_layout(F.QuantType(int(qtype)))


def _tensor_dir_name(index: int, name: str) -> str:
    """Return a unique, deterministic directory without normalising a tensor name.

    Replacing punctuation is not injective (``a/b`` and ``a.b`` used to collide).
    The ordinal makes every directory unique and the digest binds it to the name;
    the loader recomputes both, so neither field can be silently edited.
    """
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"tensor index must be a nonnegative integer, got {index!r}")
    if not isinstance(name, str) or not name:
        raise ValueError("tensor name must be a nonempty string")
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"tensor-{index:06d}-{digest}"


def _create_bundle_staging_root(final_out: pathlib.Path) -> pathlib.Path:
    """Create the sibling staging root without replacing any directory entry.

    ``Path.exists()`` follows symlinks and is false for a dangling one. Such a
    link is still an existing output chosen by the caller, so both the final and
    process-specific staging names reject it explicitly before publication.
    """
    final_out = pathlib.Path(final_out)
    final_out.parent.mkdir(parents=True, exist_ok=True)
    if final_out.exists() or final_out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output {final_out}")
    out = final_out.with_name(final_out.name + f".partial.{os.getpid()}")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to reuse partial output {out}")
    out.mkdir()
    return out


def _prepare_artifact(routes, blocks, n: int, k: int, qtype: int,
                      experts, route: str):
    """Raw GGUF block tensor -> exact resident (low, high, units) artifact."""
    layout = _target_layout(qtype)
    if route == "grouped":
        if experts is None:
            raise ValueError("grouped K-pack preparation requires an expert extent")
        artifact = routes.prepare_fully_quantized_grouped(
            blocks, n, k, qtype, experts, layout=layout)
    elif route == "dense":
        artifact = routes.prepare_fully_quantized_dense(
            blocks, n, k, qtype, layout=layout)
    else:
        raise ValueError(f"unknown GGUF pack route {route!r}")
    return layout, artifact


def _packability(qtype: int, rank: int, supported):
    """Return ``(packable, route, reason)`` without consulting tensor names.

    Rank three is GGUF's grouped ``[K,N,E]`` storage, not a dense matrix with
    an ignorable axis. The canonical K-pack policy has descriptor-aware
    grouped producers/readers for all five formats.
    """
    if int(qtype) not in supported:
        return False, None, "not a k-quant this build packs"
    if rank == 2:
        return True, "dense", None
    if rank == 3:
        if _target_layout(qtype) not in ("q4-kpack4", "kquant-kpack"):
            raise AssertionError(
                f"canonical layout for qtype={qtype} is not a descriptor-aware K-pack")
        return True, "grouped", None
    return False, None, f"{rank}-D, expected dense rank 2 or grouped rank 3"


def _tensor_geometry(shape, qtype: int):
    """Translate GGUF fast-first dimensions to the route ABI's ``(N,K,E-or-None)``."""
    dims = tuple(int(x) for x in shape)
    if len(dims) not in (2, 3):
        raise ValueError(f"GGUF tensor rank must be 2 (dense) or 3 (grouped), got shape={dims}")
    k, n = dims[:2]
    experts = dims[2] if len(dims) == 3 else None
    if k <= 0 or n <= 0 or (experts is not None and experts <= 0):
        raise ValueError(f"GGUF tensor dimensions must be positive, got shape={dims}")
    from quactlize import formats as F
    F.validate_fully_quantized_resident_geometry(qtype, n, k)
    return n, k, experts


def _route_role_authority(name: str, rank: int, route: str):
    """Prove that a tensor is the exact GGML matrix operation its route serves.

    Rank and qtype do not identify an operation: embeddings and SSM convolution
    weights can be rank two, and only recognised ``MUL_MAT_ID`` weights may use
    the grouped route. Reuse the inventory's exact llama.cpp tensor-symbol
    rules so unknown or non-matrix tensors stay visible in the skipped manifest.
    """
    from quactlize.gguf_roles import InventoryError, classify_role
    try:
        role, source = classify_role(name, rank)
    except InventoryError as exc:
        return False, f"rank-{rank} tensor has no {route} role authority: {exc}"
    expected = {
        "dense": ("dense", "MUL_MAT"),
        "grouped": ("grouped", "MUL_MAT_ID"),
    }
    if route not in expected:
        raise ValueError(f"unknown GGUF pack route {route!r}")
    if (role.route_class, role.operation) != expected[route]:
        want_route, want_operation = expected[route]
        return False, (f"rank-{rank} tensor role {role.name} from {source} is "
                       f"{role.route_class}/{role.operation}, not "
                       f"{want_route}/{want_operation}")
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
    inc = pathlib.Path(__file__).resolve().parent / "include" / "ppu_format_config.inc"
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
    """Write one tensor's arrays inside the versioned directory container."""
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
        if arrangement.layout == F.PLACED_LAYOUT_Q4_KPACK4_TRANSPOSE_V1:
            if qtype != F.QuantType.Q4_K:
                raise ValueError(
                    f"artifact {record.get('name', '<unnamed>')}: layout 1 is Q4_K-only, got {qtype.name}")
            expected = F.q4_kpack4_arrangement()
        elif arrangement.layout == F.PLACED_LAYOUT_Q4_N16K64_DIRECT_V1:
            if qtype != F.QuantType.Q4_K:
                raise ValueError(
                    f"artifact {record.get('name', '<unnamed>')}: layout 3 is Q4_K-only, got {qtype.name}")
            expected = F.q4_n16k64_direct_arrangement()
        elif arrangement.layout == F.PLACED_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1:
            if qtype == F.QuantType.Q4_K:
                raise ValueError(
                    f"artifact {record.get('name', '<unnamed>')}: layout 2 does not describe Q4_K bytes")
            expected = F.kquant_kpack_arrangement(qtype)
        else:  # validate() above is fail-closed; keep this branch explicit for future layout additions.
            raise ValueError(
                f"artifact {record.get('name', '<unnamed>')}: layout {arrangement.layout} has no restore policy")
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
    tensors = tuple(torch.from_numpy(np.load(d / f"{name}.npy", allow_pickle=False))
                    for name in _BUNDLE_ARRAYS)
    return routes.PlacedArtifact(tensors, arrangement, version)


def load_kpack_bundle(root: pathlib.Path) -> KPackBundle:
    """Load a complete production bundle, rejecting ambiguity and extra files.

    This is intentionally stricter than :func:`restore_artifact`, which remains
    a development compatibility reader for old descriptors. A product bundle
    accepts only arrangement-v2 Q4 K-pack4 or Q2/Q3/Q5/Q6 per-plane K-pack,
    validates every recorded shape and byte count, and rejects partial or
    unlisted filesystem entries.
    """
    root = pathlib.Path(root)
    manifest_path = root / KPACK_BUNDLE_MANIFEST
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"K-pack bundle root must be a real directory: {root}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"K-pack bundle is missing a regular {KPACK_BUNDLE_MANIFEST}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid K-pack bundle manifest: {exc}") from exc
    _validate_bundle_manifest(manifest)

    records = manifest["tensors"]
    expected_root = {KPACK_BUNDLE_MANIFEST, *(record["dir"] for record in records)}
    actual_root = {entry.name for entry in root.iterdir()}
    if actual_root != expected_root:
        raise ValueError(
            "K-pack bundle root entries disagree with the manifest: "
            f"missing={sorted(expected_root - actual_root)} extra={sorted(actual_root - expected_root)}")

    artifacts = {}
    for record in records:
        name = record["name"]
        directory = root / record["dir"]
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"artifact {name}: tensor entry must be a real directory")
        expected_files = {f"{array}.npy" for array in _BUNDLE_ARRAYS}
        entries = list(directory.iterdir())
        actual_files = {entry.name for entry in entries}
        if actual_files != expected_files or any(entry.is_symlink() for entry in entries):
            raise ValueError(
                f"artifact {name}: tensor files disagree with the schema: "
                f"missing={sorted(expected_files - actual_files)} extra={sorted(actual_files - expected_files)}")
        observed_hashes = _bundle_file_hashes(directory)
        if observed_hashes != record["sha256"]:
            raise ValueError(
                f"artifact {name}: array checksum mismatch: expected={record['sha256']} "
                f"observed={observed_hashes}")
        artifact = restore_artifact(root, record)
        _validate_loaded_artifact(record, artifact)
        artifacts[name] = artifact
    return KPackBundle(manifest, artifacts)


def _validate_bundle_manifest(manifest: dict) -> None:
    from quactlize import routes

    if not isinstance(manifest, dict) or set(manifest) != _BUNDLE_TOP_LEVEL_FIELDS:
        got = sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__
        raise ValueError(
            f"K-pack manifest must contain exactly {sorted(_BUNDLE_TOP_LEVEL_FIELDS)}, got {got}")
    if manifest["schema"] != KPACK_BUNDLE_SCHEMA or manifest["schema_version"] != KPACK_BUNDLE_VERSION:
        raise ValueError(
            f"unsupported K-pack bundle schema {manifest.get('schema')!r} "
            f"version {manifest.get('schema_version')!r}")
    if manifest["arrangement_version"] != routes.PLACED_ARTIFACT_VERSION_V2:
        raise ValueError("production K-pack bundles require placed arrangement version 2")
    if not isinstance(manifest["model"], str) or not manifest["model"]:
        raise ValueError("K-pack manifest model must be a nonempty string")
    selection = manifest["selection"]
    selection_fields = {"layout_policy", "packable_total", "packed", "skipped"}
    if not isinstance(selection, dict) or set(selection) != selection_fields:
        raise ValueError(f"K-pack selection must contain exactly {sorted(selection_fields)}")
    if selection["layout_policy"] != "production-kpack-only":
        raise ValueError("K-pack bundle layout_policy must be production-kpack-only")
    for field in selection_fields - {"layout_policy"}:
        _require_nonnegative_int(selection[field], f"selection.{field}")
    if not isinstance(manifest["tensors"], list) or not manifest["tensors"]:
        raise ValueError("a production K-pack bundle must contain at least one tensor")
    if not isinstance(manifest["skipped"], list):
        raise ValueError("skipped must be a list")
    if selection["packed"] != len(manifest["tensors"]):
        raise ValueError("selection.packed disagrees with tensors length")
    if selection["skipped"] != len(manifest["skipped"]):
        raise ValueError("selection.skipped disagrees with skipped length")
    if selection["packable_total"] != selection["packed"]:
        raise ValueError("selection.packable_total must equal packed for a complete product bundle")
    _validate_omission_records(manifest["skipped"], "skipped")

    names, directories = set(), set()
    for index, record in enumerate(manifest["tensors"]):
        _validate_bundle_record(record, index)
        if record["name"] in names:
            raise ValueError(f"duplicate tensor name in K-pack manifest: {record['name']}")
        if record["dir"] in directories:
            raise ValueError(f"duplicate tensor directory in K-pack manifest: {record['dir']}")
        names.add(record["name"])
        directories.add(record["dir"])


def _validate_bundle_record(record: dict, index: int) -> None:
    from quactlize import formats as F
    from quactlize import routes

    if not isinstance(record, dict) or set(record) != _BUNDLE_TENSOR_FIELDS:
        raise ValueError(
            f"tensor record {index} must contain exactly {sorted(_BUNDLE_TENSOR_FIELDS)}")
    name = record["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"tensor record {index} has an invalid name")
    if record["dir"] != _tensor_dir_name(index, name):
        raise ValueError(f"artifact {name}: directory is not the canonical collision-safe name")
    try:
        qtype = F.QuantType(_require_nonnegative_int(record["ggml_type"], f"artifact {name}.ggml_type"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"artifact {name}: unsupported ggml_type {record.get('ggml_type')!r}") from exc
    if qtype not in (F.QuantType.Q2_K, F.QuantType.Q3_K, F.QuantType.Q4_K,
                     F.QuantType.Q5_K, F.QuantType.Q6_K):
        raise ValueError(f"artifact {name}: {qtype.name} is not a production K-pack qtype")
    if record["type_name"] != qtype.name:
        raise ValueError(f"artifact {name}: type_name disagrees with ggml_type")
    route = record["route_class"]
    if route not in ("dense", "grouped"):
        raise ValueError(f"artifact {name}: route_class must be dense or grouped")
    rank = _require_nonnegative_int(record["rank"], f"artifact {name}.rank")
    if (route, rank) not in (("dense", 2), ("grouped", 3)):
        raise ValueError(f"artifact {name}: route_class/rank disagree")
    n = _require_positive_int(record["n"], f"artifact {name}.n")
    k = _require_positive_int(record["k"], f"artifact {name}.k")
    experts = record["experts"]
    if route == "dense":
        if experts is not None:
            raise ValueError(f"artifact {name}: dense tensor must record experts=null")
        expert_count = 1
    else:
        expert_count = _require_positive_int(experts, f"artifact {name}.experts")
    F.validate_fully_quantized_resident_geometry(qtype, n, k)
    layout = _target_layout(qtype)
    if record["layout_name"] != layout:
        raise ValueError(f"artifact {name}: layout_name must be canonical {layout}")
    if record["arrangement_version"] != routes.PLACED_ARTIFACT_VERSION_V2:
        raise ValueError(f"artifact {name}: production bundle requires arrangement_version=2")
    low_bits, high_bits = F.placed_code_planes(qtype)
    if record["plane_packs"] != {
            "low": 16 // low_bits, "high": 16 // high_bits if high_bits else 0}:
        raise ValueError(f"artifact {name}: plane_packs disagree with {qtype.name}")
    expected = (F.q4_kpack4_arrangement() if qtype == F.QuantType.Q4_K
                else F.kquant_kpack_arrangement(qtype))
    raw = record["arrangement"]
    if not isinstance(raw, dict) or set(raw) != set(F.PlacedArrangementV2._fields):
        raise ValueError(f"artifact {name}: arrangement has the wrong fields")
    try:
        arrangement = F.PlacedArrangementV2(
            *(_require_nonnegative_int(raw[field], f"artifact {name}.arrangement.{field}")
              for field in F.PlacedArrangementV2._fields))
        arrangement.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact {name}: invalid production arrangement: {exc}") from exc
    if arrangement != expected:
        raise ValueError(f"artifact {name}: arrangement is not canonical for {qtype.name}")
    shapes = record["shapes"]
    if not isinstance(shapes, dict) or set(shapes) != set(_BUNDLE_ARRAYS):
        raise ValueError(f"artifact {name}: shapes must name low/high/units exactly")
    for array, shape in shapes.items():
        if (not isinstance(shape, list) or
                any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape)):
            raise ValueError(f"artifact {name}: invalid {array} shape {shape!r}")
    expected_shapes = _canonical_bundle_shapes(qtype, route, expert_count, n, k)
    for array, expected_shape in expected_shapes.items():
        if shapes[array] != expected_shape:
            raise ValueError(
                f"artifact {name}: {array} shape must be canonical {expected_shape}, "
                f"got {shapes[array]}")
    hashes = record["sha256"]
    if (not isinstance(hashes, dict) or set(hashes) != set(_BUNDLE_ARRAYS) or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in hashes.values())):
        raise ValueError(f"artifact {name}: sha256 must contain exact lowercase low/high/units digests")


def _validate_loaded_artifact(record: dict, artifact) -> None:
    import torch

    for array, tensor in zip(_BUNDLE_ARRAYS, artifact):
        if tensor.dtype != torch.uint8:
            raise ValueError(f"artifact {record['name']}: {array} dtype must be uint8, got {tensor.dtype}")
        if list(tensor.shape) != record["shapes"][array]:
            raise ValueError(
                f"artifact {record['name']}: {array} shape {list(tensor.shape)} disagrees with manifest "
                f"{record['shapes'][array]}")


def _canonical_bundle_shapes(qtype, route: str, experts: int, n: int, k: int) -> dict:
    """Exact tensor ABI emitted by the dense and grouped v2 producers."""
    from quactlize import formats as F

    qtype = F.QuantType(qtype)
    low_bits, high_bits = F.placed_code_planes(qtype)
    prefix = [experts, n]
    low = prefix + [k * low_bits // 8]
    high = prefix + [k * high_bits // 8] if high_bits else [0]
    metadata_bytes = F.BLOCKS[qtype].scale_meta_bytes
    superblocks_per_unit = 1 if metadata_bytes % 4 == 0 else 2
    unit_shape = [k // (256 * superblocks_per_unit), n,
                  metadata_bytes * superblocks_per_unit]
    units = unit_shape if route == "dense" else [experts] + unit_shape
    return {"low": low, "high": high, "units": units}


def _bundle_file_hashes(directory: pathlib.Path) -> dict:
    hashes = {}
    for name in _BUNDLE_ARRAYS:
        digest = hashlib.sha256()
        with (pathlib.Path(directory) / f"{name}.npy").open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return hashes


def _validate_omission_records(records: list, field: str) -> None:
    required = {"name", "type_name", "reason"}
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != required or
                any(not isinstance(record[key], str) or not record[key] for key in required)):
            raise ValueError(f"{field}[{index}] must contain nonempty name/type_name/reason strings")


def _require_nonnegative_int(value, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _require_positive_int(value, field: str) -> int:
    value = _require_nonnegative_int(value, field)
    if value == 0:
        raise ValueError(f"{field} must be positive")
    return value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "KPACK_BUNDLE_SCHEMA",
    "KPACK_BUNDLE_VERSION",
    "KPACK_BUNDLE_MANIFEST",
    "KPackBundle",
    "load_kpack_bundle",
    "_target_layout",
    "_prepare_artifact",
    "_packability",
    "_tensor_geometry",
    "_create_bundle_staging_root",
    "_canonical_bundle_shapes",
    "_route_role_authority",
    "_low_bits",
    "_high_bits",
    "format_registry",
    "_tile_k",
    "_write",
    "restore_artifact",
]
