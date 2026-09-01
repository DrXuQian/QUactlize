#!/usr/bin/env python3
"""GGUF -> production K-pack bundle, with one placed artifact per tensor.

The directory container is a named, versioned interchange boundary. It is not
a rewritten GGUF file: one headerless ``weights.bin`` stores a 128-byte-aligned
resident region per tensor, while ``manifest.json`` binds every low/high/units
span to its exact arrangement descriptor and the source GGUF's size and SHA-256.
``load_kpack_bundle(..., source=MODEL)`` validates both authorities before
returning any artifact suitable for cache reuse.

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
import stat
import sys
import time
from collections import Counter
from typing import NamedTuple


KPACK_BUNDLE_SCHEMA = "quactlize.kquant-kpack.bundle"
KPACK_BUNDLE_VERSION = 3
KPACK_BUNDLE_MANIFEST = "manifest.json"
KPACK_BUNDLE_WEIGHTS = "weights.bin"
KPACK_BUNDLE_ALIGNMENT = 128
_BUNDLE_TOP_LEVEL_FIELDS = {
    "schema", "schema_version", "arrangement_version", "model", "selection",
    "source", "storage", "tensors", "skipped",
}
_BUNDLE_SOURCE_FIELDS = {"format", "size_bytes", "sha256"}
_BUNDLE_STORAGE_FIELDS = {"file", "size_bytes", "alignment_bytes", "sha256"}
_BUNDLE_TENSOR_FIELDS = {
    "name", "ggml_type", "type_name", "route_class", "layout_name",
    "plane_packs", "rank", "n", "k", "experts", "arrangement_version",
    "arrangement", "source_tensor", "region", "spans",
}
_BUNDLE_SOURCE_TENSOR_FIELDS = {
    "index", "data_offset", "size_bytes", "sha256", "binding_sha256",
}
_BUNDLE_REGION_FIELDS = {"offset_bytes", "size_bytes"}
_BUNDLE_SPAN_FIELDS = {"offset_bytes", "size_bytes", "shape", "sha256"}
_BUNDLE_ARRAYS = ("low", "high", "units")


class KPackBundle(NamedTuple):
    """A validated manifest and its descriptor-carrying resident artifacts."""

    manifest: dict
    artifacts: dict


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_regular_file_nofollow(path: pathlib.Path, label: str) -> bytes:
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) |
             getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be one readable regular file: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a real regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read()
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise ValueError(f"{label} changed while it was being read")
        if len(payload) != before.st_size:
            raise ValueError(f"{label} was truncated while it was being read")
        return payload
    finally:
        os.close(descriptor)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .gguf file")
    ap.add_argument("out", help="output directory; created, and only finished artifacts land in it")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be packed, touching no device and writing nothing")
    a = ap.parse_args(argv)

    try:
        from gguf import GGUFEndian, GGUFReader
    except ImportError:
        print("needs the official gguf package: pip install gguf", file=sys.stderr)
        return 2

    from quactlize import formats as F

    source_identity = None if a.dry_run else _source_file_identity(a.model)
    reader = GGUFReader(a.model)
    if reader.endianess != GGUFEndian.LITTLE:
        print("K-pack b16 storage requires a little-endian source GGUF", file=sys.stderr)
        return 4
    # THE SUPPORTED SET COMES FROM THE FORMAT TABLE, not from a list here. A second list is a second place to
    # forget a format, and this file would forget it silently -- the tensor would land in "skipped" looking like
    # an unsupported type rather than like an omission.
    supported = {int(q): q for q in (F.QuantType.Q2_K, F.QuantType.Q3_K, F.QuantType.Q4_K,
                                     F.QuantType.Q5_K, F.QuantType.Q6_K)}

    seen, packable, skipped = Counter(), [], []
    route_mix = Counter()
    for tensor_index, t in enumerate(reader.tensors):
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
                packable.append((tensor_index, t, route))
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
        int(t.tensor_type) for _index, t, _route in todo
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
    weights_path = out / KPACK_BUNDLE_WEIGHTS
    with weights_path.open("xb") as weights:
        for i, (tensor_index, t, route) in enumerate(todo):
            qtype = int(t.tensor_type)
            n, k, experts = _tensor_geometry(t.shape, qtype)
            # GGUF dimensions are fast-first [K,N,(E)].  Flattening therefore leaves one expert's N rows
            # contiguous and experts adjacent, exactly the grouped producer's [E*N*(K/256), type_size] ABI.
            block_rows = (experts or 1) * n * (k // 256)
            blocks = torch.from_numpy(t.data.reshape(block_rows, -1).copy())
            layout, artifact = _prepare_artifact(
                routes, blocks, n, k, qtype, experts, route)

            # THE ARRANGEMENT IS NOT INFERRED FROM THE OUTPUT. The selected policy
            # names one producer, that producer returns the descriptor for the bytes
            # it built, and this exact expected value prevents a manifest from
            # relabelling Xplane as K-pack (or one K-pack mapping as another).
            arr = artifact.arrangement
            expected = (F.q4_kpack4_arrangement() if layout == "q4-kpack4"
                        else F.kquant_kpack_arrangement(qtype))
            if arr != expected:
                raise RuntimeError(
                    f"{t.name}: producer returned arrangement {arr}, pack plan expected {expected}; refusing to "
                    f"write a manifest that describes different bytes from the ones just produced")
            region, spans = _append_bundle_artifact(weights, artifact)
            low_bits, high_bits = F.placed_code_planes(qtype)
            block = F.BLOCKS[F.QuantType(qtype)]
            raw_bytes = (experts or 1) * n * (k // block.weights) * block.block_bytes
            if region["size_bytes"] != raw_bytes:
                raise RuntimeError(
                    f"{t.name}: resident region is {region['size_bytes']} bytes but GGUF owns {raw_bytes}; "
                    "the K-pack cache must remain byte-neutral")
            # Bind the manifest to the exact immutable snapshot consumed by
            # the producer, not to a second read through GGUFReader's mmap.
            source_tensor = _source_tensor_identity(tensor_index, t, blocks.numpy())
            record = {"name": t.name, "ggml_type": qtype, "type_name": t.tensor_type.name,
                      "route_class": route, "layout_name": layout,
                      "plane_packs": {"low": 16 // low_bits,
                                      "high": 16 // high_bits if high_bits else 0},
                      "rank": len(t.shape), "n": n, "k": k, "experts": experts,
                      "arrangement_version": artifact.arrangement_version,
                      "arrangement": arr._asdict(), "source_tensor": source_tensor,
                      "region": region, "spans": spans}
            source_tensor["binding_sha256"] = _source_tensor_binding(record)
            manifest.append(record)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"  packed {i+1}/{len(todo)}  ({time.time()-t0:.1f}s)")
        weights.flush()
        os.fsync(weights.fileno())

    skipped_manifest = [
        {"name": name, "type_name": type_name, "reason": reason}
        for name, type_name, reason in skipped
    ]
    final_source_identity, final_tensor_hashes = _source_file_identity_and_ranges(
        a.model, [(record["source_tensor"]["data_offset"],
                   record["source_tensor"]["size_bytes"], record["name"])
                  for record in manifest])
    if final_source_identity != source_identity:
        raise RuntimeError(
            f"source GGUF changed while K-pack artifacts were being produced: {a.model}; "
            "refusing to publish a bundle whose source authority is ambiguous")
    for record in manifest:
        observed = final_tensor_hashes[record["name"]]
        if observed != record["source_tensor"]["sha256"]:
            raise RuntimeError(
                f"source tensor {record['name']} changed while K-pack artifacts were being produced: "
                f"expected={record['source_tensor']['sha256']} observed={observed}")
    bundle_manifest = {
        "schema": KPACK_BUNDLE_SCHEMA,
        "schema_version": KPACK_BUNDLE_VERSION,
        "arrangement_version": 2,
        "model": a.model,
        "source": source_identity,
        "storage": _bundle_storage_identity(weights_path),
        "selection": {"layout_policy": "production-kpack-only",
                      "packable_total": len(packable), "packed": len(manifest),
                      "skipped": len(skipped_manifest)},
        "tensors": manifest,
        "skipped": skipped_manifest,
    }
    _validate_bundle_manifest(bundle_manifest)
    manifest_path = out / KPACK_BUNDLE_MANIFEST
    with manifest_path.open("xb") as manifest_file:
        manifest_file.write((json.dumps(bundle_manifest, indent=2) + "\n").encode("utf-8"))
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    _read_bundle_blob_nofollow(weights_path, bundle_manifest, capture_payloads=False)
    _fsync_directory(out)
    _publish_bundle_noreplace(out, final_out)
    _fsync_directory(final_out.parent)
    print(f"\nwrote {len(manifest)} artifact(s) in {KPACK_BUNDLE_WEIGHTS} + manifest.json to {final_out}")
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


def _publish_bundle_noreplace(staging: pathlib.Path, final_out: pathlib.Path) -> None:
    """Atomically publish one complete bundle without replacing any entry.

    ``os.rename``/``Path.rename`` may replace an empty target directory on
    Linux, so a preflight existence check alone is racy.  Product publication
    requires the kernel's NOREPLACE operation; lack of that operation is an
    error rather than permission to weaken the contract.
    """
    import ctypes
    import errno

    staging = pathlib.Path(staging)
    final_out = pathlib.Path(final_out)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "atomic no-replace bundle publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if renameat2(at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(final_out), rename_noreplace) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(error, f"refusing to overwrite existing output {final_out}", final_out)
        raise OSError(error, os.strerror(error), final_out)


def _fsync_directory(path: pathlib.Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _align_up(value: int, alignment: int = KPACK_BUNDLE_ALIGNMENT) -> int:
    if (not isinstance(value, int) or isinstance(value, bool) or value < 0 or
            not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0 or
            alignment & (alignment - 1)):
        raise ValueError(f"invalid alignment request value={value!r} alignment={alignment!r}")
    return (value + alignment - 1) & -alignment


def _write_zero_padding(stream, target: int) -> None:
    current = stream.tell()
    if target < current:
        raise ValueError(f"cannot pad backwards from {current} to {target}")
    remaining = target - current
    while remaining:
        chunk = min(remaining, 4096)
        stream.write(b"\0" * chunk)
        remaining -= chunk


def _append_bundle_artifact(weights, artifact) -> tuple[dict, dict]:
    """Append one contiguous resident region and return its manifest spans."""
    import torch

    region_offset = _align_up(weights.tell())
    _write_zero_padding(weights, region_offset)
    spans = {}
    for name, tensor in zip(_BUNDLE_ARRAYS, artifact):
        if tensor.dtype != torch.uint8 or tensor.device.type != "cpu":
            raise ValueError(f"K-pack {name} must be a CPU uint8 tensor")
        tensor = tensor.contiguous()
        span_offset = _align_up(weights.tell() - region_offset)
        _write_zero_padding(weights, region_offset + span_offset)
        payload = memoryview(tensor.numpy()).cast("B")
        weights.write(payload)
        spans[name] = {
            "offset_bytes": span_offset,
            "size_bytes": len(payload),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    region_size = _align_up(weights.tell() - region_offset)
    _write_zero_padding(weights, region_offset + region_size)
    return ({"offset_bytes": region_offset, "size_bytes": region_size}, spans)


def _bundle_storage_identity(path: pathlib.Path) -> dict:
    identity = _source_file_identity(path)
    return {
        "file": KPACK_BUNDLE_WEIGHTS,
        "size_bytes": identity["size_bytes"],
        "alignment_bytes": KPACK_BUNDLE_ALIGNMENT,
        "sha256": identity["sha256"],
    }


def _write(d: pathlib.Path, low, high, units) -> None:
    """Development compatibility writer for pre-blob artifact fixtures."""
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


def load_kpack_bundle(root: pathlib.Path, *, source=None) -> KPackBundle:
    """Load a complete production bundle, rejecting ambiguity and extra files.

    This is intentionally stricter than :func:`restore_artifact`, which remains
    a development compatibility reader for old descriptors. A product bundle
    accepts only arrangement-v2 Q4 K-pack4 or Q2/Q3/Q5/Q6 per-plane K-pack,
    validates every recorded shape and byte count, and rejects partial or
    unlisted filesystem entries. With no ``source`` this proves only that the
    sidecar is internally intact; callers deciding a persistent cache hit must
    pass the current GGUF so its source authority is checked before artifacts
    are returned.
    """
    root = pathlib.Path(root)
    manifest_path = root / KPACK_BUNDLE_MANIFEST
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"K-pack bundle root must be a real directory: {root}")
    try:
        manifest = json.loads(
            _read_regular_file_nofollow(
                manifest_path, f"K-pack bundle {KPACK_BUNDLE_MANIFEST}").decode("utf-8"),
            object_pairs_hook=_strict_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid K-pack bundle manifest: {exc}") from exc
    _validate_bundle_manifest(manifest)
    if source is not None:
        validate_kpack_bundle_source(manifest, source)

    records = manifest["tensors"]
    expected_root = {KPACK_BUNDLE_MANIFEST, KPACK_BUNDLE_WEIGHTS}
    actual_root = {entry.name for entry in root.iterdir()}
    if actual_root != expected_root:
        raise ValueError(
            "K-pack bundle root entries disagree with the manifest: "
            f"missing={sorted(expected_root - actual_root)} extra={sorted(actual_root - expected_root)}")

    weights_path = root / KPACK_BUNDLE_WEIGHTS
    payloads = _read_bundle_blob_nofollow(weights_path, manifest, capture_payloads=True)

    import numpy as np
    import torch
    from quactlize import formats as F
    from quactlize import routes

    artifacts = {}
    for record in records:
        tensors = []
        for array in _BUNDLE_ARRAYS:
            span = record["spans"][array]
            payload = payloads[record["name"]][array]
            value = np.frombuffer(payload, dtype=np.uint8).reshape(span["shape"])
            tensors.append(torch.from_numpy(value))
        raw = record["arrangement"]
        arrangement = F.PlacedArrangementV2(
            *(raw[field] for field in F.PlacedArrangementV2._fields))
        artifact = routes.PlacedArtifact(tuple(tensors), arrangement, record["arrangement_version"])
        _validate_loaded_artifact(record, artifact)
        artifacts[record["name"]] = artifact
    return KPackBundle(manifest, artifacts)


def _read_bundle_blob_nofollow(path: pathlib.Path, manifest: dict, *, capture_payloads: bool) -> dict:
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) |
             getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"K-pack bundle {KPACK_BUNDLE_WEIGHTS} must be one readable regular file: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"K-pack bundle {KPACK_BUNDLE_WEIGHTS} must be a real regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as weights:
            payloads = _validate_bundle_blob(
                weights, before.st_size, manifest, capture_payloads=capture_payloads)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise ValueError(f"K-pack bundle {KPACK_BUNDLE_WEIGHTS} changed while it was being read")
        return payloads
    finally:
        os.close(descriptor)


def _read_blob_segment(
        stream, length: int, whole, *, expected_hash=None, padding_label=None,
        capture_payload=False):
    digest = hashlib.sha256() if expected_hash is not None else None
    payload = bytearray() if digest is not None and capture_payload else None
    remaining = length
    while remaining:
        chunk = stream.read(min(4 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError(f"K-pack {KPACK_BUNDLE_WEIGHTS} is truncated")
        whole.update(chunk)
        if digest is not None:
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
        elif padding_label is not None and any(chunk):
            raise ValueError(f"K-pack {padding_label} padding must be zero")
        remaining -= len(chunk)
    if digest is not None and digest.hexdigest() != expected_hash:
        raise ValueError(f"K-pack {padding_label} span checksum mismatch")
    return payload


def _validate_bundle_blob(
        stream, observed_size: int, manifest: dict, *, capture_payloads: bool) -> dict:
    storage = manifest["storage"]
    if observed_size != storage["size_bytes"]:
        raise ValueError(
            f"K-pack storage size mismatch: expected={storage['size_bytes']} observed={observed_size}")
    whole = hashlib.sha256()
    position = 0
    payloads = {}
    stream.seek(0)
    for record in manifest["tensors"]:
        region = record["region"]
        if region["offset_bytes"] != position:
            raise ValueError(f"artifact {record['name']}: region is not in canonical file order")
        region_end = region["offset_bytes"] + region["size_bytes"]
        record_payloads = {}
        for array in _BUNDLE_ARRAYS:
            span = record["spans"][array]
            span_start = region["offset_bytes"] + span["offset_bytes"]
            _read_blob_segment(
                stream, span_start - position, whole,
                padding_label=f"artifact {record['name']} before {array}")
            position = span_start
            record_payloads[array] = _read_blob_segment(
                stream, span["size_bytes"], whole,
                expected_hash=span["sha256"],
                padding_label=f"artifact {record['name']} {array}",
                capture_payload=capture_payloads)
            position += span["size_bytes"]
        _read_blob_segment(
            stream, region_end - position, whole,
            padding_label=f"artifact {record['name']} trailing")
        position = region_end
        if capture_payloads:
            payloads[record["name"]] = record_payloads
    if position != storage["size_bytes"] or stream.read(1):
        raise ValueError("K-pack storage has an unlisted tail")
    if whole.hexdigest() != storage["sha256"]:
        raise ValueError("K-pack storage checksum mismatch")
    return payloads


def _validate_bundle_manifest(manifest: dict) -> None:
    from quactlize import routes

    if isinstance(manifest, dict) and manifest.get("schema") == KPACK_BUNDLE_SCHEMA:
        if manifest.get("schema_version") == 1:
            raise ValueError("K-pack bundle schema v1 is source-unbound; repack it from the source GGUF")
        if manifest.get("schema_version") == 2:
            raise ValueError("K-pack bundle schema v2 uses the retired NPY carrier; repack it as an aligned blob")
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
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != _BUNDLE_SOURCE_FIELDS:
        raise ValueError(
            f"K-pack source authority must contain exactly {sorted(_BUNDLE_SOURCE_FIELDS)}")
    if source["format"] != "gguf":
        raise ValueError("K-pack source authority format must be gguf")
    _require_positive_int(source["size_bytes"], "source.size_bytes")
    if (not isinstance(source["sha256"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
        raise ValueError("source.sha256 must be an exact lowercase SHA-256 digest")
    storage = manifest["storage"]
    if not isinstance(storage, dict) or set(storage) != _BUNDLE_STORAGE_FIELDS:
        raise ValueError(
            f"K-pack storage must contain exactly {sorted(_BUNDLE_STORAGE_FIELDS)}")
    if storage["file"] != KPACK_BUNDLE_WEIGHTS:
        raise ValueError(f"K-pack storage.file must be {KPACK_BUNDLE_WEIGHTS}")
    _require_positive_int(storage["size_bytes"], "storage.size_bytes")
    if storage["alignment_bytes"] != KPACK_BUNDLE_ALIGNMENT:
        raise ValueError(f"K-pack storage alignment must be {KPACK_BUNDLE_ALIGNMENT} bytes")
    if (not isinstance(storage["sha256"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", storage["sha256"])):
        raise ValueError("storage.sha256 must be an exact lowercase SHA-256 digest")
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
    skipped_names = _validate_omission_records(manifest["skipped"], "skipped")

    names = set()
    expected_region_offset = 0
    previous_source_index = -1
    previous_source_end = 0
    for index, record in enumerate(manifest["tensors"]):
        _validate_bundle_record(record, index)
        if record["name"] in names:
            raise ValueError(f"duplicate tensor name in K-pack manifest: {record['name']}")
        if record["region"]["offset_bytes"] != expected_region_offset:
            raise ValueError(f"artifact {record['name']}: region is not in canonical manifest order")
        source_tensor = record["source_tensor"]
        if source_tensor["index"] <= previous_source_index:
            raise ValueError("K-pack source tensor indices must be strictly increasing")
        if source_tensor["data_offset"] < previous_source_end:
            raise ValueError("K-pack source tensor byte ranges must be ordered and disjoint")
        expected_region_offset += record["region"]["size_bytes"]
        previous_source_index = source_tensor["index"]
        previous_source_end = source_tensor["data_offset"] + source_tensor["size_bytes"]
        names.add(record["name"])
    overlap = names & skipped_names
    if overlap:
        raise ValueError(f"K-pack tensors and skipped inventory overlap: {sorted(overlap)}")
    if expected_region_offset != storage["size_bytes"]:
        raise ValueError("K-pack tensor regions do not cover storage.size_bytes exactly")


def _validate_bundle_record(record: dict, index: int) -> None:
    from quactlize import formats as F
    from quactlize import routes

    if not isinstance(record, dict) or set(record) != _BUNDLE_TENSOR_FIELDS:
        raise ValueError(
            f"tensor record {index} must contain exactly {sorted(_BUNDLE_TENSOR_FIELDS)}")
    name = record["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"tensor record {index} has an invalid name")
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
    raw_bytes = expert_count * n * (k // F.BLOCKS[qtype].weights) * F.BLOCKS[qtype].block_bytes
    source_tensor = record["source_tensor"]
    if not isinstance(source_tensor, dict) or set(source_tensor) != _BUNDLE_SOURCE_TENSOR_FIELDS:
        raise ValueError(f"artifact {name}: source_tensor has the wrong fields")
    _require_nonnegative_int(source_tensor["index"], f"artifact {name}.source_tensor.index")
    _require_nonnegative_int(source_tensor["data_offset"], f"artifact {name}.source_tensor.data_offset")
    source_size = _require_positive_int(
        source_tensor["size_bytes"], f"artifact {name}.source_tensor.size_bytes")
    if source_size != raw_bytes:
        raise ValueError(
            f"artifact {name}: source tensor has {source_size} bytes, expected canonical GGUF size {raw_bytes}")
    for field in ("sha256", "binding_sha256"):
        if (not isinstance(source_tensor[field], str) or
                not re.fullmatch(r"[0-9a-f]{64}", source_tensor[field])):
            raise ValueError(f"artifact {name}: source_tensor.{field} must be exact lowercase hex")
    if source_tensor["binding_sha256"] != _source_tensor_binding(record):
        raise ValueError(f"artifact {name}: source tensor binding disagrees with its tensor identity")
    region = record["region"]
    if not isinstance(region, dict) or set(region) != _BUNDLE_REGION_FIELDS:
        raise ValueError(f"artifact {name}: region has the wrong fields")
    region_offset = _require_nonnegative_int(region["offset_bytes"], f"artifact {name}.region.offset_bytes")
    region_size = _require_positive_int(region["size_bytes"], f"artifact {name}.region.size_bytes")
    if region_offset % KPACK_BUNDLE_ALIGNMENT or region_size % KPACK_BUNDLE_ALIGNMENT:
        raise ValueError(f"artifact {name}: region must be {KPACK_BUNDLE_ALIGNMENT}-byte aligned")
    spans = record["spans"]
    if not isinstance(spans, dict) or set(spans) != set(_BUNDLE_ARRAYS):
        raise ValueError(f"artifact {name}: spans must name low/high/units exactly")
    expected_shapes = _canonical_bundle_shapes(qtype, route, expert_count, n, k)
    cursor = 0
    for array, expected_shape in expected_shapes.items():
        span = spans[array]
        if not isinstance(span, dict) or set(span) != _BUNDLE_SPAN_FIELDS:
            raise ValueError(f"artifact {name}: {array} span has the wrong fields")
        offset = _require_nonnegative_int(
            span["offset_bytes"], f"artifact {name}.{array}.offset_bytes")
        size = _require_nonnegative_int(span["size_bytes"], f"artifact {name}.{array}.size_bytes")
        if offset != _align_up(cursor):
            raise ValueError(f"artifact {name}: {array} span offset is not canonical")
        shape = span["shape"]
        if (not isinstance(shape, list) or
                any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape)):
            raise ValueError(f"artifact {name}: invalid {array} shape {shape!r}")
        if shape != expected_shape:
            raise ValueError(
                f"artifact {name}: {array} shape must be canonical {expected_shape}, "
                f"got {shape}")
        expected_size = 1
        for extent in shape:
            expected_size *= extent
        if size != expected_size:
            raise ValueError(
                f"artifact {name}: {array} size must be canonical {expected_size}, got {size}")
        if (not isinstance(span["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", span["sha256"])):
            raise ValueError(f"artifact {name}: {array} sha256 must be exact lowercase hex")
        cursor = offset + size
    if region_size != _align_up(cursor):
        raise ValueError(f"artifact {name}: region size does not exactly cover its canonical spans")
    if region_size != raw_bytes:
        raise ValueError(
            f"artifact {name}: region must remain byte-neutral with its {raw_bytes}-byte GGUF tensor")


def _validate_loaded_artifact(record: dict, artifact) -> None:
    import torch

    for array, tensor in zip(_BUNDLE_ARRAYS, artifact):
        if tensor.dtype != torch.uint8:
            raise ValueError(f"artifact {record['name']}: {array} dtype must be uint8, got {tensor.dtype}")
        if list(tensor.shape) != record["spans"][array]["shape"]:
            raise ValueError(
                f"artifact {record['name']}: {array} shape {list(tensor.shape)} disagrees with manifest "
                f"{record['spans'][array]['shape']}")


def _canonical_bundle_shapes(qtype, route: str, experts: int, n: int, k: int) -> dict:
    """Exact tensor ABI emitted by the dense and grouped v2 producers."""
    from quactlize import formats as F

    qtype = F.QuantType(qtype)
    low_bits, high_bits = F.placed_code_planes(qtype)
    prefix = [experts, n]
    low = prefix + [k * low_bits // 8]
    high = prefix + [k * high_bits // 8] if high_bits else [0]
    packed_unit = F.packed_unit_layout(qtype)
    unit_shape = [k // (256 * packed_unit.superblocks_per_unit), n,
                  packed_unit.unit_bytes]
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


def _source_tensor_identity(index: int, tensor, payload=None) -> dict:
    index = _require_nonnegative_int(index, "source tensor index")
    try:
        offset = int(tensor.data_offset)
        size = int(tensor.n_bytes)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"source tensor {getattr(tensor, 'name', index)!r} lacks GGUF offset/size authority") from exc
    _require_nonnegative_int(offset, "source tensor data_offset")
    _require_positive_int(size, "source tensor size_bytes")
    payload = memoryview(tensor.data if payload is None else payload).cast("B")
    if len(payload) != size:
        raise ValueError(
            f"source tensor {tensor.name}: reader n_bytes={size} but mapped payload has {len(payload)} bytes")
    return {
        "index": index,
        "data_offset": offset,
        "size_bytes": size,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _source_tensor_binding(record: dict) -> str:
    source = record["source_tensor"]
    payload = [
        record["name"], record["ggml_type"], record["rank"], record["n"], record["k"], record["experts"],
        source["index"], source["data_offset"], source["size_bytes"], source["sha256"],
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_file_identity_and_ranges(path, ranges) -> tuple[dict, dict]:
    """Hash one stable regular-file snapshot for persistent-cache authority.

    Symlinks are accepted because the target bytes, rather than a path or inode,
    are the authority. Opening nonblocking and checking the opened descriptor
    avoids a path-check/open race accepting or blocking on a FIFO or device.
    Named byte ranges are hashed during the same sequential pass, so tensor
    records are bound without rereading the model.
    """
    path = pathlib.Path(path)
    digest = hashlib.sha256()
    canonical_ranges = []
    keys = set()
    for offset, length, key in ranges:
        offset = _require_nonnegative_int(offset, f"source range {key}.offset")
        length = _require_positive_int(length, f"source range {key}.size")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError(f"source range keys must be unique nonempty strings: {key!r}")
        keys.add(key)
        canonical_ranges.append((offset, offset + length, key))
    canonical_ranges.sort()
    for previous, current in zip(canonical_ranges, canonical_ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"source tensor ranges overlap: {previous[2]} and {current[2]}")
    range_digests = {key: hashlib.sha256() for _start, _end, key in canonical_ranges}
    range_counts = {key: 0 for _start, _end, key in canonical_ranges}
    range_cursor = 0
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open source GGUF {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"source GGUF must resolve to a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                chunk_start = size
                chunk_end = chunk_start + len(chunk)
                digest.update(chunk)
                while (range_cursor < len(canonical_ranges) and
                       canonical_ranges[range_cursor][1] <= chunk_start):
                    range_cursor += 1
                current = range_cursor
                while current < len(canonical_ranges):
                    start, end, key = canonical_ranges[current]
                    if start >= chunk_end:
                        break
                    lo = max(start, chunk_start) - chunk_start
                    hi = min(end, chunk_end) - chunk_start
                    range_digests[key].update(chunk[lo:hi])
                    range_counts[key] += hi - lo
                    current += 1
                while (range_cursor < len(canonical_ranges) and
                       canonical_ranges[range_cursor][1] <= chunk_end):
                    range_cursor += 1
                size = chunk_end
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"cannot read source GGUF {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or size != after.st_size:
        raise ValueError(f"source GGUF changed while it was being hashed: {path}")
    if size == 0:
        raise ValueError(f"source GGUF must not be empty: {path}")
    for start, end, key in canonical_ranges:
        if end > size or range_counts[key] != end - start:
            raise ValueError(f"source tensor range {key} is outside {size}-byte GGUF")
    return ({"format": "gguf", "size_bytes": size, "sha256": digest.hexdigest()},
            {key: value.hexdigest() for key, value in range_digests.items()})


def _source_file_identity(path) -> dict:
    identity, _ranges = _source_file_identity_and_ranges(path, [])
    return identity


def validate_kpack_bundle_source(manifest: dict, source) -> dict:
    """Prove that ``source`` is the exact GGUF used to build ``manifest``.

    ``manifest['model']`` is a label, not authority: an identical GGUF may move between pack and
    deployment. Size plus SHA-256 binds the content and rejects a stale sidecar
    when another model replaces the source at the same path.
    """
    _validate_bundle_manifest(manifest)
    ranges = [(record["source_tensor"]["data_offset"], record["source_tensor"]["size_bytes"], record["name"])
              for record in manifest["tensors"]]
    observed, tensor_hashes = _source_file_identity_and_ranges(source, ranges)
    if observed != manifest["source"]:
        raise ValueError(
            f"K-pack bundle source mismatch for {source}: "
            f"expected={manifest['source']} observed={observed}")
    for record in manifest["tensors"]:
        expected = record["source_tensor"]["sha256"]
        got = tensor_hashes[record["name"]]
        if got != expected:
            raise ValueError(
                f"K-pack source tensor mismatch for {record['name']}: expected={expected} observed={got}")
    return observed


def _validate_omission_records(records: list, field: str) -> set[str]:
    required = {"name", "type_name", "reason"}
    names = set()
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != required or
                any(not isinstance(record[key], str) or not record[key] for key in required)):
            raise ValueError(f"{field}[{index}] must contain nonempty name/type_name/reason strings")
        if record["name"] in names:
            raise ValueError(f"duplicate tensor name in {field}: {record['name']}")
        names.add(record["name"])
    return names


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
    "KPACK_BUNDLE_WEIGHTS",
    "KPACK_BUNDLE_ALIGNMENT",
    "KPackBundle",
    "load_kpack_bundle",
    "validate_kpack_bundle_source",
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
    "_append_bundle_artifact",
    "restore_artifact",
]
