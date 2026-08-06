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

    python3 tools/pack_gguf.py MODEL.gguf OUT_DIR [--limit N] [--dry-run]

Needs the device library (the placement lives there), so it runs on the box:
    QUACTLIZE_PPU_LIB=<...>/libquactlize_ppu.so python3 tools/pack_gguf.py ...
"""
import argparse
import json
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
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be packed, touching no device and writing nothing")
    a = ap.parse_args()

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
    for t in reader.tensors:
        tt = int(t.tensor_type)
        seen[t.tensor_type.name] += 1
        if tt in supported and len(t.shape) == 2:
            packable.append(t)
        else:
            why = "not a k-quant this build packs" if tt not in supported else f"{len(t.shape)}-D, expected 2-D"
            skipped.append((t.name, t.tensor_type.name, why))

    print(f"model      {a.model}")
    print(f"tensors    {sum(seen.values())}  ->  {len(packable)} packable, {len(skipped)} skipped")
    print(f"type mix   {dict(seen)}")
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

    backend = quactlize.gguf_backend()
    if not backend.startswith("ppu"):
        print(f"\nrefusing to pack on backend '{backend}'.\n"
              f"  The placement lives in the device library, so a host fallback would either fail or -- worse --\n"
              f"  produce something. Set QUACTLIZE_PPU_LIB to a built libquactlize_ppu.so.", file=sys.stderr)
        return 3

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest, t0 = [], time.time()
    todo = packable[:a.limit] if a.limit else packable
    for i, t in enumerate(todo):
        n, k = int(t.shape[1]), int(t.shape[0])         # gguf reports (k, n); the routes want (n, k)
        qtype = int(t.tensor_type)
        blocks = torch.from_numpy(t.data.reshape(n * (k // 256), -1).copy())
        low, high, units = routes.prepare_fully_quantized_dense(blocks, n, k, qtype, tile_k=_tile_k(qtype))

        stem = t.name.replace("/", "_").replace(".", "_")
        tmp = out / (stem + ".partial")
        tmp.mkdir(exist_ok=True)
        _write(tmp, low, high, units)
        tmp.rename(out / stem)                          # a directory that exists is a directory that finished

        # THE ARRANGEMENT IS NOT A CHOICE THIS TOOL MAKES. tile_k comes from the format, F follows from tile_k
        # and the width by the same expression the consumer uses (formats.fold_for). The SAME _tile_k value is
        # passed to the producer above, so the manifest describes bytes that were actually built -- which was
        # not true before the *_for_tile ops existed (INBOX 027/028): the producer pinned F=1 whatever this
        # recorded, and a manifest naming an unbuildable arrangement reads as a capability.
        arr = F.PlacedArrangement(bits=_low_bits(qtype), tile_k=_tile_k(qtype), high_bits=_high_bits(qtype))
        manifest.append({"name": t.name, "dir": stem, "ggml_type": qtype, "type_name": t.tensor_type.name,
                         "n": n, "k": k, "arrangement": arr._asdict(),
                         "fold": [arr.fold, arr.high_fold],  # derived, printed for a human; readers re-derive
                         "shapes": {"low": list(low.shape), "high": list(high.shape), "units": list(units.shape)}})
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  packed {i+1}/{len(todo)}  ({time.time()-t0:.1f}s)")

    (out / "manifest.json").write_text(json.dumps({"model": a.model, "tensors": manifest}, indent=2) + "\n")
    print(f"\nwrote {len(manifest)} artifact(s) + manifest.json to {out}")
    print("The manifest carries the ARRANGEMENT per tensor, not per format: dense and MoE want different folds,\n"
          "and a reader that has to infer it is a reader that will one day infer wrongly.")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
