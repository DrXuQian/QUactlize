#!/usr/bin/env python3
"""Does changing TileK change the OFFLINE BYTES, or only the kernel that reads them?

    QUACTLIZE_PPU_LIB=<...>/libquactlize_ppu.so python3 tools/tilek_artifact_shared.py

WHY THIS QUESTION DECIDES SOMETHING EXPENSIVE. TileK is a build-time constant (BENCH_TSK) and deliberately NOT a
sweep axis -- ci/local_gates.py's lint_tactic_cannot_change_offline_layout enforces that no config row carries
it, because a tactic that changed the layout would invalidate every artifact on disk. The consequence: a binary
built at TileK=64 can never select a TileK=256 winner however many rows it sweeps. And docs/BACKTEST.md D4 --
the 37.5% tensor-core figure at the decode band -- is `i4 16x32:256`, TileK 256, while the dense sweep runs at
TileK=64.

So "small batch wants a different TileK from large batch" reads as "two copies of the weights", which for a 30B
model is not a trade anyone makes. THAT CONCLUSION HAS AN UNCHECKED PREMISE. The placement is built around the
fold, and fold_for(bits, tk) is 1 for int4 at tk in {64, 128, 256} -- identical. If nothing else in the
placement depends on tk, the two artifacts are the same bytes, and the "conflict" costs one extra BINARY and no
extra storage.

IT ASKS BY PACKING TWICE AND COMPARING BYTES. An earlier version of this file went through recover() to test
prepare/recover-at-different-TileK as a round trip -- which is a more elegant property and needs an inverse that
routes.py does not expose (the .so exports quactlize_ppu_recover_fully_quantized_v1; nothing wires it to
Python). Two packings and a memcmp answer the same question with the API that exists. The missing inverse is
worth fixing on its own -- see memory/format-needs-both-dequant-kernels -- but not on the path to this answer.

Needs the device library because the placement lives there. A host reimplementation of it here would be a second
spelling of exactly the thing under test.
"""
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TILEKS = (64, 128, 256)


if __name__ == "__main__":
    # The registry parser lives in tools/pack_gguf.py; importing it by path keeps ONE reader of the .inc rather
    # than a second parse here. This is the same rule the packer's own docstring states.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import importlib.util
    _spec = importlib.util.spec_from_file_location("pack_gguf", pathlib.Path(__file__).resolve().parent / "pack_gguf.py")
    _pg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pg)

    try:
        import torch
    except ImportError:
        print("needs torch", file=sys.stderr)
        raise SystemExit(2)
    import quactlize
    from quactlize import formats as F, routes

    backend = quactlize.gguf_backend()
    if not backend.startswith("ppu"):
        print(f"refusing to answer on backend '{backend}': the placement lives in the device library, and a host\n"
              f"fallback would be a second implementation of the thing being tested. Set QUACTLIZE_PPU_LIB.",
              file=sys.stderr)
        raise SystemExit(3)

    reg = _pg.format_registry()
    n, k = 128, 1024
    print(f"n={n} k={k}. SAME bytes at two TileKs means one weight file serves both, so supporting a small-batch\n"
          f"TileK and a large-batch TileK costs a second BINARY and no extra storage.\n")
    rc = 0
    for qtype, row in sorted(reg.items()):
        bits = row["low_bits"]
        # THE BLOCK SIZE COMES FROM formats.BLOCKS, per format. An earlier draft hardcoded 210 -- which is Q6_K's
        # and wrong for the other four -- under a comment claiming it was read from somewhere. A comment that
        # describes what the code should do rather than what it does is the failure mode this whole repo keeps
        # paying for, and it was in a file written to catch exactly that class of thing.
        layout = F.BLOCKS[F.QuantType(qtype)]
        nblk = n * (k // layout.weights)
        blocks = torch.randint(0, 256, (nblk, layout.block_bytes), dtype=torch.uint8)
        packed = {}
        print(f"{row['name']} (low_bits={bits}):")
        for tk in TILEKS:
            try:
                packed[tk] = routes.prepare_fully_quantized_dense(
                    blocks, n, k, qtype, tile_k=tk, layout="xplane")
            except Exception as e:                                # noqa: BLE001
                print(f"  tk {tk:>3}   REFUSED: {type(e).__name__}: {e}")
        for a, b in itertools.combinations(TILEKS, 2):
            if a not in packed or b not in packed:
                continue
            if F.fold_for(bits, a) != F.fold_for(bits, b):
                verdict = f"DIFFERENT by construction (fold {F.fold_for(bits, a)} vs {F.fold_for(bits, b)})"
            else:
                same = all(torch.equal(x, y) for x, y in zip(packed[a], packed[b]))
                verdict = "SAME artifact" if same else "DIFFERENT artifact (fold matches, placement does not)"
            print(f"  tk {a:>3} vs {b:>3}   {verdict}")
    print("\nSAME  -> build two binaries, ship one weight file, let the tactic cache pick per M.\n"
          "DIFFERENT -> two artifacts, and whether a tensor gets both is a storage decision. That is exactly what\n"
          "the offline-format planner exists to put in front of a human rather than take silently.")
    raise SystemExit(rc)
