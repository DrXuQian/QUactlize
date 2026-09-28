"""Identity of the reviewed PPU CUTLASS 4.3.0 source snapshot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

BACKEND = "ppu17-cutlass430"
REVISION = "0eb6a3b218977892e2b6cd30d59880c440e54fd6"
ARCHIVE_SHA256 = "d0fba143142cf0329474f6db21e9ee6dafd0c3987238423d1b2ac1fa863ebf22"
TREE_SHA256 = "0cbc390af07a84190daa5b9df02d80b28545cad5b0497febeebdcd79e5a75771"


def tree_sha256(root):
    root = Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("CUTLASS symlink escapes its source snapshot: " + str(relative))
            kind, data = "link", os.readlink(path).encode()
        elif path.is_file():
            kind, data = "file", path.read_bytes()
        elif path.is_dir():
            continue
        else:
            raise ValueError("unsupported CUTLASS source entry: " + str(relative))
        row = [relative.as_posix(), kind, hashlib.sha256(data).hexdigest()]
        digest.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def require_cutlass(root):
    """Accept only the supplied archive's complete tree, with or without .git.

    The revision is the archive's declared upstream identity, not a fabricated
    local Git commit. Content hashes are authoritative for extracted archives.
    Build products must live outside this immutable source directory.
    """
    root = Path(root).resolve(strict=True)
    version = (root / "include/cutlass/version.h").read_text()
    values = [re.search(r"^#define CUTLASS_" + name + r"\s+(\d+)\s*$", version, re.M)
              for name in ("MAJOR", "MINOR", "PATCH")]
    if any(value is None for value in values) or tuple(int(v[1]) for v in values) != (4, 3, 0):
        raise ValueError("PPU1.7 requires the reviewed PPU CUTLASS 4.3.0 snapshot")
    actual = tree_sha256(root)
    if actual != TREE_SHA256:
        raise ValueError("PPU CUTLASS 4.3.0 source tree differs from the reviewed archive")
    return {"version": "4.3.0", "revision": REVISION, "authority": "ARCHIVE_TREE_SHA256",
            "archive_sha256": ARCHIVE_SHA256, "tree_sha256": actual}
