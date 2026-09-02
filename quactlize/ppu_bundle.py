"""Validate an installed six-library PPU runtime bundle.

The bundle is a deployment unit, not a build-directory convention.  One
manifest binds the exact source, toolchain identity, compile-time format role,
filename, size and digest of the default Q4 ScaleFirst library plus FMT0..4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


SCHEMA = "quactlize.ppu-runtime-bundle"
SCHEMA_VERSION = 1
SOURCE_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
SDK_ARCHIVE_SHA256 = "63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd"
SDK_RELEASE = "2.1.1-a5c56e"

MANIFEST_FIELDS = {"schema", "schema_version", "source", "toolchain", "libraries"}
SOURCE_FIELDS = {"commit", "tree_state", "submodules"}
SUBMODULE_FIELDS = {"commit", "path"}
TOOLCHAIN_FIELDS = {"arch", "sdk_release", "sdk_archive_sha256", "hgcc"}
LIBRARY_FIELDS = {
    "role", "filename", "packed_scale", "packed_format", "qtype",
    "dense_only", "size", "sha256", "definitions",
}


@dataclass(frozen=True)
class LibraryRole:
    role: str
    filename: str
    packed_scale: int
    packed_format: Optional[int]
    qtype: int

    @property
    def definitions(self) -> list[str]:
        result = [
            f"PPU_PACKED_SCALE={self.packed_scale}",
            f"QUACTLIZE_DENSE_ONLY={self.qtype}",
        ]
        if self.packed_format is not None:
            result.append(f"PPU_PACKED_FORMAT={self.packed_format}")
        return result


LIBRARY_ROLES = (
    LibraryRole("default", "libquactlize_ppu.so", 0, None, 12),
    LibraryRole("fmt0", "libquactlize_ppu_fmt0.so", 1, 0, 12),
    LibraryRole("fmt1", "libquactlize_ppu_fmt1.so", 1, 1, 13),
    LibraryRole("fmt2", "libquactlize_ppu_fmt2.so", 1, 2, 10),
    LibraryRole("fmt3", "libquactlize_ppu_fmt3.so", 1, 3, 11),
    LibraryRole("fmt4", "libquactlize_ppu_fmt4.so", 1, 4, 14),
)

# Name the predecessor contract separately, but require the successor exports
# below for current bundle admission. Frozen legacy artifacts retain their own
# versioned verifier; weakening this one would let an older library masquerade
# as a measured-policy release.
LEGACY_REQUIRED_EXPORTS = {
    "quactlize_ppu_build_packed_format_v1",
    "quactlize_ppu_canonical_arrangement_v2",
    "quactlize_ppu_prepare_dense_for_arrangement_v2",
    "quactlize_ppu_recover_dense_for_arrangement_v2",
    "quactlize_ppu_units_bytes",
    "quactlize_ppu_prepare_units",
    "quactlize_ppu_prepare_units_grouped",
    "quactlize_ppu_prepare_fully_quantized_for_arrangement_v2",
    "quactlize_ppu_recover_fully_quantized_for_arrangement_v2",
    "quactlize_ppu_dense_lowbit_for_arrangement_v2",
    "quactlize_ppu_dense_lowbit_dev_for_arrangement_v2",
    "quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2",
    "quactlize_ppu_dense_fully_quantized_for_arrangement_v2",
    "quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2",
    "quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2",
    "quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4",
    "quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2",
    "quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2",
}

SELECTED_CONFIG_REQUIRED_EXPORTS = {
    "quactlize_ppu_dense_fully_quantized_selected_config_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_selected_config_for_arrangement_v2",
}

ANY_M_REQUIRED_EXPORTS = {
    "quactlize_ppu_dense_fully_quantized_any_m_valid_for_arrangement_v2",
    "quactlize_ppu_grouped_fully_quantized_any_m_valid_for_arrangement_v2",
}

SCALEFIRST_PREPASS_REQUIRED_EXPORTS = {
    "quactlize_ppu_kquant_scalefirst_metadata_plane_bytes_for_arrangement_v2",
    "quactlize_ppu_kquant_scalefirst_prepass_dev_for_arrangement_v2",
    "quactlize_ppu_q4_kpack4_scalefirst_metadata_plane_bytes_for_arrangement_v2",
    "quactlize_ppu_q4_kpack4_scalefirst_prepass_dev_for_arrangement_v2",
}

REQUIRED_EXPORTS = (
    LEGACY_REQUIRED_EXPORTS |
    SELECTED_CONFIG_REQUIRED_EXPORTS |
    ANY_M_REQUIRED_EXPORTS |
    SCALEFIRST_PREPASS_REQUIRED_EXPORTS
)


class BundleError(ValueError):
    pass


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _run(command: list[str]) -> str:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode:
        raise BundleError(
            f"binary inspection failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout}")
    return proc.stdout


def _validate_manifest(data: object) -> list[dict]:
    if not isinstance(data, dict):
        raise BundleError("manifest root must be an object")
    if set(data) != MANIFEST_FIELDS:
        raise BundleError(
            f"manifest fields differ: got={sorted(data)} want={sorted(MANIFEST_FIELDS)}")
    if data.get("schema") != SCHEMA or data.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("unsupported PPU runtime bundle schema")
    source = data.get("source")
    if not isinstance(source, dict) or set(source) != SOURCE_FIELDS:
        raise BundleError("source fields must be exact")
    if not SOURCE_SHA_RE.fullmatch(str(source.get("commit", ""))):
        raise BundleError("source.commit must be one exact 40-hex Git identity")
    if source.get("tree_state") != "clean":
        raise BundleError("runtime bundles must be built from a clean tracked tree")
    toolchain = data.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != TOOLCHAIN_FIELDS:
        raise BundleError("toolchain fields must be exact")
    if toolchain.get("arch") != "ppu0010":
        raise BundleError("toolchain.arch must be ppu0010")
    if toolchain.get("sdk_release") != SDK_RELEASE:
        raise BundleError(f"toolchain.sdk_release must be {SDK_RELEASE}")
    if toolchain.get("sdk_archive_sha256") != SDK_ARCHIVE_SHA256:
        raise BundleError("bundle was not built with the admitted PPU SDK archive")
    if not isinstance(toolchain.get("hgcc"), str) or not toolchain["hgcc"].strip():
        raise BundleError("toolchain.hgcc must record the compiler identity")
    submodules = source.get("submodules")
    if not isinstance(submodules, list):
        raise BundleError("source.submodules must record the recursive gitlinks")
    for item in submodules:
        if (not isinstance(item, dict) or set(item) != SUBMODULE_FIELDS or
                not SOURCE_SHA_RE.fullmatch(str(item.get("commit", ""))) or
                not isinstance(item.get("path"), str) or not item["path"]):
            raise BundleError("source.submodules contains an invalid gitlink record")
    libraries = data.get("libraries")
    if not isinstance(libraries, list) or len(libraries) != len(LIBRARY_ROLES):
        raise BundleError("manifest must contain exactly default plus FMT0..4")
    by_role = {}
    for entry in libraries:
        if not isinstance(entry, dict) or set(entry) != LIBRARY_FIELDS:
            raise BundleError("each library entry must have the exact admitted fields")
        if not isinstance(entry.get("role"), str):
            raise BundleError("each library entry must be an object with a role")
        if entry["role"] in by_role:
            raise BundleError(f"duplicate library role {entry['role']}")
        by_role[entry["role"]] = entry
    for expected in LIBRARY_ROLES:
        got = by_role.get(expected.role)
        if got is None:
            raise BundleError(f"missing library role {expected.role}")
        identity = (got.get("filename"), got.get("packed_scale"),
                    got.get("packed_format"), got.get("qtype"))
        want = (expected.filename, expected.packed_scale,
                expected.packed_format, expected.qtype)
        if identity != want:
            raise BundleError(f"{expected.role} identity mismatch: got={identity} want={want}")
        if got.get("dense_only") != expected.qtype:
            raise BundleError(f"{expected.role} must be built with QUACTLIZE_DENSE_ONLY={expected.qtype}")
        if got.get("definitions") != expected.definitions:
            raise BundleError(
                f"{expected.role} definitions differ: "
                f"got={got.get('definitions')!r} want={expected.definitions!r}")
        if not isinstance(got.get("size"), int) or got["size"] <= 0:
            raise BundleError(f"{expected.role} has an invalid size")
        if not re.fullmatch(r"[0-9a-f]{64}", str(got.get("sha256", ""))):
            raise BundleError(f"{expected.role} has an invalid SHA-256")
    return [by_role[role.role] for role in LIBRARY_ROLES]


def verify_bundle(root: pathlib.Path, *, sdk_root: Optional[pathlib.Path] = None,
                  inspect_binaries: bool = True) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read bundle manifest: {exc}") from exc
    libraries = _validate_manifest(data)
    expected_names = {"manifest.json", *(role.filename for role in LIBRARY_ROLES)}
    present_names = {path.name for path in root.iterdir()}
    if present_names != expected_names:
        raise BundleError(
            f"bundle root inventory differs: got={sorted(present_names)} "
            f"want={sorted(expected_names)}")
    if inspect_binaries and sdk_root is None:
        raise BundleError("binary inspection requires --ppu-sdk")
    hgobjdump = sdk_root / "bin" / "hgobjdump" if sdk_root else None
    release_file = sdk_root / "release.yaml" if sdk_root else None
    if inspect_binaries:
        try:
            release_lines = release_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise BundleError(f"cannot read SDK release receipt {release_file}: {exc}") from exc
        versions = [line.split(":", 1)[1].strip() for line in release_lines
                    if line.startswith("version:")]
        if versions != [SDK_RELEASE]:
            raise BundleError(
                f"SDK release receipt differs: got={versions!r} want={[SDK_RELEASE]!r}")
    if inspect_binaries and (not hgobjdump.is_file() or not hgobjdump.stat().st_mode & 0o111):
        raise BundleError(f"missing executable hgobjdump: {hgobjdump}")
    for entry in libraries:
        path = root / entry["filename"]
        if path.is_symlink() or not path.is_file():
            raise BundleError(f"{entry['role']} is missing or not a regular file: {path}")
        if path.stat().st_size != entry["size"]:
            raise BundleError(f"{entry['role']} size differs from manifest")
        if _sha256(path) != entry["sha256"]:
            raise BundleError(f"{entry['role']} SHA-256 differs from manifest")
        if inspect_binaries:
            symbols = {
                line.split()[-1] for line in _run(
                    ["nm", "-D", "--defined-only", str(path)]).splitlines()
                if line.split()
            }
            missing = sorted(REQUIRED_EXPORTS - symbols)
            if missing:
                raise BundleError(f"{entry['role']} is missing required exports: {missing}")
            image = _run([str(hgobjdump), "--list-elf", str(path)])
            if not re.search(r"^Func [0-9]+:", image, re.MULTILINE):
                raise BundleError(f"{entry['role']} contains no inspectable PPU kernel image")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("--ppu-sdk", type=pathlib.Path,
                        help="PPU SDK root used for embedded-image inspection")
    parser.add_argument("--manifest-only", action="store_true",
                        help="validate schema, files and hashes without ELF/image inspection")
    args = parser.parse_args(argv)
    try:
        data = verify_bundle(args.bundle, sdk_root=args.ppu_sdk,
                             inspect_binaries=not args.manifest_only)
    except BundleError as exc:
        print(f"[ppu-runtime-bundle] FAIL: {exc}")
        return 1
    print("[ppu-runtime-bundle] PASS "
          f"source={data['source']['commit']} arch={data['toolchain']['arch']} libraries=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
