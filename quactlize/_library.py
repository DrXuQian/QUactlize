"""Installed PPU resources. Discovery is file-only: no Torch, dlopen or GPU calls."""
import hashlib
import json
from pathlib import Path

from ._version import __version__

PUBLIC_LIBRARY = "libncp_quactlize.so"
PUBLIC_EXPORTS = (
    "quactlize_ppu_kpack_canonical_arrangement_v1",
    "quactlize_ppu_kpack_sizes_for_arrangement_v1",
    "quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2",
    "quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1",
    *("quactlize_ppu_dense_" + s for s in (
        "supported_v1", "create_v1", "create_with_options_v1", "prepare_v1", "prepare_candidate_v1",
        "run_v1", "destroy_v1", "error_v1")),
    *("quactlize_ppu_moe_matmul_" + s for s in (
        "supported_v1", "create_v1", "prepare_v1", "run_v1", "destroy_v1", "error_v1")),
    *("quactlize_ppu_ffn_" + s for s in (
        "supported_v1", "create_v1", "prepare_v1", "run_v1", "run_v2", "destroy_v1", "error_v1")),
    *("quactlize_ppu_matmul_" + s for s in (
        "query_v1", "prepare_v1", "packed_query_v1", "packed_prepare_v1", "run_v1", "dequant_v1",
        "destroy_v1", "error_v1")),
    "quactlize_ppu_router_run_v1", "quactlize_ppu_swiglu_run_v1", "quactlize_ppu_merge_run_v1",
    *("quactlize_serving_" + s for s in (
        "prepare_v1", "error_v1", "router_v2", "swiglu_v2", "gather_rows_v1", "merge_v1")),
)
REQUIRED_PAYLOADS = (
    PUBLIC_LIBRARY,
    *("quactlize-runtime/" + name for name in (
        "libquactlize_kpack_dispatch.so", "libquactlize_ppu_execution.so", "libquactlize_ppu_prefill.so",
        "final-selection.json", "prefill-confirmed.json", "dense-prefill.json", "moe-prefill.json",
        "prefill-runtime.json", "jit/tools/kpack_jit.py", "jit/quactlize/runtime/compiler.py",
        "jit/third_party/actlize/include/cute/arch/mma_ppu0010.hpp")),
    *("include/" + name for name in (
        "quactlize/packing/api.h", "quactlize/dense/api.h", "quactlize/moe/matmul.h",
        "quactlize/ffn/api.h", "quactlize/compat/api.h", "quactlize/compat/serving.h",
        "quactlize_ppu_config.h")),
)


class LibraryError(RuntimeError):
    """An installed native package is missing, incompatible or incomplete."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resource(root, name):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise LibraryError(f"Invalid package resource path: {name!r}")
    if not path.is_file():
        raise LibraryError(f"Missing Quactlize package resource: {path}; reinstall the complete PPU wheel")
    return path


def _json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (OSError, ValueError) as exc:
        raise LibraryError(f"Cannot read Quactlize metadata {path}: {exc}") from exc


def verify_payloads(root, manifest):
    """Shared file-only verification; the build additionally checks actual ELF exports."""
    if manifest.get("schema") != "quactlize.ppu-library.v1" or manifest.get("library") != PUBLIC_LIBRARY:
        raise LibraryError(f"Invalid public library manifest in {root}")
    payloads = manifest.get("payloads", {})
    if not isinstance(payloads, dict):
        raise LibraryError(f"Invalid payload inventory in {root / 'manifest.json'}")
    missing = set(REQUIRED_PAYLOADS) - payloads.keys()
    if missing:
        raise LibraryError("Incomplete PPU bundle manifest: " + ", ".join(sorted(missing)))
    for name, expected in payloads.items():
        path = _resource(root, name)
        if file_sha256(path) != expected:
            raise LibraryError(f"Quactlize payload hash mismatch: {path}")


def get_library_info(target=None):
    """Describe this wheel's library without loading it or querying a device.

    Omitting target is allowed only for the wheel's explicit, unambiguous default.
    No environment, source-checkout or system-library fallback is performed.
    """
    root = Path(__file__).resolve().parent / "_native"
    info = _json(_resource(root, "wheel.json"))
    if info.get("schema") != "quactlize.ppu-wheel.v1" or info.get("version") != __version__:
        raise LibraryError(f"Wheel metadata/version mismatch: {root / 'wheel.json'}")
    targets = info.get("supported_targets", [])
    chosen = target if target is not None else info.get("default_target")
    if not chosen or chosen not in targets:
        raise LibraryError(f"Quactlize target {chosen!r} is unavailable; this wheel supports {targets}")
    manifest_path = _resource(root, "manifest.json")
    if file_sha256(manifest_path) != info.get("manifest_sha256"):
        raise LibraryError(f"Quactlize manifest hash mismatch: {manifest_path}")
    manifest = _json(manifest_path)
    if manifest.get("build_info") != info.get("build_info"):
        raise LibraryError(f"Python/native build identity mismatch: {manifest_path}")
    for name, expected in info['build_info']['source_hashes'].items():
        if name.startswith('quactlize/') and name.endswith('.py'):
            path = _resource(root.parent, name[len('quactlize/'):])
            if file_sha256(path) != expected:
                raise LibraryError(f"Python/native source hash mismatch: {path}")
    if set(PUBLIC_EXPORTS) - set(info.get("verified_exports", [])):
        raise LibraryError(f"Wheel is missing required public ABI exports: {root / 'wheel.json'}")
    verify_payloads(root, manifest)
    return {**info, "target": chosen, "library_path": str(root / PUBLIC_LIBRARY),
            "manifest_path": str(manifest_path)}


def get_library_path(target=None):
    """Return the absolute pathlib.Path of this installed wheel's public PPU DSO."""
    return Path(get_library_info(target)["library_path"])


def resolve_library(path=None, target=None):
    if path is None:
        return get_library_path(target)
    if target is not None:
        raise ValueError("target selects an installed wheel; use either path or target, not both")
    # Preserve explicit developer/NVIDIA paths, without claiming their target or provenance.
    return Path(path).expanduser().resolve(strict=True)


def load_library(path=None, target=None):
    import ctypes
    resolved = resolve_library(path, target)
    try:
        return ctypes.CDLL(str(resolved), mode=ctypes.RTLD_LOCAL)
    except OSError as exc:
        raise LibraryError(f"Cannot load Quactlize DSO {resolved}: {exc}. "
                           "Use the matching PPU SDK/runtime; no alternate DSO was loaded.") from exc


def bind_symbol(library, name, args, result):
    try:
        function = getattr(library, name)
    except AttributeError as exc:
        raise LibraryError(f"Quactlize ABI symbol {name} is missing from {library._name}; "
                           "rebuild/install the complete matching public library") from exc
    function.argtypes, function.restype = args, result
    return function
