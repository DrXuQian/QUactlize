#!/usr/bin/env python3
"""Build the immutable real-GGUF shape authority for internal sweeps.

The resolver owns model identity, the exact ordered shard set, workload axes,
and TP policy.  GGUF headers own tensor names, dimensions, qtypes, split
metadata, and MoE expert metadata.  This inventory joins those authorities and
fails closed when they disagree.

Only recognised matrix operations become sweep cells:

* dense GGML ``MUL_MAT`` weights are rank-2 ``[K,N]`` tensors;
* grouped GGML ``MUL_MAT_ID`` weights are rank-3 ``[K,N,E]`` tensors;
* recognised lookup/non-matmul tensors remain visible in tensor statistics but
  never enter a matrix denominator; and
* an unknown rank-2/rank-3 tensor remains visible as ``UNSUPPORTED`` and never
  becomes an implicit dense GEMM.  A known name with the wrong rank, or an
  ambiguous role rule, still fails closed.

Typical use::

  python3 -B tools/gguf_internal_shape_inventory.py --self-test
  python3 -B tools/gguf_internal_shape_inventory.py \
    --resolved /workspace/internal-sweep/resolved-models.json \
    --output-dir /workspace/internal-sweep/inventory
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable


ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "quactlize" / "include" / "ppu_format_config.inc"
SCHEMA = "quactlize.gguf_internal_shape_inventory.v2"
RESOLVED_SCHEMA = "quactlize.internal_sweep.resolved_models.v1"
REQUIRED_QTYPE_AUDIT = (6, 8, 10, 11, 12, 13, 14, 20)
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ROUTING_FIXTURE = "token-topk-hot16x4-wor-sm64-s44-v1"
ROUTING_SEED = 0x51554143544C0044
ROUTING_SOURCE = ROOT / "benchmarks" / "moe_router_fixture.hpp"


class InventoryError(ValueError):
    """The inputs are too ambiguous or inconsistent to publish an inventory."""


_SCALAR_FORMAT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

_QTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0",
    7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    39: "MXFP4", 40: "NVFP4", 41: "Q1_0",
}

# For formats not owned by ppu_format_config.inc, these are the official GGML
# block semantics.  ``quant_group_size`` is used in the shape folder; the
# independent storage block is the admission boundary for a TP split along K.
# K-quants are overridden from the shipping registry for group size and retain
# their official 256-code GGUF superblock for storage alignment.
_OFFICIAL_GGUF_TRAITS: dict[int, tuple[int, int, str]] = {
    0: (0, 1, "official-gguf-trait:unquantized"),
    1: (0, 1, "official-gguf-trait:unquantized"),
    2: (32, 32, "official-gguf-trait:block32"),
    3: (32, 32, "official-gguf-trait:block32"),
    6: (32, 32, "official-gguf-trait:block32"),
    7: (32, 32, "official-gguf-trait:block32"),
    8: (32, 32, "official-gguf-trait:block32"),
    9: (32, 32, "official-gguf-trait:block32"),
    15: (256, 256, "official-gguf-trait:qk-k-256"),
    16: (256, 256, "official-gguf-trait:qk-k-256"),
    17: (256, 256, "official-gguf-trait:qk-k-256"),
    18: (256, 256, "official-gguf-trait:qk-k-256"),
    19: (256, 256, "official-gguf-trait:qk-k-256"),
    20: (32, 32, "official-gguf-trait:iq4-nl-block32"),
    21: (256, 256, "official-gguf-trait:qk-k-256"),
    22: (256, 256, "official-gguf-trait:qk-k-256"),
    23: (256, 256, "official-gguf-trait:qk-k-256"),
    24: (0, 1, "official-gguf-trait:unquantized"),
    25: (0, 1, "official-gguf-trait:unquantized"),
    26: (0, 1, "official-gguf-trait:unquantized"),
    27: (0, 1, "official-gguf-trait:unquantized"),
    28: (0, 1, "official-gguf-trait:unquantized"),
    29: (256, 256, "official-gguf-trait:qk-k-256"),
    30: (0, 1, "official-gguf-trait:unquantized"),
}


@dataclass(frozen=True)
class Role:
    name: str
    route_class: str       # dense, grouped, embedding, non_matmul
    operation: str         # MUL_MAT, MUL_MAT_ID, GET_ROWS, SSM_CONV
    rank: int
    tp_policy_role: str


@dataclass(frozen=True)
class RoleRule:
    pattern: re.Pattern[str]
    role: Role
    source_symbol: str


def _rule(pattern: str, name: str, route_class: str, operation: str, rank: int,
          tp_policy_role: str, source_symbol: str) -> RoleRule:
    return RoleRule(re.compile(pattern), Role(name, route_class, operation, rank,
                                              tp_policy_role), source_symbol)


# This is data rather than a chain of fallbacks: every recognised name is tied
# to the llama.cpp tensor symbol/op that established its semantics.  The
# qwen35-specific rows prevent new 2-D/3-D matrices from disappearing merely
# because an older Q/K/V/FFN registry did not know their names.
ROLE_RULES: tuple[RoleRule, ...] = (
    _rule(r"^blk\.\d+\.attn_q\.weight$", "attn_q", "dense", "MUL_MAT", 2,
          "attn_q", "LLM_TENSOR_ATTN_Q"),
    _rule(r"^blk\.\d+\.attn_k\.weight$", "attn_k", "dense", "MUL_MAT", 2,
          "attn_k", "LLM_TENSOR_ATTN_K"),
    _rule(r"^blk\.\d+\.attn_v\.weight$", "attn_v", "dense", "MUL_MAT", 2,
          "attn_v", "LLM_TENSOR_ATTN_V"),
    _rule(r"^blk\.\d+\.attn_qkv\.weight$", "attn_qkv", "dense", "MUL_MAT", 2,
          "attn_q", "LLM_TENSOR_ATTN_QKV"),
    _rule(r"^blk\.\d+\.attn_gate\.weight$", "attn_gate", "dense", "MUL_MAT", 2,
          "attn_q", "LLM_TENSOR_ATTN_GATE"),
    _rule(r"^blk\.\d+\.attn_output\.weight$", "attn_o", "dense", "MUL_MAT", 2,
          "attn_o", "LLM_TENSOR_ATTN_OUT"),
    _rule(r"^blk\.\d+\.ffn_gate\.weight$", "ffn_gate", "dense", "MUL_MAT", 2,
          "ffn_gate", "LLM_TENSOR_FFN_GATE"),
    _rule(r"^blk\.\d+\.ffn_up\.weight$", "ffn_up", "dense", "MUL_MAT", 2,
          "ffn_up", "LLM_TENSOR_FFN_UP"),
    _rule(r"^blk\.\d+\.ffn_down\.weight$", "ffn_down", "dense", "MUL_MAT", 2,
          "ffn_down", "LLM_TENSOR_FFN_DOWN"),
    _rule(r"^blk\.\d+\.ffn_gate_inp\.weight$", "moe_router", "dense", "MUL_MAT", 2,
          "moe_router", "LLM_TENSOR_FFN_GATE_INP"),
    _rule(r"^blk\.\d+\.ffn_gate_inp_shexp\.weight$", "shared_expert_router", "dense",
          "MUL_MAT", 2, "moe_router", "LLM_TENSOR_FFN_GATE_INP_SHEXP"),
    _rule(r"^blk\.\d+\.ffn_gate_exps\.weight$", "moe_expert_gate", "grouped",
          "MUL_MAT_ID", 3, "moe_expert_gate", "LLM_TENSOR_FFN_GATE_EXPS"),
    _rule(r"^blk\.\d+\.ffn_up_exps\.weight$", "moe_expert_up", "grouped",
          "MUL_MAT_ID", 3, "moe_expert_up", "LLM_TENSOR_FFN_UP_EXPS"),
    _rule(r"^blk\.\d+\.ffn_gate_up_exps\.weight$", "moe_expert_gate_up", "grouped",
          "MUL_MAT_ID", 3, "moe_expert_up", "LLM_TENSOR_FFN_GATE_UP_EXPS"),
    _rule(r"^blk\.\d+\.ffn_down_exps\.weight$", "moe_expert_down", "grouped",
          "MUL_MAT_ID", 3, "moe_expert_down", "LLM_TENSOR_FFN_DOWN_EXPS"),
    _rule(r"^blk\.\d+\.ffn_gate_shexp\.weight$", "shared_expert_gate", "dense",
          "MUL_MAT", 2, "shared_expert_gate", "LLM_TENSOR_FFN_GATE_SHEXP"),
    _rule(r"^blk\.\d+\.ffn_up_shexp\.weight$", "shared_expert_up", "dense",
          "MUL_MAT", 2, "shared_expert_up", "LLM_TENSOR_FFN_UP_SHEXP"),
    _rule(r"^blk\.\d+\.ffn_down_shexp\.weight$", "shared_expert_down", "dense",
          "MUL_MAT", 2, "shared_expert_down", "LLM_TENSOR_FFN_DOWN_SHEXP"),
    _rule(r"^blk\.\d+\.ssm_beta\.weight$", "ssm_beta", "dense", "MUL_MAT", 2,
          "attn_q", "LLM_TENSOR_SSM_BETA"),
    _rule(r"^blk\.\d+\.ssm_alpha\.weight$", "ssm_alpha", "dense", "MUL_MAT", 2,
          "attn_q", "LLM_TENSOR_SSM_ALPHA"),
    _rule(r"^blk\.\d+\.ssm_out\.weight$", "ssm_out", "dense", "MUL_MAT", 2,
          "attn_o", "LLM_TENSOR_SSM_OUT"),
    _rule(r"^blk\.\d+\.nextn\.eh_proj\.weight$", "nextn_eh_proj", "dense",
          "MUL_MAT", 2, "attn_q", "LLM_TENSOR_NEXTN_EH_PROJ"),
    _rule(r"^blk\.\d+\.nextn\.shared_head_head\.weight$", "nextn_shared_head",
          "dense", "MUL_MAT", 2, "lm_head", "LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD"),
    _rule(r"^nextn\.pre_projection\.weight$", "nextn_proj_pre", "dense", "MUL_MAT", 2,
          "nextn_proj_pre", "LLM_TENSOR_NEXTN_PROJ_PRE"),
    _rule(r"^nextn\.post_projection\.weight$", "nextn_proj_post", "dense", "MUL_MAT", 2,
          "nextn_proj_post", "LLM_TENSOR_NEXTN_PROJ_POST"),
    _rule(r"^output\.weight$", "lm_head", "dense", "MUL_MAT", 2,
          "lm_head", "LLM_TENSOR_OUTPUT"),
    _rule(r"^token_embd\.weight$", "token_embedding", "embedding", "GET_ROWS", 2,
          "token_embedding", "LLM_TENSOR_TOKEN_EMBD"),
    _rule(r"^position_embd\.weight$", "position_embedding", "embedding", "GET_ROWS", 2,
          "position_embedding", "LLM_TENSOR_POS_EMBD"),
    _rule(r"^token_types\.weight$", "token_type_embedding", "embedding", "GET_ROWS", 2,
          "token_type_embedding", "LLM_TENSOR_TOKEN_TYPES"),
    _rule(r"^blk\.\d+\.nextn\.embed_tokens\.weight$", "nextn_token_embedding",
          "embedding", "GET_ROWS", 2, "nextn_token_embedding", "LLM_TENSOR_NEXTN_EMBED_TOKENS"),
    _rule(r"^blk\.\d+\.ssm_conv1d\.weight$", "ssm_conv1d", "non_matmul",
          "SSM_CONV", 2, "ssm_conv1d", "LLM_TENSOR_SSM_CONV1D"),
)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise InventoryError(f"truncated GGUF header: wanted {size} bytes, got {len(data)}")
    return data


def _u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _string(stream: BinaryIO) -> str:
    length = _u64(stream)
    try:
        return _read_exact(stream, length).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InventoryError("GGUF header contains non-UTF-8 text") from exc


def _capture_metadata_key(key: str) -> bool:
    return (key in {"general.name", "general.architecture", *SPLIT_KEYS}
            or key.endswith(".expert_count")
            or key.endswith(".expert_used_count"))


def _read_or_skip_metadata(stream: BinaryIO, value_type: int, capture: bool) -> Any:
    if value_type == 8:
        value = _string(stream)
        return value if capture else None
    if value_type == 9:
        element_type, count = _u32(stream), _u64(stream)
        if element_type == 9:
            raise InventoryError("nested GGUF metadata arrays are forbidden")
        for _ in range(count):
            _read_or_skip_metadata(stream, element_type, False)
        return None
    fmt = _SCALAR_FORMAT.get(value_type)
    if fmt is None:
        raise InventoryError(f"unknown GGUF metadata value type {value_type}")
    value = struct.unpack(fmt, _read_exact(stream, struct.calcsize(fmt)))[0]
    return value if capture else None


def _sha256_range(stream: BinaryIO, start: int, end: int) -> str:
    saved = stream.tell()
    stream.seek(start)
    digest = hashlib.sha256()
    remaining = end - start
    while remaining:
        chunk = _read_exact(stream, min(1024 * 1024, remaining))
        digest.update(chunk)
        remaining -= len(chunk)
    stream.seek(saved)
    return digest.hexdigest()


def read_gguf_header(stream: BinaryIO, source: str) -> dict[str, Any]:
    if not stream.seekable():
        raise InventoryError(f"{source}: GGUF header source must be seekable")
    if _read_exact(stream, 4) != b"GGUF":
        raise InventoryError(f"{source}: not a GGUF file")
    version = _u32(stream)
    if version not in (2, 3):
        raise InventoryError(f"{source}: unsupported GGUF version {version}")
    tensor_count, metadata_count = _u64(stream), _u64(stream)
    metadata: dict[str, Any] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for _ in range(metadata_count):
        key = _string(stream)
        if key in fingerprints:
            raise InventoryError(f"{source}: duplicate GGUF metadata key {key!r}")
        value_type = _u32(stream)
        start = stream.tell()
        capture = _capture_metadata_key(key)
        value = _read_or_skip_metadata(stream, value_type, capture)
        end = stream.tell()
        fingerprints[key] = {
            "value_type": value_type,
            "encoded_bytes": end - start,
            "sha256": _sha256_range(stream, start, end),
        }
        if capture:
            metadata[key] = value

    tensors = []
    seen = set()
    for ordinal in range(tensor_count):
        name = _string(stream)
        if name in seen:
            raise InventoryError(f"{source}: duplicate tensor name {name!r}")
        seen.add(name)
        ndims = _u32(stream)
        if ndims == 0:
            raise InventoryError(f"{source}: tensor {name!r} has zero dimensions")
        dims = [_u64(stream) for _ in range(ndims)]
        if any(dim <= 0 for dim in dims):
            raise InventoryError(f"{source}: tensor {name!r} has invalid dimensions {dims}")
        qtype, offset = _u32(stream), _u64(stream)
        tensors.append({"ordinal": ordinal, "name": name, "dims_gguf": dims,
                        "qtype": qtype, "offset": offset})
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "metadata_fingerprints": fingerprints,
        "tensors": tensors,
    }


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qtype_name(qtype: int) -> str:
    return _QTYPE_NAMES.get(qtype, f"UNKNOWN_QTYPE_{qtype}")


def match_role(name: str, rank: int) -> tuple[Role, str] | None:
    matches = [rule for rule in ROLE_RULES if rule.pattern.fullmatch(name)]
    if not matches:
        return None
    if len(matches) != 1:
        raise InventoryError(
            f"ambiguous rank-{rank} tensor role for {name!r}; matches="
            f"{[rule.source_symbol for rule in matches]}")
    rule = matches[0]
    if rule.role.rank != rank:
        raise InventoryError(
            f"tensor {name!r} is {rank}-D but {rule.source_symbol}/{rule.role.operation} "
            f"requires {rule.role.rank}-D")
    return rule.role, rule.source_symbol


def classify_role(name: str, rank: int) -> tuple[Role, str]:
    match = match_role(name, rank)
    if match is None:
        raise InventoryError(
            f"unknown rank-{rank} tensor role for {name!r}; add an exact role/op rule")
    return match


_REGISTRY_RE = re.compile(
    r'^\s*X\(\s*([A-Za-z0-9_]+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,'
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)


def load_format_registry(path: pathlib.Path = REGISTRY) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        match = _REGISTRY_RE.match(line)
        if not match:
            continue
        ident, name, qtype, low, high, group, sf_tk, fq_tk, packed = match.groups()
        qtype_i = int(qtype)
        if qtype_i in rows:
            raise InventoryError(f"{path}:{lineno}: duplicate qtype {qtype_i}")
        rows[qtype_i] = {
            "id": ident, "name": name, "qtype": qtype_i,
            "low_bits": int(low), "high_bits": int(high),
            "group_size": int(group), "scale_first_tile_k": int(sf_tk),
            "fully_quantized_tile_k": int(fq_tk), "packed_format": int(packed),
        }
    if set(rows) != {10, 11, 12, 13, 14}:
        raise InventoryError(f"{path}: expected shipping qtypes 10..14, found {sorted(rows)}")
    return rows


def qtype_traits(qtype: int, registry: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if qtype in registry:
        return {
            "quant_group_size": registry[qtype]["group_size"],
            "quant_group_size_known": True,
            "storage_block_k": 256,
            "source": "ppu_format_config.inc+official-gguf-kquant-superblock256",
        }
    official = _OFFICIAL_GGUF_TRAITS.get(qtype)
    if official is None:
        return {
            "quant_group_size": 0,
            "quant_group_size_known": False,
            "storage_block_k": 0,
            "source": "UNKNOWN",
        }
    group, block, source = official
    return {
        "quant_group_size": group,
        "quant_group_size_known": True,
        "storage_block_k": block,
        "source": source,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InventoryError(f"{field} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InventoryError(f"{field} must be a nonnegative integer, got {value!r}")
    return value


def _positive_axis(values: Any, field: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise InventoryError(f"{field} must be a nonempty integer list")
    result = [_positive_int(value, f"{field}[{index}]") for index, value in enumerate(values)]
    if len(result) != len(set(result)):
        raise InventoryError(f"{field} contains duplicates: {result}")
    return result


def validate_resolved(document: Any, source: str) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema") != RESOLVED_SCHEMA:
        raise InventoryError(f"{source}: expected schema {RESOLVED_SCHEMA}")
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise InventoryError(f"{source}: models must be a nonempty list")
    shape_directory = document.get("shape_directory")
    if not isinstance(shape_directory, dict) or set(shape_directory) != {"dense", "grouped"}:
        raise InventoryError(f"{source}: shape_directory must contain dense and grouped")
    workload_axes = document.get("workload_axes")
    if not isinstance(workload_axes, dict):
        raise InventoryError(f"{source}: workload_axes must be an object")
    dense = workload_axes.get("dense", {})
    grouped = workload_axes.get("grouped", {})
    _positive_axis(dense.get("decode_m"), "workload_axes.dense.decode_m")
    _positive_axis(dense.get("prefill_m"), "workload_axes.dense.prefill_m")
    _positive_axis(grouped.get("decode_tokens"), "workload_axes.grouped.decode_tokens")
    _positive_axis(grouped.get("prefill_tokens"), "workload_axes.grouped.prefill_tokens")
    if grouped.get("expert_count_source") != "gguf:{architecture}.expert_count":
        raise InventoryError(f"{source}: grouped expert_count_source is not GGUF-owned")
    if grouped.get("top_k_source") != "gguf:{architecture}.expert_used_count":
        raise InventoryError(f"{source}: grouped top_k_source is not GGUF-owned")
    profiles = grouped.get("ragged_profiles")
    if not isinstance(profiles, list) or not profiles or any(
            not isinstance(value, str) or not SAFE_SEGMENT.fullmatch(value) for value in profiles):
        raise InventoryError(f"{source}: grouped.ragged_profiles must be folder-safe strings")

    seen_models = set()
    seen_paths = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise InventoryError(f"{source}: models[{index}] is not an object")
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not SAFE_SEGMENT.fullmatch(model_id):
            raise InventoryError(f"{source}: invalid model_id={model_id!r}")
        if model_id in seen_models:
            raise InventoryError(f"{source}: duplicate model_id={model_id}")
        seen_models.add(model_id)
        _positive_int(model.get("tp_world_size"), f"{model_id}.tp_world_size")
        kinds = model.get("problem_kinds")
        if not isinstance(kinds, list) or not kinds or any(
                kind not in {"dense", "grouped"} for kind in kinds):
            raise InventoryError(f"{model_id}: invalid problem_kinds={kinds!r}")
        policy = model.get("tp_policy_definition")
        if not isinstance(policy, dict) or policy.get("default_axis") not in {
                "n", "k", "replicated"}:
            raise InventoryError(f"{model_id}: invalid tp_policy_definition")
        role_axes = policy.get("role_partition_axis", {})
        if not isinstance(role_axes, dict) or any(axis not in {"n", "k", "replicated"}
                                                   for axis in role_axes.values()):
            raise InventoryError(f"{model_id}: invalid role_partition_axis")
        files = model.get("files")
        if not isinstance(files, list) or not files:
            raise InventoryError(f"{model_id}: files must be nonempty")
        for file_index, row in enumerate(files):
            if not isinstance(row, dict):
                raise InventoryError(f"{model_id}.files[{file_index}] is not an object")
            path = pathlib.Path(str(row.get("path", "")))
            if not path.is_absolute():
                raise InventoryError(f"{model_id}: file path is not absolute: {path}")
            if path in seen_paths:
                raise InventoryError(f"resolved models reuse the same GGUF path: {path}")
            seen_paths.add(path)
            _nonnegative_int(row.get("size"), f"{model_id}.files[{file_index}].size")
            digest = row.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise InventoryError(f"{model_id}: invalid file sha256 for {path}")
        fileset = model.get("fileset_sha256")
        if not isinstance(fileset, str) or not re.fullmatch(r"[0-9a-f]{64}", fileset):
            raise InventoryError(f"{model_id}: invalid fileset_sha256")
    return document


def load_resolved(path: pathlib.Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read resolved model spec {path}: {exc}") from exc
    return validate_resolved(document, str(path))


def _load_shard(path: pathlib.Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise InventoryError(f"resolved GGUF is not a regular file: {path}")
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_size != expected["size"] or actual_hash != expected["sha256"]:
        raise InventoryError(
            f"resolved GGUF changed after resolution: {path}; "
            f"size={actual_size}/{expected['size']} sha256={actual_hash}/{expected['sha256']}")
    with path.open("rb") as stream:
        header = read_gguf_header(stream, str(path))
    return {"path": str(path), "size": actual_size, "sha256": actual_hash, "header": header}


ShardProvider = Callable[[pathlib.Path, dict[str, Any]], dict[str, Any]]


def _fileset_sha256(shards: list[dict[str, Any]]) -> str:
    return _identity_sha256([{"size": shard["size"], "sha256": shard["sha256"]}
                             for shard in shards])


def _metadata_authorities(shards: list[dict[str, Any]], model_id: str) -> tuple[str, str, dict[str, Any]]:
    first = shards[0]["header"]
    first_fingerprints = first["metadata_fingerprints"]
    for shard in shards[1:]:
        current = shard["header"]["metadata_fingerprints"]
        if set(current) != set(first_fingerprints):
            missing = sorted(set(first_fingerprints) - set(current))
            extra = sorted(set(current) - set(first_fingerprints))
            raise InventoryError(
                f"{model_id}: shard metadata key mismatch missing={missing[:20]} extra={extra[:20]}")
        for key in first_fingerprints:
            if key == "split.no":
                if current[key]["value_type"] != first_fingerprints[key]["value_type"]:
                    raise InventoryError(f"{model_id}: split.no metadata type differs across shards")
            elif current[key] != first_fingerprints[key]:
                raise InventoryError(f"{model_id}: metadata {key!r} differs across shards")
    metadata = first["metadata"]
    general_name = metadata.get("general.name")
    architecture = metadata.get("general.architecture")
    if not isinstance(general_name, str) or not general_name:
        raise InventoryError(f"{model_id}: missing nonempty general.name")
    if not isinstance(architecture, str) or not architecture:
        raise InventoryError(f"{model_id}: missing nonempty general.architecture")
    return general_name, architecture, metadata


def _normalise_split(shards: list[dict[str, Any]], model_id: str) -> tuple[list[dict[str, Any]], int]:
    presence = [{key for key in SPLIT_KEYS if key in shard["header"]["metadata"]}
                for shard in shards]
    if all(not keys for keys in presence):
        if len(shards) != 1:
            raise InventoryError(f"{model_id}: multiple GGUF files have no split metadata")
        total = shards[0]["header"]["tensor_count"]
        return [{**shards[0], "split_no": 0, "split_count": 1,
                 "split_tensors_count": total}], total
    for index, keys in enumerate(presence):
        if keys != set(SPLIT_KEYS):
            raise InventoryError(
                f"{model_id}: shard {index} has partial split metadata {sorted(keys)}; "
                f"required={list(SPLIT_KEYS)}")
    rows = []
    expected_count: int | None = None
    expected_total: int | None = None
    for ordinal, shard in enumerate(shards):
        metadata = shard["header"]["metadata"]
        split_no = _nonnegative_int(metadata["split.no"], f"{model_id}.split.no")
        split_count = _positive_int(metadata["split.count"], f"{model_id}.split.count")
        split_total = _positive_int(metadata["split.tensors.count"],
                                    f"{model_id}.split.tensors.count")
        if split_no != ordinal:
            raise InventoryError(
                f"{model_id}: ordered resolved files require split.no={ordinal}, got {split_no}")
        if expected_count is None:
            expected_count, expected_total = split_count, split_total
        if split_count != expected_count or split_total != expected_total:
            raise InventoryError(f"{model_id}: split count/total differs across shards")
        rows.append({**shard, "split_no": split_no, "split_count": split_count,
                     "split_tensors_count": split_total})
    assert expected_count is not None and expected_total is not None
    if expected_count != len(shards):
        raise InventoryError(
            f"{model_id}: split.count={expected_count} but resolved files={len(shards)}")
    observed_total = sum(shard["header"]["tensor_count"] for shard in shards)
    if observed_total != expected_total:
        raise InventoryError(
            f"{model_id}: observed tensors={observed_total} != split.tensors.count={expected_total}")
    return rows, expected_total


def _model_shards(model: dict[str, Any], provider: ShardProvider) -> tuple[list[dict[str, Any]], int]:
    loaded = []
    for expected in model["files"]:
        path = pathlib.Path(expected["path"])
        shard = provider(path, expected)
        if shard["size"] != expected["size"] or shard["sha256"] != expected["sha256"]:
            raise InventoryError(f"{model['model_id']}: provider returned stale identity for {path}")
        loaded.append(shard)
    actual_fileset = _fileset_sha256(loaded)
    if actual_fileset != model["fileset_sha256"]:
        raise InventoryError(
            f"{model['model_id']}: fileset sha mismatch {actual_fileset} != {model['fileset_sha256']}")
    return _normalise_split(loaded, model["model_id"])


def route_for(role: Role, band: str) -> str:
    if role.route_class == "grouped":
        return "grouped_fully_quantized" if band == "decode" else "grouped_scalefirst"
    return "dense_fully_quantized" if band == "decode" else "dense_scalefirst"


def support_for(role: Role, qtype: int, band: str, registry: dict[int, dict[str, Any]],
                tp_reason: str | None) -> tuple[str, str]:
    if tp_reason:
        return "UNSUPPORTED", tp_reason
    if qtype in registry:
        return "SUPPORTED", "SHIPPING_KQUANT_ROUTE"
    if role.route_class == "grouped" and qtype == 8:
        return "UNSUPPORTED", "Q8_GROUPED_ROUTE_NOT_REGISTERED"
    if qtype == 8 and band == "prefill":
        return "SUPPORTED", "CONTROLLED_Q8_SCALEFIRST_ROUTE"
    if qtype == 8:
        return "UNSUPPORTED", "Q8_HAS_NO_FULLY_QUANTIZED_DECODE_ROUTE"
    if qtype in (2, 3, 6, 7):
        return "UNSUPPORTED", "LEGACY_GGUF_QTYPE_NOT_REGISTERED"
    if qtype == 20:
        return "UNSUPPORTED", "IQ4_NL_NOT_REGISTERED"
    return "UNSUPPORTED", "QTYPE_NOT_REGISTERED"


def _tp_shape(model: dict[str, Any], role: Role, qtype_trait: dict[str, Any],
              logical_n: int, logical_k: int) -> dict[str, Any]:
    world = model["tp_world_size"]
    policy = model["tp_policy_definition"]
    role_axes = policy.get("role_partition_axis", {})
    axis = role_axes.get(role.tp_policy_role, policy["default_axis"])
    if axis == "n":
        if logical_n % world:
            raise InventoryError(
                f"{model['model_id']}:{role.name}: N={logical_n} not divisible by TP={world}")
        local_n, local_k = logical_n // world, logical_k
    elif axis == "k":
        if logical_k % world:
            raise InventoryError(
                f"{model['model_id']}:{role.name}: K={logical_k} not divisible by TP={world}")
        local_n, local_k = logical_n, logical_k // world
    else:
        local_n, local_k = logical_n, logical_k
    admission_reason = None
    if axis == "k" and world > 1:
        block = qtype_trait["storage_block_k"]
        if block <= 0:
            admission_reason = "TP_K_STORAGE_BLOCK_UNKNOWN"
        elif local_k % block:
            admission_reason = "TP_K_PARTITION_NOT_BLOCK_ALIGNED"
    return {
        "world_size": world,
        "rank": 0,
        "rank_scope": "representative-symmetric-local-kernel",
        "policy": model["tp_policy"],
        "partition_axis": axis,
        "policy_role": role.tp_policy_role,
        "local_n": local_n,
        "local_k": local_k,
        "measurement_scope": policy.get(
            "measurement_scope", "per-rank-local-kernel; inter-rank collectives excluded"),
        "admission_reason": admission_reason,
    }


def _splitmix64(state: int) -> tuple[int, int]:
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return state, value & mask


def _routing_fixture(experts: int, top_k: int, tokens: int) -> dict[str, Any]:
    """Bit-for-bit Python transcription of moe_router_fixture.hpp v1."""
    if experts <= 0 or tokens <= 0 or top_k <= 0 or top_k > experts:
        raise InventoryError(
            f"invalid routing fixture E={experts} top_k={top_k} tokens={tokens}")
    counts = [0] * experts
    state = ROUTING_SEED
    hot_experts = min(16, experts)
    token_routes: list[list[int]] = []
    for _token in range(tokens):
        selected = [False] * experts
        remaining_weight = hot_experts * 4 + (experts - hot_experts)
        picks: list[int] = []
        for _pick in range(top_k):
            state, random_value = _splitmix64(state)
            lottery = random_value % remaining_weight
            chosen = -1
            for expert in range(experts):
                if selected[expert]:
                    continue
                weight = 4 if expert < hot_experts else 1
                if lottery < weight:
                    chosen = expert
                    break
                lottery -= weight
            if chosen < 0:
                raise InventoryError("versioned routing fixture failed to choose an expert")
            selected[chosen] = True
            counts[chosen] += 1
            picks.append(chosen)
            remaining_weight -= 4 if chosen < hot_experts else 1
        if len(picks) != len(set(picks)):
            raise InventoryError("versioned routing fixture selected one expert twice for a token")
        token_routes.append(picks)
    row_offsets = [0]
    for rows in counts:
        row_offsets.append(row_offsets[-1] + rows)
    fixture = {
        "fixture": ROUTING_FIXTURE,
        "fixture_source": str(ROUTING_SOURCE.relative_to(ROOT)),
        "fixture_source_sha256": sha256_file(ROUTING_SOURCE),
        "seed": f"0x{ROUTING_SEED:016x}",
        "experts": experts,
        "top_k": top_k,
        "tokens": tokens,
        "total_rows": tokens * top_k,
        "active": sum(rows > 0 for rows in counts),
        "zero": sum(rows == 0 for rows in counts),
        "min_rows": min(counts),
        "max_rows": max(counts),
        "rows_per_expert": counts,
        "row_offsets": row_offsets,
        "token_routes_sha256": _identity_sha256(token_routes),
        "row_offsets_sha256": _identity_sha256(row_offsets),
    }
    _validate_routing_fixture(fixture)
    return fixture


def _validate_routing_fixture(fixture: dict[str, Any]) -> None:
    experts = fixture["experts"]
    top_k = fixture["top_k"]
    tokens = fixture["tokens"]
    rows = fixture["rows_per_expert"]
    offsets = fixture["row_offsets"]
    if len(rows) != experts or len(offsets) != experts + 1:
        raise InventoryError("routing fixture rows/offsets cardinality mismatch")
    if offsets[0] != 0 or any(offsets[i + 1] - offsets[i] != rows[i]
                              for i in range(experts)):
        raise InventoryError("routing fixture row_offsets do not prefix-sum rows_per_expert")
    if offsets[-1] != tokens * top_k or fixture["total_rows"] != offsets[-1]:
        raise InventoryError("routing fixture total is not tokens*top_k")
    if fixture["active"] != sum(value > 0 for value in rows):
        raise InventoryError("routing fixture active count does not match its histogram")
    if fixture["row_offsets_sha256"] != _identity_sha256(offsets):
        raise InventoryError("routing fixture row_offsets hash mismatch")


def _effective_routing_profiles(grouped_axes: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    declared = grouped_axes["ragged_profiles"]
    if declared == [ROUTING_FIXTURE] or set(declared) == {ROUTING_FIXTURE}:
        return [ROUTING_FIXTURE], {"catalog_authority": "NATIVE_VERSIONED_FIXTURE"}
    raise InventoryError(
        f"grouped ragged profiles lack an implemented versioned authority: {declared}")


def _shape_folder(template: str, values: dict[str, Any]) -> str:
    try:
        folder = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise InventoryError(f"invalid shape_directory template {template!r}: {exc}") from exc
    if not SAFE_SEGMENT.fullmatch(folder):
        raise InventoryError(f"shape directory is not folder-safe: {folder!r}")
    return folder


def _dedup_fields(cell: dict[str, Any]) -> dict[str, Any]:
    grouped = cell.get("grouped")
    return {
        "model_id": cell["model_id"],
        "tp_world": cell["tp"]["world_size"],
        "tp_rank": cell["tp"]["rank"],
        "tp_partition": cell["tp"]["partition_axis"],
        "problem_kind": cell["problem_kind"],
        "experts": None if grouped is None else grouped["experts"],
        "top_k": None if grouped is None else grouped["top_k"],
        "active": None if grouped is None else grouped["active"],
        "ragged": None if grouped is None else grouped["ragged_profile"],
        "route": cell["route"],
        "qtype": cell["qtype"],
        "M": cell["M"], "N": cell["N"], "K": cell["K"], "L": cell["L"],
    }


def _dedup_key(fields: dict[str, Any]) -> str:
    return _identity_sha256(fields)


def _validate_sweep_source_provenance(row: dict[str, Any]) -> None:
    bindings = row.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise InventoryError("sweep shape lacks source_bindings")
    physical = sorted({binding.get("storage_tensor") for binding in bindings})
    logical = sorted({binding.get("logical_consumer") for binding in bindings})
    if any(not isinstance(name, str) or not name for name in [*physical, *logical]):
        raise InventoryError("sweep shape source binding contains an empty tensor name")
    if row.get("source_tensors") != physical:
        raise InventoryError(
            "sweep source_tensors does not name its physical storage tensors")
    if row.get("logical_consumer_tensors") != logical:
        raise InventoryError(
            "sweep logical_consumer_tensors differs from source bindings")
    triples = row.get("sources")
    if not isinstance(triples, list) or not triples or sorted(
            {source[0] for source in triples}) != physical:
        raise InventoryError("sweep source triples do not name physical storage tensors")
    for binding in bindings:
        if binding.get("logical_alias") is True and \
                binding["storage_tensor"] == binding["logical_consumer"]:
            raise InventoryError("logical alias lost its distinct physical storage tensor")


def _expert_authority(model_id: str, architecture: str, metadata: dict[str, Any],
                      required: bool) -> tuple[int | None, int | None, dict[str, str]]:
    expert_key = f"{architecture}.expert_count"
    top_k_key = f"{architecture}.expert_used_count"
    experts = metadata.get(expert_key)
    top_k = metadata.get(top_k_key)
    if not required and experts is None and top_k is None:
        return None, None, {"experts": expert_key, "top_k": top_k_key}
    experts_i = _positive_int(experts, f"{model_id}:{expert_key}")
    top_k_i = _positive_int(top_k, f"{model_id}:{top_k_key}")
    if top_k_i > experts_i:
        raise InventoryError(f"{model_id}: per-token top_k {top_k_i} exceeds E={experts_i}")
    return experts_i, top_k_i, {"experts": expert_key, "top_k": top_k_key}


def inventory_resolved(document: dict[str, Any], provider: ShardProvider = _load_shard) -> dict[str, Any]:
    document = validate_resolved(document, "<resolved-models>")
    registry = load_format_registry()
    dense_axes = document["workload_axes"]["dense"]
    grouped_axes = document["workload_axes"]["grouped"]
    effective_profiles, routing_profile_compatibility = _effective_routing_profiles(grouped_axes)
    routing_fixtures: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    model_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    for model in document["models"]:
        model_id = model["model_id"]
        shards, declared_tensor_count = _model_shards(model, provider)
        general_name, architecture, metadata = _metadata_authorities(shards, model_id)
        all_tensors: list[dict[str, Any]] = []
        seen_names: dict[str, int] = {}
        for shard in shards:
            for tensor in shard["header"]["tensors"]:
                name = tensor["name"]
                if name in seen_names:
                    raise InventoryError(
                        f"{model_id}: tensor {name!r} appears in shards "
                        f"{seen_names[name]} and {shard['split_no']}")
                seen_names[name] = shard["split_no"]
                all_tensors.append({**tensor, "split_no": shard["split_no"],
                                    "shard_path": shard["path"],
                                    "shard_sha256": shard["sha256"],
                                    "logical_alias": False,
                                    "storage_tensor": name})
        if len(all_tensors) != declared_tensor_count:
            raise InventoryError(
                f"{model_id}: assembled tensor count {len(all_tensors)} != declared {declared_tensor_count}")

        logical_tensors = list(all_tensors)
        output_present = "output.weight" in seen_names
        token_embedding = next(
            (tensor for tensor in all_tensors if tensor["name"] == "token_embd.weight"), None)
        tied_output = False
        if not output_present and token_embedding is not None:
            # llama_model_qwen3/qwen35 make output optional and alias it to the
            # token embedding.  The physical tensor must remain a GET_ROWS row,
            # while this logical alias contributes the otherwise-missing lm-head
            # MUL_MAT denominator.
            logical_tensors.append({**token_embedding, "name": "output.weight",
                                    "logical_alias": True,
                                    "storage_tensor": "token_embd.weight"})
            tied_output = True
        ranked = [tensor for tensor in logical_tensors if len(tensor["dims_gguf"]) in (2, 3)]
        classified: list[tuple[dict[str, Any], Role, str]] = []
        unclassified: list[dict[str, Any]] = []
        for tensor in ranked:
            match = match_role(tensor["name"], len(tensor["dims_gguf"]))
            if match is None:
                unclassified.append(tensor)
            else:
                role, role_source = match
                classified.append((tensor, role, role_source))
        grouped_present = any(role.route_class == "grouped" for _, role, _ in classified)
        if grouped_present and "grouped" not in model["problem_kinds"]:
            raise InventoryError(f"{model_id}: grouped expert tensors present but problem_kinds omits grouped")
        if any(role.route_class == "dense" for _, role, _ in classified) and \
                "dense" not in model["problem_kinds"]:
            raise InventoryError(f"{model_id}: dense tensors present but problem_kinds omits dense")
        experts, top_k, expert_sources = _expert_authority(
            model_id, architecture, metadata, grouped_present or "grouped" in model["problem_kinds"])
        if grouped_present:
            assert experts is not None and top_k is not None
            for tokens in [*grouped_axes["decode_tokens"], *grouped_axes["prefill_tokens"]]:
                for profile in effective_profiles:
                    key = (experts, top_k, tokens, profile)
                    if key not in routing_fixtures:
                        fixture = _routing_fixture(experts, top_k, tokens)
                        fixture["fixture_id"] = _identity_sha256({
                            "fixture": profile, "experts": experts,
                            "top_k": top_k, "tokens": tokens,
                            "row_offsets_sha256": fixture["row_offsets_sha256"],
                        })
                        routing_fixtures[key] = fixture

        model_rows.append({
            "model_id": model_id,
            "display_name": model.get("display_name"),
            "general_name": general_name,
            "architecture": architecture,
            "fileset_sha256": model["fileset_sha256"],
            "tp_world_size": model["tp_world_size"],
            "tp_policy": model["tp_policy"],
            "problem_kinds": model["problem_kinds"],
            "expert_count": experts,
            "expert_top_k": top_k,
            "expert_metadata_sources": expert_sources,
            "declared_tensor_count": declared_tensor_count,
            "observed_tensor_count": len(all_tensors),
            "logical_tensor_count": len(logical_tensors),
            "tied_output_alias_materialized": tied_output,
            "rank_counts": dict(sorted(Counter(
                len(tensor["dims_gguf"]) for tensor in all_tensors).items())),
            "matrix_tensor_count": sum(role.route_class in {"dense", "grouped"}
                                       for _, role, _ in classified),
            "unclassified_rank2_or_rank3_count": len(unclassified),
            "shards": [{
                "path": shard["path"], "size": shard["size"],
                "sha256": shard["sha256"], "split_no": shard["split_no"],
                "split_count": shard["split_count"],
                "declared_total_tensors": shard["split_tensors_count"],
                "observed_shard_tensors": shard["header"]["tensor_count"],
                "gguf_version": shard["header"]["version"],
            } for shard in shards],
        })

        for tensor in unclassified:
            dims = list(map(int, tensor["dims_gguf"]))
            qtype = int(tensor["qtype"])
            traits = qtype_traits(qtype, registry)
            tp_unknown = model["tp_world_size"] > 1
            tensor_rows.append({
                "model_id": model_id, "fileset_sha256": model["fileset_sha256"],
                "tensor": tensor["name"], "split_no": tensor["split_no"],
                "storage_tensor": tensor["storage_tensor"],
                "logical_alias": tensor["logical_alias"],
                "tensor_ordinal_in_shard": tensor["ordinal"],
                "shard_path": tensor["shard_path"], "shard_sha256": tensor["shard_sha256"],
                "dims_gguf_fast_first": dims, "rank": len(dims),
                "logical_shape": {"N": dims[1], "K": dims[0],
                                  "E": dims[2] if len(dims) == 3 else None},
                "physical_shape": None if tp_unknown else {
                    "N": dims[1], "K": dims[0],
                    "E": dims[2] if len(dims) == 3 else None},
                "tp": {"world_size": model["tp_world_size"], "rank": 0,
                       "partition_axis": "UNKNOWN",
                       "admission_reason": "TP_PARTITION_UNKNOWN" if tp_unknown else None},
                "qtype": qtype, "qtype_name": qtype_name(qtype),
                "quantization": traits, "role": "UNCLASSIFIED",
                "role_source": "NONE", "operation": "UNKNOWN",
                "route_class": "unknown", "matmul_tensor": False,
                "status": "UNSUPPORTED",
                "reason": "TP_PARTITION_UNKNOWN" if tp_unknown else "UNCLASSIFIED_TENSOR_ROLE",
                "expanded_cell_ids": [],
            })

        for tensor, role, role_source in classified:
            dims = list(map(int, tensor["dims_gguf"]))
            qtype = int(tensor["qtype"])
            traits = qtype_traits(qtype, registry)
            if role.route_class in {"embedding", "non_matmul"}:
                tensor_rows.append({
                    "model_id": model_id, "fileset_sha256": model["fileset_sha256"],
                    "tensor": tensor["name"], "split_no": tensor["split_no"],
                    "storage_tensor": tensor["storage_tensor"],
                    "logical_alias": tensor["logical_alias"],
                    "tensor_ordinal_in_shard": tensor["ordinal"],
                    "shard_path": tensor["shard_path"], "shard_sha256": tensor["shard_sha256"],
                    "dims_gguf_fast_first": dims, "rank": len(dims),
                    "qtype": qtype, "qtype_name": qtype_name(qtype),
                    "quantization": traits, "role": role.name,
                    "role_source": role_source, "operation": role.operation,
                    "route_class": role.route_class, "matmul_tensor": False,
                    "status": "UNSUPPORTED", "reason": "ROLE_IS_NOT_MATRIX_MULTIPLY",
                    "expanded_cell_ids": [],
                })
                continue

            logical_k, logical_n = dims[0], dims[1]
            tensor_experts = dims[2] if role.route_class == "grouped" else None
            if role.route_class == "grouped":
                assert experts is not None and top_k is not None
                if tensor_experts != experts:
                    raise InventoryError(
                        f"{model_id}:{tensor['name']}: tensor E={tensor_experts} != "
                        f"{expert_sources['experts']}={experts}")
                if model["tp_policy_definition"].get("expert_axis", "replicated") != "replicated":
                    raise InventoryError(f"{model_id}: grouped expert axis must be replicated")
            tp = _tp_shape(model, role, traits, logical_n, logical_k)
            route_states: dict[str, dict[str, str]] = {}
            expanded_ids: list[str] = []
            bands = (("decode", dense_axes["decode_m"]),
                     ("prefill", dense_axes["prefill_m"])) if role.route_class == "dense" else (
                         ("decode", grouped_axes["decode_tokens"]),
                         ("prefill", grouped_axes["prefill_tokens"]))
            for band, workload_values in bands:
                route = route_for(role, band)
                status, reason = support_for(role, qtype, band, registry, tp["admission_reason"])
                route_states[band] = {"route": route, "status": status, "reason": reason}
                profiles = [None] if role.route_class == "dense" else effective_profiles
                for workload in workload_values:
                    for profile in profiles:
                        grouped_identity = None
                        active_for_folder = None
                        if role.route_class == "grouped":
                            assert experts is not None and top_k is not None and profile is not None
                            fixture = routing_fixtures[(experts, top_k, workload, profile)]
                            active = fixture["active"]
                            active_for_folder = active
                            m = fixture["total_rows"]
                            grouped_identity = {
                                "experts": experts,
                                "top_k": top_k,
                                "active": active,
                                "tokens": workload,
                                "total_rows": m,
                                "ragged_profile": profile,
                                "ragged": profile,
                                "routing_fixture_id": fixture["fixture_id"],
                                "row_offsets_sha256": fixture["row_offsets_sha256"],
                                "max_rows": fixture["max_rows"],
                            }
                            l = experts
                        else:
                            m, l = workload, 1
                        folder_values = {
                            "m": m, "n": tp["local_n"], "k": tp["local_k"],
                            "group_size": (traits["quant_group_size"]
                                           if traits["quant_group_size_known"] else "UNKNOWN"),
                            "experts": experts, "active": active_for_folder,
                            "ragged_profile": profile,
                        }
                        folder = _shape_folder(
                            document["shape_directory"][role.route_class], folder_values)
                        # The catalog declares symmetric local kernels.  Rank 0
                        # is therefore the one measured representative; world
                        # size and partition remain part of its identity.
                        cell = {
                            "model_id": model_id,
                            "fileset_sha256": model["fileset_sha256"],
                            "tensor": tensor["name"], "tensor_or_route": tensor["name"],
                            "storage_tensor": tensor["storage_tensor"],
                            "logical_alias": tensor["logical_alias"],
                            "role": role.name, "role_source": role_source,
                            "operation": role.operation,
                            "problem_kind": role.route_class,
                            "band": band, "route": route,
                            "qtype": qtype, "qtype_name": qtype_name(qtype),
                            "quant_group_size": traits["quant_group_size"],
                            "quant_group_size_known": traits["quant_group_size_known"],
                            "quantization_trait_source": traits["source"],
                            "storage_block_k": traits["storage_block_k"],
                            "M": m, "N": tp["local_n"], "K": tp["local_k"], "L": l,
                            "logical_shape": {"M": m, "N": logical_n,
                                              "K": logical_k, "L": l},
                            "physical_shape": {"M": m, "N": tp["local_n"],
                                               "K": tp["local_k"], "L": l},
                            "tp": tp,
                            "fold_n_basis": "physical_shape.N",
                            "grouped": grouped_identity,
                            "status": status, "reason": reason,
                            "shape_directory": folder,
                        }
                        identity = {
                            "model_id": model_id,
                            "fileset_sha256": model["fileset_sha256"],
                            "tensor": tensor["name"],
                            **_dedup_fields(cell),
                        }
                        cell["cell_id"] = _identity_sha256(identity)
                        fields = _dedup_fields(cell)
                        cell["dedup_key_fields"] = fields
                        cell["dedup_key"] = _dedup_key(fields)
                        expanded_ids.append(cell["cell_id"])
                        cells.append(cell)
            tensor_rows.append({
                "model_id": model_id, "fileset_sha256": model["fileset_sha256"],
                "tensor": tensor["name"], "split_no": tensor["split_no"],
                "storage_tensor": tensor["storage_tensor"],
                "logical_alias": tensor["logical_alias"],
                "tensor_ordinal_in_shard": tensor["ordinal"],
                "shard_path": tensor["shard_path"], "shard_sha256": tensor["shard_sha256"],
                "dims_gguf_fast_first": dims, "rank": len(dims),
                "logical_shape": {"N": logical_n, "K": logical_k,
                                  "E": tensor_experts},
                "physical_shape": {"N": tp["local_n"], "K": tp["local_k"],
                                   "E": tensor_experts},
                "tp": tp, "fold_n_basis": "physical_shape.N",
                "qtype": qtype, "qtype_name": qtype_name(qtype),
                "quantization": traits, "role": role.name,
                "role_source": role_source, "operation": role.operation,
                "route_class": role.route_class, "matmul_tensor": True,
                "route_states": route_states, "expanded_cell_ids": expanded_ids,
            })

    cell_ids = [cell["cell_id"] for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise InventoryError("expanded tensor cell identity collision")

    grouped_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped_cells[cell["dedup_key"]].append(cell)
    sweep_shapes = []
    for key, members in sorted(grouped_cells.items()):
        fields = members[0]["dedup_key_fields"]
        if any(member["dedup_key_fields"] != fields for member in members):
            raise InventoryError(f"dedup hash collision for {key}")
        states = {(member["status"], member["reason"]) for member in members}
        if len(states) != 1:
            raise InventoryError(f"dedup key {key} aliases incompatible support states {states}")
        folders = {member["shape_directory"] for member in members}
        if len(folders) != 1:
            raise InventoryError(f"dedup key {key} aliases shape folders {folders}")
        status, reason = next(iter(states))
        # Offline layout planning must name a physical GGUF/artifact tensor.
        # A tied LM head's logical consumer is output.weight, but its storage is
        # token_embd.weight; publishing only the former would point downstream
        # at a tensor that does not exist in the file.
        source_tensors = sorted({member["storage_tensor"] for member in members})
        logical_consumers = sorted({member["tensor"] for member in members})
        source_bindings = [{
            "storage_tensor": storage,
            "logical_consumer": logical,
            "role": role,
            "cell_id": cell_id,
            "logical_alias": logical_alias,
        } for storage, logical, role, cell_id, logical_alias in sorted({
            (member["storage_tensor"], member["tensor"], member["role"],
             member["cell_id"], member["logical_alias"])
            for member in members
        })]
        grouped_identity = members[0]["grouped"]
        group_size: int | str = members[0]["quant_group_size"]
        if not members[0]["quant_group_size_known"]:
            group_size = "UNKNOWN"
        sweep_shapes.append({
            "dedup_key": key, "dedup_key_fields": fields,
            "shape_id": key,
            "model_id": fields["model_id"], "route": fields["route"],
            "problem_kind": fields["problem_kind"],
            "problem_route": fields["problem_kind"],
            "qtype": fields["qtype"], "qtype_name": qtype_name(fields["qtype"]),
            "group_size": group_size,
            "M": fields["M"], "N": fields["N"], "K": fields["K"], "L": fields["L"],
            "M_values": sorted({int(member["M"]) for member in members}),
            "tp_world": fields["tp_world"], "tp_rank": fields["tp_rank"],
            "tp_partition": fields["tp_partition"],
            "experts": fields["experts"], "active": fields["active"],
            "top_k": fields["top_k"],
            "ragged": fields["ragged"],
            "grouped": grouped_identity,
            "shape_directory": next(iter(folders)),
            "status": status, "reason": reason,
            "source_tensors": source_tensors,
            "logical_consumer_tensors": logical_consumers,
            "source_bindings": source_bindings,
            "sources": sorted({
                (member["storage_tensor"], member["role"], member["cell_id"])
                for member in members
            }),
        })
        _validate_sweep_source_provenance(sweep_shapes[-1])

    qtype_counts = Counter(row["qtype"] for row in tensor_rows)
    status_counts = Counter(cell["status"] for cell in cells)
    gguf_hashes = {row["model_id"]: row["fileset_sha256"] for row in model_rows}
    gguf_set_sha256 = _identity_sha256(gguf_hashes)
    return {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "resolved_models_schema": RESOLVED_SCHEMA,
        "resolved_set_sha256": document.get("resolved_set_sha256"),
        "provenance": {
            "gguf_hashes": gguf_hashes,
            "gguf_set_sha256": gguf_set_sha256,
            "shape_directory": document["shape_directory"],
        },
        "shape_directory_contract": document["shape_directory"],
        "workload_axes": document["workload_axes"],
        "effective_grouped_routing_profiles": effective_profiles,
        "grouped_routing_profile_compatibility": routing_profile_compatibility,
        "grouped_workload_fixtures": sorted(
            routing_fixtures.values(),
            key=lambda row: (row["experts"], row["top_k"], row["tokens"], row["fixture"])),
        "dedup_contract": (
            "exact tuple (model_id,tp_world,tp_rank,tp_partition,problem_kind,E,top_k,active,ragged,"
            "route,qtype,M,local_N,local_K,L); sources retain tensor provenance"),
        "format_registry": str(REGISTRY.resolve()),
        "format_registry_sha256": sha256_file(REGISTRY),
        "role_registry_source": {
            "path": "llama.cpp/src/llama-arch.cpp + llama.cpp/src/models/qwen35moe.cpp",
            "semantics": "LLM tensor name + GGML op table; qwen35moe dimensions cross-checked",
        },
        "models": model_rows,
        "model_count": len(model_rows),
        "physical_tensor_count": sum(row["observed_tensor_count"] for row in model_rows),
        "rank2_or_rank3_logical_tensor_count": len(tensor_rows),
        "tensor_count": len(tensor_rows),
        "matrix_tensor_count": sum(row["matmul_tensor"] for row in tensor_rows),
        "unclassified_tensor_count": sum(row["role"] == "UNCLASSIFIED"
                                           for row in tensor_rows),
        "expanded_cell_count": len(cells),
        "deduplicated_shape_count": len(sweep_shapes),
        "qtype_counts": {
            str(qtype): {"name": qtype_name(qtype), "count": count,
                         "traits": qtype_traits(qtype, registry)}
            for qtype, count in sorted(qtype_counts.items())
        },
        "required_qtype_presence": {
            str(qtype): {"name": qtype_name(qtype), "count": qtype_counts.get(qtype, 0),
                         "present": bool(qtype_counts.get(qtype, 0)),
                         "traits": qtype_traits(qtype, registry)}
            for qtype in REQUIRED_QTYPE_AUDIT
        },
        "cell_status_counts": dict(sorted(status_counts.items())),
        "tensors": tensor_rows,
        "cells": cells,
        "sweep_shapes": sweep_shapes,
    }


def _atomic_text(path: pathlib.Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise InventoryError(f"refusing to overwrite output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    if pending.exists() or pending.is_symlink():
        raise InventoryError(f"stale partial output exists: {pending}")
    pending.write_text(text)
    pending.replace(path)


_TSV_FIELDS = (
    "model_id", "tensor", "role", "operation", "problem_kind", "band", "route",
    "qtype", "qtype_name", "quant_group_size", "M", "N", "K", "L",
    "tp_world", "tp_rank", "tp_partition", "experts", "top_k", "active", "ragged",
    "status", "reason", "shape_directory", "cell_id", "dedup_key",
)


def _cell_tsv_rows(cells: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=_TSV_FIELDS, delimiter="\t",
                            lineterminator="\n")
    writer.writeheader()
    for cell in cells:
        grouped = cell.get("grouped") or {}
        writer.writerow({
            "model_id": cell["model_id"], "tensor": cell["tensor"],
            "role": cell["role"], "operation": cell["operation"],
            "problem_kind": cell["problem_kind"], "band": cell["band"],
            "route": cell["route"], "qtype": cell["qtype"],
            "qtype_name": cell["qtype_name"],
            "quant_group_size": cell["quant_group_size"],
            "M": cell["M"], "N": cell["N"], "K": cell["K"], "L": cell["L"],
            "tp_world": cell["tp"]["world_size"], "tp_rank": cell["tp"]["rank"],
            "tp_partition": cell["tp"]["partition_axis"],
            "experts": grouped.get("experts", ""), "top_k": grouped.get("top_k", ""),
            "active": grouped.get("active", ""),
            "ragged": grouped.get("ragged_profile", ""),
            "status": cell["status"], "reason": cell["reason"],
            "shape_directory": cell["shape_directory"],
            "cell_id": cell["cell_id"], "dedup_key": cell["dedup_key"],
        })
    return stream.getvalue()


def materialise(output_dir: pathlib.Path, document: dict[str, Any]) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise InventoryError(f"output directory must not exist or must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    _atomic_text(output_dir / "inventory.json",
                 json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False,
                            allow_nan=False) + "\n")
    _atomic_text(output_dir / "cells.tsv", _cell_tsv_rows(document["cells"]))

    # Unknown matrix-shaped tensors and recognised non-matmul tensors must be
    # visible even though they have no shape directory/candidate denominator.
    # Keeping this per model prevents an empty shape tree from looking like the
    # GGUF contained no such tensors.
    for model in document["models"]:
        model_id = model["model_id"]
        unsupported = [row for row in document["tensors"]
                       if row["model_id"] == model_id and
                       row.get("status") == "UNSUPPORTED"]
        model_dir = output_dir / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(model_dir / "unsupported_tensors.json",
                     json.dumps({
                         "schema": "quactlize.gguf_unsupported_tensors.v2",
                         "model_id": model_id,
                         "count": len(unsupported),
                         "tensors": unsupported,
                     }, indent=2, sort_keys=True, ensure_ascii=False,
                        allow_nan=False) + "\n")

    by_shape: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in document["cells"]:
        by_shape[(cell["model_id"], cell["shape_directory"])].append(cell)
    shape_manifest = []
    for (model_id, folder), members in sorted(by_shape.items()):
        model_dir = output_dir / "models" / model_id
        shape_dir = model_dir / folder
        shape_dir.mkdir(parents=True)
        dedup_rows = [row for row in document["sweep_shapes"]
                      if row["model_id"] == model_id and row["shape_directory"] == folder]
        scope = {
            "schema": "quactlize.gguf_internal_shape_scope.v2",
            "model_id": model_id,
            "shape_directory": folder,
            "cell_count": len(members),
            "deduplicated_shape_count": len(dedup_rows),
            "cell_ids": sorted(cell["cell_id"] for cell in members),
            "dedup_keys": sorted(row["dedup_key"] for row in dedup_rows),
        }
        _atomic_text(shape_dir / "scope.json",
                     json.dumps(scope, indent=2, sort_keys=True) + "\n")
        _atomic_text(shape_dir / "cells.tsv", _cell_tsv_rows(members))
        shape_manifest.append(scope)
    _atomic_text(output_dir / "shape_manifest.json",
                 json.dumps({"schema": "quactlize.gguf_shape_manifest.v2",
                             "shapes": shape_manifest}, indent=2, sort_keys=True) + "\n")


def _encoded_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _synthetic_gguf(metadata: list[tuple[str, Any]],
                    tensors: list[tuple[str, tuple[int, ...], int]]) -> bytes:
    out = io.BytesIO()
    out.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata:
        out.write(_encoded_string(key))
        if isinstance(value, str):
            out.write(struct.pack("<I", 8) + _encoded_string(value))
        elif isinstance(value, bool):
            out.write(struct.pack("<I?", 7, value))
        elif isinstance(value, int) and value >= 0:
            out.write(struct.pack("<II", 4, value))
        else:
            raise AssertionError(f"unsupported synthetic metadata value {key}={value!r}")
    for ordinal, (name, dims, qtype) in enumerate(tensors):
        out.write(_encoded_string(name) + struct.pack("<I", len(dims)))
        for dim in dims:
            out.write(struct.pack("<Q", dim))
        out.write(struct.pack("<IQ", qtype, ordinal * 4096))
    return out.getvalue()


def _expect_error(call: Callable[[], Any], needle: str) -> None:
    try:
        call()
    except InventoryError as exc:
        if needle not in str(exc):
            raise AssertionError(f"negative raised wrong error: {exc}; wanted {needle!r}") from exc
    else:
        raise AssertionError(f"negative did not fail: expected {needle!r}")


def self_test() -> dict[str, Any]:
    registry = load_format_registry()
    assert qtype_traits(12, registry)["quant_group_size"] == 32
    assert qtype_traits(12, registry)["storage_block_k"] == 256
    assert qtype_traits(6, registry)["quant_group_size"] == 32
    assert qtype_traits(20, registry)["source"] == "official-gguf-trait:iq4-nl-block32"
    assert classify_role("blk.0.ffn_gate_up_exps.weight", 3)[0].name == "moe_expert_gate_up"
    assert classify_role("blk.0.ssm_conv1d.weight", 2)[0].operation == "SSM_CONV"
    _expect_error(lambda: classify_role("blk.0.ffn_gate_up_exps.weight", 2), "requires 3-D")
    assert match_role("blk.0.new_matrix.weight", 2) is None
    assert match_role("blk.0.new_experts.weight", 3) is None

    # The versioned router is a data authority, not a prose profile.  Pin its
    # published E=256/top-k=8 ladder and prove corrupt offsets cannot pass.
    expected_ladder = {
        1: (8, 1), 2: (15, 2), 4: (30, 3),
        64: (212, 12), 2048: (256, 239), 4096: (256, 447),
    }
    for tokens, (active, max_rows) in expected_ladder.items():
        fixture = _routing_fixture(256, 8, tokens)
        assert (fixture["active"], fixture["max_rows"]) == (active, max_rows)
        assert fixture["total_rows"] == tokens * 8
    corrupt_fixture = json.loads(json.dumps(_routing_fixture(256, 8, 4)))
    corrupt_fixture["row_offsets"][17] += 1
    _expect_error(lambda: _validate_routing_fixture(corrupt_fixture), "prefix-sum")
    corrupt_hash = json.loads(json.dumps(_routing_fixture(256, 8, 4)))
    corrupt_hash["row_offsets_sha256"] = "0" * 64
    _expect_error(lambda: _validate_routing_fixture(corrupt_hash), "hash mismatch")

    common = [
        ("general.name", "inventory-v2-self-test"),
        ("general.architecture", "qwen35moe"),
        ("qwen35moe.expert_count", 4),
        ("qwen35moe.expert_used_count", 2),
        ("split.count", 2),
        ("split.tensors.count", 6),
    ]
    shard0 = _synthetic_gguf(common + [("split.no", 0)], [
        ("blk.0.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_gate_up_exps.weight", (256, 16, 4), 12),
    ])
    shard1 = _synthetic_gguf(common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    paths = [pathlib.Path("/workspace/inventory-self-test-00001-of-00002.gguf"),
             pathlib.Path("/workspace/inventory-self-test-00002-of-00002.gguf")]
    payloads = {str(paths[0]): shard0, str(paths[1]): shard1}

    def file_row(path: pathlib.Path) -> dict[str, Any]:
        payload = payloads[str(path)]
        return {"path": str(path), "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    rows = [file_row(path) for path in paths]
    model = {
        "model_id": "synthetic-qwen35moe-tp2",
        "display_name": "synthetic",
        "tp_world_size": 2,
        "tp_policy": "qwen-tensor-parallel-v1",
        "problem_kinds": ["dense", "grouped"],
        "files": rows,
        "fileset_sha256": _identity_sha256(
            [{"size": row["size"], "sha256": row["sha256"]} for row in rows]),
        "tp_policy_definition": {
            "default_axis": "replicated",
            "role_partition_axis": {
                "attn_q": "n", "moe_expert_up": "n", "moe_expert_down": "k"},
            "expert_axis": "replicated",
            "measurement_scope": "per-rank-local-kernel; inter-rank collectives excluded",
        },
    }
    resolved = {
        "schema": RESOLVED_SCHEMA,
        "resolved_set_sha256": "1" * 64,
        "shape_directory": {
            "dense": "m{m}_n{n}_k{k}_g{group_size}",
            "grouped": "m{m}_n{n}_k{k}_g{group_size}_e{experts}_a{active}_{ragged_profile}",
        },
        "workload_axes": {
            "dense": {"decode_m": [1], "prefill_m": [64]},
            "grouped": {"decode_tokens": [1], "prefill_tokens": [64],
                        "expert_count_source": "gguf:{architecture}.expert_count",
                        "top_k_source": "gguf:{architecture}.expert_used_count",
                        "ragged_profiles": [ROUTING_FIXTURE]},
        },
        "models": [model],
    }

    def provider(path: pathlib.Path, expected: dict[str, Any]) -> dict[str, Any]:
        payload = payloads[str(path)]
        return {"path": str(path), "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "header": read_gguf_header(io.BytesIO(payload), str(path))}

    document = inventory_resolved(resolved, provider)
    assert document["model_count"] == 1
    assert document["models"][0]["declared_tensor_count"] == 6
    assert document["models"][0]["observed_tensor_count"] == 6
    assert document["models"][0]["logical_tensor_count"] == 7
    assert document["models"][0]["tied_output_alias_materialized"] is True
    assert [row["split_no"] for row in document["models"][0]["shards"]] == [0, 1]
    assert document["matrix_tensor_count"] == 5
    assert document["physical_tensor_count"] == 6
    assert document["tensor_count"] == 7
    assert document["unclassified_tensor_count"] == 0
    attn = next(row for row in document["tensors"] if row["role"] == "attn_q")
    assert attn["logical_shape"] == {"N": 8, "K": 256, "E": None}
    assert attn["physical_shape"] == {"N": 4, "K": 256, "E": None}
    fused = next(row for row in document["tensors"] if row["role"] == "moe_expert_gate_up")
    assert fused["logical_shape"] == {"N": 16, "K": 256, "E": 4}
    assert fused["physical_shape"] == {"N": 8, "K": 256, "E": 4}
    down = next(row for row in document["tensors"] if row["role"] == "moe_expert_down")
    assert down["physical_shape"]["K"] == 256
    assert all(state["status"] == "SUPPORTED" for state in down["route_states"].values())
    down_role = classify_role("blk.0.ffn_down_exps.weight", 3)[0]
    planted_bad_tp = _tp_shape(model, down_role, qtype_traits(12, registry), 8, 256)
    assert planted_bad_tp["local_k"] == 128
    assert planted_bad_tp["admission_reason"] == "TP_K_PARTITION_NOT_BLOCK_ALIGNED"
    lm_rows = [row for row in document["tensors"] if row["role"] == "lm_head"]
    token_rows = [row for row in document["tensors"] if row["role"] == "token_embedding"]
    assert len(lm_rows) == len(token_rows) == 1
    assert lm_rows[0]["tensor"] == "output.weight"
    assert lm_rows[0]["storage_tensor"] == "token_embd.weight"
    assert lm_rows[0]["logical_alias"] is True
    assert token_rows[0]["matmul_tensor"] is False
    grouped_cells = [cell for cell in document["cells"]
                     if cell["problem_kind"] == "grouped"]
    assert grouped_cells
    assert {cell["grouped"]["experts"] for cell in grouped_cells} == {4}
    assert {cell["grouped"]["top_k"] for cell in grouped_cells} == {2}
    assert {cell["grouped"]["active"] for cell in grouped_cells} == {2, 4}
    assert {cell["grouped"]["total_rows"] for cell in grouped_cells} == {2, 128}
    assert all(cell["grouped"]["routing_fixture_id"] for cell in grouped_cells)
    assert {cell["grouped"]["ragged"] for cell in grouped_cells} == {ROUTING_FIXTURE}
    assert all("_e4_a2_" in cell["shape_directory"] or
               "_e4_a4_" in cell["shape_directory"] for cell in grouped_cells)
    assert {cell["tp"]["rank"] for cell in document["cells"]} == {0}
    assert all(cell["tp"]["rank_scope"] == "representative-symmetric-local-kernel"
               for cell in document["cells"])
    assert len(document["sweep_shapes"]) < len(document["cells"])
    assert all(row["dedup_key_fields"]["model_id"] == model["model_id"]
               for row in document["sweep_shapes"])
    canonical_fields = {
        "model_id", "shape_id", "source_tensors", "tp_world", "tp_rank",
        "tp_partition", "problem_route", "group_size", "grouped", "qtype",
        "N", "K", "M_values",
    }
    assert all(canonical_fields <= set(row) for row in document["sweep_shapes"])
    assert all(row["shape_id"] == row["dedup_key"] and
               SAFE_SEGMENT.fullmatch(row["shape_id"]) and
               row["M_values"] == [row["M"]]
               for row in document["sweep_shapes"])
    q_sources = {"blk.0.attn_q.weight", "blk.1.attn_q.weight"}
    assert any(set(row["source_tensors"]) == q_sources
               for row in document["sweep_shapes"])
    tied_lm_shapes = [row for row in document["sweep_shapes"]
                      if row["logical_consumer_tensors"] == ["output.weight"]]
    assert tied_lm_shapes and all(
        row["source_tensors"] == ["token_embd.weight"] and
        {source[0] for source in row["sources"]} == {"token_embd.weight"}
        for row in tied_lm_shapes)
    planted_logical_only = json.loads(json.dumps(tied_lm_shapes[0]))
    planted_logical_only["source_tensors"] = ["output.weight"]
    planted_logical_only["sources"] = [
        ["output.weight", source[1], source[2]]
        for source in planted_logical_only["sources"]
    ]
    _expect_error(lambda: _validate_sweep_source_provenance(planted_logical_only),
                  "physical storage tensors")
    assert all((row["grouped"] is None) == (row["problem_route"] == "dense")
               for row in document["sweep_shapes"])
    gguf_hashes = {model["model_id"]: model["fileset_sha256"]}
    assert document["provenance"] == {
        "gguf_hashes": gguf_hashes,
        "gguf_set_sha256": _identity_sha256(gguf_hashes),
        "shape_directory": resolved["shape_directory"],
    }

    # Multi-shard negatives exercise header authority rather than filename
    # heuristics: partial metadata, wrong order, wrong total, and duplicate
    # global tensor names must all fail.
    def run_with_payloads(replacements: dict[int, bytes], model_mutator: Callable[[dict], None] | None = None):
        local_payloads = dict(payloads)
        for index, payload in replacements.items():
            local_payloads[str(paths[index])] = payload
        local_rows = [{"path": str(path), "size": len(local_payloads[str(path)]),
                       "sha256": hashlib.sha256(local_payloads[str(path)]).hexdigest()}
                      for path in paths]
        local_model = json.loads(json.dumps(model))
        local_model["files"] = local_rows
        local_model["fileset_sha256"] = _identity_sha256(
            [{"size": row["size"], "sha256": row["sha256"]} for row in local_rows])
        if model_mutator:
            model_mutator(local_model)
        local_resolved = dict(resolved)
        local_resolved["models"] = [local_model]

        def local_provider(path: pathlib.Path, expected: dict[str, Any]) -> dict[str, Any]:
            payload = local_payloads[str(path)]
            return {"path": str(path), "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "header": read_gguf_header(io.BytesIO(payload), str(path))}
        return inventory_resolved(local_resolved, local_provider)

    partial = _synthetic_gguf([entry for entry in common if entry[0] != "split.tensors.count"]
                              + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({1: partial}), "partial split metadata")
    wrong_no = _synthetic_gguf(common + [("split.no", 7)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({1: wrong_no}), "split.no=1, got 7")
    wrong_count_common = [(key, 3 if key == "split.count" else value)
                          for key, value in common]
    wrong_count0 = _synthetic_gguf(wrong_count_common + [("split.no", 0)], [
        ("blk.0.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_gate_up_exps.weight", (256, 16, 4), 12),
    ])
    wrong_count1 = _synthetic_gguf(wrong_count_common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({0: wrong_count0, 1: wrong_count1}),
                  "split.count=3 but resolved files=2")
    wrong_total_common = [(key, 7 if key == "split.tensors.count" else value)
                          for key, value in common]
    wrong_total0 = _synthetic_gguf(wrong_total_common + [("split.no", 0)], [
        ("blk.0.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_gate_up_exps.weight", (256, 16, 4), 12),
    ])
    wrong_total1 = _synthetic_gguf(wrong_total_common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({0: wrong_total0, 1: wrong_total1}),
                  "observed tensors=6")
    duplicate = _synthetic_gguf(common + [("split.no", 1)], [
        ("blk.0.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({1: duplicate}), "appears in shards")
    wrong_e = _synthetic_gguf(common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 3), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({1: wrong_e}), "tensor E=3")
    unknown_3d = _synthetic_gguf(common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.future_experts.weight", (256, 8, 4), 12),
        ("token_embd.weight", (256, 32), 12),
    ])
    unknown_document = run_with_payloads({1: unknown_3d})
    unknown_row = next(row for row in unknown_document["tensors"]
                       if row["tensor"] == "blk.0.future_experts.weight")
    assert unknown_row["status"] == "UNSUPPORTED"
    assert unknown_row["reason"] == "TP_PARTITION_UNKNOWN"
    assert unknown_row["physical_shape"] is None
    assert unknown_document["unclassified_tensor_count"] == 1
    unknown_2d = _synthetic_gguf(common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.future_dense.weight", (256, 8), 12),
        ("token_embd.weight", (256, 32), 12),
    ])
    unknown_2d_document = run_with_payloads({1: unknown_2d})
    assert any(row["tensor"] == "blk.0.future_dense.weight" and
               row["status"] == "UNSUPPORTED" for row in unknown_2d_document["tensors"])

    # If a physical output tensor exists, it wins over the optional tied alias;
    # the embedding remains a distinct GET_ROWS row and no duplicate LM head is
    # materialised.
    actual_output = _synthetic_gguf(common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("output.weight", (256, 32), 12),
        ("token_embd.weight", (256, 32), 12),
    ])
    actual_output_document = run_with_payloads({1: actual_output})
    actual_lm = [row for row in actual_output_document["tensors"]
                 if row["role"] == "lm_head"]
    assert len(actual_lm) == 1 and actual_lm[0]["logical_alias"] is False
    assert actual_lm[0]["storage_tensor"] == "output.weight"
    assert actual_output_document["models"][0]["tied_output_alias_materialized"] is False
    assert all(row["source_tensors"] == ["output.weight"] for row in
               actual_output_document["sweep_shapes"]
               if row["logical_consumer_tensors"] == ["output.weight"])

    unknown_qtype0 = _synthetic_gguf(common + [("split.no", 0)], [
        ("blk.0.attn_q.weight", (256, 8), 999),
        ("blk.0.ffn_gate_up_exps.weight", (256, 16, 4), 12),
    ])
    unknown_qtype_document = run_with_payloads({0: unknown_qtype0})
    unknown_qtype_cells = [cell for cell in unknown_qtype_document["cells"]
                           if cell["tensor"] == "blk.0.attn_q.weight"]
    assert unknown_qtype_cells and all(cell["status"] == "UNSUPPORTED" and
                                       "_gUNKNOWN" in cell["shape_directory"]
                                       for cell in unknown_qtype_cells)
    assert all(row["group_size"] == "UNKNOWN" for row in
               unknown_qtype_document["sweep_shapes"]
               if row["source_tensors"] == ["blk.0.attn_q.weight"])

    metadata_mismatch_common = [
        (key, "other-name" if key == "general.name" else value)
        for key, value in common
    ]
    metadata_mismatch = _synthetic_gguf(metadata_mismatch_common + [("split.no", 1)], [
        ("blk.1.attn_q.weight", (256, 8), 12),
        ("blk.0.ffn_down_exps.weight", (512, 8, 4), 12),
        ("blk.0.ssm_conv1d.weight", (4, 32), 1),
        ("token_embd.weight", (256, 32), 12),
    ])
    _expect_error(lambda: run_with_payloads({1: metadata_mismatch}),
                  "metadata 'general.name' differs")
    duplicate_metadata = _synthetic_gguf(common + [("general.name", "duplicate"),
                                                    ("split.no", 0)], [])
    _expect_error(lambda: read_gguf_header(io.BytesIO(duplicate_metadata), "duplicate-metadata"),
                  "duplicate GGUF metadata key")
    _expect_error(lambda: run_with_payloads({}, lambda row: row.update(
        fileset_sha256="0" * 64)), "fileset sha mismatch")

    # Dedup must not erase TP/model/grouped identity.  These planted variants
    # differ in exactly one field and therefore require distinct keys.
    base_cell = grouped_cells[0]
    base_fields = _dedup_fields(base_cell)
    variants = []
    for field, value in (("model_id", "other-model"), ("tp_world", 4),
                         ("active", 3), ("top_k", 1),
                         ("ragged", "other-ragged")):
        changed = dict(base_fields)
        changed[field] = value
        variants.append(_dedup_key(changed))
    assert len({_dedup_key(base_fields), *variants}) == 6
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved", type=pathlib.Path,
                        help="immutable output of resolve_internal_sweep_models.py")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        help="new or empty output directory")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print("[gguf-inventory-v2:self-test] PASS: multi-shard metadata/count/name "
                  "fail-close; dense/grouped/unclassified visibility; GGUF E/top-k + pinned "
                  "routing fixture; tied LM-head; TP-local/rank/block admission; canonical "
                  "downstream provenance and dedup identity")
        if args.resolved is None:
            if args.self_test:
                return 0
            raise InventoryError("--resolved is required")
        if args.output_dir is None:
            raise InventoryError("--output-dir is required")
        resolved = load_resolved(args.resolved.resolve())
        document = inventory_resolved(resolved)
        materialise(args.output_dir, document)
        print(f"[gguf-inventory-v2] status={document['status']} "
              f"models={document['model_count']} tensors={document['tensor_count']} "
              f"matmul={document['matrix_tensor_count']} cells={document['expanded_cell_count']} "
              f"dedup={document['deduplicated_shape_count']}")
        print(f"[gguf-inventory-v2] output={args.output_dir.resolve()}")
        return 0
    except (InventoryError, OSError, struct.error) as exc:
        print(f"[gguf-inventory-v2] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
