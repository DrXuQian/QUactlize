"""GGUF tensor-name authority shared by the installed packer and sweep tools.

GGUF rank alone does not identify an operation: rank-three weights may be
grouped ``MUL_MAT_ID`` operands, while rank-two tensors include both matrices
and lookup tables.  Keep the exact llama.cpp-derived name rules in the
installed package so artifact conversion does not depend on the repository's
development-only ``tools`` package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class InventoryError(ValueError):
    """A tensor name/rank cannot be assigned one unambiguous runtime role."""


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


# Every recognised name is tied to the llama.cpp tensor symbol/op that
# establishes its semantics.  Unknown matrix-shaped tensors fail closed.
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


__all__ = [
    "InventoryError", "Role", "RoleRule", "ROLE_RULES", "match_role", "classify_role",
]
