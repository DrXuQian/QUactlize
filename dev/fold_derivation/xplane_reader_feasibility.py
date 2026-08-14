#!/usr/bin/env python3
"""Exhaustive, host-only feasibility census for the shipping BC xplane reader.

The authority is the finite `arrangement_supported_v` domain plus each explicit
`ArrangementSlotPermutation` specialization in gguf_bc_vecdot.hpp.  This probe
does not propose a new placement.  It asks whether one fixed logical row can be
covered by aligned 32-bit words and, when it can, records the exact register
permutation needed to recover logical K.

The production-writer anchor is deliberately outside this parallel model:
run_xplane_reader_feasibility.sh first runs L137, which exhaustively compares
these permutations with xplane::place_from_map.  The unfolded-Q4 signature is
also checked against q4_group's independently shipped 8x4 transpose.
"""

from __future__ import annotations

import argparse
import dataclasses
import fractions
import json
import math
import pathlib
import re
import sys
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[2]
BC_HEADER = ROOT / "quactlize/include/gguf_bc_vecdot.hpp"
SCALE_HEADER = ROOT / "quactlize/include/gguf_scale_layout.hpp"


@dataclasses.dataclass(frozen=True)
class FormatTraits:
    name: str
    low_bits: int
    high_bits: int
    group_size: int


@dataclasses.dataclass(frozen=True)
class SlotSpec:
    qtype: str
    artifact_tile_k: int
    high: bool
    strides: tuple[int, ...]


class ExpressionParser:
    """Tiny fail-closed parser for arrangement_supported_v's C++ expression."""

    TOKEN = re.compile(
        r"\s*(KType::[A-Za-z0-9_]+|ArtifactTileK|T|==|\|\||&&|\?|:|\(|\)|[0-9]+|true|false)"
    )

    def __init__(self, text: str):
        self.tokens: list[str] = []
        pos = 0
        while pos < len(text):
            match = self.TOKEN.match(text, pos)
            if not match:
                raise ValueError(f"unsupported token in arrangement_supported_v at {text[pos:pos+40]!r}")
            self.tokens.append(match.group(1))
            pos = match.end()
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse(self) -> Any:
        node = self.conditional()
        if self.peek() is not None:
            raise ValueError(f"unparsed arrangement_supported_v token {self.peek()!r}")
        return node

    def conditional(self) -> Any:
        condition = self.or_expr()
        if self.peek() == "?":
            self.take("?")
            yes = self.conditional()
            self.take(":")
            no = self.conditional()
            return ("?:", condition, yes, no)
        return condition

    def or_expr(self) -> Any:
        node = self.and_expr()
        while self.peek() == "||":
            self.take()
            node = ("||", node, self.and_expr())
        return node

    def and_expr(self) -> Any:
        node = self.equality()
        while self.peek() == "&&":
            self.take()
            node = ("&&", node, self.equality())
        return node

    def equality(self) -> Any:
        node = self.primary()
        while self.peek() == "==":
            self.take()
            node = ("==", node, self.primary())
        return node

    def primary(self) -> Any:
        token = self.take()
        if token == "(":
            node = self.conditional()
            self.take(")")
            return node
        if token == "true":
            return True
        if token == "false":
            return False
        if token.isdigit():
            return int(token)
        if token in ("T", "ArtifactTileK"):
            return ("var", token)
        if token.startswith("KType::"):
            return token.removeprefix("KType::")
        raise ValueError(f"unsupported primary token {token!r}")


def eval_expr(node: Any, qtype: str, artifact_tile_k: int) -> Any:
    if isinstance(node, (bool, int, str)):
        return node
    op = node[0]
    if op == "var":
        return qtype if node[1] == "T" else artifact_tile_k
    if op == "==":
        return eval_expr(node[1], qtype, artifact_tile_k) == eval_expr(node[2], qtype, artifact_tile_k)
    if op == "||":
        return bool(eval_expr(node[1], qtype, artifact_tile_k)) or bool(eval_expr(node[2], qtype, artifact_tile_k))
    if op == "&&":
        return bool(eval_expr(node[1], qtype, artifact_tile_k)) and bool(eval_expr(node[2], qtype, artifact_tile_k))
    if op == "?:":
        branch = node[2] if bool(eval_expr(node[1], qtype, artifact_tile_k)) else node[3]
        return eval_expr(branch, qtype, artifact_tile_k)
    raise ValueError(f"unsupported expression node {node!r}")


def extract_braced_traits(text: str) -> dict[str, tuple[str | None, str]]:
    pattern = re.compile(
        r"template\s*<>\s*struct\s+Traits<KType::([A-Za-z0-9_]+)>"
        r"\s*(?::\s*Traits<KType::([A-Za-z0-9_]+)>)?\s*\{(.*?)\};",
        re.S,
    )
    return {match.group(1): (match.group(2), match.group(3)) for match in pattern.finditer(text)}


def parse_authority() -> tuple[list[str], dict[str, FormatTraits], set[tuple[str, int]], dict[tuple[str, int, bool], SlotSpec]]:
    bc = BC_HEADER.read_text()
    scale = SCALE_HEADER.read_text()

    enum_match = re.search(r"enum\s+class\s+KType\s*\{([^}]+)\}", scale)
    if not enum_match:
        raise ValueError("KType enum not found")
    qtypes = [item.strip() for item in enum_match.group(1).split(",") if item.strip()]

    bc_traits: dict[str, tuple[int, int]] = {}
    for match in re.finditer(
        r"template\s*<>\s*struct\s+Traits<KType::([A-Za-z0-9_]+)>\s*\{"
        r"\s*static\s+constexpr\s+int\s+Lo\s*=\s*([0-9]+)\s*,\s*Hi\s*=\s*([0-9]+)",
        bc,
    ):
        bc_traits[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    scale_traits = extract_braced_traits(scale)

    def group_size(name: str, seen: set[str] | None = None) -> int:
        seen = set() if seen is None else seen
        if name in seen or name not in scale_traits:
            raise ValueError(f"cannot resolve kGroupSize for {name}")
        seen.add(name)
        base, body = scale_traits[name]
        match = re.search(r"kGroupSize\s*=\s*([0-9]+)", body)
        if match:
            return int(match.group(1))
        if base:
            return group_size(base, seen)
        raise ValueError(f"kGroupSize absent for {name}")

    if set(qtypes) != set(bc_traits):
        raise ValueError(f"BC traits {sorted(bc_traits)} do not cover KType enum {qtypes}")
    traits = {
        name: FormatTraits(name, bc_traits[name][0], bc_traits[name][1], group_size(name))
        for name in qtypes
    }

    supported_match = re.search(
        r"inline\s+constexpr\s+bool\s+arrangement_supported_v\s*=\s*(.*?);", bc, re.S
    )
    if not supported_match:
        raise ValueError("arrangement_supported_v initializer not found")
    support_text = supported_match.group(1)
    support_ast = ExpressionParser(support_text).parse()
    artifact_values = sorted({int(value) for value in re.findall(r"ArtifactTileK\s*==\s*([0-9]+)", support_text)})
    if not artifact_values:
        raise ValueError("arrangement_supported_v names no finite ArtifactTileK values")
    supported = {
        (name, artifact_tile_k)
        for name in qtypes
        for artifact_tile_k in artifact_values
        if bool(eval_expr(support_ast, name, artifact_tile_k))
    }

    slots: dict[tuple[str, int, bool], SlotSpec] = {}
    slot_pattern = re.compile(
        r"QUACTLIZE_BC_SLOT\((Q[0-9]+_K),\s*([0-9]+),\s*(true|false),\s*([^;]+?)\);"
    )
    for match in slot_pattern.finditer(bc):
        values = tuple(int(value.strip(), 0) for value in match.group(4).split(","))
        if len(values) != 14:
            raise ValueError(f"{match.group(1)} A={match.group(2)} slot permutation has {len(values)} entries")
        key = (match.group(1), int(match.group(2)), match.group(3) == "true")
        if key in slots:
            raise ValueError(f"duplicate ArrangementSlotPermutation {key}")
        slots[key] = SlotSpec(*key, values)

    expected_planes = {
        (name, artifact_tile_k, high)
        for name, artifact_tile_k in supported
        for high in ([False, True] if traits[name].high_bits else [False])
    }
    if set(slots) != expected_planes:
        missing = sorted(expected_planes - set(slots))
        extra = sorted(set(slots) - expected_planes)
        raise ValueError(f"slot/support denominator mismatch: missing={missing} extra={extra}")
    return qtypes, traits, supported, slots


def delivery_fold(bits: int, artifact_tile_k: int) -> int:
    if bits == 0 or artifact_tile_k * bits // 8 >= 32:
        return 1
    return 32 // (artifact_tile_k * bits // 8)


class PlaneMap:
    N = 256
    K = 256

    def __init__(self, traits: FormatTraits, spec: SlotSpec):
        self.traits = traits
        self.spec = spec
        self.bits = traits.high_bits if spec.high else traits.low_bits
        self.cpw = 32 // self.bits
        self.fold = delivery_fold(self.bits, spec.artifact_tile_k)
        other_bits = traits.low_bits if spec.high else traits.high_bits
        other_fold = delivery_fold(other_bits, spec.artifact_tile_k) if other_bits else 1
        max_fold = max(self.fold, other_fold)
        self.wn = 16 * max_fold if max_fold > 2 else 32
        self.tn = 2 * self.wn
        self.dl = self.fold * spec.artifact_tile_k * self.bits // 256
        if self.dl < 1:
            raise ValueError(f"invalid delivery count for {spec}")

    def physical(self, n: int, k: int) -> int:
        artifact_tile_k = self.spec.artifact_tile_k
        local = (n % self.tn) * artifact_tile_k + (k % artifact_tile_k)
        slot = 0
        for bit, stride in enumerate(self.spec.strides):
            if local & (1 << bit):
                slot |= stride
        j = slot % self.cpw
        slot //= self.cpw
        wd = slot % 8
        slot //= 8
        dl = slot % self.dl
        row = slot // self.dl
        tn = n // self.tn
        artifact_ki = k // artifact_tile_k
        if self.fold > 1:
            word_row_off = 256 // self.cpw
            runs = word_row_off // 8
            rows = self.tn // self.fold
            return (
                j
                + wd * self.cpw
                + row * 256
                + tn * rows * 256
                + (artifact_ki % runs) * 8 * self.cpw
                + (artifact_ki // runs) * (self.N // self.fold) * 256
            )
        contig = artifact_tile_k * self.bits // 8
        aiu_byte = min(contig, 128)
        aiu_elem = aiu_byte * 8 // self.bits
        rows_per_supertile = 256 // aiu_elem
        return (
            j
            + wd * self.cpw
            + dl * 8 * self.cpw
            + (artifact_ki % rows_per_supertile) * aiu_elem
            + row * 256
            + tn * self.tn * 256
            + (artifact_ki // rows_per_supertile) * self.N * 256
        )

    def owners(self) -> dict[int, tuple[int, int]]:
        owners: dict[int, tuple[int, int]] = {}
        for n in range(self.N):
            for k in range(self.K):
                physical = self.physical(n, k)
                if physical in owners:
                    raise ValueError(f"non-bijective physical owner {self.spec} at {physical}")
                owners[physical] = (n, k)
        if len(owners) != self.N * self.K or min(owners) != 0 or max(owners) != self.N * self.K - 1:
            raise ValueError(
                f"physical plane is not the exact [0,{self.N*self.K}) bijection for {self.spec}: "
                f"count={len(owners)} range={min(owners)}..{max(owners)}"
            )
        return owners


def closure_for_run(plane: PlaneMap, owners: dict[int, tuple[int, int]], run: int) -> tuple[bool, Any]:
    signatures: set[tuple[int, ...]] = set()
    word_offsets: set[tuple[int, ...]] = set()
    for n in range(plane.N):
        for k0 in range(0, plane.K, run):
            physical = [plane.physical(n, k) for k in range(k0, k0 + run)]
            words = sorted({value // plane.cpw for value in physical})
            logical: list[int] = []
            for word in words:
                for j in range(plane.cpw):
                    owner = owners.get(word * plane.cpw + j)
                    if owner is None or owner[0] != n:
                        return False, "WORD_MIXES_LOGICAL_ROWS"
                    if not (k0 <= owner[1] < k0 + run):
                        return False, "WORD_MIXES_K_WINDOWS"
                    logical.append(owner[1] - k0)
            if len(words) * plane.cpw != run:
                return False, "WINDOW_DOES_NOT_FILL_WHOLE_WORDS"
            signatures.add(tuple(logical))
            word_offsets.add(tuple(word - words[0] for word in words))
    if len(signatures) != 1:
        return False, "PERMUTATION_VARIES_BY_POSITION"
    if len(word_offsets) != 1:
        return False, "WORD_OFFSETS_VARY_BY_POSITION"
    return True, (next(iter(signatures)), next(iter(word_offsets)))


def analyze_plane(traits: FormatTraits, spec: SlotSpec, plant_wrong_permutation_bit: bool = False) -> dict[str, Any]:
    plane = PlaneMap(traits, spec)
    owners = plane.owners()
    candidate_runs = [run for run in range(plane.cpw, plane.K + 1, plane.cpw) if plane.K % run == 0]
    direct_ok, direct_detail = closure_for_run(plane, owners, plane.cpw)
    closure_run: int | None = None
    signature: tuple[int, ...] = ()
    word_offsets: tuple[int, ...] = ()
    last_reason = str(direct_detail)
    for run in candidate_runs:
        ok, detail = closure_for_run(plane, owners, run)
        if ok:
            closure_run = run
            signature, word_offsets = detail
            break
        last_reason = str(detail)

    oracle_coordinates = 0
    if closure_run is not None:
        # This is a bijective, one-bit error rather than an obviously duplicate table entry:
        # every logical offset's low bit is wrong.  Only the scalar code_at oracle can reject it.
        if plant_wrong_permutation_bit:
            signature = tuple(logical_k ^ 1 for logical_k in signature)
        inverse = [-1] * closure_run
        for physical_slot, logical_k in enumerate(signature):
            if logical_k < 0 or logical_k >= closure_run or inverse[logical_k] != -1:
                raise ValueError(f"fast permutation is not bijective for {spec}")
            inverse[logical_k] = physical_slot
        for n in range(plane.N):
            for k0 in range(0, plane.K, closure_run):
                words = sorted(
                    {plane.physical(n, k) // plane.cpw for k in range(k0, k0 + closure_run)}
                )
                for logical_k in range(closure_run):
                    slot = inverse[logical_k]
                    planned = words[slot // plane.cpw] * plane.cpw + slot % plane.cpw
                    scalar = plane.physical(n, k0 + logical_k)
                    if planned != scalar:
                        planted = "PLANTED_RED wrong-permutation-bit " if plant_wrong_permutation_bit else ""
                        raise ValueError(
                            f"{planted}fast/code_at address mismatch {spec} n={n} k={k0+logical_k}: "
                            f"fast={planned} scalar={scalar}"
                        )
                    oracle_coordinates += 1

    return {
        "qtype": spec.qtype,
        "artifact_tile_k": spec.artifact_tile_k,
        "high": spec.high,
        "bits": plane.bits,
        "fold": plane.fold,
        "cpw": plane.cpw,
        "direct_cpw_same_word": direct_ok,
        "direct_reason": None if direct_ok else str(direct_detail),
        "closure_run": closure_run,
        "closure_words": None if closure_run is None else closure_run // plane.cpw,
        "word_offsets": list(word_offsets),
        "permutation": list(signature),
        "reject_reason": None if closure_run is not None else last_reason,
        "oracle_coordinates": oracle_coordinates,
        "before_position_terms_per_code": 14,
        "before_scalar_plane_loads_per_code": 1,
        "after_position_terms_per_code": None if closure_run is None else 0,
        "after_word_loads_per_code": None if closure_run is None else f"1/{plane.cpw}",
    }


def permutation_id(row: dict[str, Any]) -> str:
    if row["closure_run"] is None:
        return "-"
    return f"P{row['bits']}x{row['closure_run']}"


def aggregate(
    qtypes: list[str],
    traits: dict[str, FormatTraits],
    supported: set[tuple[str, int]],
    planes: list[dict[str, Any]],
    plant: str,
) -> dict[str, Any]:
    by_plane = {(row["qtype"], row["artifact_tile_k"], row["high"]): row for row in planes}

    arrangements: list[dict[str, Any]] = []
    for qtype in qtypes:
        for artifact_tile_k in sorted(a for name, a in supported if name == qtype):
            keys = [(qtype, artifact_tile_k, False)]
            if traits[qtype].high_bits:
                keys.append((qtype, artifact_tile_k, True))
            rows = [by_plane[key] for key in keys]
            feasible = all(row["closure_run"] is not None for row in rows)
            before_terms = sum(row["before_position_terms_per_code"] for row in rows)
            before_loads = len(rows)
            after_loads = sum(
                (fractions.Fraction(1, row["cpw"]) for row in rows), fractions.Fraction(0)
            ) if feasible else None
            arrangements.append(
                {
                    "qtype": qtype,
                    "artifact_tile_k": artifact_tile_k,
                    "planes": len(rows),
                    "fast": feasible,
                    "closure_run": math.lcm(*(row["closure_run"] for row in rows)) if feasible else None,
                    "permutations": [permutation_id(row) for row in rows],
                    "reject_reasons": sorted(
                        {
                            f"{'high' if row['high'] else 'low'}:{row['reject_reason']}"
                            for row in rows
                            if row["reject_reason"]
                        }
                    ),
                    "before_position_terms_per_code": before_terms,
                    "before_scalar_plane_loads_per_code": before_loads,
                    "after_position_terms_per_code": 0 if feasible else None,
                    "after_word_loads_per_code": (
                        f"{after_loads.numerator}/{after_loads.denominator}" if feasible else None
                    ),
                }
            )
    if plant == "missing-denominator":
        arrangements.pop()

    reported = {(row["qtype"], row["artifact_tile_k"]) for row in arrangements}
    if reported != supported:
        raise ValueError(
            "PLANTED_RED coverage denominator differs from arrangement_supported_v: "
            f"missing={sorted(supported-reported)} extra={sorted(reported-supported)}"
        )

    # q4_group is the independent positive reference.  It ships only for unfolded Q4,
    # and its p -> k formula is (p&3)*8 + (p>>2).
    q4_signature = tuple((position & 3) * 8 + (position >> 2) for position in range(32))
    for artifact_tile_k in (64, 128, 256):
        row = by_plane[("Q4_K", artifact_tile_k, False)]
        if row["fold"] != 1 or tuple(row["permutation"]) != q4_signature:
            raise ValueError(f"unfolded Q4 does not reproduce q4_group at A={artifact_tile_k}")

    fast_arrangements = sum(row["fast"] for row in arrangements)
    fast_planes = sum(row["closure_run"] is not None for row in planes)
    direct_planes = sum(row["direct_cpw_same_word"] for row in planes)
    return {
        "authority": {
            "arrangement_supported_v": str(BC_HEADER.relative_to(ROOT)),
            "arrangement_slot_permutation": str(BC_HEADER.relative_to(ROOT)),
            "production_writer_anchor": "dev/fold_derivation/run_l137_bc_arrangement_layout.sh",
            "positive_reference": "gguf_bc_vecdot.hpp::q4_group",
        },
        "coverage": {
            "fast_arrangements": fast_arrangements,
            "supported_arrangements": len(supported),
            "fast_planes": fast_planes,
            "supported_planes": len(planes),
            "direct_cpw_planes": direct_planes,
            "oracle_coordinates": sum(row["oracle_coordinates"] for row in planes),
            "classified_coordinates": len(planes) * PlaneMap.N * PlaneMap.K,
            "explicit_reject_coordinates": sum(
                PlaneMap.N * PlaneMap.K for row in planes if row["closure_run"] is None
            ),
        },
        "arrangements": arrangements,
        "planes": planes,
    }


PREREGISTRATION = """### Preregistered criteria (verbatim from INBOX 169)

* (a) **源码级**:每码指令数,逐 `(T, ArtifactTileK)`,改前/改后同表。与编译器无关,先报这个。
* (b) **覆盖**:走快读的 `(T, ArtifactTileK, High)` 组合数 / 受支持组合总数。分母来自 `arrangement_supported_v` 的枚举,**不是手写清单**。
* (c) **box 实测**:② 同 shape 的时间。**注意 ② 现在没有基线**,所以第一次跑要先补 baseline 再谈提升。

### Negative controls (verbatim from INBOX 169)

1. 快读与 `code_at` 必须在同一测试里逐码比对,**全部受支持组合、全部 (n,k)**,不是抽样。`code_at` 保留为 oracle,不许删。
2. 植入一个**字内置换错一位**的故障,必须判红。这是 `q4_group` 那类"physically contiguous 但逻辑 K 是转置"的实际失败形态。
3. 对 (b) 的分母植入一个"少枚举一个受支持组合"的故障,必须判红 —— 否则覆盖率可以靠缩分母刷。

### Scope limits (verbatim from INBOX 169)

* 不动 π,不动离线摆放,不动 artifact 字节。**prefill 的零成本性质是硬约束。**
* 不碰 ① 和 ③;它们是否退役是另一件事(见 167 末尾),本任务不预设。
* 2.75 条/对是 sm_120 的;PPU codegen 未测。本任务的 (a) 是源码级计数,**不许把它说成 PPU 实测**。
* TODO #58 与出货路无关(见第四节),**不要顺手做**。
"""


def markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Shipping BC xplane whole-word reader feasibility",
        "",
        "This is a host-only feasibility verdict for INBOX 169. It changes no kernel, placement, or artifact bytes.",
        "The denominator is evaluated from `arrangement_supported_v`; it is not a copied support list.",
        "L137 independently anchors every slot permutation to the production `xplane::place_from_map` writer.",
        "",
        "## Verdict",
        "",
        f"* Direct CPW window in one word: **{coverage['direct_cpw_planes']}/{coverage['supported_planes']} planes**.",
        f"* Fixed-row multiword closure: **{coverage['fast_planes']}/{coverage['supported_planes']} planes**.",
        f"* Complete formats (all resident planes close): **{coverage['fast_arrangements']}/{coverage['supported_arrangements']} arrangements**.",
        f"* Fast-plan versus scalar-address comparisons: **{coverage['oracle_coordinates']:,} coordinates**, exhaustive.",
        f"* Total classified domain: **{coverage['classified_coordinates']:,} coordinates**; the remaining "
        f"**{coverage['explicit_reject_coordinates']:,}** are explicit `UNSUPPORTED`, never scalar fallback.",
        "* Conclusion: **partial generalisation, not a universal reader**. The existing Q4 path is a multiword",
        "  positive: 32 logical codes close over four words and use one fixed 8x4 register permutation.",
        "  Literal consecutive-CPW blocks do not individually occupy one word.",
        "* Criterion (c) remains **NOT MEASURED**: shipping `gguf_bc_vecdot` has no box baseline yet.",
        "* Criterion (a) below counts source-level address/extraction primitives only. The post-change register",
        "  permutation has not been lowered, so this is not a PPU instruction-count claim.",
        "",
        "## Arrangement table",
        "",
        "| T | ArtifactTileK | planes | fast | closure K | permutations | source before | source after | rejection |",
        "|---|---:|---:|:---:|---:|---|---|---|---|",
    ]
    for row in report["arrangements"]:
        before = (
            f"{row['before_position_terms_per_code']} slot terms + "
            f"{row['before_scalar_plane_loads_per_code']} scalar plane loads/code"
        )
        after = (
            f"0 slot terms + {row['after_word_loads_per_code']} word loads/code"
            if row["fast"] else "explicitly unsupported"
        )
        lines.append(
            f"| {row['qtype']} | {row['artifact_tile_k']} | {row['planes']} | "
            f"{'YES' if row['fast'] else 'NO'} | {row['closure_run'] or '-'} | "
            f"{', '.join(row['permutations'])} | {before} | {after} | "
            f"{'; '.join(row['reject_reasons']) or '-'} |"
        )

    lines += [
        "",
        "## Plane table",
        "",
        "`permutation` lists logical K offsets in physical word/slot order. Consecutive chunks of CPW entries",
        "are the exact within-word permutations.",
        "",
        "| T | A | plane | bits | F | CPW | direct CPW word | minimum closed K | words | permutation | reason |",
        "|---|---:|---|---:|---:|---:|:---:|---:|---:|---|---|",
    ]
    for row in report["planes"]:
        permutation = ",".join(str(value) for value in row["permutation"]) or "-"
        lines.append(
            f"| {row['qtype']} | {row['artifact_tile_k']} | {'high' if row['high'] else 'low'} | "
            f"{row['bits']} | {row['fold']} | {row['cpw']} | "
            f"{'YES' if row['direct_cpw_same_word'] else 'NO'} | {row['closure_run'] or '-'} | "
            f"{row['closure_words'] or '-'} | `{permutation}` | {row['reject_reason'] or '-'} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Fast complete arrangements are Q2 A=64/128/256; Q3 A=128/256; Q4 A=32/64/128/256; and",
        "Q6 A=64/128. Q5 is only a partial-plane opportunity: its low plane closes, but every supported high",
        "plane word mixes logical rows, so silently retaining per-code `code_at` for the high plane would violate",
        "the preregistered property. Q2 A=32, Q3 A=64, and Q6 A=32 fail for the same row-mixing reason.",
        "",
        "The three reusable fixed permutations are P4x32, P2x64, and P1x128. P4x32 is byte-for-byte the",
        "shipping `q4_group` formula `k=(p&3)*8+(p>>2)`. The A=32 Q4 result is a feasibility result only; the",
        "current kernel intentionally does not select `q4_group` for folded Q4.",
        "",
        PREREGISTRATION.rstrip(),
        "",
    ]
    return "\n".join(lines)


def run(plant: str) -> dict[str, Any]:
    qtypes, traits, supported, slots = parse_authority()
    planes = [
        analyze_plane(
            traits[qtype],
            slots[(qtype, artifact_tile_k, high)],
            plant_wrong_permutation_bit=(
                plant == "wrong-permutation-bit"
                and qtype == "Q4_K"
                and artifact_tile_k == 64
                and not high
            ),
        )
        for qtype in qtypes
        for artifact_tile_k in sorted(a for name, a in supported if name == qtype)
        for high in ([False, True] if traits[qtype].high_bits else [False])
    ]
    return aggregate(qtypes, traits, supported, planes, plant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument(
        "--plant", choices=("none", "wrong-permutation-bit", "missing-denominator"), default="none"
    )
    args = parser.parse_args()
    try:
        report = run(args.plant)
    except Exception as exc:  # fail closed; plants intentionally arrive here
        print(f"[xplane-reader] FAIL: {exc}", file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"[xplane-reader] FAIL: plant {args.plant} escaped", file=sys.stderr)
        return 1
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    md = markdown(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload)
    else:
        print(payload, end="")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)
    coverage = report["coverage"]
    print(
        "[xplane-reader] PASS: "
        f"arrangements={coverage['fast_arrangements']}/{coverage['supported_arrangements']} "
        f"planes={coverage['fast_planes']}/{coverage['supported_planes']} "
        f"direct_cpw={coverage['direct_cpw_planes']}/{coverage['supported_planes']} "
        f"oracle_coordinates={coverage['oracle_coordinates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
