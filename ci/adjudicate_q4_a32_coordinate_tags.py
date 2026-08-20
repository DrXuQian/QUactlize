#!/usr/bin/env python3
"""Decode the exact Q4_K/A32 device coordinate-tag experiment.

The benchmark emits raw fp16 outputs.  Each arm encodes one independent part
of the source coordinate consumed by the shipping kernel; this adjudicator
combines those parts without assuming the implementation's mapping.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import pathlib
import re
import struct
import sys


K_ROUNDS = ("even", "odd", "stage")
TAIL_ROUNDS = ("even-next", "odd-next")
CODE_K_MODES = tuple(f"code-k{i}-tag" for i in range(4))
CODE_N_MODES = tuple(f"code-n{i}-tag" for i in range(3))
SCALE_GROUP_MODE = "scale-group-tag"
SCALE_N_MODE = "scale-n-tag"
ZERO_GROUP_MODE = "zero-group-tag"
ZERO_N_MODE = "zero-n-tag"
METADATA_MODES = (
    SCALE_GROUP_MODE, SCALE_N_MODE, ZERO_GROUP_MODE, ZERO_N_MODE,
)
TAG_MODES = METADATA_MODES + CODE_K_MODES + CODE_N_MODES
REQUIRED = {(mode, round_) for mode in TAG_MODES for round_ in K_ROUNDS}
TAIL_REQUIRED = {(SCALE_N_MODE, round_) for round_ in TAIL_ROUNDS}
COUNT = 64
K = 5120
N = 1024
EXPECTED_COORDINATES = 6 * COUNT * len(K_ROUNDS) + 6 * COUNT


class EvidenceError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Record:
    mode: str
    round: str
    rows: tuple[int, ...]
    row_want: tuple[int, ...]
    cols: tuple[int, ...]
    col_want: tuple[int, ...]


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def half_value(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def exact_int(bits: int, lo: int, hi: int) -> int | None:
    value = half_value(bits)
    if not math.isfinite(value) or value != math.trunc(value):
        return None
    integer = int(value)
    return integer if lo <= integer <= hi else None


def probe_k(round_: str, row: int) -> int:
    if round_ == "even":
        return 2 * row
    if round_ == "odd":
        return 2 * row + 1
    if round_ == "stage":
        return (row % (K // 128)) * 128 + 11
    if round_ == "even-next":
        return 128 + 2 * row
    if round_ == "odd-next":
        return 129 + 2 * row
    raise EvidenceError(f"unknown tag round {round_!r}")


def parse_values(text: str, field: str) -> tuple[int, ...]:
    values = tuple(int(item, 16) for item in text.split(","))
    if len(values) != COUNT:
        raise EvidenceError(f"{field}: got {len(values)} values, want {COUNT}")
    if any(value < 0 or value > 0xFFFF for value in values):
        raise EvidenceError(f"{field}: value outside raw-fp16 range")
    return values


def one_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise EvidenceError(f"{label}: got {len(matches)} records, want exactly 1")
    return matches[0]


def parse_text(text: str, label: str) -> Record:
    marker = one_match(
        re.compile(
            r"^SF_FIXTURE mode=(?P<mode>[a-z0-9-]+) first_golden=0x[0-9a-f]{4} "
            r"tag_round=(?P<round>even|odd|stage|even-next|odd-next) probe_count=64 "
            r"probe_fnv=[0-9a-f]{16} .* roundtrip=1 exact=1 isolation=1$",
            re.MULTILINE,
        ),
        text,
        f"{label}: fixture marker",
    )
    mode, round_ = marker.group("mode"), marker.group("round")
    row = one_match(
        re.compile(
            rf"^SF_TAG_ROWS mode={re.escape(mode)} tag_round={round_} n=0 "
            r"count=64 got=(?P<got>[0-9a-f,]+) want=(?P<want>[0-9a-f,]+)$",
            re.MULTILINE,
        ),
        text,
        f"{label}: row dump",
    )
    col = one_match(
        re.compile(
            rf"^SF_TAG_COLS mode={re.escape(mode)} tag_round={round_} m=0 "
            r"count=64 got=(?P<got>[0-9a-f,]+) want=(?P<want>[0-9a-f,]+)$",
            re.MULTILINE,
        ),
        text,
        f"{label}: column dump",
    )
    if not (
        re.search(r"^SF_FATAL .*state=RAW_FP16_MISMATCH step=RAW_FP16_MISMATCH", text, re.MULTILINE)
        or re.search(r"^SF_COMPLETE status=COMPLETE ", text, re.MULTILINE)
    ):
        raise EvidenceError(f"{label}: neither numeric verdict nor COMPLETE is present")
    return Record(
        mode,
        round_,
        parse_values(row.group("got"), f"{label}: rows got"),
        parse_values(row.group("want"), f"{label}: rows want"),
        parse_values(col.group("got"), f"{label}: cols got"),
        parse_values(col.group("want"), f"{label}: cols want"),
    )


def parse_log(path: pathlib.Path) -> Record:
    return parse_text(path.read_text(encoding="utf-8", errors="strict"), str(path))


def code_nibble(bits: int) -> int | None:
    decoded = exact_int(bits, -8, 7)
    return None if decoded is None else decoded + 8


def combine_code_coordinate(
    records: dict[tuple[str, str], Record], modes: tuple[str, ...],
    round_: str, index: int, axis: str, limit: int,
) -> tuple[int | None, str]:
    raw = [
        (records[(mode, round_)].rows if axis == "rows" else
         records[(mode, round_)].cols)[index]
        for mode in modes
    ]
    nibbles = [code_nibble(bits) for bits in raw]
    source = None
    if all(nibble is not None for nibble in nibbles):
        candidate = sum(int(nibble) << (4 * i)
                        for i, nibble in enumerate(nibbles))
        if 0 <= candidate < limit:
            source = candidate
    detail = ",".join(
        f"q{i}={half_value(bits):g}" for i, bits in enumerate(raw)
    )
    return source, detail


def append_result(
    lines: list[str], kind: str, round_: str, output: str,
    expected: int, source: int | None, detail: str,
) -> tuple[int, int]:
    status = "UNDECODABLE" if source is None else (
        "IDENTITY" if source == expected else "MISMATCH"
    )
    lines.append(
        f"{kind}\t{round_}\t{output}\t{expected}\t"
        f"{source if source is not None else 'NA'}\t{status}\t{detail}"
    )
    return int(status == "IDENTITY"), int(source is not None)


def tagged_coordinate(bits: int, limit: int) -> tuple[int | None, str]:
    """Decode a one-based, exactly representable fp16 coordinate tag."""
    tag = exact_int(bits, 1, limit)
    return (tag - 1 if tag is not None else None), f"tag={half_value(bits):g}"


def adjudicate(records: dict[tuple[str, str], Record]) -> tuple[str, list[str]]:
    got_keys = set(records)
    if got_keys != REQUIRED:
        missing = sorted(REQUIRED - got_keys)
        extra = sorted(got_keys - REQUIRED)
        raise EvidenceError(f"tag denominator mismatch missing={missing} extra={extra}")
    lines = ["kind\tround\toutput\tprobe\tsource\tstatus\tcomponents"]
    identity = 0
    decoded = 0
    total = 0
    for round_ in K_ROUNDS:
        scale_group = records[(SCALE_GROUP_MODE, round_)].rows
        scale_n = records[(SCALE_N_MODE, round_)].rows
        zero_group = records[(ZERO_GROUP_MODE, round_)].rows
        zero_n = records[(ZERO_N_MODE, round_)].rows
        for row in range(COUNT):
            expected_k = probe_k(round_, row)
            source_k, k_detail = combine_code_coordinate(
                records, CODE_K_MODES, round_, row, "rows", K
            )
            inc_i, inc_d = append_result(
                lines, "B_K_ROW", round_, f"m={row},n=0",
                expected_k, source_k, k_detail,
            )
            identity += inc_i; decoded += inc_d; total += 1

            source_n, n_detail = combine_code_coordinate(
                records, CODE_N_MODES, round_, row, "rows", N
            )
            inc_i, inc_d = append_result(
                lines, "B_N_ROW", round_, f"m={row},n=0",
                0, source_n, n_detail,
            )
            identity += inc_i; decoded += inc_d; total += 1

            source, detail = tagged_coordinate(scale_group[row], K // 32)
            inc_i, inc_d = append_result(
                lines, "SCALE_GROUP_ROW", round_, f"m={row},n=0",
                expected_k // 32, source, detail,
            )
            identity += inc_i; decoded += inc_d; total += 1

            source, detail = tagged_coordinate(scale_n[row], N)
            inc_i, inc_d = append_result(
                lines, "SCALE_N_ROW", round_, f"m={row},n=0",
                0, source, detail,
            )
            identity += inc_i; decoded += inc_d; total += 1

            source, detail = tagged_coordinate(zero_group[row], K // 32)
            inc_i, inc_d = append_result(
                lines, "ZERO_GROUP_ROW", round_, f"m={row},n=0",
                expected_k // 32, source, detail,
            )
            identity += inc_i; decoded += inc_d; total += 1

            source, detail = tagged_coordinate(zero_n[row], N)
            inc_i, inc_d = append_result(
                lines, "ZERO_N_ROW", round_, f"m={row},n=0",
                0, source, detail,
            )
            identity += inc_i; decoded += inc_d; total += 1

    round_ = "even"
    scale_group_cols = records[(SCALE_GROUP_MODE, round_)].cols
    scale_n_cols = records[(SCALE_N_MODE, round_)].cols
    zero_group_cols = records[(ZERO_GROUP_MODE, round_)].cols
    zero_n_cols = records[(ZERO_N_MODE, round_)].cols
    expected_k = probe_k(round_, 0)
    for n in range(COUNT):
        source_k, k_detail = combine_code_coordinate(
            records, CODE_K_MODES, round_, n, "cols", K
        )
        inc_i, inc_d = append_result(
            lines, "B_K_COL", round_, f"m=0,n={n}",
            expected_k, source_k, k_detail,
        )
        identity += inc_i; decoded += inc_d; total += 1

        source_n, n_detail = combine_code_coordinate(
            records, CODE_N_MODES, round_, n, "cols", N
        )
        inc_i, inc_d = append_result(
            lines, "B_N_COL", round_, f"m=0,n={n}", n, source_n, n_detail,
        )
        identity += inc_i; decoded += inc_d; total += 1

        source, detail = tagged_coordinate(scale_group_cols[n], K // 32)
        inc_i, inc_d = append_result(
            lines, "SCALE_GROUP_COL", round_, f"m=0,n={n}",
            expected_k // 32, source, detail,
        )
        identity += inc_i; decoded += inc_d; total += 1

        source, detail = tagged_coordinate(scale_n_cols[n], N)
        inc_i, inc_d = append_result(
            lines, "SCALE_N_COL", round_, f"m=0,n={n}", n, source, detail,
        )
        identity += inc_i; decoded += inc_d; total += 1

        source, detail = tagged_coordinate(zero_group_cols[n], K // 32)
        inc_i, inc_d = append_result(
            lines, "ZERO_GROUP_COL", round_, f"m=0,n={n}",
            expected_k // 32, source, detail,
        )
        identity += inc_i; decoded += inc_d; total += 1

        source, detail = tagged_coordinate(zero_n_cols[n], N)
        inc_i, inc_d = append_result(
            lines, "ZERO_N_COL", round_, f"m=0,n={n}", n, source, detail,
        )
        identity += inc_i; decoded += inc_d; total += 1
    if total != EXPECTED_COORDINATES:
        raise EvidenceError(
            f"coordinate denominator is {total}, want {EXPECTED_COORDINATES}"
        )
    verdict = "IDENTITY" if identity == total else (
        "NONIDENTITY" if decoded == total else "UNDECODABLE"
    )
    lines.append(
        f"summary\t-\t-\t-\t-\t{verdict}\t"
        f"identity={identity}/{total},decoded={decoded}/{total}"
    )
    return verdict, lines


def summarize_raw(values: tuple[int, ...]) -> str:
    counts = collections.Counter(values)
    return ",".join(
        f"0x{bits:04x}:{count}" for bits, count in sorted(counts.items())
    )


def adjudicate_tail(
    records: dict[tuple[str, str], Record],
) -> tuple[str, list[str], list[str]]:
    """Classify the same local TK128 coordinates one K tile later.

    The original device table lost rows 48..63 in both even/odd rounds,
    exactly local K=96..127.  Moving those impulses by +128 distinguishes a
    recurring final-delivery failure from a first-tile prologue/lifetime
    failure without changing the shipping specialization.
    """
    got_keys = set(records)
    if got_keys != TAIL_REQUIRED:
        missing = sorted(TAIL_REQUIRED - got_keys)
        extra = sorted(got_keys - TAIL_REQUIRED)
        raise EvidenceError(
            f"tail denominator mismatch missing={missing} extra={extra}"
        )
    lines = ["round\trow\tprobe_k\tgot\twant\tstatus"]
    summaries: list[str] = []
    full_counts: list[int] = []
    prefix_counts: list[int] = []
    tail_counts: list[int] = []
    for round_ in TAIL_ROUNDS:
        record = records[(SCALE_N_MODE, round_)]
        full = prefix = tail = 0
        for row, (got, want) in enumerate(zip(record.rows, record.row_want)):
            exact = got == want
            full += int(exact)
            if row < 48:
                prefix += int(exact)
            else:
                tail += int(exact)
            lines.append(
                f"{round_}\t{row}\t{probe_k(round_, row)}\t"
                f"0x{got:04x}\t0x{want:04x}\t"
                f"{'IDENTITY' if exact else 'MISMATCH'}"
            )
        full_counts.append(full)
        prefix_counts.append(prefix)
        tail_counts.append(tail)
        summaries.append(
            f"Q4_A32_TAIL_ROUND round={round_} identity={full}/64 "
            f"prefix={prefix}/48 tail={tail}/16 "
            f"tail_got={summarize_raw(record.rows[48:])}"
        )
    if full_counts == [64, 64]:
        verdict = "NEXT_TILE_LIVE"
    elif prefix_counts == [48, 48] and tail_counts == [0, 0]:
        verdict = "EVERY_TILE_LAST_DELIVERY_BAD"
    else:
        verdict = "MIXED"
    lines.append(
        "summary\t-\t-\t-\t-\t" + verdict +
        f" full={sum(full_counts)}/128 tail={sum(tail_counts)}/32"
    )
    return verdict, lines, summaries


def identity_records() -> dict[tuple[str, str], Record]:
    out: dict[tuple[str, str], Record] = {}
    for mode, round_ in sorted(REQUIRED):
        rows: list[int] = []
        cols: list[int] = []
        for row in range(COUNT):
            k = probe_k(round_, row)
            if mode in (SCALE_GROUP_MODE, ZERO_GROUP_MODE):
                value = k // 32 + 1
            elif mode in (SCALE_N_MODE, ZERO_N_MODE):
                value = 1
            elif mode in CODE_K_MODES:
                value = ((k >> (4 * CODE_K_MODES.index(mode))) & 15) - 8
            else:
                value = -8  # output n=0 in every code-N nibble.
            rows.append(half_bits(float(value)))
        for n in range(COUNT):
            k = probe_k(round_, 0)
            if mode in (SCALE_GROUP_MODE, ZERO_GROUP_MODE):
                value = k // 32 + 1
            elif mode in (SCALE_N_MODE, ZERO_N_MODE):
                value = n + 1
            elif mode in CODE_K_MODES:
                value = ((k >> (4 * CODE_K_MODES.index(mode))) & 15) - 8
            else:
                value = ((n >> (4 * CODE_N_MODES.index(mode))) & 15) - 8
            cols.append(half_bits(float(value)))
        out[(mode, round_)] = Record(
            mode, round_, tuple(rows), tuple(rows), tuple(cols), tuple(cols)
        )
    return out


def tail_identity_records() -> dict[tuple[str, str], Record]:
    out: dict[tuple[str, str], Record] = {}
    rows = tuple(half_bits(1.0) for _ in range(COUNT))
    cols = tuple(half_bits(float(n + 1)) for n in range(COUNT))
    for round_ in TAIL_ROUNDS:
        out[(SCALE_N_MODE, round_)] = Record(
            SCALE_N_MODE, round_, rows, rows, cols, cols
        )
    return out


def self_test() -> None:
    records = identity_records()
    verdict, _ = adjudicate(records)
    if verdict != "IDENTITY":
        raise EvidenceError("identity fixture did not adjudicate IDENTITY")
    key = ("code-k0-tag", "even")
    planted = dict(records)
    old = planted[key]
    rows = list(old.rows)
    rows[0] = half_bits(-7.0)  # source k low nibble moved by exactly one.
    planted[key] = dataclasses.replace(old, rows=tuple(rows))
    verdict, _ = adjudicate(planted)
    if verdict != "NONIDENTITY":
        raise EvidenceError("one-bit coordinate plant was not detected")
    key = (ZERO_GROUP_MODE, "odd")
    planted = dict(records)
    old = planted[key]
    rows = list(old.rows)
    rows[1] = half_bits(2.0)  # expected one-based group tag is 1.
    planted[key] = dataclasses.replace(old, rows=tuple(rows))
    verdict, _ = adjudicate(planted)
    if verdict != "NONIDENTITY":
        raise EvidenceError("independent zero-coordinate plant was not detected")
    missing = dict(records)
    del missing[("code-k3-tag", "stage")]
    try:
        adjudicate(missing)
    except EvidenceError:
        pass
    else:
        raise EvidenceError("missing tag combination did not fail closed")
    extra = dict(records)
    extra[(SCALE_N_MODE, "diagonal")] = dataclasses.replace(
        records[(SCALE_N_MODE, "even")], round="diagonal"
    )
    try:
        adjudicate(extra)
    except EvidenceError:
        pass
    else:
        raise EvidenceError("wrong-round tag did not fail closed")
    values = ",".join(["bc00"] * COUNT)
    parse_fixture = (
        "SF_FIXTURE mode=scale-group-tag first_golden=0x3c00 tag_round=even "
        "probe_count=64 probe_fnv=0123456789abcdef fixture=bound "
        "roundtrip=1 exact=1 isolation=1\n"
        f"SF_TAG_ROWS mode=scale-group-tag tag_round=even n=0 count=64 "
        f"got={values} want={values}\n"
        f"SF_TAG_COLS mode=scale-group-tag tag_round=even m=0 count=64 "
        f"got={values} want={values}\n"
        "SF_FATAL symbol=x state=RAW_FP16_MISMATCH "
        "step=RAW_FP16_MISMATCH\n"
    )
    parse_text(parse_fixture, "self-test")
    try:
        parse_text(parse_fixture.replace("tag_round=even", "tag_round=odd", 1),
                   "wrong-marker-round")
    except EvidenceError:
        pass
    else:
        raise EvidenceError("marker/data round mismatch did not fail closed")
    next_fixture = parse_fixture.replace(
        "tag_round=even", "tag_round=even-next"
    )
    parse_text(next_fixture, "next-tile-round")

    tail_records = tail_identity_records()
    verdict, _, _ = adjudicate_tail(tail_records)
    if verdict != "NEXT_TILE_LIVE":
        raise EvidenceError("next-tile identity fixture did not remain live")
    recurring = dict(tail_records)
    for key, old in tuple(recurring.items()):
        rows = list(old.rows)
        rows[48:] = [half_bits(0.0)] * 16
        recurring[key] = dataclasses.replace(old, rows=tuple(rows))
    verdict, _, _ = adjudicate_tail(recurring)
    if verdict != "EVERY_TILE_LAST_DELIVERY_BAD":
        raise EvidenceError("recurring final-delivery plant was not classified")
    missing_tail = dict(tail_records)
    del missing_tail[(SCALE_N_MODE, "odd-next")]
    try:
        adjudicate_tail(missing_tail)
    except EvidenceError:
        pass
    else:
        raise EvidenceError("missing shifted-tail round did not fail closed")
    print(
        "[q4-a32-tags:self-test] PASS: identity; one-bit B and independent "
        "zero-map plants=NONIDENTITY; missing denominator, wrong round and "
        "marker/data mismatch=RED; shifted-tail live/recurring classified "
        "and missing shifted round=RED"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tail-bisect", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test or (not args.logs and args.out is None):
            self_test()
            return 0
        if not args.logs or args.out is None:
            raise EvidenceError("logs and --out are required")
        records: dict[tuple[str, str], Record] = {}
        for path in args.logs:
            record = parse_log(path)
            key = (record.mode, record.round)
            if key in records:
                raise EvidenceError(f"duplicate tag combination {key}")
            records[key] = record
        if args.tail_bisect:
            verdict, lines, summaries = adjudicate_tail(records)
        else:
            verdict, lines = adjudicate(records)
            summaries = []
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.tail_bisect:
            for summary in summaries:
                print(summary)
            print(
                f"Q4_A32_TAIL_BISECT verdict={verdict} "
                f"table={args.out}"
            )
            return 0
        summary = lines[-1].split("\t")[-1]
        print(f"Q4_A32_COORDINATE_MAP verdict={verdict} {summary} table={args.out}")
        observations = [
            line for line in lines[1:-1]
            if "\tMISMATCH\t" in line or "\tUNDECODABLE\t" in line
        ]
        for line in observations[:64]:
            print(f"Q4_A32_MAP_OBSERVATION {line}")
        if len(observations) > 64:
            print(
                f"Q4_A32_MAP_OBSERVATION omitted={len(observations) - 64} "
                f"full_table={args.out}"
            )
        return 0
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"[q4-a32-tags] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
