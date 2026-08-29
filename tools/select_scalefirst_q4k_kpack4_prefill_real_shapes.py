#!/usr/bin/env python3
"""Materialize the matched 15-shape ScaleFirst Xplane/K-pack4 ship board."""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import select_scalefirst_q4k_kpack4_prefill_ab as base  # noqa: E402


SCHEMA = "quactlize.scalefirst-q4k-kpack4-prefill-real.v2"
ARM_SCHEMA = "quactlize.scalefirst-q4k-kpack4-prefill-real-arm.v2"
FAMILIES = (
    (1024, 5120),
    (5120, 8192),
    (5120, 25600),
    (8192, 5120),
    (25600, 5120),
)
PREFILL_M = (64, 2048, 4096)
SHAPES = tuple((m, n, k) for n, k in FAMILIES for m in PREFILL_M)

# Union of the established large-M controls and the M64 winner families from
# the earlier Xplane/K-pack4 searches.  Every row exists in both type spaces;
# comparing the same rows isolates the offline byte map while still allowing
# each layout to choose its own persistent grid.
CANDIDATES = (
    (32, 128, 256, 32, 16, 2),
    (32, 128, 256, 32, 16, 3),
    (64, 64, 64, 64, 32, 3),
    (64, 64, 256, 64, 16, 2),
    (64, 128, 64, 64, 16, 6),
    (64, 128, 256, 64, 16, 2),
    (128, 64, 64, 64, 32, 3),
)


def materialize(output: pathlib.Path) -> dict:
    return base.materialize(
        output, CANDIDATES, SHAPES, SCHEMA, ARM_SCHEMA,
        "adjudicate one unified K-pack4 offline format on all real persistent ScaleFirst prefill shapes")


def validate(value: dict) -> None:
    base.validate_bundle(value, CANDIDATES, SHAPES, SCHEMA)
    if value.get("purpose") != (
            "adjudicate one unified K-pack4 offline format on all real "
            "persistent ScaleFirst prefill shapes"):
        raise base.SelectError("real-shape purpose differs")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qz-sf-kpack4-real-select-") as temp:
        value = materialize(pathlib.Path(temp) / "generated")
        validate(value)
        if len(value["shapes"]) != 15 or len(value["candidates"]) != 7:
            raise AssertionError("real-shape denominator differs")
        plants = []
        broken = copy.deepcopy(value)
        broken["shapes"].pop()
        plants.append(broken)
        broken = copy.deepcopy(value)
        broken["candidates"].pop()
        plants.append(broken)
        broken = copy.deepcopy(value)
        broken["arms"][1]["weight_mapping_id"] = "0x0"
        plants.append(broken)
        for broken in plants:
            try:
                validate(broken)
            except base.SelectError:
                pass
            else:
                raise AssertionError("real-shape selector negative stayed green")
    print("[sf-kpack4-prefill-real-select:self-test] PASS 15 shapes, "
          "five families, M=64/2048/4096 and seven matched tactics; "
          "three plants RED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    run = sub.add_parser("materialize")
    run.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
        else:
            value = materialize(args.out_dir)
            validate(value)
        return 0
    except (AssertionError, OSError, base.SelectError) as error:
        print(f"[sf-kpack4-prefill-real-select] FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
