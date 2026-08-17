#!/usr/bin/env python3
"""Contract test for the GGUF inventory -> internal-sweep bridge.

This deliberately calls the real FullyQuantized plan materializer.  A local
inventory self-test that merely re-states its own schema would not prove the
consumer can ingest it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_fully_quantized_internal_sweep as fq  # noqa: E402
import gguf_internal_shape_inventory as inventory  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def expect_error(call: Callable[[], Any], needle: str) -> None:
    try:
        call()
    except ValueError as exc:
        if needle not in str(exc):
            raise AssertionError(
                f"negative raised wrong error: {exc}; wanted {needle!r}") from exc
    else:
        raise AssertionError(f"negative did not fail: expected {needle!r}")


def main() -> int:
    # JSON round-trip first: tuples or Python-only sentinels are not a durable
    # inventory even when the in-memory producer/consumer happen to agree.
    document = json.loads(json.dumps(inventory.self_test(), ensure_ascii=False))
    manifest_sha = hashlib.sha256(canonical(document).encode("utf-8")).hexdigest()
    plan = fq._materialize_spec(  # pylint: disable=protected-access
        document, pathlib.Path("/workspace/synthetic-inventory-v2.json"), manifest_sha)

    expected_cells = sum(len(row["M_values"]) for row in document["sweep_shapes"])
    assert len(plan["cells"]) == expected_cells
    assert plan["provenance"]["gguf_hashes"] == document["provenance"]["gguf_hashes"]
    assert plan["provenance"]["gguf_set_sha256"] == document["provenance"][
        "gguf_set_sha256"]
    assert {cell["tp_rank"] for cell in plan["cells"]} == {0}
    assert {cell["problem_route"] for cell in plan["cells"]} == {"dense", "grouped"}

    resolved = {
        "schema": fq.RESOLVED_MODELS_SCHEMA,
        "shape_directory": document["provenance"]["shape_directory"],
        "workload_axes": document["workload_axes"],
        "resolved_set_sha256": document["resolved_set_sha256"],
        "models": [{
            "model_id": document["models"][0]["model_id"],
            "fileset_sha256": document["models"][0]["fileset_sha256"],
            "tp_world_size": 2,
        }],
    }
    fq._validate_resolved_document(plan, resolved, "synthetic-resolved")  # noqa: SLF001

    bad_set = copy.deepcopy(document)
    bad_set["provenance"]["gguf_set_sha256"] = "0" * 64
    expect_error(lambda: fq._materialize_spec(  # pylint: disable=protected-access
        bad_set, pathlib.Path("/workspace/bad-set.json"), "1" * 64),
        "does not bind gguf_hashes")

    missing_source = copy.deepcopy(document)
    del missing_source["sweep_shapes"][0]["sources"]
    expect_error(lambda: fq._materialize_spec(  # pylint: disable=protected-access
        missing_source, pathlib.Path("/workspace/missing-source.json"), "2" * 64),
        "lacks canonical fields")

    print("[gguf-inventory-v2:bridge] PASS: real FQ consumer accepted canonical "
          "multi-GGUF/model/TP/grouped rows; aggregate-hash, source-provenance, "
          "and missing-source negatives red after JSON round-trip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
