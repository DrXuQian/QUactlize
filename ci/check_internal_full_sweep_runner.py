#!/usr/bin/env python3
"""Exercise catalog -> GGUF inventory -> four-board merge as one command.

The real component runners are device-only.  This local check replaces only
their measurement step with hash-bound synthetic COMPLETE summaries; the
catalog resolver, GGUF parser, input freezing, merger, and model/shape folder
materialisation are the production implementations.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = pathlib.Path("/workspace").resolve()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def write_fake_component(path: pathlib.Path) -> None:
    path.write_text(r'''#!/usr/bin/env python3
import collections, copy, hashlib, json, os, pathlib, sys

root = pathlib.Path(sys.argv[1])
component = sys.argv[2]
output = pathlib.Path(os.environ["OUT"]) / "results" / "summary.json"
spec_path = pathlib.Path(os.environ["INTERNAL_SWEEP_SPEC"])
sys.path.insert(0, str(root / "tools"))
import merge_internal_full_sweep as merger

spec = json.loads(spec_path.read_text())
doc = merger.synthetic_doc(component)
if component == "fully_quantized":
    doc["cells"] = [cell for cell in doc["cells"]
                    if str(cell["qtype"]).upper() not in {"8", "Q8", "Q8_0"}]
base_cells = doc["cells"]
doc["cells"] = []
for row in spec["sweep_shapes"]:
    for base in base_cells:
        cell = copy.deepcopy(base)
        cell.update({
            "model_id": row["model_id"], "shape_id": row["shape_id"],
            "source_tensors": row["source_tensors"],
            "tp_world": row["tp_world"], "tp_rank": row["tp_rank"],
            "tp_partition": row["tp_partition"],
            "problem_route": row["problem_route"],
            "group_size": row["group_size"], "grouped": row["grouped"],
            "qtype": row["qtype"], "tensor": row["source_tensors"][0],
            "shape": {"m": row["M"], "n": row["N"], "k": row["K"], "l": row["L"]},
        })
        doc["cells"].append(cell)
if os.environ.get("FAKE_FOREIGN_MODEL") == "1":
    for cell in doc["cells"]:
        cell["model_id"] = "model-b"
if os.environ.get("FAKE_GROUPED_DRIFT") == "1":
    changed = 0
    for cell in doc["cells"]:
        if cell["problem_route"] == "grouped":
            cell["grouped"] = copy.deepcopy(cell["grouped"])
            cell["grouped"]["active"] += 1
            changed += 1
    if changed == 0:
        raise SystemExit("grouped-drift negative had no grouped cells")
doc["expected_cells"] = len(doc["cells"])
doc["status_counts"] = dict(collections.Counter(
    cell["status"] for cell in doc["cells"]))
root_sha = __import__("subprocess").check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
actlize_sha = __import__("subprocess").check_output(
    ["git", "-C", str(root / "third_party/actlize"), "rev-parse", "HEAD"],
    text=True).strip()
doc["provenance"].update({
    "root_sha": root_sha, "actlize_sha": actlize_sha,
    "shape_manifest_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    "gguf_hashes": spec["provenance"]["gguf_hashes"],
    "gguf_set_sha256": spec["provenance"]["gguf_set_sha256"],
    "shape_directory": spec["provenance"]["shape_directory"],
    "orchestration_attempt_id": os.environ["INTERNAL_SWEEP_ATTEMPT_ID"],
})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
''')
    path.chmod(0o755)


def run() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import gguf_internal_shape_inventory as inventory  # pylint: disable=import-outside-toplevel

    base = WORKSPACE / f"quactlize-internal-full-runner-selftest-{os.getpid()}"
    if base.exists():
        raise AssertionError(f"self-test path unexpectedly exists: {base}")
    base.mkdir()
    try:
        gguf = base / "model.gguf"
        gguf.write_bytes(inventory._synthetic_gguf(  # pylint: disable=protected-access
            [("general.name", "runner-self-test"),
             ("general.architecture", "qwen35moe"),
             ("qwen35moe.expert_count", 4),
             ("qwen35moe.expert_used_count", 2)],
            [("blk.0.attn_q.weight", (256, 32), 12),
             ("blk.0.ffn_up_exps.weight", (256, 32, 4), 12),
             # Visibility-only GET_ROWS row: model TP identity applies, but it
             # must not invent a per-row matrix partition/shape.
             ("token_embd.weight", (256, 32), 12),
             # A real model contains non-matrix ranks.  Keep one here so the
             # validator cannot equate all-rank logical_tensor_count with the
             # deliberately rank-2/3-only tensors[] publication.
             ("blk.0.attn_norm.weight", (32,), 1)]))

        catalog = json.loads((ROOT / "benchmarks/internal_sweep_models.json").read_text())
        catalog["workload_axes"]["dense"] = {"decode_m": [1], "prefill_m": [64]}
        catalog["workload_axes"]["grouped"].update(
            {"decode_tokens": [1], "prefill_tokens": [64]})
        model = copy.deepcopy(catalog["models"][0])
        model.update({
            "model_id": "model-a", "display_name": "runner self-test",
            "binding_env": "QUACTLIZE_RUNNER_SELFTEST_GGUF",
            "problem_kinds": ["dense", "grouped"], "tp_world_size": 1,
            "tp_policy": "replicated-v1",
        })
        catalog["models"] = [model]
        catalog_path = base / "catalog.json"
        catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")

        fake = base / "fake-component.py"
        write_fake_component(fake)
        wrapper = base / "fake-component.sh"
        wrapper.write_text(
            "#!/usr/bin/env bash\nset -u\n"
            f'exec python3 -B "{fake}" "{ROOT}" "${{INTERNAL_SWEEP_COMPONENT}}"\n')
        wrapper.chmod(0o755)

        bundle = base / "bundle"
        env = dict(os.environ)
        env.update({
            "OUT": str(bundle),
            "INTERNAL_SWEEP_CATALOG": str(catalog_path),
            "QUACTLIZE_RUNNER_SELFTEST_GGUF": str(gguf),
            "SCALEFIRST_RUNNER": str(wrapper),
            "FULLY_QUANTIZED_RUNNER": str(wrapper),
            "INTERNAL_SWEEP_DEV_MODE": "1",
            "RESUME": "0",
        })
        for name in ("GGUF_SET", "INTERNAL_SWEEP_SPEC"):
            env.pop(name, None)

        production_override_env = dict(env, OUT=str(base / "production-override"))
        production_override_env.pop("INTERNAL_SWEEP_DEV_MODE")
        production_override = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=production_override_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if (production_override.returncode == 0 or
                "overrides require INTERNAL_SWEEP_DEV_MODE=1"
                not in production_override.stdout):
            raise AssertionError(
                "custom one-model catalog masqueraded as production:\n" +
                production_override.stdout)

        result = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise AssertionError(
                f"one-command synthetic sweep failed rc={result.returncode}\n{result.stdout}")
        if "[internal-full-sweep] DEVELOPMENT-COMPLETE" not in result.stdout:
            raise AssertionError("development runner returned zero without its scoped witness")

        frozen = json.loads((bundle / "inputs/resolved-models.json").read_text())
        spec = json.loads((bundle / "inputs/inventory/inventory.json").read_text())
        assert [item["model_id"] for item in frozen["models"]] == ["model-a"]
        assert spec["provenance"]["gguf_hashes"] == {
            "model-a": frozen["models"][0]["fileset_sha256"]}
        validator = ROOT / "tools/validate_internal_sweep_authorities.py"
        missing_ranked_row = copy.deepcopy(spec)
        missing_ranked_row["tensors"].pop()
        missing_ranked_row["tensor_count"] = len(missing_ranked_row["tensors"])
        missing_ranked_row["rank2_or_rank3_logical_tensor_count"] = len(
            missing_ranked_row["tensors"])
        missing_ranked_row["matrix_tensor_count"] = sum(
            bool(row.get("matmul_tensor")) for row in missing_ranked_row["tensors"])
        missing_ranked_row["unclassified_tensor_count"] = sum(
            row.get("role") == "UNCLASSIFIED" for row in missing_ranked_row["tensors"])
        extra_ranked_row = copy.deepcopy(spec)
        extra_ranked_row["tensors"].append(copy.deepcopy(extra_ranked_row["tensors"][0]))
        extra_ranked_row["tensor_count"] = len(extra_ranked_row["tensors"])
        extra_ranked_row["rank2_or_rank3_logical_tensor_count"] = len(
            extra_ranked_row["tensors"])
        extra_ranked_row["matrix_tensor_count"] = sum(
            bool(row.get("matmul_tensor")) for row in extra_ranked_row["tensors"])
        extra_ranked_row["unclassified_tensor_count"] = sum(
            row.get("role") == "UNCLASSIFIED" for row in extra_ranked_row["tensors"])
        missing_matmul_tp = copy.deepcopy(spec)
        next(row for row in missing_matmul_tp["tensors"]
             if row["matmul_tensor"])["tp"] = None
        for label, planted, needle in (
                ("empty-denominator",
                 {**spec, "cells": [], "sweep_shapes": [], "tensors": []},
                 "does not describe rows"),
                ("missing-rank2-row", missing_ranked_row,
                 "rank-2/3 tensor count differs from tensor rows"),
                ("extra-rank2-row", extra_ranked_row,
                 "rank-2/3 tensor count differs from tensor rows"),
                ("missing-matmul-tp", missing_matmul_tp,
                 "lacks required TP identity"),
                ("tp-drift",
                 {**spec, "models": [{**spec["models"][0], "tp_world_size": 999}]},
                 "tp_world_size differs")):
            planted_path = base / f"{label}.json"
            planted_path.write_text(json.dumps(planted, indent=2, sort_keys=True) + "\n")
            checked = subprocess.run(
                [sys.executable, "-B", str(validator),
                 "--catalog", str(bundle / "inputs/catalog.json"),
                 "--resolved", str(bundle / "inputs/resolved-models.json"),
                 "--inventory", str(planted_path)],
                cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False)
            if checked.returncode == 0 or needle not in checked.stdout:
                raise AssertionError(
                    f"authority validator accepted {label}:\n{checked.stdout}")
        model_root = bundle / "results/models/model-a"
        shape_dirs = sorted(path.name for path in model_root.iterdir() if path.is_dir())
        expected_dirs = sorted({row["shape_directory"]
                                for row in spec["sweep_shapes"]})
        assert shape_dirs == expected_dirs
        for folder in shape_dirs:
            shape_root = model_root / folder
            for name in ("cells.tsv", "winners.tsv", "scope.json"):
                assert (shape_root / name).is_file()
        dense_m1 = next(row for row in spec["sweep_shapes"]
                        if row["problem_route"] == "dense" and row["M"] == 1)
        winners = (model_root / dense_m1["shape_directory"] / "winners.tsv").read_text()
        for board in (
                "SCALEFIRST_FULL_OUTPUT", "SCALEFIRST_SPLITK_PRODUCER_ONLY",
                "FULLY_QUANTIZED_FULL_OUTPUT", "FULLY_QUANTIZED_SPLITK_PRODUCER_ONLY"):
            assert board in winners

        original_catalog = catalog_path.read_text()

        # A finished bundle is self-contained.  Removing the external catalog
        # must not invalidate an exact resume, and no component is rerun.
        catalog_path.unlink()
        env["RESUME"] = "1"
        resumed = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if (resumed.returncode != 0 or
                "DEVELOPMENT-COMPLETE (idempotent resume" not in resumed.stdout):
            raise AssertionError(
                "self-contained completed resume failed:\n" + resumed.stdout)
        catalog_path.write_text(original_catalog)

        shape_cells = model_root / shape_dirs[0] / "cells.tsv"
        saved_shape_cells = shape_cells.read_bytes()
        shape_cells.unlink()
        damaged = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if damaged.returncode == 0 or "results manifest mismatch" not in damaged.stdout:
            raise AssertionError("missing per-shape result escaped completion:\n" + damaged.stdout)
        shape_cells.write_bytes(saved_shape_cells)

        # A pre-existing results directory without completion.json is a
        # recoverable interrupted publication: merge into a new attempt, keep
        # the old tree, then atomically publish the replacement.
        (bundle / "completion.json").unlink()
        partial_env = dict(env, OUT=str(bundle), RESUME="1")
        recovered = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=partial_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if recovered.returncode != 0:
            raise AssertionError("partial results did not recover:\n" + recovered.stdout)
        preserved = list((bundle / "attempts").glob("*/preexisting-results"))
        if len(preserved) != 1 or not (bundle / "completion.json").is_file():
            raise AssertionError("partial result tree was not preserved and republished")

        # Importing a one-model authority under the production three-model
        # catalog must fail before a component attempt can start.
        mismatch_bundle = base / "catalog-mismatch"
        mismatch_env = dict(env, OUT=str(mismatch_bundle), RESUME="0",
                            INTERNAL_SWEEP_CATALOG=str(
                                ROOT / "benchmarks/internal_sweep_models.json"),
                            GGUF_SET=str(bundle / "inputs/resolved-models.json"),
                            INTERNAL_SWEEP_SPEC=str(
                                bundle / "inputs/inventory/inventory.json"))
        mismatch = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=mismatch_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if (mismatch.returncode == 0 or
                "resolved model set was not produced from the frozen catalog"
                not in mismatch.stdout or (mismatch_bundle / "attempts").exists()):
            raise AssertionError(
                "one-model import passed as the production model set:\n" + mismatch.stdout)

        foreign_bundle = base / "foreign-summary"
        foreign_env = dict(env, OUT=str(foreign_bundle), RESUME="0",
                           FAKE_FOREIGN_MODEL="1")
        foreign = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=foreign_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if foreign.returncode == 0 or "shape membership differs" not in foreign.stdout:
            raise AssertionError(
                "foreign component cells escaped frozen authority:\n" + foreign.stdout)

        grouped_bundle = base / "grouped-drift"
        grouped_env = dict(env, OUT=str(grouped_bundle), RESUME="0",
                           FAKE_GROUPED_DRIFT="1")
        grouped_drift = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=grouped_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if (grouped_drift.returncode == 0 or
                "differs from frozen grouped" not in grouped_drift.stdout):
            raise AssertionError(
                "grouped E/active/top-k/ragged drift escaped frozen authority:\n" +
                grouped_drift.stdout)

        # A zero-returning component cannot bless copied summaries from a
        # previous attempt.  The second invocation reaches the attempt-ID seam
        # and must reject both summaries as stale.
        noop = base / "noop-component.sh"
        noop.write_text("#!/usr/bin/env bash\nexit 0\n")
        noop.chmod(0o755)
        stale_bundle = base / "stale-summary"
        stale_env = dict(env, OUT=str(stale_bundle), RESUME="0",
                         SCALEFIRST_RUNNER=str(noop),
                         FULLY_QUANTIZED_RUNNER=str(noop))
        first_noop = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=stale_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if first_noop.returncode == 0:
            raise AssertionError("no-op components unexpectedly completed a fresh run")
        shutil.copytree(bundle / "scale-first", stale_bundle / "scale-first")
        shutil.copytree(bundle / "fully-quantized", stale_bundle / "fully-quantized")
        stale_env["RESUME"] = "1"
        stale = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=stale_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if stale.returncode == 0 or "summary is stale" not in stale.stdout:
            raise AssertionError("no-op runners reused stale summaries:\n" + stale.stdout)

        # A truncated immutable provenance record is not "nonempty enough".
        saved_completion = (bundle / "completion.json").read_text()
        saved_provenance = (bundle / "orchestration.provenance.txt").read_text()
        (bundle / "completion.json").unlink()
        (bundle / "orchestration.provenance.txt").write_text(
            "schema=quactlize.internal_full_sweep.run.v2\n")
        truncated_env = dict(env, OUT=str(bundle), RESUME="1")
        truncated = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=truncated_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if truncated.returncode == 0 or "provenance is truncated" not in truncated.stdout:
            raise AssertionError("truncated provenance was accepted:\n" + truncated.stdout)
        (bundle / "orchestration.provenance.txt").write_text(saved_provenance)
        (bundle / "completion.json").write_text(saved_completion)

        # Plant a write target that cannot be opened as a file.  The runner
        # must stop at the authority write rather than continue to measurement.
        write_bundle = base / "write-failure"
        (write_bundle / "inputs/input-state.sha256.partial").mkdir(parents=True)
        write_env = dict(env, OUT=str(write_bundle), RESUME="1")
        write_failure = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=write_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if write_failure.returncode == 0 or (write_bundle / "attempts").exists():
            raise AssertionError("authority write failure continued to measurement")

        # Change only the catalog authority and resume.  The frozen model/spec
        # still look complete, so this catches a fail-open input-state check.
        planted = json.loads(original_catalog)
        planted["description"] += " planted drift"
        catalog_path.write_text(json.dumps(planted, indent=2, sort_keys=True) + "\n")
        env["RESUME"] = "1"
        negative = subprocess.run(
            ["bash", str(ROOT / "tools/run_internal_full_sweep_box.sh")],
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if negative.returncode == 0 or "differs from bundle-frozen" not in negative.stdout:
            raise AssertionError(
                "catalog drift did not fail before resumed measurement:\n" + negative.stdout)

        print("[internal-full-sweep-runner:self-test] PASS: one-command catalog -> "
              "GGUF inventory -> four boards -> model/shape folder; self-contained "
              "idempotent/partial resume; production/dev isolation; model-set/"
              "all-rank-vs-rank2/3 inventory count +/- negatives+TP/component-authority/"
              "results-manifest/stale-summary/"
              "grouped-identity/catalog-drift/provenance/write negatives red")
    finally:
        # The target is one exact PID-named /workspace child created above.
        if base.parent != WORKSPACE or not base.name.startswith(
                "quactlize-internal-full-runner-selftest-"):
            raise AssertionError(f"unsafe self-test cleanup target: {base}")
        shutil.rmtree(base)


if __name__ == "__main__":
    try:
        run()
    except (AssertionError, OSError, ValueError) as error:
        print(f"[internal-full-sweep-runner:self-test] FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
