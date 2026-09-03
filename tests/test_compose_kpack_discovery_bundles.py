from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compose_kpack_discovery_bundles as compose  # noqa: E402
import fq_dense_structural_proof as structural  # noqa: E402
import run_kpack_discovery_worker as worker  # noqa: E402


def test_compose_kpack_discovery_bundles_self_test() -> None:
    compose.self_test()


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_structural_fixture(root: pathlib.Path,
                             inspector_sha: str) -> tuple[pathlib.Path,
                                                          pathlib.Path,
                                                          pathlib.Path,
                                                          str]:
    sf, fq, output = compose._fixture(root)
    bundle = json.loads(fq.read_text(encoding="utf-8"))
    key, native = next(
        (key, row) for key, row in bundle["shards"].items()
        if row["operator"] == "dense")
    manifest_path = fq.parent / native["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary_path = fq.parent / native["binary"]
    binary_path.chmod(0o755)
    old_receipt = json.loads(
        (fq.parent / native["binary_receipt"]).read_text(encoding="utf-8"))
    compile_hashes = {
        "build_make_sha256": "1" * 64,
        "payload_inspector_output_sha256": inspector_sha,
        "registry_sha256": "3" * 64,
        "unit_sources": [{"path": "units/u.cu", "sha256": "4" * 64}],
        "unit_objects": [{"path": "ppu_targets/ppu_obj/u.o",
                          "sha256": "5" * 64}],
        "link_file_sha256": "6" * 64,
        "link_argv_sha256": "7" * 64,
        "census_source_sha256": "8" * 64,
        "census_compile_argv_sha256": "9" * 64,
        "census_object_sha256": "a" * 64,
        "stub_source_sha256": "b" * 64,
        "stub_compile_argv_sha256": "c" * 64,
        "stub_object_sha256": "d" * 64,
        "census_link_argv_sha256": "e" * 64,
        "census_binary_sha256": "f" * 64,
        "census_stdout_sha256": "0" * 64,
        "nm_path": "/usr/bin/nm", "nm_sha256": "1" * 64,
        "nm_output_sha256": "2" * 64,
    }
    parents = manifest["dense_tc_parents"]
    proof = {
        "schema": structural.PROOF_SCHEMA,
        "payload_kind": structural.PAYLOAD_KIND,
        "shard": {field: copy.deepcopy(native[field])
                  for field in structural.SHARD_FIELDS},
        "manifest_sha256": _sha(manifest_path),
        "binary_sha256": _sha(binary_path),
        "source_authority": {
            "build_input_authority_sha256":
                old_receipt["build_input_authority_sha256"],
            "source_sha": old_receipt["source_sha"],
            "source_tree": old_receipt["source_tree"],
            "submodules": old_receipt["submodules"],
            "sdk_compiler_sha256": old_receipt["sdk_compiler_sha256"],
            "sdk_inspector_sha256": old_receipt["sdk_inspector_sha256"],
            "host_cxx_sha256": "3" * 64,
        },
        "repair_authority": {
            "source_sha": "4" * 40, "source_tree": "5" * 40,
            "tool_path": "tools/fq_dense_structural_proof.py",
            "tool_sha256": "6" * 64,
        },
        "compile_authority": compile_hashes,
        "shared_memory_limit_bytes": structural.SMEM_LIMIT,
        "rows": [{
            "parent_id": row["static_candidate_id"],
            "symbol": row["symbol"],
            "runtime_variants": ["TC_S1", "TC_S2", "TC_S4", "TC_S8"],
            "shipping_smem": structural.SMEM_LIMIT + 1,
            "split_smem": structural.SMEM_LIMIT + 1,
        } for row in parents],
        "all_rows_shipping_shared_storage": True,
    }
    proof_rel = f"payloads/{key}/structural-proof.json"
    proof_path = fq.parent / proof_rel
    proof_path.write_text(compose._encoded(proof), encoding="utf-8")
    receipt = {
        **old_receipt,
        "schema": compose.fq_index.STRUCTURAL_RECEIPT_SCHEMA,
        "payload_kind": structural.PAYLOAD_KIND,
        "device_arch": "NO_DEVICE_KERNEL",
        "inspector_output_sha256": inspector_sha,
        "structural_proof": proof_rel,
        "structural_proof_sha256": _sha(proof_path),
    }
    receipt_path = fq.parent / native["binary_receipt"]
    receipt_path.write_text(compose._encoded(receipt), encoding="utf-8")
    native.update({
        "payload_kind": structural.PAYLOAD_KIND,
        "device_arch": "NO_DEVICE_KERNEL",
        "inspector_output_sha256": inspector_sha,
        "structural_proof": proof_rel,
        "structural_proof_sha256": _sha(proof_path),
        "binary_receipt_sha256": _sha(receipt_path),
    })
    fq.write_text(compose._encoded(bundle), encoding="utf-8")
    return sf, fq, output, key


def test_compose_propagates_only_proved_fq_dense_structural_payload(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    inspector_output = "host-only image\n"
    inspector_sha = hashlib.sha256(inspector_output.encode()).hexdigest()
    sf, fq, output, key = _make_structural_fixture(tmp_path, inspector_sha)
    monkeypatch.setattr(
        compose, "_inspect_structural_binary",
        lambda _binary, _source: (inspector_output, inspector_sha))

    document = compose.compose_document(
        output=output, scalefirst_bundle=sf, fully_quantized_bundle=fq)
    row = next(row for row in document["shards"]
               if row["native_shard_key"] == key)
    assert row["payload_kind"] == structural.PAYLOAD_KIND
    assert row["parent_count"] == len(row["parent_ids"])
    assert set(row["files"]) == {
        "manifest", "binary", "binary_receipt", "structural_proof"}
    compose.write_composite(output, document)
    resolved = worker.resolve_native_shard(output, document, row["shard_key"])
    assert resolved.payload_kind == structural.PAYLOAD_KIND
    assert resolved.parent_count == row["parent_count"]

    # The exception remains narrow: changing the native row to grouped must
    # fail before the no-image marker can be treated as an executable shard.
    bundle = json.loads(fq.read_text(encoding="utf-8"))
    bundle["shards"][key]["operator"] = "grouped"
    fq.write_text(compose._encoded(bundle), encoding="utf-8")
    try:
        compose.compose_document(
            output=output, scalefirst_bundle=sf, fully_quantized_bundle=fq)
    except compose.BundleError:
        pass
    else:
        raise AssertionError("grouped structural payload stayed green")


def test_structural_inspector_hashes_full_output_and_rejects_ppu_image(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    sdk = tmp_path / "sdk"
    inspector = sdk / "bin/hgobjdump"
    inspector.parent.mkdir(parents=True)
    binary = tmp_path / "payload"
    binary.write_bytes(b"payload\n")
    output = "payload:\tfile format elf64-x86-64\n"
    inspector.write_text(
        "#!/bin/sh\nprintf '%s' '" + output + "'\n", encoding="utf-8")
    inspector.chmod(0o755)
    source = {"sdk": {"inspector": {
        "path": "bin/hgobjdump", "sha256": _sha(inspector)}}}
    monkeypatch.setenv("PPU_SDK", str(sdk))
    observed, digest = compose._inspect_structural_binary(binary, source)
    assert observed == output
    assert digest == hashlib.sha256(output.encode()).hexdigest()

    ppu_output = (output + "\nELF FILE 1 (PPU ppu0010)\n")
    inspector.write_text(
        "#!/bin/sh\nprintf '%s' '" + ppu_output + "'\n", encoding="utf-8")
    inspector.chmod(0o755)
    source["sdk"]["inspector"]["sha256"] = _sha(inspector)
    try:
        compose._inspect_structural_binary(binary, source)
    except compose.BundleError as exc:
        assert "unexpectedly has a PPU image" in str(exc)
    else:
        raise AssertionError("structural payload with PPU image stayed green")

    for invalid_output in ("", "garbage\n"):
        inspector.write_text(
            "#!/bin/sh\nprintf '%s' '" + invalid_output + "'\n",
            encoding="utf-8")
        inspector.chmod(0o755)
        source["sdk"]["inspector"]["sha256"] = _sha(inspector)
        try:
            compose._inspect_structural_binary(binary, source)
        except compose.BundleError as exc:
            assert "host ELF inspector identity" in str(exc)
        else:
            raise AssertionError("empty/garbage inspector output stayed green")


def test_compose_rejects_structural_receipt_and_symbol_drift(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    output_text = "kernel:\tfile format elf64-x86-64\n"
    inspector_sha = hashlib.sha256(output_text.encode()).hexdigest()
    monkeypatch.setattr(
        compose, "_inspect_structural_binary",
        lambda binary, _source: (
            f"{binary.name}:\tfile format elf64-x86-64\n", inspector_sha))
    for mode in ("receipt", "symbol"):
        sf, fq, output, key = _make_structural_fixture(
            tmp_path / mode, inspector_sha)
        bundle = json.loads(fq.read_text(encoding="utf-8"))
        native = bundle["shards"][key]
        receipt_path = fq.parent / native["binary_receipt"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if mode == "receipt":
            receipt["structural_proof_sha256"] = "9" * 64
        else:
            proof_path = fq.parent / native["structural_proof"]
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            proof["rows"][0]["symbol"] = "fqk_tc_q10_planted_symbol"
            proof_path.write_text(compose._encoded(proof), encoding="utf-8")
            proof_sha = _sha(proof_path)
            native["structural_proof_sha256"] = proof_sha
            receipt["structural_proof_sha256"] = proof_sha
        receipt_path.write_text(compose._encoded(receipt), encoding="utf-8")
        native["binary_receipt_sha256"] = _sha(receipt_path)
        fq.write_text(compose._encoded(bundle), encoding="utf-8")
        try:
            compose.compose_document(
                output=output, scalefirst_bundle=sf,
                fully_quantized_bundle=fq)
        except compose.BundleError:
            pass
        else:
            raise AssertionError(f"structural {mode} drift stayed green")


def _run_dense_validator(*args: pathlib.Path | str) -> subprocess.CompletedProcess:
    shell = (ROOT / "tools/run_fully_quantized_kpack_discovery_box.sh") \
        .read_text(encoding="utf-8")
    function = shell.split("validate_dense_evidence() {", 1)[1].split(
        "\n}\n\nmain()", 1)[0]
    program = "validate_dense_evidence() {" + function + "\n}\n" + \
        "validate_dense_evidence \"$@\"\n"
    return subprocess.run(
        ["bash", "-c", program, "validator", *map(str, args)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_structural_runner_rejects_unknown_shape_duplicate_done_and_proof_toctou(
        tmp_path: pathlib.Path) -> None:
    inspector_sha = "7" * 64
    _sf, fq, _output, key = _make_structural_fixture(tmp_path, inspector_sha)
    bundle = json.loads(fq.read_text(encoding="utf-8"))
    native = bundle["shards"][key]
    manifest = fq.parent / native["manifest"]
    binary = fq.parent / native["binary"]
    proof = fq.parent / native["structural_proof"]
    manifest_doc = json.loads(manifest.read_text(encoding="utf-8"))
    symbols = [row["symbol"] for row in manifest_doc["dense_tc_parents"]]
    shape = "1x16x128"
    shapes = tmp_path / "shapes.txt"
    shapes.write_text(shape + "\n", encoding="utf-8")
    cells = "".join(
        f"FQ_TC_CELL shape={shape} symbol={symbol} S={split} "
        "state=SHIPPING_SHARED_STORAGE raw_bad=0\n"
        for symbol in symbols for split in (1, 2, 4, 8))
    done = (f"FQ_SHAPE_DONE q={manifest_doc['identity']['qtype']} "
            f"shape={shape} typed_rows={len(symbols)} "
            f"selected_rows={len(symbols)} status=PASS\n")
    log = tmp_path / "valid.log"
    log.write_text(cells + done, encoding="utf-8")
    common = (manifest, shapes, structural.PAYLOAD_KIND, proof, key, "screen",
              ROOT, fq, fq.parent, binary)
    result = _run_dense_validator(log, *common)
    assert result.returncode == 0, result.stdout
    assert "STRUCTURAL_CENSUS_NO_DEVICE_KERNEL" in result.stdout

    unknown = tmp_path / "unknown.log"
    unknown.write_text(
        log.read_text(encoding="utf-8") +
        f"FQ_TC_CELL shape=2x16x128 symbol={symbols[0]} S=1 "
        "state=SHIPPING_SHARED_STORAGE raw_bad=0\n", encoding="utf-8")
    assert _run_dense_validator(unknown, *common).returncode != 0

    duplicate = tmp_path / "duplicate.log"
    duplicate.write_text(
        log.read_text(encoding="utf-8") + done, encoding="utf-8")
    assert _run_dense_validator(duplicate, *common).returncode != 0

    proof.write_text("{}\n", encoding="utf-8")
    stale = _run_dense_validator(log, *common)
    assert stale.returncode != 0
    assert "changed after preflight" in stale.stdout
