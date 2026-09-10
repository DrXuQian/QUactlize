import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.resolve_kpack_batched_models import resolve_plan
from tools.run_kpack_batched_bench import command, validate_plan


ROOT = Path(__file__).resolve().parents[1]


def plan(root):
    return dict(model_root=str(root), prompts=[512], generations=[16],
                parallel=1, batch=512, ubatch=512, models=[
                    dict(name="subject", directory="model", devices="0", split="none"),
                    dict(name="absent", directory="other-model", devices="0", split="none")])


def fixture(root, name="model/custom-name.gguf"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")  # Resolver only inspects paths/stat; inventory owns GGUF headers.
    return path


def test_selected_directory_and_benchmark_share_exact_path(tmp_path):
    path = fixture(tmp_path)
    fixture(tmp_path, "model/mmproj-F16.gguf")
    original = plan(tmp_path)
    resolved = resolve_plan(validate_plan(original), ["subject"])
    model, = resolved["models"]
    assert model["path"] == str(path) and model["files"] == [str(path)]
    assert "path" not in original["models"][0]
    assert len(original["models"]) == 2  # Unselected missing model is not searched.
    argv = command("/bench", model, resolved, 1, [], "reference", tmp_path / "cache")
    assert argv[argv.index("-m") + 1] == str(path)
    assert resolve_plan(resolved) == resolved


def test_nested_complete_split_uses_first_shard(tmp_path):
    files = [fixture(tmp_path, f"model/Q4_K_M/weights-{i:05d}-of-00003.gguf")
             for i in (1, 2, 3)]
    resolved = resolve_plan(plan(tmp_path), ["subject"])
    assert resolved["models"][0]["path"] == str(files[0])
    assert resolved["models"][0]["files"] == list(map(str, files))
    assert resolve_plan(resolved) == resolved


@pytest.mark.parametrize("extra", [".cache/stale.gguf", "nested/other.gguf", ".hidden.gguf"])
def test_visible_bf16_shards_are_not_mixed_with_other_directories(tmp_path, extra):
    files = [fixture(tmp_path, f"model/Qwen3.5-35B-A3B-BF16-{i:05d}-of-00002.gguf")
             for i in (1, 2)]
    fixture(tmp_path, "model/" + extra)
    model, = resolve_plan(plan(tmp_path), ["subject"])["models"]
    assert model["files"] == list(map(str, files))


@pytest.mark.parametrize("target_name", ["downloaded.gguf", "blob-without-extension"])
def test_split_symlinks_keep_their_public_filenames(tmp_path, target_name):
    first = fixture(tmp_path, "model/weights-00001-of-00002.gguf")
    target = fixture(tmp_path, "blobs/" + target_name)
    second = tmp_path / "model/weights-00002-of-00002.gguf"
    second.symlink_to(target)
    source = plan(tmp_path)
    resolved = resolve_plan(source, ["subject"])
    assert resolved["models"][0]["files"] == [str(first), str(second)]
    assert resolve_plan(resolved) == resolved
    source["models"][0]["filename"] = first.name
    assert resolve_plan(source, ["subject"])["models"][0]["files"] == [str(first), str(second)]


def test_explicit_bf16_filename_selects_only_its_complete_family(tmp_path):
    first = fixture(tmp_path, "model/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf")
    fixture(tmp_path, "model/other.gguf")
    source = plan(tmp_path)
    source["models"][0]["filename"] = first.name
    with pytest.raises(ValueError, match="incomplete"):
        resolve_plan(source, ["subject"])
    second = fixture(tmp_path, "model/Qwen3.5-35B-A3B-BF16-00002-of-00002.gguf")
    model, = resolve_plan(source, ["subject"])["models"]
    assert model["files"] == [str(first), str(second)]


def test_incomplete_direct_set_does_not_borrow_nested_shard(tmp_path):
    fixture(tmp_path, "model/weights-00001-of-00002.gguf")
    fixture(tmp_path, "model/old/weights-00002-of-00002.gguf")
    with pytest.raises(ValueError, match="incomplete"):
        resolve_plan(plan(tmp_path), ["subject"])


def test_nested_discovery_excludes_hidden_cache(tmp_path):
    fixture(tmp_path, "model/.cache/old.gguf")
    chosen = fixture(tmp_path, "model/Q4_K_M/weights.gguf")
    model, = resolve_plan(plan(tmp_path), ["subject"])["models"]
    assert model["files"] == [str(chosen)]


@pytest.mark.parametrize("filenames,reason", [
    (["Q4.gguf", "Q8.gguf"], "multiple unsplit"),
    (["a-00001-of-00003.gguf", "a-00003-of-00003.gguf"], "incomplete"),
    (["a-00001-of-00001.gguf", "b-00001-of-00001.gguf"], "multiple GGUF split groups"),
    (["a-00001-of-00001.gguf", "b.gguf"], "mixes split and unsplit"),
    (["weights.safetensors", "config.json"], "no .gguf files"),
])
def test_no_arbitrary_file_or_format_fallback(tmp_path, filenames, reason):
    for name in filenames:
        fixture(tmp_path, "model/" + name)
    with pytest.raises(ValueError, match=reason):
        resolve_plan(plan(tmp_path), ["subject"])


def test_does_not_search_bench_model_zoo_as_fallback(tmp_path):
    fixture(tmp_path, "bench_model_zoo/model/weights.gguf")
    with pytest.raises(ValueError, match="binding path does not exist"):
        resolve_plan(plan(tmp_path), ["subject"])


def test_explicit_file_resolves_ambiguity(tmp_path):
    chosen = fixture(tmp_path, "model/Q4.gguf")
    fixture(tmp_path, "model/Q8.gguf")
    source = plan(tmp_path)
    source["models"][0]["path"] = str(chosen)
    assert resolve_plan(source, ["subject"])["models"][0]["path"] == str(chosen)


def test_qwen3_catalog_selects_base_even_with_eagle3_present(tmp_path):
    source = json.loads((ROOT / "tools/kpack_batched_models.json").read_text())
    fixture(tmp_path, "Qwen3-32B-GGUF/Qwen3-32B-eagle3.gguf")
    # The explicit base file must exist; the other GGUF is never a fallback.
    with pytest.raises(ValueError, match="binding path does not exist"):
        resolve_plan(source, ["qwen3-32b-base"], tmp_path)
    chosen = fixture(tmp_path, "Qwen3-32B-GGUF/Qwen3-32b.gguf")
    resolved = resolve_plan(source, ["qwen3-32b-base"], tmp_path)
    model, = resolved["models"]
    assert model["path"] == str(chosen) and model["files"] == [str(chosen)]
    assert resolve_plan(resolved) == resolved


def test_empty_file_and_unknown_model_are_errors(tmp_path):
    fixture(tmp_path).write_bytes(b"")
    with pytest.raises(ValueError, match="empty GGUF"):
        resolve_plan(plan(tmp_path), ["subject"])
    with pytest.raises(ValueError, match="unknown model names"):
        resolve_plan(plan(tmp_path), ["typo"])


def test_root_override_is_a_replacement_not_an_extra_search_root(tmp_path):
    fixture(tmp_path)
    replacement = fixture(tmp_path, "new-root/model/new-name.gguf")
    resolved = resolve_plan(plan(tmp_path), ["subject"], tmp_path / "new-root")
    assert resolved["models"][0]["path"] == str(replacement)


def test_cli_resolves_without_sdk_and_never_overwrites_receipt(tmp_path):
    path = fixture(tmp_path)
    source, output = tmp_path / "input.json", tmp_path / "resolved.json"
    source.write_text(json.dumps(plan(tmp_path)))
    argv = [sys.executable, str(ROOT / "tools/resolve_kpack_batched_models.py"),
            "--plan", str(source), "--model", "subject", "--output", str(output)]
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert f"shards=1 path={path}" in proc.stdout
    assert json.loads(output.read_text())["models"][0]["path"] == str(path)
    before = output.read_bytes()
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode == 1 and "File exists" in proc.stderr
    assert output.read_bytes() == before


def test_box_catalog_uses_only_requested_parent_root():
    source = json.loads((ROOT / "tools/kpack_batched_models.json").read_text())
    assert source["model_root"] == "/sim/eec/shared/AI_workspace/llm-models"
    assert len(source["models"]) == 5
    assert all(not Path(m["directory"]).is_absolute() and ".." not in Path(m["directory"]).parts
               for m in source["models"])
    assert source["models"][1]["directory"] == "Qwen3.5-35B-A3B-Q4_K_M-GGUF"
    assert source["models"][0]["filename"] == "Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf"
    runner = (ROOT / "tools/run_kpack_fusion_box.sh").read_text()
    assert runner.index("stage=model-paths") < runner.index("stage=device-gates")
    assert '--plan "$RUN/results/model-plan.json" "${MODEL_ARGS[@]}"' in runner
    assert '"$RUN/results/model-plan.json" "$MODEL_NAME"' in runner
