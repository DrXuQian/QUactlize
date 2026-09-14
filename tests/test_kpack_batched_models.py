import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.resolve_kpack_batched_models import resolve_plan
from tools import run_kpack_batched_bench as bench
from tools.run_kpack_batched_bench import command, validate_plan, sequence, parse_row, progress_line


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


def test_qwen3_q4_catalog_uses_exact_underscore_directory(tmp_path):
    source = json.loads((ROOT / "tools/kpack_batched_models.json").read_text())
    fixture(tmp_path, "Qwen3-32B-Q4_K_M-GGUF/Qwen3-32B-Q4_K_M.gguf")
    with pytest.raises(ValueError, match="binding path does not exist"):
        resolve_plan(source, ["qwen3-32b-q4km"], tmp_path)
    chosen = fixture(tmp_path, "Qwen3-32B-Q4_K_M_GGUF/Qwen3-32B-Q4_K_M.gguf")
    model, = resolve_plan(source, ["qwen3-32b-q4km"], tmp_path)["models"]
    assert model["path"] == str(chosen) and model["files"] == [str(chosen)]


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


def test_focused_int4_plan_has_only_2048_and_short_decode():
    focused = validate_plan(json.loads((ROOT / "tools/kpack_batched_int4_2048.json").read_text()))
    full = json.loads((ROOT / "tools/kpack_batched_models.json").read_text())
    assert focused["prompts"] == [2048] and focused["generations"] == [128]
    assert focused["batch"] == focused["ubatch"] == 2048 and focused["parallel"] == 1
    assert {m["name"] for m in focused["models"]} == {"qwen35-35b-q4km", "qwen3-32b-q4km"}
    assert focused["model_root"] == full["model_root"]
    by_name = {m["name"]: m for m in full["models"]}
    for model in focused["models"]:
        assert model["directory"] == by_name[model["name"]]["directory"]
        assert model["split"] == "none"
    assert sequence(focused, 1) == [(2048, 128, 0), (2048, 128, 1)]
    assert 3 * sum(tg for _, tg, _ in sequence(focused, 1)) == 768


def test_final_model_runner_uses_joint_ci_caller_and_pinned_runtime():
    text = (ROOT / 'tools/run_kpack_q4_model_box.sh').read_text()
    subprocess.run(['bash', '-n', str(ROOT/'tools/run_kpack_q4_model_box.sh')], check=True)
    assert '\n(\n' in text and 'trap finish EXIT' in text
    assert 'lfs pull origin' in text and 'kpack_q4_model_artifact.json' in text
    assert 'dev/quactlize-v0.3.0' in text and 'cmake --build' not in text
    assert 'build_kpack_model_ci.py' in text and 'NCP_LIB_DIR:-/sim/eec/shared/junfu.qx/ncp_flash_lib' in text
    assert 'BUILD_DIR="$LLAMA_DIR/build-ci"' in text and 'BUILD_DIR="$BUNDLE/llama"' not in text
    assert 'caller-ci-build.json' in text and 'JOBS=${JOBS:-192}' in text
    assert text.index('stage=ci-build') < text.index('stage=mixed-decode-gate')
    assert '--mixed' in text and 'cells=80' in text
    assert text.index('stage=mixed-decode-gate') < text.index('stage=model-numerical') < text.index('stage=model-benchmark') < text.index('stage=model-trace')
    assert '--order abba --require-selected' in text
    assert 'kpack-fusion-v3/dispatch' not in text
    assert '--exclude=\'*.asysrep\'' in text
    assert 'DrXuQian/llama.cpp.git' in text and 'ggml-org/llama.cpp.git' not in text
    receipt = json.loads((ROOT/'tools/kpack_q4_model_artifact.json').read_text())
    assert receipt['branch'] == 'artifacts/kpack-model-v1'
    assert receipt['llama_branch'] == 'dev/quactlize-v0.3.0'
    assert len(receipt['commit']) == len(receipt['llama_commit']) == 40
    assert len(receipt['llama_ci_commit']) == 40 and receipt['llama_ci_commit'] != receipt['llama_commit']
    assert len(receipt['manifest_sha256']) == 64 and receipt['lfs_payloads'] == 25


def test_joint_ci_checkout_preserves_dirty_source_and_nested_submodules(tmp_path):
    from tools.build_kpack_model_ci import clone_checkout, git
    def repository(name):
        root = tmp_path / name
        root.mkdir()
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        git(root, 'config', 'user.name', 'Test')
        git(root, 'config', 'user.email', 'test@example.invalid')
        (root / 'source').write_text(name)
        git(root, 'add', 'source')
        git(root, 'commit', '-qm', 'initial')
        return root
    leaf, child, source = [repository(name) for name in ('leaf', 'child', 'source')]
    git(child, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(leaf), 'nested')
    git(child, 'commit', '-qam', 'nested')
    git(source, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(child), 'third_party/library')
    git(source, 'commit', '-qam', 'library')
    git(source, '-c', 'protocol.file.allow=always', 'submodule', 'update', '--init', '--recursive')
    pin = git(source, 'rev-parse', 'HEAD')
    (source / 'source').write_text('local edit')
    (source / 'third_party/library/source').write_text('local submodule patch')
    before = git(source, 'status', '--porcelain')
    target = tmp_path / 'isolated'
    clone_checkout(source, target, pin)
    assert git(target, 'rev-parse', 'HEAD') == pin
    assert (target / 'source').read_text() == 'source'
    assert (target / 'third_party/library/source').read_text() == 'child'
    assert (target / 'third_party/library/.git').is_file()
    assert (target / 'third_party/library/nested/.git').is_file()
    assert git(target, 'status', '--porcelain') == ''
    assert git(source, 'status', '--porcelain') == before
    assert (source / 'third_party/library/source').read_text() == 'local submodule patch'
    with pytest.raises(subprocess.CalledProcessError):
        clone_checkout(source, target, pin)  # Never reset or clear an existing build checkout.


@pytest.mark.parametrize('modules', [None, '', '# No submodules\n', '[other]\n    enabled = true\n'])
def test_joint_ci_checkout_without_submodule_entries(tmp_path, modules):
    from tools.build_kpack_model_ci import clone_checkout, git
    source, target = tmp_path / 'source', tmp_path / 'isolated'
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    git(source, 'config', 'user.name', 'Test')
    git(source, 'config', 'user.email', 'test@example.invalid')
    (source / 'source.txt').write_text('committed source')
    if modules is not None:
        (source / '.gitmodules').write_text(modules)
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'fixture')
    pin = git(source, 'rev-parse', 'HEAD')
    clone_checkout(source, target, pin)
    assert git(target, 'rev-parse', 'HEAD') == pin
    assert git(target, 'status', '--porcelain') == ''
    assert (target / 'source.txt').read_text() == 'committed source'


def test_joint_ci_checkout_still_rejects_malformed_gitmodules(tmp_path):
    from tools.build_kpack_model_ci import clone_checkout, git
    source, target = tmp_path / 'source', tmp_path / 'isolated'
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    git(source, 'config', 'user.name', 'Test')
    git(source, 'config', 'user.email', 'test@example.invalid')
    (source / '.gitmodules').write_text('[submodule broken\n')
    git(source, 'add', '.gitmodules')
    git(source, 'commit', '-qm', 'malformed fixture')
    with pytest.raises(subprocess.CalledProcessError) as failure:
        clone_checkout(source, target, git(source, 'rev-parse', 'HEAD'))
    assert failure.value.returncode != 1


def test_joint_ci_receipt_requires_hooks_libraries_and_jit_headers(tmp_path, monkeypatch):
    from tools import build_kpack_model_ci as ci
    llama, ncp, sdk = [tmp_path / name for name in ('llama', 'ncp', 'sdk')]
    build = llama / 'build-ci'
    binary = build / 'bin'
    binary.mkdir(parents=True)
    flags = dict(GGML_CUDA='ON', GGML_USE_PPU='ON', GGML_NCP_QUACTLIZE='ON',
        GGML_NCP_FA='ON', GGML_NCP_MOE='ON', GGML_NCP_GDN='OFF',
        CMAKE_CUDA_COMPILER=str(sdk / 'CUDA_SDK/bin/nvcc'))
    cache = ''.join(f'{key}:STRING={value}\n' for key, value in flags.items())
    (build / 'CMakeCache.txt').write_text(cache)
    for name in ('llama-server', 'llama-batched-bench', 'llama-perplexity',
                 'libncp_fa.so', 'libncp_moe.so', 'libggml-cuda.so'):
        (binary / name).write_bytes(b'fixture')
    header = binary / 'deep_gemm/include/deep_gemm/gemm.cuh'
    header.parent.mkdir(parents=True)
    header.write_text('header')
    monkeypatch.setattr(ci, 'git', lambda path, *args: 'a'*40)
    receipt = ci.build_receipt(llama, ncp, sdk, 192)
    assert receipt['requested_jobs'] == 192 and receipt['device_admission'] == 'PENDING'
    assert 'bin/deep_gemm/include/deep_gemm/gemm.cuh' in receipt['files']
    for key in ('GGML_NCP_FA', 'GGML_NCP_MOE', 'GGML_NCP_QUACTLIZE'):
        (build / 'CMakeCache.txt').write_text(cache.replace(key+':STRING=ON', key+':STRING=OFF'))
        with pytest.raises(ValueError, match='CMake profile'):
            ci.build_receipt(llama, ncp, sdk, 192)
    (build / 'CMakeCache.txt').write_text(cache)
    header.unlink()
    with pytest.raises(ValueError, match='JIT include tree'):
        ci.build_receipt(llama, ncp, sdk, 192)


def test_runtime_publication_excludes_historical_llama_binaries(tmp_path, monkeypatch):
    from tools import publish_kpack_dispatch as publisher
    from tools.verify_kpack_dispatch import verify
    from quactlize.runtime.compiler import sha, source_contract
    src = tmp_path / 'historical'
    src.mkdir()
    identity = dict(kernel='kernel', generator='generator', flags=[], sdk='sdk', host='host')
    manifest = dict(schema='quactlize.kpack-native-dispatch.v1', modules=[], jit_required=True,
                    jit_source_identity=identity, jit_source_contract=source_contract(identity))
    for name, field in (('libquactlize_kpack_dispatch.so', 'dispatch_sha256'),
                        ('libquactlize_ppu_execution.so', 'execution_sha256')):
        (src / name).write_bytes(b'Quactlize fixture')
        manifest[field] = sha(src / name)
    old = dict(schema='quactlize.q4-model-deployment.v1', llama_source_commit='a'*40, files={}, links={})
    for name in ('llama-server', 'llama-batched-bench', 'llama-perplexity', 'libggml-cuda.so'):
        path = src / 'llama/bin' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'historical caller fixture')
        old['files'][str(path.relative_to(src))] = sha(path)
    pack = src / 'pack/libquactlize_ppu_pack.so'
    pack.parent.mkdir()
    pack.write_bytes(b'Quactlize producer fixture')
    (pack.parent / 'manifest.json').write_text(json.dumps(dict(
        schema='quactlize.kpack-device-pack-build.v1', library=pack.name, sha256=sha(pack))))
    for path in (pack, pack.parent / 'manifest.json'):
        old['files'][str(path.relative_to(src))] = sha(path)
    manifest['model'] = old
    (src / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(publisher, 'ROOT', tmp_path)
    dst = tmp_path / 'prebuilt/ppu0010/runtime-only'
    result = publisher.publish(src, dst)
    assert 'model' not in result and 'pack' in result
    assert not (dst / 'llama').exists()
    assert (src / 'llama/bin/llama-server').is_file()  # Historical payloads are not deleted.
    assert result['dispatch_sha256'] == manifest['dispatch_sha256']
    assert result['execution_sha256'] == manifest['execution_sha256']
    assert verify(dst)['pack']['files']['pack/libquactlize_ppu_pack.so'] == sha(pack)
    result['pack']['files']['llama/bin/llama-server'] = 'b'*64
    (dst / 'manifest.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='packer package payload set'):
        verify(dst)
    result['pack']['files'].pop('llama/bin/llama-server')
    (dst / 'manifest.json').write_text(json.dumps(result))
    (dst / 'pack/libquactlize_ppu_pack.so').write_bytes(b'wrong producer')
    with pytest.raises(ValueError, match='packer payload differs'):
        verify(dst)
    extra = src / 'llama-server'
    extra.write_bytes(b'caller cannot enter as a gate payload')
    manifest['decode_io_gate'] = dict(simt_binaries=[dict(path=extra.name, sha256=sha(extra))])
    (src / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='caller binary is outside'):
        publisher.publish(src, tmp_path / 'prebuilt/ppu0010/rejected')
    assert not (tmp_path / 'prebuilt/ppu0010/rejected').exists()


def test_publisher_has_no_llama_binary_cli_and_checkout_ignores_caller_payloads():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/publish_kpack_dispatch.py'),
        '--build', '/unused', '--output', '/unused', '--llama-build', '/unused'], capture_output=True, text=True)
    assert result.returncode == 2 and 'unrecognized arguments: --llama-build' in result.stderr
    for path in ('prebuilt/ppu0010/test/llama/bin/llama-server',
                 'prebuilt/ppu0010/test/libggml-cuda.so.0', 'prebuilt/ppu0010/test/llama-server'):
        subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', path], check=True)


def test_model_numerical_metrics_bind_actual_coverage_and_reject_nonfinite():
    from tools.run_kpack_model_validation import numerical_metrics
    metrics = dict(ppl=r'Final estimate: PPL =\s*(\S+)', mean_kld=r'Mean KLD:\s*(\S+)',
                   max_kld=r'Maximum KLD:\s*(\S+)', ppl_ratio=r'Ratio:\s*(\S+)', same_top_pct=r'Same top:\s*(\S+)')
    text = 'perplexity: calculating perplexity over 2 chunks, n_ctx=2048, batch_size=2048, n_seq=1\nFinal estimate: PPL = 1.25'
    assert numerical_metrics(text, metrics, 2048, 2048, 2, True) == dict(ppl=1.25)
    for bad in (text.replace('batch_size=2048', 'batch_size=128'), text.replace('1.25', 'nan'),
                text.replace('chunks', 'missing'), text+'\nCUDA error: invalid image'):
        with pytest.raises(ValueError):
            numerical_metrics(bad, metrics, 2048, 2048, 2, True)
    kld = 'kl_divergence: computing over 2 chunks, n_ctx=256, batch_size=1, n_seq=1\nMean KLD: 0.01\nMaximum KLD: 0.02\nRatio: 1.002\nSame top: 99.8'
    assert numerical_metrics(kld, metrics, 1, 256, 2, False)['mean_kld'] == .01
    with pytest.raises(ValueError):
        numerical_metrics(kld.replace('Maximum KLD', 'Missing'), metrics, 1, 256, 2, False)


def test_model_trace_reuses_reference_tokens_and_never_times_profiler(tmp_path, monkeypatch):
    from tools import run_kpack_model_validation as model
    commands = []
    def fake_run(argv, log):
        commands.append(argv)
        output = Path(argv[argv.index('--output')+1]); output.mkdir()
        (output/'proof.json').write_text(json.dumps(dict(input_tokens_sha256='a'*64, missing_ops=[])))
    monkeypatch.setattr(model, 'run', fake_run)
    args = SimpleNamespace(llama=Path('/source'), build=Path('/build'), cache=tmp_path/'cache',
        bundle=Path('/bundle'), asys=Path('/asys'), inspector=Path('/inspect'), jit_cache=tmp_path/'jit')
    model.traces(args, dict(name='model', path='/model.gguf'), tmp_path, tmp_path/'inventory.json')
    assert len(commands)==2
    ref, candidate = commands
    assert '--proof-tokens' not in ref
    assert candidate[candidate.index('--proof-tokens')+1] == tmp_path/'reference/proof-request/input-tokens.json'
    assert all('--proof-only' in c and c[c.index('--proof-prompt')+1]==2048 for c in commands)


def test_progress_distinguishes_per_token_from_total_time():
    source = json.loads((ROOT / "tools/kpack_batched_int4_2048.json").read_text())
    row = dict(n_kv_max=2176, pp=2048, tg=128, pl=1, n_batch=2048, n_ubatch=2048,
               flash_attn=1, is_pp_shared=0, n_kv=2176,
               t_pp=.2048, t_tg=1.28, speed_pp=10000, speed_tg=100)
    parsed = parse_row(json.dumps(row), (2048, 128, 1), source)
    line = progress_line(source["models"][0], "1-kpack", parsed, 2, 2)
    assert "phase=measured" in line and "completed=2/2" in line
    for field in ("prefill_us_per_token=100.000", "decode_us_per_token=10000.000",
                  "prefill_total_ms=204.800", "decode_total_ms=1280.000"):
        assert field in line
    assert "prefill_us=" not in line and "decode_us=" not in line


@pytest.mark.parametrize("arm", ["reference", "kpack"])
def test_benchmark_collects_compute_receipts_without_per_kernel_debug(arm, tmp_path):
    source = plan(tmp_path)
    model = source["models"][0] | {"path": "/weights.gguf"}
    argv = command("/bench", model, source, 1, ["blk.0.ffn_down.weight"], arm, tmp_path)
    # GGML_LOG_INFO enters common_log_default_callback at TRACE (4), not
    # common LOG_INF's level 3. DEBUG (5) would add per-kernel route traffic.
    assert argv.count("--verbosity") == 1
    assert argv[argv.index("--verbosity") + 1] == "4"


@pytest.mark.parametrize("kind", ["missing", "dense", "legacy", "wrong-q8"])
def test_compute_receipts_stay_required_and_are_preserved(kind, tmp_path, monkeypatch, capsys):
    source = plan(tmp_path)
    model = source["models"][0] | {"path": "/weights.gguf"}
    evidence = dict(plans=[], fallbacks=[])
    if kind in ("dense", "wrong-q8"):
        evidence["plans"] = [dict(op="dense", q="8" if kind == "wrong-q8" else "12")]
    elif kind == "legacy":
        evidence["fallbacks"] = ["blk.0.ffn_down.weight: native policy miss"]
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "quactlize_native", SimpleNamespace(
        model_selection=lambda args, text: evidence.copy()))
    row = dict(n_kv_max=528, pp=512, tg=16, pl=1, n_batch=512, n_ubatch=512,
               flash_attn=1, is_pp_shared=0, n_kv=528,
               t_pp=.512, t_tg=.016, speed_pp=1000, speed_tg=1000)
    # Run a real, short host subprocess: valid timing rows alone must not
    # satisfy native route admission. The kernel-plan verifier is a separate
    # boundary; this test exercises run_arm's admission and failure receipts.
    transcript = json.dumps(row) + "\n" + json.dumps(row) + "\n"
    if kind == "dense":
        transcript += "[kpack-pair] blk.0.ffn_gate_up_exps.weight <- {gate,up} N=1024 GPU_PACK\n"
        transcript += "[quactlize-moe] merged=1 rows=8 shared_prepare=1 reduce_scatter=1\n"
    elif kind == "legacy":
        transcript += "[quactlize-moe] merged=0 rows=8 shared_prepare=1 reduce_scatter=1\n"
    monkeypatch.setattr(bench, "command", lambda *args: [
        sys.executable, "-c", "print(" + repr(transcript) + ", end='')"])
    args = SimpleNamespace(cache_root=tmp_path / "cache", repeats=1, binary=Path(sys.executable),
                           llama_dir=tmp_path, bundle=tmp_path, manifest={}, jit_cache=tmp_path)
    inv = dict(eligible=["blk.0.ffn_down.weight"], q8=["blk.0.ffn_down.weight"]
               if kind == "wrong-q8" else [])
    if kind == "missing":
        with pytest.raises(ValueError, match="no selected or legacy K-pack compute plan") as failure:
            bench.run_arm(args, model, source, inv, "kpack", tmp_path, 1)
        assert str(tmp_path / "1-kpack.log") in str(failure.value)
    elif kind == "wrong-q8":
        with pytest.raises(ValueError, match="Q8_0 W8A16 compute evidence missing"):
            bench.run_arm(args, model, source, inv, "kpack", tmp_path, 1)
    else:
        result = bench.run_arm(args, model, source, inv, "kpack", tmp_path, 1)
        assert result["coverage"] == ("PARTIAL_NATIVE" if kind == "legacy" else "SELECTED_PLANS")
    receipt = json.loads((tmp_path / "1-kpack.selection.json").read_text())
    assert receipt["plans"] == evidence["plans"] and receipt["fallbacks"] == evidence["fallbacks"]
    assert receipt["plan_admission"] == ("FAIL" if kind in ("missing", "wrong-q8") else "PASS")
    assert receipt["fusion"] == dict(paired_weights=int(kind == "dense"),
        merged_chain_plans=int(kind == "dense"), separate_chain_plans=int(kind == "legacy"))
    if kind in ("dense", "legacy"):
        output = capsys.readouterr().out
        assert "BATCHED_MODEL_FUSION model=subject arm=1-kpack" in output
        assert f"paired_weights={int(kind == 'dense')}" in output
        assert f"merged_chain_plans={int(kind == 'dense')}" in output
        assert "scope=PLAN_RECEIPTS" in output
    assert json.loads((tmp_path / "1-kpack.process.json").read_text())["rc"] == 0
    records = json.loads((tmp_path / "1-kpack.timings.json").read_text())
    assert [r["phase"] for r in records] == ["warmup", "measured"]
