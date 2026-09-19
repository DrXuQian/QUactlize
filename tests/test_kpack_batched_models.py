import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.resolve_kpack_batched_models import resolve_plan
from tools import run_kpack_batched_bench as bench
from tools.run_kpack_batched_bench import command, validate_plan, sequence, parse_row, progress_line


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('compute', [None, 'fp16', 'bf16'])
@pytest.mark.parametrize('op', ['dense', 'grouped'])
@pytest.mark.parametrize('rows', [1, 8, 9, 16352])
def test_benchmark_bf16_scope_is_grouped_only(compute, op, rows):
    actual = bench.expected_activation(dict(op=op, rows=str(rows)), compute)
    assert actual == ('BF16' if compute == 'bf16' and op == 'grouped' else 'FP16')


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
    assert 'if [[ -n ${LLAMA_CI_DIR:-} ]]' in text and 'CI_SOURCE_ARGS+=(--local-llama)' in text
    assert 'CI_SOURCE_ARGS+=(--llama-revision "${INFO[4]}")' in text
    assert 'git -C "$LLAMA_DIR" diff --cached --quiet' not in text
    assert 'BUILD_DIR=$(realpath -e -- "${LLAMA_CI_BUILD_DIR:-$RUN/ci/llama-build}")' in text
    assert 'CI_SOURCE_ARGS+=(--reuse-llama-build "$LLAMA_CI_BUILD_DIR")' in text
    assert 'LLAMA_CI_BUILD_DIR requires LLAMA_CI_DIR' in text
    assert 'NCP_BUILD_ARGS+=(--reuse-ncp-build "$NCP_CI_DIR")' in text
    assert '$SDK/targets/x86_64-linux/lib' in text
    assert 'git -C "$LLAMA_DIR" rev-parse HEAD > "$RUN/results/llama-source.txt"' in text
    assert text.index('stage=ci-build') < text.index('stage=mixed-decode-gate')
    assert '--mixed' in text and 'cells=80' in text
    assert text.index('stage=mixed-decode-gate') < text.index('stage=model-numerical') < text.index('stage=model-benchmark') < text.index('stage=model-trace')
    assert '--order abba --require-selected' in text
    assert 'kpack-fusion-v3/dispatch' not in text
    assert '--exclude=\'*.asysrep\'' in text
    assert 'DrXuQian/llama.cpp.git' in text and 'ggml-org/llama.cpp.git' not in text
    receipt = json.loads((ROOT/'tools/kpack_q4_model_artifact.json').read_text())
    caller_branches = {'artifacts/kpack-model-runtime-v1': 'dev/quactlize-v0.3.0',
                      'artifacts/kpack-model-paired-n4-v1': 'dev/quactlize-gate-up-v0.3.0',
                      'artifacts/kpack-model-readers-v1': 'dev/quactlize-gate-up-v0.3.0',
                      'artifacts/kpack-model-prepare-v1': 'dev/quactlize-tp2-v0.3.0',
                      'artifacts/kpack-model-tp2-scale-v1': 'dev/quactlize-tp2-v0.3.0',
                      'artifacts/kpack-model-selected-v1': 'dev/quactlize-selected-decode'}
    assert receipt['branch'] in caller_branches
    assert receipt['path'] == 'prebuilt/ppu0010/kpack-model-runtime-v1'
    assert receipt['llama_branch'] == caller_branches[receipt['branch']]
    assert 'llama_commit' not in receipt and 'test ! -e "$BUNDLE/llama"' in text
    assert len(receipt['commit']) == len(receipt['llama_ci_commit']) == 40
    assert len(receipt['manifest_sha256']) == 64
    # BF16 capability modules are a separate, bounded device gate. The
    # publication validator checks the actual payload closure; it is no
    # longer the historical twelve-file F16-only package.
    assert isinstance(receipt['lfs_payloads'],int) and receipt['lfs_payloads']>0
    if receipt['branch']=='artifacts/kpack-model-selected-v1':
        assert receipt['selected_entry_gate'] is True
        assert 'tools/run_selected_decode_numerics.py' in text
    if receipt['branch'] in ('artifacts/kpack-model-readers-v1','artifacts/kpack-model-prepare-v1',
                            'artifacts/kpack-model-tp2-scale-v1'):
        assert receipt['model_reader_gate'] is True
        assert text.index('stage=model-reader-integration') < text.index('stage=model-benchmark')
        assert 'tools/run_model_gemv_integration.py' in text
        assert 'update the llama reader trace parser' in text
    if receipt['branch'] in ('artifacts/kpack-model-prepare-v1','artifacts/kpack-model-tp2-scale-v1'):
        assert receipt['prepare_integration_gate'] is True
        assert 'dev/moe_prepare/run_integration.py' in text
        assert text.index('stage=prepare-integration') < text.index('stage=model-benchmark')


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


@pytest.mark.parametrize('cache_state', ['no-checkout', 'staged', 'head-moved'])
def test_pinned_caller_ignores_source_cache_index_and_head(tmp_path, monkeypatch, cache_state):
    from tools import build_kpack_model_ci as ci
    def repository(name):
        path = tmp_path / name
        subprocess.run(['git', 'init', '-q', str(path)], check=True)
        ci.git(path, 'config', 'user.name', 'Test')
        ci.git(path, 'config', 'user.email', 'test@example.invalid')
        (path / 'source.cu').write_text('committed')
        ci.git(path, 'add', '.')
        ci.git(path, 'commit', '-qm', 'fixture')
        return path
    ncp, original = repository('ncp'), repository('llama')
    ncp_rev = ci.git(ncp, 'rev-parse', 'HEAD')
    entry = '.aoneci/scripts/build.sh'
    script = original / entry
    script.parent.mkdir(parents=True)
    script.write_text('# fixture uses ${LLAMA_BUILD_DIR}\n')
    (original / '.aoneci/NCP_LIB_VERSION').write_text(ncp_rev + '\n')
    ci.git(original, 'add', '.')
    ci.git(original, 'commit', '-qm', 'CI entry')
    pin = ci.git(original, 'rev-parse', 'HEAD')
    if cache_state == 'head-moved':
        script.write_text('# newer source is not the requested commit\n')
        ci.git(original, 'commit', '-qam', 'newer source')
    source = tmp_path / ('llama-model-source-' + pin[:10])
    subprocess.run(['git', 'clone', '--no-checkout', str(original), str(source)], check=True)
    if cache_state == 'staged':
        ci.git(source, 'checkout', '--detach', pin)
        (source / entry).write_text('# staged changes must not enter the pinned build\n')
        ci.git(source, 'add', entry)
    else:
        assert not (source / entry).exists()
    # The original runner passed the worktree check and then stopped at the index check.
    assert subprocess.run(['git', '-C', str(source), 'diff', '--quiet']).returncode == 0
    assert subprocess.run(['git', '-C', str(source), 'diff', '--cached', '--quiet']).returncode == 1
    before = (ci.git(source, 'status', '--porcelain'), ci.git(source, 'rev-parse', 'HEAD'),
              ci.git(source, 'diff', '--cached', '--binary'))
    runner = (ROOT / 'tools/run_kpack_q4_model_box.sh').read_text()
    caller_block = 'CI_SOURCE_ARGS=()' + runner.split('CI_SOURCE_ARGS=()', 1)[1].split('    stage=ci-build', 1)[0]
    env = dict(os.environ, RESULT_DIR=str(tmp_path), LLAMA_CI_DIR='', PIN=pin)
    subprocess.run(['bash', '-ec', 'INFO=(unused unused unused unused "$PIN")\n' + caller_block],
                   env=env, check=True)
    sdk = tmp_path / 'sdk'
    compiler = sdk / 'CUDA_SDK/bin/nvcc'
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    output, receipt = tmp_path / 'ci', tmp_path / 'receipt.json'
    run = subprocess.run
    builds = []
    def checked_run(command, *args, **kwargs):
        if command[0] != 'bash':
            return run(command, *args, **kwargs)
        builds.append(command)
        llama = output / 'llama'
        assert command == ['bash', str(llama / entry)]
        assert '${LLAMA_BUILD_DIR' in (llama / entry).read_text()
        assert ci.git(llama, 'rev-parse', 'HEAD') == pin
        assert not ci.git(llama, 'status', '--porcelain')
        assert kwargs['env']['LLAMA_BUILD_DIR'] == str(llama / 'build-ci')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(ci.subprocess, 'run', checked_run)
    monkeypatch.setattr(ci, 'build_receipt', lambda l, n, s, jobs, build: dict(
        llama_source_commit=ci.git(l, 'rev-parse', 'HEAD'), ncp_source_commit=ncp_rev, build=str(build)))
    monkeypatch.setattr(sys, 'argv', ['build_kpack_model_ci.py', '--llama', str(source), '--llama-revision', pin,
        '--ncp', str(ncp), '--sdk', str(sdk), '--output', str(output), '--receipt', str(receipt)])
    ci.main()
    assert len(builds) == 1
    result = json.loads(receipt.read_text())
    assert result['llama_source_mode'] == 'PINNED_CHECKOUT' and result['llama_source_commit'] == pin
    assert before == (ci.git(source, 'status', '--porcelain'), ci.git(source, 'rev-parse', 'HEAD'),
                      ci.git(source, 'diff', '--cached', '--binary'))


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


@pytest.mark.parametrize('reuse_ncp', [False, True])
@pytest.mark.parametrize('reuse_llama', [False, True])
def test_local_llama_override_uses_dirty_worktree_and_external_build(tmp_path, monkeypatch, reuse_ncp, reuse_llama):
    from tools import build_kpack_model_ci as ci
    def repository(name):
        root = tmp_path / name
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        ci.git(root, 'config', 'user.name', 'Test')
        ci.git(root, 'config', 'user.email', 'test@example.invalid')
        (root / 'source.cu').write_text('committed')
        ci.git(root, 'add', '.')
        ci.git(root, 'commit', '-qm', 'fixture')
        return root
    llama, ncp = repository('llama'), repository('ncp')
    ncp_rev = ci.git(ncp, 'rev-parse', 'HEAD')
    script = llama / '.aoneci/scripts/build.sh'
    script.parent.mkdir(parents=True)
    script.write_text('# fixture consumes ${LLAMA_BUILD_DIR} and ${LLAMA_BUILD_REUSE}\n')
    (llama / '.aoneci/NCP_LIB_VERSION').write_text(ncp_rev + '\n')
    ci.git(llama, 'add', '.')
    ci.git(llama, 'commit', '-qm', 'CI entry')
    llama_rev = ci.git(llama, 'rev-parse', 'HEAD')
    (llama / 'source.cu').write_text('local edit')
    (llama / 'local.cu').write_text('untracked input')
    old = llama / 'build-ci/keep'
    old.parent.mkdir()
    old.write_text('previous build')
    sdk = tmp_path / 'sdk'
    compiler = sdk / 'CUDA_SDK/bin/nvcc'
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    output, receipt = tmp_path / 'ci', tmp_path / 'receipt.json'
    argv = ['build_kpack_model_ci.py', '--llama', str(llama), '--ncp', str(ncp), '--sdk', str(sdk),
            '--output', str(output), '--receipt', str(receipt), '--local-llama', '--jobs', '192']
    caller_build = old.parent if reuse_llama else output / 'llama-build'
    if reuse_llama:
        argv += ['--reuse-llama-build', str(caller_build)]
        flags = ci.caller_profile(sdk) | dict(CMAKE_HOME_DIRECTORY=str(llama), CMAKE_BUILD_TYPE='Release',
            CMAKE_CUDA_ARCHITECTURES='OFF', LLAMA_BUILD_TESTS='ON', LLAMA_BUILD_EXAMPLES='ON', LLAMA_BUILD_SERVER='ON')
        cache = caller_build / 'CMakeCache.txt'
        original = ''.join(f'{k}:STRING={v}\n' for k, v in flags.items())
        for key in ('CMAKE_HOME_DIRECTORY', 'CMAKE_CUDA_COMPILER', 'GGML_NCP_MOE', 'LLAMA_BUILD_TESTS'):
            cache.write_text(original.replace(f'{key}:STRING={flags[key]}', f'{key}:STRING=wrong'))
            with pytest.raises(ValueError, match='llama reuse'):
                ci.reusable_llama_build(caller_build, llama, sdk)
        cache.write_text(original)
    if reuse_ncp:
        argv += ['--reuse-ncp-build', str(ncp)]
        (ncp / 'build').mkdir()
        flags = dict(CMAKE_BUILD_TYPE='Release', NCP_BUILD_FA='ON', NCP_BUILD_MOE='ON',
                     NCP_BUILD_GDN='OFF', CMAKE_CUDA_ARCHITECTURES='OFF',
                     CMAKE_HOME_DIRECTORY=str(ncp), CMAKE_CUDA_COMPILER=str(compiler))
        (ncp / 'build/CMakeCache.txt').write_text(''.join(f'{k}:STRING={v}\n' for k, v in flags.items()))
        (ncp / 'build/completed.o').write_bytes(b'keep completed object')
    clone, run = ci.clone_checkout, subprocess.run
    clones, builds = [], []
    def checked_clone(source, target, revision):
        clones.append(source)
        return clone(source, target, revision)
    def checked_run(command, *args, **kwargs):
        if command[0] != 'bash':
            return run(command, *args, **kwargs)
        builds.append(command)
        assert command == ['bash', str(script)]
        assert kwargs['cwd'] == llama
        assert kwargs['env']['LLAMA_CI_DIR'] == str(llama)
        assert kwargs['env']['LLAMA_BUILD_DIR'] == str(caller_build)
        assert kwargs['env']['LLAMA_BUILD_REUSE'] == str(int(reuse_llama))
        assert kwargs['env']['NCP_LIB_DIR'] == str(ncp if reuse_ncp else output / 'ncp_flash_lib')
        assert kwargs['env']['JOBS'] == '192'
        assert (llama / 'source.cu').read_text() == 'local edit'
        assert (llama / 'local.cu').is_file()
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(ci, 'clone_checkout', checked_clone)
    monkeypatch.setattr(ci.subprocess, 'run', checked_run)
    monkeypatch.setattr(ci, 'build_receipt', lambda l, n, s, jobs, build: dict(
        llama_source_commit=llama_rev, ncp_source_commit=ncp_rev, build=str(build)))
    monkeypatch.setattr(sys, 'argv', argv)
    ci.main()
    assert clones == ([] if reuse_ncp else [ncp]) and len(builds) == 1
    result = json.loads(receipt.read_text())
    assert result['llama_source_mode'] == 'LOCAL_WORKTREE'
    assert result['llama_worktree']['directory'] == str(llama)
    assert 'source.cu' in result['llama_worktree']['status']
    assert old.read_text() == 'previous build'
    assert result['build'] == str(caller_build)
    assert result['llama_build_mode'] == ('REUSE_BUILD' if reuse_llama else 'FRESH_BUILD')
    assert result['ncp_build_mode'] == ('REUSE_BUILD' if reuse_ncp else 'FRESH_CHECKOUT')
    if reuse_ncp:
        assert (ncp / 'build/completed.o').read_bytes() == b'keep completed object'
        cache = ncp / 'build/CMakeCache.txt'
        original = cache.read_text()
        for wrong in ('revision', 'source', 'compiler', 'profile'):
            cache.write_text(original)
            pin = ncp_rev
            if wrong == 'revision': pin = 'a' * 40
            if wrong == 'source': cache.write_text(original.replace(str(ncp), str(tmp_path / 'other')))
            if wrong == 'compiler': cache.write_text(original.replace(str(compiler), str(script)))
            if wrong == 'profile': cache.write_text(original.replace('NCP_BUILD_MOE:STRING=ON', 'NCP_BUILD_MOE:STRING=OFF'))
            with pytest.raises(ValueError, match='NCP reuse'):
                ci.reusable_ncp_build(ncp, pin, sdk)
        cache.write_text(original)
    script.write_text('# old CI without external build support\n')
    with pytest.raises(ValueError, match='LLAMA_BUILD_DIR support is required'):
        ci.main()
    assert len(builds) == 1


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


@pytest.mark.parametrize('fault', [None, 'command', 'rc', 'incomplete', 'corpus', 'symlink', 'output'])
def test_numerical_log_reuse_never_skips_validation_or_relaunches(tmp_path, fault):
    from tools.run_kpack_model_validation import reuse_completed
    import hashlib
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    previous, target = old / 'b1-kpack.log', new / 'b1-kpack.log'
    previous.write_text('completed log retained verbatim')
    for root in (old, new): (root / 'corpus.txt').write_text('same corpus')
    argv = ['/bin/llama-perplexity', '-m', '/model.gguf', '-f', str(old / 'corpus.txt'), '-b', '1']
    expected = argv.copy(); expected[expected.index('-f') + 1] = str(new / 'corpus.txt')
    if fault == 'command': argv[-1] = '128'
    (old / 'b1-kpack.command.json').write_text(json.dumps(dict(argv=argv)))
    if fault != 'incomplete':
        (old / 'b1-kpack.process.json').write_text(json.dumps(dict(rc=1 if fault == 'rc' else 0)))
    if fault == 'corpus': (new / 'corpus.txt').write_text('different corpus')
    if fault == 'symlink':
        previous.rename(old / 'kept.log'); previous.symlink_to(old / 'kept.log')
    if fault == 'output': target.write_text('keep this too')
    before = previous.read_bytes()
    if fault:
        with pytest.raises(ValueError): reuse_completed(expected, target, previous)
    else:
        assert reuse_completed(expected, target, previous) == before.decode()
        receipt = json.loads(target.with_suffix('.reuse.json').read_text())
        assert receipt['checks'] == 'PENDING_RECHECK' and receipt['timing'] == 'NOT_REMEASURED'
        assert receipt['sha256'] == hashlib.sha256(before).hexdigest()
        assert json.loads(target.with_suffix('.command.json').read_text())['argv'] == argv
    assert previous.read_bytes() == before
    if fault == 'output': assert target.read_text() == 'keep this too'
    assert reuse_completed(expected, new / 'absent.log', old / 'absent.log') is None


@pytest.mark.parametrize('fault', [None, 'metrics', 'coverage', 'selection'])
def test_reused_numerical_calls_still_check_metrics_coverage_and_selection(tmp_path, monkeypatch, fault):
    from tools import run_kpack_model_validation as validation
    checked = []
    def metrics(*_args):
        checked.append('metrics')
        if fault == 'metrics': raise ValueError('nonfinite numerical metrics')
        return {}
    def coverage(*_args):
        checked.append('coverage')
        if fault == 'coverage': raise ValueError('missing coverage')
        return {}
    def selection(*_args):
        checked.append('selection')
        return dict(fully_selected=fault != 'selection')
    monkeypatch.setattr(validation, 'reuse_completed', lambda *_args: 'saved log')
    monkeypatch.setattr(validation, 'run', lambda *_args: pytest.fail('a completed call was relaunched'))
    monkeypatch.setattr(validation, 'numerical_metrics', metrics)
    corpus = tmp_path / 'input'; corpus.write_text('corpus')
    args = SimpleNamespace(corpus=corpus, logits=tmp_path / 'logits', build=tmp_path / 'build',
        cache=tmp_path / 'cache', reuse_from=tmp_path / 'old')
    call = lambda: validation.numerical(args, dict(name='model', path='/model.gguf'),
        dict(eligible=['weight']), tmp_path, ({}, lambda *_args: 'corpus', coverage, selection))
    if fault:
        with pytest.raises(ValueError): call()
    else:
        assert len(call()) == 6
        assert checked.count('metrics') == checked.count('coverage') == 6
        assert checked.count('selection') == 2


def continuation_fixture(tmp_path, monkeypatch):
    from tools import resume_kpack_q4_model as resume
    from quactlize.runtime.compiler import sha
    previous, sdk, llama = [tmp_path / name for name in ('run', 'sdk', 'llama')]
    result = previous / 'results'; result.mkdir(parents=True)
    (llama / 'tests').mkdir(parents=True)
    (llama / 'tests/quactlize_native.py').write_text('compute scope mismatch:')
    (result / 'runner-status.txt').write_text('runner_rc=1 stage=model-numerical\n')
    (result / 'numerical').mkdir()
    (result / 'numerical/status.json').write_text(json.dumps([dict(status='FAIL',
        error='explicit BF16 request fell back to FP16 compute')]))
    root = tmp_path / 'root'; (root / 'tools').mkdir(parents=True)
    pin = dict(commit='a'*40, path='prebuilt/ppu0010/kpack-model-runtime-v1')
    bundle = tmp_path / 'quactlize-model-artifact-aaaaaaaaaa' / pin['path']; bundle.mkdir(parents=True)
    for path in (bundle / 'manifest.json', result / 'bundle-manifest.json'): path.write_text('{}')
    pin['manifest_sha256'] = sha(bundle / 'manifest.json')
    (root / 'tools/kpack_q4_model_artifact.json').write_text(json.dumps(pin))
    build = tmp_path / 'build'; (build / 'bin').mkdir(parents=True)
    files = {}
    for name in ('llama-server', 'llama-batched-bench', 'llama-perplexity', 'libggml-cuda.so', 'libncp_fa.so', 'libncp_moe.so'):
        file = build / 'bin' / name; file.write_text('original binary')
        files['bin/' + name] = sha(file)
    (result / 'caller-ci-build.json').write_text(json.dumps(dict(build=str(build),
        entry='.aoneci/scripts/build.sh', files=files)))
    for name in ('production-q8.json', 'bf16-metadata/summary.json', 'bf16/summary.json',
                 'bf16-selected-q4/summary.json', 'bf16-matched-prefill/summary.json', 'mixed-chain/summary.json'):
        path = result / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status='PASS')))
    monkeypatch.setattr(resume, 'ROOT', root)
    monkeypatch.setattr(resume, 'verify', lambda *_a, **_k: {})
    return resume, previous, llama, sdk, bundle, build


@pytest.mark.parametrize('fault', [None, 'execution', 'compute', 'mapping', 'policy', 'base'])
def test_dispatcher_only_retry_rejects_device_or_contract_changes(fault):
    from tools.resume_kpack_q4_model import check_dispatcher_refresh
    import copy
    old = dict(dispatch_sha256='old', host_command=['old'], execution_sha256='same',
        compute_contract={'metadata':'bf16'}, policy_hashes={'quactlize/dispatch/policy.hpp':'old','other':'same'},
        smallm_matched_policy={'sha256':'same'})
    current=copy.deepcopy(old)
    current.update(dispatch_sha256='new',host_command=['new'],dispatcher_refresh=dict(
        base_manifest_sha256='base', changed_policy_inputs=['quactlize/dispatch/policy.hpp'],
        scope='DENSE_N_EXTENSION_PREDICTED', gpu_compilations=0))
    current['policy_hashes']['quactlize/dispatch/policy.hpp']='new'
    if fault=='execution':current['execution_sha256']='new'
    if fault=='compute':current['compute_contract']['metadata']='f16'
    if fault=='mapping':current['smallm_matched_policy']['sha256']='new'
    if fault=='policy':current['policy_hashes']['other']='new'
    if fault=='base':current['dispatcher_refresh']['base_manifest_sha256']='wrong'
    if fault:
        with pytest.raises(ValueError):check_dispatcher_refresh(old,current,'base')
    else:check_dispatcher_refresh(old,current,'base')


@pytest.mark.parametrize('fault', [None, 'numerical', 'fallback', 'origin'])
def test_prefill_retry_accepts_only_the_uploaded_policy_failure(tmp_path, monkeypatch, fault):
    from quactlize.runtime.compiler import sha
    resume, original, *_ = continuation_fixture(tmp_path, monkeypatch)
    previous=tmp_path/'retry';result=previous/'results';result.mkdir(parents=True)
    log=result/'numerical/model';log.mkdir(parents=True)
    (result/'status.json').write_text(json.dumps(dict(status='FAIL',phase='numerical')))
    (result/'numerical/status.json').write_text(json.dumps([dict(model='model',status='FAIL',
        error='NaN' if fault=='numerical' else 'model numerical run has missing selected operations or a legacy fallback')]))
    text='[quactlize] output.weight: native policy miss, retain legacy K-pack FQ (no same-family policy choice)'
    if fault=='fallback':text=text.replace('output.weight','blk.0.weight')
    (log/'b2048-kpack-reference.log').write_text(text+'\n')
    (result/'continuation.json').write_text(json.dumps(dict(previous=str(original),
        runtime_manifest_sha256='wrong' if fault=='origin' else sha(original/'results/bundle-manifest.json'))))
    if fault:
        with pytest.raises(ValueError):resume.repaired_source(previous)
    else:assert resume.repaired_source(previous)==original


def test_changed_dispatcher_reruns_both_native_evaluations(tmp_path, monkeypatch):
    from tools import run_kpack_model_validation as validation
    calls=[]
    def reused(argv, log, previous):
        calls.append(('reused', log.name));return 'saved reference'
    def launched(argv, log, env):
        calls.append(('launched', log.name));return 'new native'
    monkeypatch.setattr(validation, 'reuse_completed', reused)
    monkeypatch.setattr(validation, 'run', launched)
    monkeypatch.setattr(validation, 'numerical_metrics', lambda *_args: {})
    corpus=tmp_path/'input';corpus.write_text('corpus')
    args=SimpleNamespace(corpus=corpus,logits=tmp_path/'logits',build=tmp_path/'build',
        cache=tmp_path/'cache',reuse_from=tmp_path/'old',rerun_native=True)
    records=validation.numerical(args,dict(name='model',path='/model.gguf'),dict(eligible=['weight']),
        tmp_path,({},lambda *_args:'corpus',lambda *_args:{},lambda *_args:dict(fully_selected=True)))
    assert len(records)==6
    assert [name for kind,name in calls if kind=='launched']==['b1-kpack-reference.log','b2048-kpack-reference.log']
    assert len([name for kind,name in calls if kind=='reused'])==4


@pytest.mark.parametrize('fault', [None, 'stage', 'error', 'later', 'manifest', 'binary', 'checker', 'gate', 'receipt'])
def test_model_continuation_binds_prior_run(tmp_path, monkeypatch, fault):
    resume, previous, llama, sdk, bundle, build = continuation_fixture(tmp_path, monkeypatch)
    result = previous / 'results'
    if fault == 'stage': (result / 'runner-status.txt').write_text('runner_rc=1 stage=ci-build')
    if fault == 'error': (result / 'numerical/status.json').write_text(json.dumps([dict(status='FAIL', error='NaN')]))
    if fault == 'later': (result / 'trace').mkdir()
    if fault == 'manifest': (result / 'bundle-manifest.json').write_text('{"changed":true}')
    if fault == 'binary': (build / 'bin/llama-perplexity').write_text('rebuilt')
    if fault == 'checker': (llama / 'tests/quactlize_native.py').write_text('obsolete')
    if fault == 'gate': (result / 'bf16/summary.json').write_text('{"status":"FAIL"}')
    if fault == 'receipt':
        file = result / 'caller-ci-build.json'; receipt = json.loads(file.read_text()); receipt['files'] = {}
        file.write_text(json.dumps(receipt))
    if fault:
        with pytest.raises(ValueError): resume.inputs(previous, llama, sdk)
    else:
        actual_bundle, actual_build, gates = resume.inputs(previous, llama, sdk)
        assert actual_bundle == bundle and actual_build == build and len(gates) == 6


def test_model_continuation_has_no_build_and_uses_fresh_outputs(tmp_path, monkeypatch):
    resume, previous, llama, sdk, bundle, build = continuation_fixture(tmp_path, monkeypatch)
    (sdk / 'asight/bin').mkdir(parents=True)
    for name in ('asys', 'acu'):
        path = sdk / 'asight/bin' / name; path.write_text('profiler fixture'); path.chmod(0o755)
    for name in ('kpack-model-cache', 'kpack-model-jit-cache'): (tmp_path / name).mkdir()
    output = tmp_path / 'continued'; (output / 'results').mkdir(parents=True)
    commands = []
    monkeypatch.delenv('CACHE_DIR', raising=False); monkeypatch.delenv('JIT_CACHE', raising=False)
    monkeypatch.setattr(resume, 'run', lambda argv, log, env: commands.append((list(map(str, argv)), log, env)))
    resume.continuation(SimpleNamespace(previous=previous, llama=llama, sdk=sdk, device='0', corpus=tmp_path/'corpus'), output)
    assert [log.name for _, log, _ in commands] == ['numerical.log', 'benchmark.log', 'trace.log', 'acu.log']
    assert '--reuse-from' in commands[0][0] and '--order' in commands[1][0]
    assert all(env['QUACTLIZE_KPACK_COMPUTE'] == 'bf16' for _, _, env in commands)
    assert all('build_kpack_model_ci.py' not in ' '.join(argv) for argv, _, _ in commands)
    assert (output / 'results/prior/runner-status.txt').read_bytes() == (previous / 'results/runner-status.txt').read_bytes()
    subprocess.run(['bash', '-n', str(ROOT / 'tools/resume_kpack_q4_model_box.sh')], check=True)


def performance_continuation_fixture(tmp_path, monkeypatch):
    resume, previous, llama, sdk, bundle, build = continuation_fixture(tmp_path, monkeypatch)
    result=previous/'results'
    (result/'runner-status.txt').write_text('runner_rc=1 stage=paired-model-proof\n')
    pin_path=resume.ROOT/'tools/kpack_q4_model_artifact.json'
    pin=json.loads(pin_path.read_text());pin['model_reader_gate']=True
    pin_path.write_text(json.dumps(pin))
    manifest=dict(paired_gate_up={},execution_sha256='current-execution')
    monkeypatch.setattr(resume,'verify',lambda *_a,**_k:manifest)
    plan=dict(models=[dict(name='qwen35-35b-q4km')],prompts=[2048],generations=[128])
    (result/'model-plan.json').write_text(json.dumps(plan))
    ci=json.loads((result/'caller-ci-build.json').read_text())
    protocol=dict(plan=plan,first_pass_excluded=True,order='abba',repeats=2,
                  production_fusion='MOE_CHAIN_AND_GPU_GATE_UP_PAIR',
                  binary_sha256=ci['files']['bin/llama-batched-bench'])
    path=result/'benchmark/results';path.mkdir(parents=True)
    (path/'protocol.json').write_text(json.dumps(protocol))
    (path/'status.json').write_text('[{"status":"FAIL","reason":"out of memory"}]')
    for name in ('paired-integration','model-readers'):(result/name).mkdir()
    (result/'paired-integration/summary.json').write_text('{"status":"PASS"}')
    rows=[dict(point=name,status='PASS',rc=0,admitted_m1_bitdiff=0,
               execution_sha256=manifest['execution_sha256'],manifest_sha256=pin['manifest_sha256'])
          for name in ('q4-paired-routed','q5-routed-down','q8-paired-shared','q8-shared-down',
                       'q8-ssm-out','q8-qkv','q8-attn-gate')]
    (result/'model-readers/summary.json').write_text(json.dumps(dict(status='PASS',records=rows)))
    return resume,previous,llama,sdk,bundle,build


@pytest.mark.parametrize('fault',[None,'stage','warmup','plan','binary','reader-bits','reader-image',
                                'reader-count','reader-duplicate','reader-status','paired'])
def test_performance_resume_requires_same_artifacts_and_completed_gates(tmp_path,monkeypatch,fault):
    resume,previous,llama,sdk,bundle,build=performance_continuation_fixture(tmp_path,monkeypatch)
    result=previous/'results'
    if fault=='stage':(result/'runner-status.txt').write_text('runner_rc=1 stage=model-reader-integration')
    if fault in ('warmup','plan','binary'):
        p=result/'benchmark/results/protocol.json';data=json.loads(p.read_text())
        if fault=='warmup':data['first_pass_excluded']=False
        if fault=='plan':data['plan']['prompts']=[1024]
        if fault=='binary':data['binary_sha256']='changed'
        p.write_text(json.dumps(data))
    if fault and fault.startswith('reader-'):
        p=result/'model-readers/summary.json';data=json.loads(p.read_text());rows=data['records']
        if fault=='reader-bits':rows[0]['admitted_m1_bitdiff']=1
        if fault=='reader-image':rows[0]['execution_sha256']='changed'
        if fault=='reader-count':rows.pop()
        if fault=='reader-duplicate':rows[-1]=rows[0]
        if fault=='reader-status':rows[0]['status']='FAIL'
        p.write_text(json.dumps(data))
    if fault=='paired':(result/'paired-integration/summary.json').write_text('{"status":"FAIL"}')
    if fault:
        with pytest.raises(ValueError):resume.inputs(previous,llama,sdk,performance_only=True)
    else:
        found,caller,gates=resume.inputs(previous,llama,sdk,performance_only=True)
        assert found==bundle and caller==build and len(gates)==8
    assert json.loads((result/'benchmark/results/status.json').read_text())[0]['status']=='FAIL'


@pytest.mark.parametrize('acu',[False,True])
def test_performance_resume_preserves_prior_and_runs_only_model_phases(tmp_path,monkeypatch,acu):
    resume,previous,llama,sdk,bundle,build=performance_continuation_fixture(tmp_path,monkeypatch)
    (sdk/'asight/bin').mkdir(parents=True)
    for name in ('asys','acu'):
        p=sdk/'asight/bin'/name;p.write_text('host fixture');p.chmod(0o755)
    for name in ('kpack-model-cache','kpack-model-jit-cache'):(tmp_path/name).mkdir()
    output=tmp_path/'continued';(output/'results').mkdir(parents=True)
    monkeypatch.delenv('CACHE_DIR',raising=False);monkeypatch.delenv('JIT_CACHE',raising=False)
    commands=[]
    monkeypatch.setattr(resume,'run',lambda argv,log,env:commands.append((list(map(str,argv)),log,env)))
    resume.continuation(SimpleNamespace(previous=previous,llama=llama,sdk=sdk,device='1',
        corpus=tmp_path/'corpus',performance_only=True,profile_acu=acu),output)
    expected=['benchmark.log','trace.log','paired-model-proof.log']+(['acu.log'] if acu else [])
    assert [log.name for _,log,_ in commands]==expected
    assert all(env['CUDA_VISIBLE_DEVICES']=='1' and env['QUACTLIZE_KPACK_GATE_UP']=='1' and
               env['QUACTLIZE_KPACK_COMPUTE']=='bf16' for _,_,env in commands)
    assert all('build_kpack' not in ' '.join(argv) and 'run_model_gemv_integration' not in ' '.join(argv)
               for argv,_,_ in commands)
    assert (output/'results/prior/runner-status.txt').read_bytes()==(previous/'results/runner-status.txt').read_bytes()
    receipt=json.loads((output/'results/continuation.json').read_text())
    assert receipt['compile']=='NONE' and receipt['numerical_scope']=='NOT_RETESTED'


def trace_continuation_fixture(tmp_path, monkeypatch):
    values = performance_continuation_fixture(tmp_path, monkeypatch)
    resume, previous, llama, sdk, bundle, build = values
    result = previous / 'results'
    (result / 'runner-status.txt').write_text('runner_rc=1 stage=model-trace\n')
    source = plan(tmp_path)
    source['models'] = source['models'][:1]
    (result / 'model-plan.json').write_text(json.dumps(source))
    root = result / 'benchmark/results'
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol.update(plan=source, production_fusion='MOE_CHAIN')
    (root / 'protocol.json').write_text(json.dumps(protocol))
    model, = source['models']
    folder = root / model['name']; folder.mkdir()
    status = []
    for label in ('0-reference', '1-kpack', '2-kpack', '3-reference'):
        arm = label.split('-', 1)[1]
        status.append(dict(model=model['name'], arm=arm, status='PASS'))
        rows = []
        for point in sequence(source, 2):
            pp, tg, repeat = point
            raw = dict(n_kv_max=pp+tg, pp=pp, tg=tg, pl=1, n_batch=source['batch'],
                       n_ubatch=source['ubatch'], flash_attn=1, is_pp_shared=0,
                       n_kv=pp+tg, t_pp=1, t_tg=1, speed_pp=pp, speed_tg=tg)
            rows.append(parse_row(json.dumps(raw), point, source))
        for suffix, data in (
            ('process', dict(rc=0)), ('timings', rows),
            ('command', dict(devices='0', compute='fp16' if arm=='kpack' else 'REFERENCE')),
            ('selection', dict(plan_admission='PASS', fully_selected=True, paired_plans=[])),
        ):
            (folder / f'{label}.{suffix}.json').write_text(json.dumps(data))
    (root / 'status.json').write_text(json.dumps(status))
    # This FP16 run has no BF16 or paired component admission to reuse.
    for name in ('bf16-metadata', 'bf16', 'bf16-selected-q4', 'bf16-matched-prefill', 'paired-integration'):
        (result / name / 'summary.json').unlink()
    return values


@pytest.mark.parametrize('fault', [None, 'failed-arm', 'missing-arm', 'process', 'selection',
                                  'nan', 'warmup', 'sample-count', 'denominator', 'compute', 'device'])
def test_trace_only_rechecks_all_saved_abba_evidence(tmp_path, monkeypatch, fault):
    resume, previous, llama, sdk, bundle, build = trace_continuation_fixture(tmp_path, monkeypatch)
    root = previous / 'results/benchmark/results'
    suffix = {'failed-arm':'status', 'missing-arm':'status', 'process':'process',
              'selection':'selection', 'compute':'command', 'device':'command'}.get(fault, 'timings')
    file = root / ('status.json' if suffix=='status' else f'subject/1-kpack.{suffix}.json')
    data = json.loads(file.read_text())
    if fault=='failed-arm': data[0]['status']='FAIL'
    if fault=='missing-arm': data.pop()
    if fault=='process': data['rc']=15
    if fault=='selection': data['fully_selected']=False
    if fault=='nan': data[1]['t_pp']=float('nan')
    if fault=='warmup': data[0]['phase']='measured'
    if fault=='sample-count': data.pop()
    if fault=='denominator': data[1]['speed_tg']=1
    if fault=='compute': data['compute']='bf16'
    if fault=='device': data['devices']='1'
    file.write_text(json.dumps(data))
    if fault:
        with pytest.raises(ValueError): resume.inputs(previous, llama, sdk, performance_only=True, trace_only=True)
    else:
        found, caller, gates = resume.inputs(previous, llama, sdk, performance_only=True, trace_only=True)
        assert found==bundle and caller==build and len(gates)==3


@pytest.mark.parametrize('device', ['0', '1'])
def test_trace_only_does_not_build_or_remeasure_benchmark(tmp_path, monkeypatch, device):
    resume, previous, llama, sdk, bundle, build = trace_continuation_fixture(tmp_path, monkeypatch)
    (sdk / 'asight/bin').mkdir(parents=True)
    for name in ('asys', 'acu'):
        file = sdk / 'asight/bin' / name; file.write_text('profiler'); file.chmod(0o755)
    for name in ('kpack-model-cache', 'kpack-model-jit-cache'): (tmp_path / name).mkdir()
    monkeypatch.delenv('CACHE_DIR', raising=False); monkeypatch.delenv('JIT_CACHE', raising=False)
    calls = []
    monkeypatch.setattr(resume, 'run', lambda argv, log, env: calls.append((list(map(str, argv)), log, env)))
    output = tmp_path / 'continued'; (output / 'results').mkdir(parents=True)
    args = SimpleNamespace(previous=previous, llama=llama, sdk=sdk, device=device,
                           corpus=tmp_path/'corpus', performance_only=True, trace_only=True)
    if device=='1':
        with pytest.raises(ValueError, match='original single-device'): resume.continuation(args, output)
        assert not calls
    else:
        resume.continuation(args, output)
        assert [log.name for _, log, _ in calls]==['trace.log']
        env = calls[0][2]
        assert env['QUACTLIZE_KPACK_COMPUTE']=='fp16' and env['QUACTLIZE_KPACK_GATE_UP']=='0'
        assert env['DG_JIT_HGCC_COMPILER']==str(sdk/'bin/hgcc')
        receipt = json.loads((output/'results/continuation.json').read_text())
        assert receipt['compile']=='NONE' and receipt['benchmark_scope']=='REUSED_UNCHANGED'
        assert not (output/'results/benchmark').exists()


def test_first_nonfinite_reuses_read_only_original_and_reference_only_changes_placement():
    from tools.run_kpack_first_nonfinite import command
    argv = ['/bin/llama-perplexity', '-m', '/model.gguf', '-c', '256', '-b', '1', '-ub', '1',
            '--chunks', '2', '-f', '/corpus.txt', '--no-warmup', '-ot', '^blk.*=CUDA0_KPACK',
            '--kpack-cache', '/cache', '--kl-divergence', '--kl-divergence-base', '/existing-reference']
    assert command(argv) == argv
    reference = command(argv, True)
    assert reference[reference.index('-ot') + 1] == '^blk.*=CUDA0'
    assert '--kpack-cache' not in reference
    assert reference[reference.index('--kl-divergence-base') + 1] == '/existing-reference'
    assert '--save-all-logits' not in reference and '--no-warmup' in reference
    for old, new in (('256', '512'), ('--no-warmup', '--warmup'), ('--kl-divergence', '--save-all-logits')):
        with pytest.raises(ValueError):
            command([new if x == old else x for x in argv])


@pytest.mark.parametrize('mode', ['logits', 'tensors'])
def test_first_nonfinite_requires_complete_coverage_or_an_explicit_stop(mode):
    from tools.run_kpack_first_nonfinite import result
    start = f'LLAMA_NUMERICAL_DEBUG mode={mode} callback={int(mode == "tensors")} timing_valid=0\n'
    complete = f'LLAMA_NUMERICAL_COMPLETE mode={mode} nodes=2560 logits=256 verdict=NO_NONFINITE_OBSERVED\n'
    reason = 'NONFINITE_TENSOR' if mode == 'tensors' else 'NONFINITE_LOGITS'
    stop = f'LLAMA_NUMERICAL_STOP mode={mode} chunk=1 batch=129 position=128 nodes=3 logits=1 reason={reason}\n'
    assert result(start + stop, 86, mode)['verdict'] == 'NONFINITE_FOUND'
    assert result(start + complete, 0, mode)['verdict'] == 'NO_NONFINITE_OBSERVED'
    for text, rc in ((start, 0), (start + stop, 0), (start + complete, 86),
                     (start + complete.replace('logits=256', 'logits=128'), 0),
                     (start + complete.replace('logits=256', 'logits=0'), 0),
                     (start + stop + stop, 86), (start + stop + complete, 86),
                     (complete, 0), (start + stop.replace(reason, 'OTHER'), 86)):
        with pytest.raises(ValueError):
            result(text, rc, mode)
    if mode == 'tensors':
        with pytest.raises(ValueError):
            result(start + complete.replace('nodes=2560', 'nodes=0'), 0, mode)


def test_first_nonfinite_preserves_expected_nonzero_child_exit(tmp_path):
    from tools.run_kpack_first_nonfinite import run, result
    text = ('LLAMA_NUMERICAL_DEBUG mode=logits callback=0 timing_valid=0\n'
            'LLAMA_NUMERICAL_STOP mode=logits chunk=1 batch=129 position=128 nodes=0 logits=1 reason=NONFINITE_LOGITS')
    log = tmp_path / 'native-logits.log'
    output, rc = run([sys.executable, '-c', f'print({text!r}); raise SystemExit(86)'], os.environ.copy(), log)
    assert rc == 86 and result(output, rc, 'logits')['verdict'] == 'NONFINITE_FOUND'
    assert json.loads(log.with_suffix('.process.json').read_text())['rc'] == 86
    assert log.read_text().strip() == text


def test_first_nonfinite_preflight_error_reaches_console(tmp_path, capsys):
    from tools.run_kpack_first_nonfinite import checked_step
    log = tmp_path / 'verify.log'
    with pytest.raises(ValueError, match='verify failed rc=1'):
        checked_step([sys.executable, '-c', 'print("JIT checkout differs: kernel"); raise SystemExit(1)'],
                     os.environ.copy(), log)
    assert 'JIT checkout differs: kernel' in capsys.readouterr().err
    assert 'JIT checkout differs: kernel' in log.read_text()


def test_first_nonfinite_frozen_source_uses_exact_git_blobs(tmp_path):
    from tools.run_kpack_first_nonfinite import frozen_jit_source
    commit = 'd93b11867813df4723179f11ae583aa4ebdb078a'
    before = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True)
    output = tmp_path / 'source'
    receipt = frozen_jit_source(ROOT, commit, output)
    assert receipt['actlize_commit'] == '423253c00df333ead6fb72ea623d526f24f56b5a'
    for name in ('quactlize/runtime/moe_chain.cuh', 'quactlize/decode/compiler.py', 'tools/kpack_jit.py'):
        expected = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+name])
        assert (output/name).read_bytes() == expected
    assert (output/'third_party/actlize/include/cute/tensor.hpp').is_file()
    assert not (output/'prebuilt').exists() and not (output/'.git').exists()
    assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True) == before
    with pytest.raises(FileExistsError):
        frozen_jit_source(ROOT, commit, output)
    with pytest.raises(ValueError, match='exact commit'):
        frozen_jit_source(ROOT, 'develop', tmp_path/'mutable')


def test_first_token_snapshot_does_not_masquerade_as_model_pass():
    from tools.run_kpack_first_nonfinite import snapshot_result
    header = 'LLAMA_NUMERICAL_DEBUG mode=tensors callback=1 timing_valid=0\n'
    nodes = ('LLAMA_NUMERICAL_TENSOR role=node-1 tensor="ffn_swiglu-2"\n'
             'LLAMA_NUMERICAL_TENSOR role=node-2 tensor="ffn_out-2"\n')
    complete = 'LLAMA_NUMERICAL_SNAPSHOT_COMPLETE chunk=1 batch=1 position=0 nodes=2 target="ffn_out-2"\n'
    text = header + nodes + complete
    assert snapshot_result(text, 87, 'ffn_out-2')['verdict'] == 'SNAPSHOT_COMPLETE_NOT_ACCURACY_ADMISSION'
    stop = 'LLAMA_NUMERICAL_STOP mode=tensors chunk=1 batch=1 position=0 nodes=2 logits=0 reason=NONFINITE_TENSOR\n'
    assert snapshot_result(header + nodes + stop, 86, 'ffn_out-2')['verdict'] == 'NONFINITE_FOUND'
    for bad, rc in ((text, 0), (text.replace('batch=1', 'batch=2'), 87),
                    (text.replace('node-2', 'node-3'), 87),
                    (text.replace('target="ffn_out-2"', 'target="wrong"'), 87),
                    (header+nodes+stop.replace('NONFINITE_TENSOR', 'SNAPSHOT_LIMIT'), 86)):
        with pytest.raises(ValueError):
            snapshot_result(bad, rc, 'ffn_out-2')


def test_paired_range_snapshots_keep_large_finite_values_and_nonfinite_failures(tmp_path):
    import numpy as np
    from tools.analyze_kpack_activation_range import compare, snapshots
    for arm in ('reference-tensors', 'native-tensors'):
        directory = tmp_path / (arm+'-tensors'); directory.mkdir()
        rows = []
        for node, name in enumerate(('ffn_swiglu-2', 'ffn_out-2'), 1):
            values = np.array([1, 243383.484375] if node==1 else [3, -4], dtype='<f4')
            if node==2 and arm=='native-tensors': values[:] = np.inf
            (directory/f'node-{node}.bin').write_bytes(values.tobytes())
            rows.append(f'LLAMA_NUMERICAL_TENSOR role=node-{node} tensor="{name}" op=MUL_MAT type=f32 '
                        'ne=2,1,1,1 nb=4,8,8,8 count=2 nan=0\n')
        (tmp_path/f'{arm}.log').write_text(''.join(rows))
    result = compare(tmp_path, 'ffn_out-2')
    assert result['rows'][0]['reference']['f16_new_nonfinite'] == 1
    assert result['rows'][0]['relative_linf'] == 0
    point = result['rows'][0]['paired_points'][-1]
    assert point['index'] == 1
    assert point['reference']['value'] == point['native']['value'] == 243383.484375
    assert point['native']['f16_bits'] == '0x7c00'
    assert point['native']['operands'] == 'MISSING_SNAPSHOT'
    assert result['rows'][1]['native']['nonfinite'] == 2
    assert result['rows'][1]['relative_linf'] is None
    assert 'NOT_ACCURACY' in result['scope']
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError): compare(tmp_path, 'not-present')
    path = tmp_path/'native-tensors-tensors/node-1.bin'
    path.write_bytes(path.read_bytes()[:4])
    with pytest.raises(ValueError): snapshots(tmp_path, 'native-tensors')


@pytest.mark.parametrize('reference_large', [False, True])
def test_paired_swiglu_captures_same_index_and_independent_operands(tmp_path, reference_large):
    import numpy as np
    from tools.analyze_kpack_activation_range import compare
    for arm in ('reference-tensors', 'native-tensors'):
        directory = tmp_path / (arm+'-tensors'); directory.mkdir()
        gate = np.array([1, 500], dtype='<f4')
        up = np.array([2, 500 if arm=='native-tensors' or reference_large else 3], dtype='<f4')
        product = gate / (np.float32(1) + np.exp(-gate)) * up
        rows = []
        for node, (name, values) in enumerate((('ffn_gate-2', gate), ('ffn_up-2', up), ('ffn_swiglu-2', product)), 1):
            (directory/f'node-{node}.bin').write_bytes(values.tobytes())
            op = 'GLU' if node == 3 else 'MUL_MAT'
            rows.append(f'LLAMA_NUMERICAL_TENSOR role=node-{node} tensor="{name}" op={op} type=f32 '
                        'ne=2,1,1,1 nb=4,8,8,8 count=2 nan=0\n')
        (tmp_path/f'{arm}.log').write_text(''.join(rows))
    result = compare(tmp_path, 'ffn_swiglu-2')
    swiglu = result['rows'][-1]
    assert swiglu['reference']['f16_new_nonfinite'] == int(reference_large)
    assert swiglu['native']['f16_new_nonfinite'] == 1
    point = swiglu['paired_points'][-1]
    assert point['index'] == 1
    for arm in ('reference', 'native'):
        actual = point[arm]
        assert actual['gate']['value'] == 500
        assert actual['swiglu_recomputed_f32']['f32_bits'] == actual['f32_bits']
    assert point['native']['value'] == 250000
    assert point['reference']['value'] == (250000 if reference_large else 1500)
    json.dumps(result, allow_nan=False)


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


def successful_extension_fixture(tmp_path, monkeypatch):
    values = performance_continuation_fixture(tmp_path, monkeypatch)
    resume, previous, llama, sdk, bundle, build = values
    result = previous/'results'
    (result/'runner-status.txt').write_text('runner_rc=0 stage=complete\n')
    (result/'benchmark/results/status.json').write_text('[{"status":"PASS"}]')
    (result/'trace').mkdir()
    (result/'trace/status.json').write_text('[{"status":"PASS"}]')
    old_model = fixture(tmp_path, 'original.gguf')
    old = dict(source='original', prompts=[2048], generations=[128], parallel=1, batch=2048,
               ubatch=2048, models=[dict(name='qwen35-35b-q4km', path=str(old_model),
                                       devices='0', split='none')])
    (result/'model-plan.json').write_text(json.dumps(old))
    protocol_path=result/'benchmark/results/protocol.json'
    protocol=json.loads(protocol_path.read_text());protocol['plan']=old
    protocol_path.write_text(json.dumps(protocol))
    new=old | dict(source='extension-fixture', model_root=str(tmp_path/'models'), models=[
        dict(name='dense-control',directory='dense',devices='0',split='none'),
        dict(name='moe-control',directory='moe',devices='0',split='none')])
    for model in new['models']:
        fixture(Path(new['model_root']),model['directory']+'/weights.gguf')
    plan_path=tmp_path/'extension.json';plan_path.write_text(json.dumps(new))
    return (*values,plan_path)


def test_other_models_keep_workload_and_preserve_122b_two_device_requirement():
    plan=json.loads((ROOT/'tools/kpack_batched_other_int4_2048.json').read_text())
    full=json.loads((ROOT/'tools/kpack_batched_models.json').read_text())
    assert plan['prompts']==[2048] and plan['generations']==[128] and plan['parallel']==1
    assert plan['batch']==plan['ubatch']==2048
    assert {m['name'] for m in plan['models']}=={'qwen3-32b-q4km','qwen35-122b-q4km'}
    for model in plan['models']:
        if model['name']=='qwen35-122b-q4km':
            assert model['split']=='tensor' and model['devices']=='0,1' and model['tensor_split']=='1,1'
            argv=command('/bench',model|dict(path='/122b.gguf'),plan,2,['blk.0.ffn_down_exps.weight'],'kpack',Path('/cache'))
            assert argv[argv.index('-sm')+1]=='tensor' and argv[argv.index('-ts')+1]=='1,1'
            assert argv[argv.index('-ot')+1].endswith('=CUDA0_KPACK')
            assert argv[argv.index('--kpack-cache')+1] == '/cache'
            native=command('/bench',model|dict(path='/122b.gguf'),plan,2,['blk.0.ffn_down_exps.weight'],'reference',Path('/cache'))
            assert '-ot' not in native
        else:
            assert model['split']=='none' and model['devices']=='0' and 'tensor_split' not in model
        assert model['directory']==next(m['directory'] for m in full['models'] if m['name']==model['name'])


@pytest.mark.parametrize('fault', [None, 'device', 'geometry', 'grouped', 'fallback', 'producer'])
def test_tp2_requires_selected_local_geometry_on_both_devices(fault):
    lines=[]
    for dev in (0,1):
        for op,experts in (('dense',1),('grouped',256)):
            fields=f'tensor={op}.weight device={dev} q=12 n=512 k=3072 experts={experts}'
            if not (fault=='producer' and dev==1):
                lines.append('[quactlize-shard] '+fields+' bytes=1024 producer=GPU')
            if fault=='device' and dev==1 or fault=='grouped' and op=='grouped' and dev==1: continue
            if fault=='geometry' and dev==1: fields=fields.replace('k=3072','k=1536')
            selected='0' if fault=='fallback' and dev==1 else '1'
            lines.append('[quactlize-device-plan] '+fields+f' op={op} rows=8 selected={selected}')
    if fault:
        with pytest.raises(ValueError):bench.tp2_evidence('\n'.join(lines),['dense','grouped'])
    else:
        assert bench.tp2_evidence('\n'.join(lines),['dense','grouped'])['status']=='PASS'


def test_tp2_plan_and_runner_do_not_bypass_meta_or_cache_admission(tmp_path):
    source=json.loads((ROOT/'tools/kpack_batched_tp2_122b.json').read_text())
    validate_plan(source)
    model,=source['models']
    assert model['filename'].endswith('00001-of-00003.gguf')
    for index in (1,2,3):
        fixture(tmp_path,model['directory']+'/'+model['filename'].replace('00001-of-',f'{index:05d}-of-'))
    resolved=resolve_plan(source, model_root=tmp_path)
    assert len(resolved['models'][0]['files'])==3
    script=(ROOT/'tools/run_kpack_tp2_box.sh').read_text()
    subprocess.run(['bash','-n',str(ROOT/'tools/run_kpack_tp2_box.sh')],check=True)
    assert 'build_kpack_model_ci.py' in script and 'JOBS=${JOBS:-192}' in script
    assert 'test-quactlize-scheduler" --tp2-cache-write' in script
    assert 'test-quactlize-scheduler" --tp2-cache-read' in script
    assert script.count("grep -qx 'KPACK_TP2_SEGMENTED_ALL PASS formats=6 cases=6 segments=3 replays=3'") == 2
    assert 'cache-disabled-for-tp' not in script
    assert script.index('stage=two-device-cache-cold') < script.index('stage=two-device-cache-hot') < \
        script.index('stage=model-numerical') < script.index('stage=model-perf')
    assert '--order abba --require-selected' in script
    assert 'DG_JIT_HGCC_COMPILER="$SDK/bin/hgcc"' in script
    assert 'PPU_SDK_HOME="$SDK/CUDA_SDK"' in script
    assert script.index('TP2_MODE == communication') < script.index('stage=two-device-cache-cold')
    assert 'tools/run_kpack_tp2_comm.py' in script
    assert 'PCCL_ENABLE_EXT_KERNEL=' not in script


@pytest.mark.parametrize('state',['unset','missing','empty','present'])
def test_tp2_legacy_bundle_precheck_names_missing_input(tmp_path,state):
    source=(ROOT/'tools/run_kpack_tp2_box.sh').read_text()
    start=source.index('    if [[ -z ${QUACTLIZE_PPU_BUNDLE:-} ]]')
    block=source[start:source.index('    ASYS=',start)]
    env=dict(os.environ)
    env.pop('QUACTLIZE_PPU_BUNDLE',None)
    bundle=tmp_path/'six-library'
    if state!='unset': env['QUACTLIZE_PPU_BUNDLE']=str(bundle)
    if state in ('empty','present'):
        bundle.mkdir()
        (bundle/'manifest.json').write_text('{}' if state=='present' else '')
    result=subprocess.run(['bash','-euc',block],env=env,text=True,capture_output=True)
    assert (result.returncode==0)==(state=='present')
    if state=='unset': assert 'missing QUACTLIZE_PPU_BUNDLE' in result.stderr
    if state in ('missing','empty'): assert str(bundle/'manifest.json') in result.stderr


def comm_fixture(arm='copy', count=512):
    lines=[]
    for replay in range(3):
        for rank in (0,1):
            lines.append(f'KPACK_TP2_COMM_LOCAL arm={arm} rank={rank} replay={replay} error=0 synchronized=1 status=PASS')
        lines.append(f'KPACK_TP2_COMM_REDUCE_BEGIN arm={arm} count={count} replay={replay}')
        for rank in (0,1):
            lines.append(f'KPACK_TP2_COMM_SUM arm={arm} rank={rank} replay={replay} error=0 status=PASS')
    return '\n'.join(lines)+f'\nKPACK_TP2_COMM PASS arm={arm} count={count} replays=3\n'


@pytest.mark.parametrize('fault',[None,'missing','duplicate','nan','tolerance','rc','fallback','shape'])
def test_tp2_communication_requires_both_ranks_and_all_replays(fault):
    from tools.run_kpack_tp2_comm import evidence
    text=comm_fixture()
    if fault=='missing': text=text.replace('rank=1 replay=2','rank=1 replay=3')
    if fault=='duplicate': text+=text.splitlines()[0]+'\n'
    if fault=='nan': text=text.replace('error=0','error=nan',1)
    if fault=='tolerance': text=text.replace('error=0','error=0.5',1)
    if fault=='fallback': text+='NCCL init failed: falling back to internal AllReduce\n'
    if fault=='shape': text=text.replace('count=512','count=3072')
    result=evidence(text,'copy',512,1 if fault=='rc' else 0)
    assert (result['status']=='PASS')==(fault is None)


def q4_comm_fixture(arm, fail=False):
    if arm == 'meta':
        text = ('KPACK_TP2_MISMATCH q=12 experts=4 tokens=1 split=K replay=0 relative=0.1823166\n' if fail else
                'KPACK_TP2_SINGLE PASS q=12 experts=4 tokens=1 split=K replays=3\n')
    else:
        text = 'KPACK_TP2_COMM_SHAPE q=12 n=512 global_k=1024 local_k=512 experts=4 tokens=1 topk=2\n'
        for rank in (0,1): text += f'KPACK_TP2_COMM_WEIGHT rank={rank} q=12 bytes=589824 hash={rank:016x}\n'
        for replay in range(1 if fail else 3):
            for rank in (0,1):
                text += f'KPACK_TP2_COMM_ORACLE rank={rank} replay={replay} input_hash={replay:016x} ids_hash={replay:016x} golden_hash={rank:016x}\n'
        if fail:
            for rank in (0,1):
                text += f'KPACK_TP2_COMM_LOCAL_BEGIN arm={arm} rank={rank} replay=0\n'
                text += f'KPACK_TP2_COMM_LOCAL arm={arm} rank={rank} replay=0 error=0.1 synchronized=1 status=FAIL\n'
        else: text += comm_fixture(arm,1024)
    if arm != 'raw':
        for rank in (0,1):
            text += '[quactlize-plan] tensor=tp-weight op=grouped route=gemv reader=simt-reuse q=12 rows=2 n=512 k=512 variant=0 columns=4 warps=4 values=4 split=1 policy=11 activation=BF16 scale_resident=0 experts=4 channels=2 topk=2\n'
            text += f'[quactlize-device-plan] tensor=tp-weight device={rank} op=grouped q=12 rows=2 n=512 k=512 experts=4 selected=1\n'
    return text


@pytest.mark.parametrize('fault',[None,'local','meta','raw','weights','oracle','shape','selection','coverage'])
def test_tp2_q4_local_diagnostic_separates_numeric_boundaries(fault):
    from tools.run_kpack_tp2_comm import q4_evidence,q4_verdict
    results=[]
    for arm in ('raw','kpack','meta'):
        fail = (fault=='local' and arm=='kpack') or fault==arm
        text = q4_comm_fixture(arm,fail)
        if arm=='kpack':
            if fault=='weights': text=text.replace('bytes=589824 hash=0000000000000000','bytes=589824 hash=1111111111111111')
            if fault=='oracle': text=text.replace('input_hash=0000000000000000','input_hash=1111111111111111')
            if fault=='shape': text=text.replace('local_k=512','local_k=1024')
            if fault=='selection': text=text.replace('variant=0','variant=3')
            if fault=='coverage': text=text.replace('rank=1 replay=2','rank=1 replay=3')
        results.append(dict(arm=arm,**q4_evidence(text,arm,int(fail))))
    wanted={None:'FRESH_PROCESS_NOT_REPRODUCED','local':'LOCAL_KPACK_PACKING_OR_COMPUTE_FAILED',
            'meta':'META_PATH_ONLY_FAILURE','raw':'RAW_CONTROL_NOT_ADMITTED',
            'weights':'FIXTURE_IDENTITY_FAILED','oracle':'FIXTURE_IDENTITY_FAILED',
            'shape':'LOCAL_OR_COLLECTIVE_NOT_ADMITTED','selection':'FIXTURE_IDENTITY_FAILED',
            'coverage':'LOCAL_OR_COLLECTIVE_NOT_ADMITTED'}
    assert q4_verdict(results)==wanted[fault]


def test_tp2_q4_local_runner_has_three_fresh_children_and_no_policy_mutation(tmp_path,monkeypatch):
    from tools import run_kpack_tp2_comm as comm
    calls=[]
    monkeypatch.delenv('GGML_CUDA_ALLREDUCE',raising=False)
    monkeypatch.setenv('QUACTLIZE_KPACK_COMPUTE','fp16')
    def start(command,**kwargs):
        arm='meta' if command[-1]=='--tp2-q4-meta' else command[-1]
        calls.append((command,kwargs['env']))
        kwargs['stdout'].write(q4_comm_fixture(arm,arm=='kpack'))
        return SimpleNamespace(wait=lambda **kw:int(arm=='kpack'))
    monkeypatch.setattr(comm.subprocess,'Popen',start)
    comm.run_q4(Path('/build/bin/test-quactlize-scheduler'),tmp_path/'results')
    assert len(calls)==3 and os.environ['QUACTLIZE_KPACK_COMPUTE']=='fp16'
    assert all(env['QUACTLIZE_KPACK_COMPUTE']=='bf16' and env['QUACTLIZE_KPACK_ROUTE']=='auto' for _,env in calls)
    summary=json.loads((tmp_path/'results/summary.json').read_text())
    assert summary['device_admission']=='PENDING'
    assert summary['verdict']=='LOCAL_KPACK_PACKING_OR_COMPUTE_FAILED'


def test_tp2_communication_continues_failures_and_extension_is_child_local(tmp_path,monkeypatch):
    from tools import run_kpack_tp2_comm as comm
    calls=[]
    monkeypatch.delenv('GGML_CUDA_ALLREDUCE',raising=False)
    monkeypatch.delenv('PCCL_ENABLE_EXT_KERNEL',raising=False)
    def start(command,**kwargs):
        calls.append((command,kwargs['env']))
        arm,count=command[-2:]
        if len(calls)==1:
            kwargs['stdout'].write(comm_fixture(arm,int(count)).split('KPACK_TP2_COMM_SUM',1)[0] +
                "[drv_extension.cc:379] PCCL ERROR invalid device function\n")
        else: kwargs['stdout'].write(comm_fixture(arm,int(count)))
        return SimpleNamespace(wait=lambda **kw: -6 if len(calls)==1 else 0)
    monkeypatch.setattr(comm.subprocess,'Popen',start)
    comm.run(Path('/build/bin/test-quactlize-scheduler'),tmp_path/'results')
    assert len(calls)==7
    result=json.loads((tmp_path/'results/summary.json').read_text())
    assert result['status']=='DIAGNOSTIC_COMPLETE' and result['device_admission']=='PENDING'
    assert result['cases'][0]['status']=='COLLECTIVE_FAILED_AFTER_LOCAL_PASS'
    assert result['cases'][0]['extension_invalid_function']
    assert all(x['status']=='PASS' for x in result['cases'][1:])
    assert 'PCCL_ENABLE_EXT_KERNEL' not in os.environ
    for index,(_,env) in enumerate(calls):
        assert env.get('PCCL_ENABLE_EXT_KERNEL')==('0' if index>=5 else None)


def test_tp2_communication_timeout_stops_only_its_own_child_and_continues(tmp_path,monkeypatch):
    from tools import run_kpack_tp2_comm as comm
    children=[]
    killed=[]
    monkeypatch.delenv('GGML_CUDA_ALLREDUCE',raising=False)
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=1000+len(children)
            self.command=command
            self.calls=0
            children.append(self)
        def wait(self,**kwargs):
            self.calls+=1
            if self.calls==1: raise subprocess.TimeoutExpired(self.command,1)
            return -9
    monkeypatch.setattr(comm.subprocess,'Popen',Child)
    monkeypatch.setattr(comm.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    comm.run(Path('/build/bin/test-quactlize-scheduler'),tmp_path/'results',timeout=1)
    assert killed==[(child.pid,comm.signal.SIGKILL) for child in children]
    result=json.loads((tmp_path/'results/summary.json').read_text())
    assert len(result['cases'])==7 and all(x['status']=='TIMEOUT' for x in result['cases'])


@pytest.mark.parametrize('fault',[None,'local-fail','global-pass','kpack-global-pass','missing-symbol',
                                  'wrong-scope','old-seven','duplicate','different-path','local-global-symbol'])
def test_tp2_wrapper_verdict_requires_local_green_and_two_exact_global_negatives(fault):
    from tools.run_kpack_tp2_comm import cases,wrapper_verdict
    results=[]
    for arm,count,extension,scope in cases(True):
        negative=scope=='global'
        results.append(dict(arm=arm,count=count,extension=extension,wrapper_scope=scope,
            status='COLLECTIVE_FAILED_AFTER_LOCAL_PASS' if negative else 'PASS',
            extension_invalid_function=negative,
            wrapper_scopes=[(scope,'/sdk/lib/libhggc_wrapper.so')] if scope!='none' else [],
            global_symbols=['phase=after-wrapper name=hggcLaunchKernel library='+
                            ('/sdk/lib/libhggc_wrapper.so' if negative else 'NOT_GLOBAL')]))
    if fault=='local-fail': results[7]['status']='SETUP_FAILED'
    if fault=='global-pass': results[8]['status']='PASS'
    if fault=='kpack-global-pass': results[9]['status']='PASS'
    if fault=='missing-symbol': results[8]['global_symbols']=[]
    if fault=='wrong-scope': results[8]['wrapper_scopes'][0]=('local','/sdk/lib/libhggc_wrapper.so')
    if fault=='different-path': results[8]['wrapper_scopes'][0]=('global','/foreign/lib/libhggc_wrapper.so')
    if fault=='local-global-symbol': results[7]['global_symbols']=results[8]['global_symbols']
    if fault=='old-seven': results=results[:7]
    if fault=='duplicate': results[8]=results[7]
    verdict=wrapper_verdict(results)
    assert (verdict=='GLOBAL_WRAPPER_CAUSAL_CANDIDATE_PASS')==(fault is None)


def test_tp2_wrapper_ab_uses_one_binary_and_unchanged_producer_arguments(tmp_path,monkeypatch):
    from tools import run_kpack_tp2_comm as comm
    commands=[]
    monkeypatch.delenv('GGML_CUDA_ALLREDUCE',raising=False)
    def start(command,**kwargs):
        commands.append(command)
        arm,count=command[2:4]
        scope=command[4] if len(command)==5 else 'none'
        text=comm_fixture(arm,int(count))
        if scope!='none':
            text=f'KPACK_TP2_COMM_WRAPPER scope={scope} path=/sdk/lib/libhggc_wrapper.so\n'+text
        text='KPACK_TP2_COMM_SYMBOL phase=after-wrapper name=hggcLaunchKernel library='+(
            '/sdk/lib/libhggc_wrapper.so' if scope=='global' else 'NOT_GLOBAL')+'\n'+text
        if scope=='global':
            text=text.split('KPACK_TP2_COMM_SUM',1)[0]+'[drv_extension.cc:379] invalid device function\n'
        kwargs['stdout'].write(text)
        return SimpleNamespace(wait=lambda **kw:-6 if scope=='global' else 0)
    monkeypatch.setattr(comm.subprocess,'Popen',start)
    comm.run(Path('/build/bin/test-quactlize-scheduler'),tmp_path/'results',wrapper_ab=True)
    result=json.loads((tmp_path/'results/summary.json').read_text())
    assert len(commands)==10
    assert commands[7][:4]==commands[8][:4]==commands[0]
    assert commands[9][:4]==commands[4]
    assert result['wrapper_verdict']=='GLOBAL_WRAPPER_CAUSAL_CANDIDATE_PASS'
    assert result['device_admission']=='PENDING'


@pytest.mark.parametrize('hot', [False, True])
@pytest.mark.parametrize('fault', [None, 'device', 'producer', 'receipt', 'count', 'miss'])
def test_tp2_cache_evidence_proves_new_process_reload(hot, fault):
    from tools.run_kpack_batched_bench import tp2_cache_evidence
    producer = 'CACHE' if hot else 'GPU'
    lines = [f'[quactlize-shard] tensor=w device={d} q=12 n=512 k=1024 experts=2 producer={producer}'
             for d in (0, 1)]
    receipt = '[kpack-cache] cache_uploads=2 resident_misses=0' if hot else '[kpack-cache] published: total_seconds=1'
    if fault == 'device': lines.pop()
    if fault == 'producer': lines[1] = lines[1].replace(producer, 'GPU' if hot else 'CACHE')
    if fault == 'receipt': receipt = ''
    if fault == 'count': receipt = receipt.replace('uploads=2', 'uploads=1') if hot else ''
    if fault == 'miss': receipt = receipt.replace('misses=0', 'misses=1') if hot else ''
    text = '\n'.join(lines + [receipt])
    if fault:
        with pytest.raises(ValueError): tp2_cache_evidence(text, hot)
    else:
        result = tp2_cache_evidence(text, hot)
        assert result['status'] == 'PASS' and result['mode'] == ('hot' if hot else 'cold')


@pytest.mark.parametrize('fault', [None, 'compiler-missing', 'compiler-not-executable', 'native-header', 'cuda-header'])
def test_tp2_sdk_precheck_uses_separate_native_and_cuda_headers(tmp_path, fault):
    sdk=tmp_path/'PPU_SDK'
    compiler=sdk/'bin/hgcc'
    native=sdk/'targets/x86_64-linux/include/hggc_runtime_api.h'
    cuda=sdk/'CUDA_SDK/targets/x86_64-linux/include/cuda_runtime_api.h'
    for path in (compiler,native,cuda): path.parent.mkdir(parents=True,exist_ok=True)
    (sdk/'include').symlink_to('targets/x86_64-linux/include',target_is_directory=True)
    (sdk/'CUDA_SDK/include').symlink_to('targets/x86_64-linux/include',target_is_directory=True)
    if fault!='compiler-missing':
        compiler.write_text('#!/bin/sh\nexit 0\n')
        compiler.chmod(0o644 if fault=='compiler-not-executable' else 0o755)
    if fault!='native-header':native.write_text('// Native PPU API\n')
    if fault!='cuda-header':cuda.write_text('// CUDA compatibility API\n')
    # The official layout does not put the native runtime API in CUDA_SDK.
    assert not (sdk/'CUDA_SDK/include/hggc_runtime_api.h').exists()
    script=(ROOT/'tools/run_kpack_tp2_box.sh').read_text()
    block='    export PPU_SDK='+script.split('    export PPU_SDK=',1)[1].split('    export LD_LIBRARY_PATH=',1)[0]
    result=subprocess.run(['bash','-euc',block],env=dict(os.environ,SDK=str(sdk)),text=True,capture_output=True)
    if fault:
        assert result.returncode!=0 and 'KPACK_TP2_SDK PASS' not in result.stdout
        expected=compiler if fault.startswith('compiler-') else sdk/('include/hggc_runtime_api.h' if fault=='native-header' else 'CUDA_SDK/include/cuda_runtime_api.h')
        assert str(expected) in result.stderr
    else:
        assert result.returncode==0,result.stderr
        assert 'KPACK_TP2_SDK PASS' in result.stdout


@pytest.mark.parametrize('fault',[None,'stage','benchmark','trace','empty-trace','binary','prepare','prepare-count'])
def test_model_extension_reuses_only_completed_unchanged_inputs(tmp_path,monkeypatch,fault):
    resume,previous,llama,sdk,bundle,build,_=successful_extension_fixture(tmp_path,monkeypatch)
    result=previous/'results'
    if fault=='stage':(result/'runner-status.txt').write_text('runner_rc=1 stage=model-trace\n')
    if fault in ('benchmark','trace','empty-trace'):
        path=result/('benchmark/results/status.json' if fault=='benchmark' else 'trace/status.json')
        path.write_text('[]' if fault=='empty-trace' else '[{"status":"FAIL"}]')
    if fault=='binary':(build/'bin/llama-server').write_text('different binary')
    if fault and fault.startswith('prepare'):
        pin_path=resume.ROOT/'tools/kpack_q4_model_artifact.json'
        pin=json.loads(pin_path.read_text());pin['prepare_integration_gate']=True
        pin_path.write_text(json.dumps(pin))
        (result/'prepare-integration').mkdir()
        (result/'prepare-integration/summary.json').write_text(json.dumps(dict(
            status='FAIL' if fault=='prepare' else 'PASS',expected=16,
            passed=15 if fault=='prepare-count' else 16,router_edge_cases=56)))
    if fault:
        with pytest.raises(ValueError):resume.inputs(previous,llama,sdk,performance_only=True,extend_models=True)
    else:
        actual,caller,gates=resume.inputs(previous,llama,sdk,performance_only=True,extend_models=True)
        assert actual==bundle and caller==build and len(gates)==8


@pytest.mark.parametrize('fault',[None,'workload','existing','alias','tensor','empty'])
def test_model_extension_changes_only_new_model_bindings(tmp_path,monkeypatch,fault):
    resume,previous,*_,source=successful_extension_fixture(tmp_path,monkeypatch)
    plan=json.loads(source.read_text())
    if fault=='workload':plan['prompts']=[1024]
    if fault=='existing':plan['models'][0]['name']='qwen35-35b-q4km'
    if fault=='alias':plan['models'][0]['path']=str(tmp_path/'original.gguf')
    if fault=='tensor':plan['models'][1]|=dict(split='tensor',devices='0,1',tensor_split='1,1')
    if fault=='empty':plan['models']=[]
    source.write_text(json.dumps(plan))
    original=(previous/'results/model-plan.json').read_bytes()
    if fault:
        with pytest.raises(ValueError):resume.extension_plan(previous,source,'1')
    else:
        resolved=resume.extension_plan(previous,source,'1')
        assert len(resolved['models'])==2 and all(m['devices']=='1' for m in resolved['models'])
        assert all(Path(m['path']).is_file() for m in resolved['models'])
    assert (previous/'results/model-plan.json').read_bytes()==original


@pytest.mark.parametrize('fault',[None,'command','shape','split'])
def test_original_device_override_is_bound_to_all_abba_commands(tmp_path,monkeypatch,fault):
    resume,previous,llama,sdk,*_=successful_extension_fixture(tmp_path,monkeypatch)
    path=previous/'results/benchmark/results/protocol.json'
    protocol=json.loads(path.read_text());model=protocol['plan']['models'][0]
    model['devices']='1'
    if fault=='shape':protocol['plan']['prompts']=[1024]
    if fault=='split':model['split']='tensor'
    path.write_text(json.dumps(protocol))
    directory=path.parent/model['name'];directory.mkdir()
    for label in ('0-reference','1-kpack','2-kpack','3-reference'):
        (directory/(label+'.command.json')).write_text(json.dumps(dict(
            devices='0' if fault=='command' and label=='2-kpack' else '1')))
    if fault:
        with pytest.raises(ValueError):resume.inputs(previous,llama,sdk,performance_only=True,extend_models=True)
    else:resume.inputs(previous,llama,sdk,performance_only=True,extend_models=True)


@pytest.mark.parametrize('failure',[False,True])
def test_extended_models_only_run_benchmark_and_traces_and_preserve_results(tmp_path,monkeypatch,failure):
    resume,previous,llama,sdk,bundle,build,source=successful_extension_fixture(tmp_path,monkeypatch)
    (sdk/'asight/bin').mkdir(parents=True)
    for name in ('asys','acu'):
        file=sdk/'asight/bin'/name;file.write_text('host fixture');file.chmod(0o755)
    for name in ('kpack-model-cache','kpack-model-jit-cache'):(tmp_path/name).mkdir()
    output=tmp_path/'extension';(output/'results').mkdir(parents=True)
    monkeypatch.delenv('CACHE_DIR',raising=False);monkeypatch.delenv('JIT_CACHE',raising=False)
    commands=[]
    def run(argv,log,env):
        commands.append((list(map(str,argv)),log,env))
        if failure and log.name=='benchmark.log':raise ValueError('device failure')
    monkeypatch.setattr(resume,'run',run)
    args=SimpleNamespace(previous=previous,llama=llama,sdk=sdk,device='1',corpus=tmp_path/'corpus',
                         performance_only=True,profile_acu=False,extend_plan=source)
    if failure:
        with pytest.raises(ValueError,match='failed phases: benchmark'):resume.continuation(args,output)
    else:resume.continuation(args,output)
    assert [log.name for _,log,_ in commands]==['benchmark.log','trace.log','paired-model-proof.log']
    assert '--selected-scope' in commands[-1][0]
    assert all('build_kpack' not in ' '.join(argv) and 'run_model_gemv_integration' not in ' '.join(argv)
               for argv,_,_ in commands)
    new=output/'results/model-plan.json'
    assert all(str(new) in argv for argv,_,_ in commands[:2])
    assert all(env['CUDA_VISIBLE_DEVICES']=='1' for _,_,env in commands)
    receipt=json.loads((output/'results/continuation.json').read_text())
    assert receipt['compile']=='NONE' and len(receipt['new_models'])==2
    assert (output/'results/prior/model-plan.json').read_bytes()==(previous/'results/model-plan.json').read_bytes()


@pytest.mark.parametrize('fault',[None,'count','index','duplicate','missing'])
def test_inventory_includes_later_gguf_shards_without_reading_weights(tmp_path,monkeypatch,fault):
    files=[fixture(tmp_path,f'weights-{i:05d}-of-00002.gguf') for i in (1,2)]
    seen=[]
    tensors=[dict(name='blk.0.attn_q.weight',qtype=8,dims_gguf=[2048,512]),
             dict(name='blk.1.ffn_down_exps.weight',qtype=12,dims_gguf=[2048,512,256])]
    def header(stream,path):
        seen.append(path);index=files.index(Path(path))
        return dict(metadata={'split.count':1 if fault=='count' else 2,
                              'split.no':0 if fault=='index' else index},
                    tensors=[tensors[0 if fault=='duplicate' else index]])
    monkeypatch.setattr(bench,'read_gguf_header',header)
    if fault=='missing':files[1].unlink()
    if fault:
        with pytest.raises(ValueError):bench.inventory(files[0])
    else:
        inv=bench.inventory(files[0])
        assert inv['eligible']==[tensor['name'] for tensor in tensors]
        assert inv['operators']==['dense','grouped'] and inv['shards']==2 and inv['header_only']
        assert inv['q8']==[tensors[0]['name']] and seen==list(map(str,files))


def trace_replay_fixture(tmp_path):
    diagnostic, previous, llama, sdk, build, bundle, cache, jit = [tmp_path / p for p in (
        'diagnostic', 'previous', 'llama', 'sdk', 'build', 'bundle',
        'cache/qwen35-35b-q4km', 'jit')]
    for path in (diagnostic/'diagnostic/results', previous/'results/numerical/qwen35-35b-q4km',
                 llama/'tests', sdk/'asight/bin', sdk/'bin', build/'bin', bundle, cache, jit):
        path.mkdir(parents=True)
    model = tmp_path / 'qwen35.gguf'
    model.write_bytes(b'GGUF')
    for path in (build/'bin/llama-server', sdk/'asight/bin/asys', sdk/'bin/hgobjdump',
                 llama/'tests/quactlize_native.py'):
        path.write_text('fixture')
        path.chmod(0o755)
    for path in (bundle/'manifest.json', previous/'results/bundle-manifest.json'):
        path.write_text('{}')
    values = {
        diagnostic/'diagnostic/results/caller-ci-build.json': dict(build=str(build)),
        diagnostic/'diagnostic/results/inputs.json': dict(previous=str(previous)),
        diagnostic/'diagnostic/results/environment.json': dict(PPU_SDK=str(sdk),
            CUDA_VISIBLE_DEVICES='0', QUACTLIZE_KPACK_EXECUTION=str(bundle),
            QUACTLIZE_KPACK_JIT_CACHE=str(jit)),
        previous/'results/model-plan.json': dict(models=[
            dict(name='qwen35-35b-q4km', path=str(model)),
            dict(name='qwen3-32b-q4km', path='/must-not-load.gguf')]),
        previous/'results/numerical/qwen35-35b-q4km/b1-kpack-reference.command.json':
            dict(argv=['/old/perplexity', '-m', str(model), '--kpack-cache', str(cache)]),
    }
    for path, value in values.items():
        path.write_text(json.dumps(value))
    return diagnostic, previous, llama, sdk, build, bundle


def test_model_asys_reuses_diagnostic_build_without_numerical_callback(tmp_path, monkeypatch):
    from tools import run_kpack_model_trace as trace
    diagnostic, previous, llama, sdk, build, bundle = trace_replay_fixture(tmp_path)
    for key in ('LLAMA_NUMERICAL_DEBUG', 'LLAMA_NUMERICAL_DUMP_DIR',
                'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION',
                'QUACTLIZE_KPACK_PREFILL_POLICY', 'LLAMA_ARG_BATCH'):
        monkeypatch.setenv(key, 'must-not-leak')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    args, model, env, prior = trace.inputs(diagnostic, 'qwen35-35b-q4km', llama, sdk)
    assert args.build == build and args.bundle == bundle and prior == previous
    assert model['name'] == 'qwen35-35b-q4km' and env['CUDA_VISIBLE_DEVICES'] == '3'
    assert env['LD_LIBRARY_PATH'].startswith(str(build/'bin') + ':')
    assert not any(k.startswith(('LLAMA_NUMERICAL_', 'LLAMA_ARG_')) for k in env)
    assert 'GGML_CUDA_DISABLE_GRAPHS' not in env and 'GGML_CUDA_DISABLE_FUSION' not in env
    assert 'QUACTLIZE_KPACK_PREFILL_POLICY' not in env
    with pytest.raises(ValueError, match='exactly one model'):
        trace.inputs(diagnostic, 'absent', llama, sdk)
    (bundle/'manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='bundle differs'):
        trace.inputs(diagnostic, 'qwen35-35b-q4km', llama, sdk)


@pytest.mark.parametrize('tp2', [False, True])
@pytest.mark.parametrize('fault', [None, 'binary', 'runtime', 'legacy', 'devices', 'model', 'cache'])
def test_model_asys_reuses_completed_benchmark_including_tp2(tmp_path, monkeypatch, tp2, fault):
    from tools import run_kpack_model_trace as trace
    _, previous, llama, sdk, bundle, build = trace_continuation_fixture(tmp_path, monkeypatch)
    results = previous / 'results'
    protocol_file = results / 'benchmark/results/protocol.json'
    protocol = json.loads(protocol_file.read_text())
    model = protocol['plan']['models'][0]
    model['path'] = str(fixture(tmp_path))
    if tp2:
        model.update(devices='0,1', split='tensor', tensor_split='1,1')
    protocol_file.write_text(json.dumps(protocol))
    for label in ('0-reference', '1-kpack', '2-kpack', '3-reference'):
        path = protocol_file.parent / model['name'] / (label + '.command.json')
        value = json.loads(path.read_text()); value['devices'] = model['devices']
        path.write_text(json.dumps(value))
    cache, jit, legacy = [tmp_path / name for name in ('cache/subject', 'jit', 'legacy')]
    for p in (cache, jit, legacy): p.mkdir(parents=True)
    for p in (cache / 'manifest.json', legacy / 'manifest.json', results / 'compatibility-bundle-manifest.json'):
        p.write_text('{}')
    before = results / 'trace/subject/reference.command.json'
    before.parent.mkdir(parents=True)
    before.write_text(json.dumps(dict(argv=['python', 'never-execute-saved-command',
        '--model', '/different.gguf' if fault == 'model' else model['path'],
        '--binary', str(build / 'bin/llama-server'), '--bundle', str(bundle),
        '--cache', str(cache), '--jit-cache', str(jit)])))
    monkeypatch.setenv('QUACTLIZE_PPU_BUNDLE', str(legacy))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3' if fault == 'devices' else model['devices'])
    monkeypatch.setenv('LLAMA_NUMERICAL_DEBUG', 'tensors')
    if fault == 'binary': (build / 'bin/llama-server').write_text('changed')
    if fault == 'runtime': (bundle / 'manifest.json').write_text('{"changed":true}')
    if fault == 'legacy': (legacy / 'manifest.json').write_text('{"changed":true}')
    if fault == 'cache': (cache / 'manifest.json').unlink()
    if fault:
        with pytest.raises(ValueError): trace.benchmark_inputs(previous, 'subject', llama, sdk)
    else:
        args, found, env, prior = trace.benchmark_inputs(previous, 'subject', llama, sdk)
        assert args.build == build and args.bundle == bundle and prior == previous
        assert found == model and args.cache == cache.parent and args.jit_cache == jit
        assert env['CUDA_VISIBLE_DEVICES'] == model['devices']
        assert env['QUACTLIZE_KPACK_COMPUTE'] == 'fp16'
        assert 'LLAMA_NUMERICAL_DEBUG' not in env


@pytest.mark.parametrize('kind', ['timeout', 'runtime', 'none', 'mixed'])
def test_model_asys_private_retry_is_only_for_session_creation(tmp_path, kind):
    from tools.run_kpack_model_trace import session_creation_failed
    if kind != 'none':
        folder = tmp_path / 'reference/asys-preflight'; folder.mkdir(parents=True)
        (folder / 'summary.json').write_text(json.dumps(dict(status='FAIL', attempts=[
            dict(session_creation_failure=kind in ('timeout', 'mixed')),
            dict(session_creation_failure=kind == 'timeout')])))
    assert session_creation_failed(tmp_path) == (kind == 'timeout')


@pytest.mark.parametrize('pid,mount_ns,pid_ns', [(15, 'new-mnt', 'new-pid'),
    (1, 'old-mnt', 'new-pid'), (1, 'new-mnt', 'old-pid')])
def test_private_profiler_cannot_mount_in_parent_namespace(monkeypatch, pid, mount_ns, pid_ns):
    from tools import kpack_asys_scope as scope
    monkeypatch.setattr(scope.os, 'getpid', lambda: pid)
    monkeypatch.setattr(scope.os, 'readlink', lambda path: mount_ns if path.endswith('mnt') else pid_ns)
    monkeypatch.setattr(scope.subprocess, 'run', lambda *a, **k: pytest.fail('must not mount'))
    args = SimpleNamespace(parent_mount='old-mnt', parent_pid='old-pid')
    with pytest.raises(ValueError, match='outside the private'): scope.enter(args)


def test_private_profiler_rejects_paths_hidden_by_its_tmp_mount(tmp_path):
    from tools import kpack_asys_scope as scope
    with pytest.raises(ValueError, match='outside /tmp'):
        scope.private_command(['/usr/bin/true'], tmp_path / 'profiler', [Path('/tmp/model.gguf')])
    assert not (tmp_path / 'profiler').exists()


def test_model_asys_recovers_in_private_scope_only_after_session_timeout(tmp_path, monkeypatch):
    from tools import run_kpack_model_trace as trace
    diagnostic, _, llama, sdk, _, _ = trace_replay_fixture(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(trace, 'verify', lambda bundle, **kwargs: {})
    monkeypatch.setattr(trace, 'inventory', lambda path: dict(eligible=['weight'], operators=['grouped']))
    calls = []
    def capture(args, model, output, inventory):
        calls.append('shared')
        folder = output / 'reference/asys-preflight'; folder.mkdir(parents=True)
        (folder / 'summary.json').write_text(json.dumps(dict(status='FAIL', attempts=[
            dict(session_creation_failure=True)])))
        raise ValueError('session timeout')
    def private(args, model, output, env):
        calls.append('private')
        assert 'LLAMA_NUMERICAL_DEBUG' not in env
        return output / 'private' / model['name']
    monkeypatch.setattr(trace, 'traces', capture)
    monkeypatch.setattr(trace, 'private_trace', private)
    monkeypatch.setattr(sys, 'argv', ['trace', '--diagnostic', str(diagnostic), '--llama', str(llama),
        '--sdk', str(sdk), '--result-root', str(tmp_path)])
    assert trace.main() == 0
    assert calls == ['shared', 'private']
    status, = tmp_path.glob('kpack-model-asys.*/results/status.json')
    assert json.loads(status.read_text())['reports'].endswith('/private/qwen35-35b-q4km')


@pytest.mark.parametrize('fail', [False, True])
def test_model_asys_archives_logs_not_reports_and_restores_environment(tmp_path, monkeypatch, fail):
    import tarfile
    from tools import run_kpack_model_trace as trace
    diagnostic, previous, llama, sdk, build, bundle = trace_replay_fixture(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('LLAMA_NUMERICAL_DEBUG', 'tensors')
    monkeypatch.setattr(trace, 'verify', lambda bundle, **kwargs: {})
    monkeypatch.setattr(trace, 'inventory', lambda path: dict(eligible=['weight'], operators=['grouped']))
    calls = []
    def capture(args, model, output, inventory):
        calls.append((args, model))
        assert 'LLAMA_NUMERICAL_DEBUG' not in os.environ
        for arm in ('reference', 'native'):
            folder = output / arm
            folder.mkdir()
            for name in ('proof.asysrep', 'proof.sqlite', 'kernel-times.json'):
                (folder/name).write_text('fixture')
        if fail:
            raise ValueError('mock capture failure')
    monkeypatch.setattr(trace, 'traces', capture)
    monkeypatch.setattr(sys, 'argv', ['trace', '--diagnostic', str(diagnostic), '--llama', str(llama),
        '--sdk', str(sdk), '--result-root', str(tmp_path)])
    if fail:
        with pytest.raises(ValueError, match='mock capture failure'):
            trace.main()
    else:
        assert trace.main() == 0
    assert len(calls) == 1 and calls[0][1]['name'] == 'qwen35-35b-q4km'
    assert os.environ['LLAMA_NUMERICAL_DEBUG'] == 'tensors'
    archive, = tmp_path.glob('kpack-model-asys.*.results.tgz')
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert 'results/native/kernel-times.json' in names
        assert not any(n.endswith(('.asysrep', '.sqlite')) for n in names)
        status = json.load(tar.extractfile('results/status.json'))
        assert status['status'] == ('FAIL' if fail else 'TRACE_PAIR_COMPLETE')
        assert status['accuracy_admission'] == 'NOT_RETESTED'
        assert status['performance_admission'] == 'NOT_ADMITTED_BY_PROFILER'


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
