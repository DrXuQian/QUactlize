"""Host-only orchestration tests; mocked captures do not admit PPU execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_kpack_reference_trace.sh"
STUB = r'''
import json, os, sys
from pathlib import Path
if Path(sys.argv[0]).name == 'git':
    print('a' * 40)
elif '--help' in sys.argv:
    print('--proof-arm --proof-tokens')
else:
    def option(key): return sys.argv[sys.argv.index(key)+1]
    arm = option('--proof-arm')
    with open(os.environ['TRACE_STEPS'], 'a') as stream:
        stream.write(json.dumps(dict(arm=arm, argv=sys.argv[1:],
            policy=os.environ.get('QUACTLIZE_KPACK_GEMV_POLICY'))) + '\n')
    assert '--proof-only' in sys.argv
    assert json.loads(Path(option('--proof-tokens')).read_text()) == {'2048': list(range(2048))}
    assert 'GGML_CUDA_DISABLE_GRAPHS' not in os.environ
    assert 'GGML_CUDA_DISABLE_FUSION' not in os.environ
    out = Path(option('--output'))
    (out / 'proof-request').mkdir(parents=True)
    for name in ('proof.asysrep', 'proof.sqlite', 'kernel-times.json'):
        (out / name).write_text('fixture')
    fault = os.environ.get('TRACE_FAULT')
    if fault == arm + '-failure': sys.exit(7)
    proof = dict(input_tokens_sha256='tokens', request_sha256='request', prompt_tokens=2048,
        generated_tokens=16, prefill_token_batch=2048, response_sha256='response',
        kernel_execution='REFERENCE_COMPUTE_OBSERVED' if arm == 'reference' else 'PASS_SHORT_REQUEST')
    props = dict(model_path=option('--model'), build_info='fixture-build')
    if arm == 'native':
        if fault == 'request-mismatch': proof['request_sha256'] = 'different'
        if fault == 'build-mismatch': props['build_info'] = 'different'
        if fault == 'continuation-differs': proof['response_sha256'] = 'different'
    (out/'proof.json').write_text(json.dumps(proof))
    (out/'proof-request'/f'0-{arm}.props.json').write_text(json.dumps(props))
'''


@pytest.mark.parametrize("fault,trace_arm", [(fault, "both") for fault in (
    "", "reference-failure", "request-mismatch", "build-mismatch", "continuation-differs")]
    + [("", "native"), ("native-failure", "native"), ("", "reference")])
def test_matched_trace_reuses_inputs_and_preserves_failures(tmp_path, fault, trace_arm):
    repo, llama, sdk, prior, legacy, commands, results = [tmp_path / name for name in (
        "repo", "llama", "sdk", "prior run", "legacy", "bin", "results")]
    for p in (repo / "tools", llama / "tests", sdk / "bin", sdk / "asight/bin",
              prior / "results/model-proof/proof-request", prior / "results/q8_simt",
              prior / "cache", prior / "jit-cache", legacy, commands, results):
        p.mkdir(parents=True)
    runner = repo / "tools" / RUNNER.name
    shutil.copy2(RUNNER, runner)
    (repo / "tools/verify_kpack_dispatch.py").write_text("print('mock verify')\n")
    (llama / "tests/quactlize_native.py").write_text(STUB)
    for p in (commands / "git", commands / "llama-server", sdk / "bin/hgobjdump", sdk / "asight/bin/asys"):
        p.write_text("#!" + sys.executable + "\n" + STUB)
        p.chmod(0o755)
    (sdk / "envsetup.sh").write_text('export PATH="$PPU_SDK/bin:$PATH"\n')
    # The selected Python must survive SDK setup changing PATH.
    (sdk / "bin/python3").write_text("#!/bin/sh\nexit 88\n")
    (sdk / "bin/python3").chmod(0o755)
    model = tmp_path / "model.gguf"
    pack = tmp_path / "pack.so"
    for p in (model, pack, legacy / "manifest.json", prior / "results/model-trace-inventory.json",
              prior / "results/q8_simt/gemv-policy.tsv"):
        p.write_text("fixture")
    request = prior / "results/model-proof/proof-request"
    (request / "input-tokens.json").write_text(json.dumps({"2048": list(range(2048))}))
    command = [str(sdk / "asight/bin/asys"), "launch", str(commands / "llama-server"),
               "-m", str(model), "--kpack-cache", str(prior / "cache")]
    (request / "0-native.command.json").write_text(json.dumps(command))
    (request.parent / "protocol.json").write_text(json.dumps(dict(
        proof_parameters=dict(prompts=[2048], generate=16), jit_cache=str(prior / "jit-cache"))))
    env = {k: v for k, v in os.environ.items() if not k.startswith("QUACTLIZE_")}
    env.update(LLAMA_DIR=str(llama), PPU_SDK=str(sdk), PYTHON=sys.executable,
               QUACTLIZE_PPU_BUNDLE=str(legacy), QUACTLIZE_PPU_PACK_LIBRARY=str(pack),
               PATH=str(commands) + os.pathsep + env["PATH"], RESULT_ROOT=str(results),
               CUDA_VISIBLE_DEVICES="0", TRACE_STEPS=str(tmp_path / "steps"), TRACE_FAULT=fault,
               TRACE_ARM=trace_arm,
               GGML_CUDA_DISABLE_GRAPHS="1", GGML_CUDA_DISABLE_FUSION="1")
    run = subprocess.run(["bash", str(runner), str(prior)], env=env, text=True,
                         capture_output=True, timeout=30)
    success = fault in ("", "continuation-differs")
    assert (run.returncode == 0) == success, run.stdout + run.stderr
    steps = [json.loads(line) for line in (tmp_path / "steps").read_text().splitlines()]
    expected_arms = ["reference", "native"] if trace_arm == "both" else [trace_arm]
    assert [s["arm"] for s in steps] == expected_arms
    for step in steps:
        argv = step["argv"]
        assert argv[argv.index("--binary")+1] == str(commands / "llama-server")
        assert argv[argv.index("--model")+1] == str(model)
        assert argv[argv.index("--proof-tokens")+1] == str(request / "input-tokens.json")
        assert argv[argv.index("--jit-cache")+1] == str(prior / "jit-cache")
        assert step["policy"] == str(prior / "results/q8_simt/gemv-policy.tsv")
    output, = [p / "results" for p in results.iterdir() if p.is_dir()]
    archive, = results.glob("*.results.tgz")
    with tarfile.open(archive) as tf:
        names = tf.getnames()
        assert "results/runner-status.txt" in names
        assert f"results/{expected_arms[-1]}/kernel-times.json" in names
        assert not any(n.endswith((".asysrep", ".sqlite")) for n in names)
    for arm in expected_arms:
        assert (output / arm / "proof.asysrep").is_file()
    if trace_arm != "both":
        assert not (output / ("reference" if trace_arm == "native" else "native")).exists()
    assert (output / "summary.json").exists() == success
    if success:
        summary = json.loads((output / "summary.json").read_text())
        assert summary["performance_admission"] == "NOT_MEASURED_BY_PROFILER"
        if trace_arm == "both":
            assert summary["same_generated_text"] == (fault != "continuation-differs")
            assert summary["pair_comparison"] == "PASS"
        else:
            assert summary["status"] == "TRACE_ARM_COMPLETE"
            assert summary["pair_comparison"] == "NOT_RUN"


def test_trace_help_and_missing_input_do_not_start_work(tmp_path):
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    help_result = subprocess.run(["bash", str(RUNNER), "--help"], text=True, capture_output=True, check=True)
    assert "No build or config sweep" in help_result.stdout
    assert "\\x27" not in help_result.stdout
    missing = subprocess.run(["bash", str(RUNNER), str(tmp_path / "missing")], text=True, capture_output=True)
    assert missing.returncode != 0
    assert "calling shell preserved" in missing.stdout
