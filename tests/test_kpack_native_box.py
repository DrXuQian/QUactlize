"""CPU orchestration contracts; stubs do not admit device execution."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

STUB = r'''
import json, os, sys
from pathlib import Path
PATTERN = r"^(blk\.[0-9]+\.ffn_[a-z0-9_]+_exps\.weight|output\.weight)$"
if __name__ == "__main__":
    name = Path(sys.argv[0]).name
    with open(os.environ["STEPS"], "a") as stream:
        stream.write(json.dumps(dict(name=name, argv=sys.argv[1:],
            jit=os.environ.get("QUACTLIZE_KPACK_JIT_HELPER"),
            execution=os.environ.get("QUACTLIZE_KPACK_EXECUTION"))) + "\n")
    def option(flag):
        return Path(sys.argv[sys.argv.index(flag)+1])
    def gate(path):
        path.mkdir(parents=True)
        (path / "summary.json").write_text("{}")
    if name == "run_kpack_native_gate.py": gate(option("--output"))
    elif name == "reuse_kpack_native_gates.py": gate(option("--output") / "native-gate")
    elif name == "export_kpack_prefill_policy.py": option("--output").write_text("KPACK_PREFILL_POLICY_V2_PER_CALL\n")
    elif name == "kpack_jit.py" and sys.argv[1] == "plan":
        option("--output").mkdir()
        (option("--output") / "plan.json").write_text("{}")
    elif name == "kpack_jit.py" and sys.argv[1] == "prewarm": option("--receipt").write_text("{}")
    elif name == "quactlize_native.py": gate(option("--output"))
    elif name == "git" and "rev-parse" in sys.argv: print("a"*40)
'''


@pytest.mark.parametrize("jit,resume", [(True, False), (True, True), (False, False)])
def test_model_runner_uses_bound_python_and_keeps_device_tests_isolated(tmp_path, jit, resume):
    repo, llama, sdk, build, legacy, cache, commands, results = [
        tmp_path / p for p in ("repo", "llama", "sdk", "build", "legacy", "weights", "bin", "results")]
    for p in (repo / "tools", llama / "tests", sdk / "bin", sdk / "asight/bin",
              build / "bin", legacy, cache, commands, results):
        p.mkdir(parents=True)
    runner = repo / "tools/run_kpack_native_box.sh"
    shutil.copy2(ROOT / "tools/run_kpack_native_box.sh", runner)
    for name in ("verify_kpack_dispatch", "run_kpack_native_gate", "reuse_kpack_native_gates",
                 "export_kpack_prefill_policy", "kpack_jit"):
        (repo / "tools" / (name + ".py")).write_text(STUB)
    for name in ("quactlize_native", "test-quactlize-native"):
        (llama / "tests" / (name + ".py")).write_text(STUB)
    bundle = repo / "prebuilt/ppu0010" / ("kpack-jit-v2" if jit else "kpack-native-v1")
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps(dict(jit_required=jit, jit_source_contract="b"*64)))
    for p in (cache / "manifest.json", legacy / "manifest.json", tmp_path / "model.gguf", tmp_path / "pack.so"):
        p.write_text("fixture")
    for name in ("llama-server", "llama-completion", "libggml-cuda.so", "libllama.so"):
        (build / "bin" / name).write_text("fixture")
    for i in range(5):
        (legacy / f"libquactlize_ppu_fmt{i}.so").write_text("fixture")
    (build / "CMakeCache.txt").write_text("GGML_USE_PPU:BOOL=ON\nGGML_NCP_QUACTLIZE:BOOL=ON\n")
    for path in [commands / name for name in ("git", "cmake", "ctest", "c++filt")] + [sdk / "bin/hgobjdump", sdk / "asight/bin/asys"]:
        path.write_text("#!" + sys.executable + "\n" + STUB)
        path.chmod(0o755)
    # SDK setup may prepend a different Python. The runner must retain the
    # interpreter selected before setup, for both the gate and JIT children.
    (sdk / "bin/python3").write_text("#!/bin/sh\nexit 88\n")
    (sdk / "bin/python3").chmod(0o755)
    (sdk / "envsetup.sh").write_text('export PATH="$PPU_SDK/bin:$PATH"\n')
    env = dict(os.environ, LLAMA_DIR=str(llama), MODEL=str(tmp_path / "model.gguf"),
               PPU_SDK=str(sdk), QUACTLIZE_PPU_BUNDLE=str(legacy),
               QUACTLIZE_PPU_PACK_LIBRARY=str(tmp_path / "pack.so"), CACHE_DIR=str(cache),
               BUILD_DIR=str(build), RESULT_ROOT=str(results), PYTHON=sys.executable,
               PATH=str(commands) + os.pathsep + os.environ["PATH"], STEPS=str(tmp_path / "steps"),
               CUDA_VISIBLE_DEVICES="0", RUN_GEMV_GATE="0")
    for name in ("QUACTLIZE_KPACK_EXECUTION", "RESUME_RUN", "JIT_CACHE", "ASYS"):
        env.pop(name, None)
    if not jit:
        env["QUACTLIZE_KPACK_EXECUTION"] = str(bundle)
    if resume:
        env["RESUME_RUN"] = str(tmp_path / "old")
    run = subprocess.run(["bash", str(runner)], env=env, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    steps = [json.loads(line) for line in (tmp_path / "steps").read_text().splitlines()]
    model = next(s for s in steps if s["name"] == "quactlize_native.py")
    assert model["execution"] == str(bundle)
    assert ("--jit-cache" in model["argv"]) == jit
    ctest = next(s for s in steps if s["name"] == "ctest")
    assert ctest["jit"] is None and ctest["execution"] is None
    gate_name = "reuse_kpack_native_gates.py" if resume else "run_kpack_native_gate.py"
    gate = next(s for s in steps if s["name"] == gate_name)
    assert ("--native-only" in gate["argv"]) == resume
    if not resume:
        assert ("--jit-cache" in gate["argv"]) == jit
    prewarm = [s for s in steps if s["name"] == "kpack_jit.py"]
    assert len(prewarm) == (2 if jit else 0)
    if jit:
        assert prewarm[0]["argv"][0] == "plan" and "--tensor-pattern" in prewarm[0]["argv"]
        assert prewarm[1]["argv"][0] == "prewarm" and "--source-contract" in prewarm[1]["argv"]
        assert steps.index(prewarm[1]) < steps.index(model)
    assert list(results.glob("*.results.tgz"))
