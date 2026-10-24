"""Host-only tests: no claim of model execution, accuracy or device performance."""
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_dense_model_ab as runner


def args(tmp_path):
    return SimpleNamespace(sdk=tmp_path / 'sdk', device='7', pp=2048, tg=128,
                           repeats=3, trace_repeats=2, threads=16, threads_batch=32,
                           jit_cache=None)


def record(pp=2048, tg=128, t_pp=0.4, t_tg=1.024):
    return dict(n_kv_max=2304, n_batch=pp, n_ubatch=pp, flash_attn=1,
                is_pp_shared=0, pp=pp, tg=tg, pl=1, n_kv=pp+tg,
                t_pp=t_pp, t_tg=t_tg, speed_pp=pp/t_pp, speed_tg=tg/t_tg)


def test_complete_warmup_is_excluded_not_just_initial_64_tokens():
    rows = [record(t_pp=100, t_tg=100), record(), record(t_pp=0.6, t_tg=1.28), record()]
    text = '\n'.join('1.0 I ' + json.dumps(r) for r in rows)
    result = runner.parse_rows(text, 2048, 128, 3)
    assert result['pp_total_ms'] == 400
    assert result['tpot_ms'] == 8
    assert result['measured_passes'] == 3
    assert [r['excluded'] for r in result['rows']] == [True, False, False, False]


@pytest.mark.parametrize('change', [dict(pl=2), dict(n_ubatch=512), dict(pp=1024),
                                   dict(t_pp=float('nan')), dict(speed_tg=1), dict(n_kv_max=2048)])
def test_bad_workload_or_timers_reject(change):
    row = record() | change
    with pytest.raises(ValueError):
        runner.parse_rows('\n'.join([json.dumps(row)] * 3), 2048, 128, 2)


def test_missing_and_duplicate_rows_reject():
    for count in (0, 1, 2, 4):
        with pytest.raises(ValueError):
            runner.parse_rows('\n'.join([json.dumps(record())] * count), 2048, 128, 2)


def test_no_cross_build_or_old_poc_environment(tmp_path, monkeypatch):
    for k, v in dict(QUACTLIZE_KPACK_EXECUTION='/old', QUACTLIZE_PPU_PACK_LIBRARY='/wrong.so',
                     QUACTLIZE_KPACK_COMPUTE='bf16', LLAMA_ARG_MODEL='wrong.gguf',
                     GGML_CUDA_DISABLE_GRAPHS='1', GGML_NCP_MOE_LIB='/wrong.so').items():
        monkeypatch.setenv(k, v)
    a = args(tmp_path)
    for arm in ('reference', 'quactlize'):
        build = tmp_path / arm
        env = runner.environment(a, build, arm)
        assert env['LD_LIBRARY_PATH'].split(':')[0] == str(build / 'bin')
        assert env['CUDA_VISIBLE_DEVICES'] == '7'
        assert env['GGML_QUACTLIZE_DENSE'] == str(int(arm == 'quactlize'))
        assert 'GGML_CUDA_DISABLE_GRAPHS' not in env
        assert 'QUACTLIZE_KPACK_EXECUTION' not in env
        assert 'LLAMA_ARG_MODEL' not in env
        assert 'GGML_NCP_MOE_LIB' not in env
        assert 'QUACTLIZE_KPACK_COMPUTE' not in env
        if arm == 'quactlize':
            assert env['QUACTLIZE_PPU_PACK_LIBRARY'] == str(build / 'bin/libquactlize_ppu.so')
        else:
            assert 'QUACTLIZE_PPU_PACK_LIBRARY' not in env


def test_workload_and_binary_are_explicit(tmp_path):
    a = args(tmp_path)
    for phase, count in [('perf', 4), ('trace', 3)]:
        cmd = runner.command(a, tmp_path / 'build', tmp_path / 'model.gguf', phase)
        assert cmd[0] == str(tmp_path / 'build/bin/llama-batched-bench')
        assert cmd[cmd.index('-npp') + 1] == ','.join(['2048'] * count)
        assert cmd[cmd.index('-npl') + 1] == '1'
        assert cmd[cmd.index('-sm') + 1] == 'none'
        assert '-ot' not in cmd and '--kpack-cache' not in cmd


def test_candidate_package_does_not_override_reference_or_caller(tmp_path):
    a=args(tmp_path); a.quactlize_package=tmp_path/'fresh-package';a.cublas_accumulator='f32'
    ref=runner.environment(a,tmp_path/'reference','reference')
    candidate=runner.environment(a,tmp_path/'candidate','quactlize')
    assert 'QUACTLIZE_PPU_PACK_LIBRARY' not in ref
    assert ref['GGML_CUDA_CUBLAS_ACCUMULATOR']==candidate['GGML_CUDA_CUBLAS_ACCUMULATOR']=='f32'
    assert str(a.quactlize_package) not in ref['LD_LIBRARY_PATH']
    assert candidate['QUACTLIZE_PPU_PACK_LIBRARY']==str(a.quactlize_package/'libquactlize_ppu.so')
    assert candidate['LD_LIBRARY_PATH'].split(':')[:3]==[
        str(a.quactlize_package),str(a.quactlize_package/'quactlize-runtime'),str(tmp_path/'candidate/bin')]
    assert runner.command(a,tmp_path/'candidate',tmp_path/'model.gguf','perf')[0]==str(tmp_path/'candidate/bin/llama-batched-bench')


@pytest.mark.parametrize('wrong_sdk',[False,True])
def test_fresh_package_snapshot_binds_private_payload_not_old_install(tmp_path,monkeypatch,wrong_sdk):
    from tools import build_ppu_library
    from quactlize.runtime import native
    a=args(tmp_path);a.quactlize_package=tmp_path/'fresh'
    build=tmp_path/'caller';(build/'bin').mkdir(parents=True)
    (build/'bin/llama-batched-bench').write_bytes(b'host fixture, not an executable')
    files=['libquactlize_ppu.so','quactlize-runtime/libquactlize_ppu_prefill.so',
           'quactlize-runtime/dense-prefill.json','quactlize-runtime/jit/kernel.hpp']
    for name in [*files,'manifest.json']:
        path=a.quactlize_package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    manifest=dict(sdk='a',payloads={name:'not-a-real-device-package' for name in files})
    monkeypatch.setattr(build_ppu_library,'verify',lambda path:manifest)
    monkeypatch.setattr(native,'sdk_identity',lambda path:'b' if wrong_sdk else 'a')
    def command(argv,env,log):
        assert env['QUACTLIZE_PPU_PACK_LIBRARY']==str(a.quactlize_package/'libquactlize_ppu.so')
        log.write_text('--output-format -npp -ntg --fit');return 0
    monkeypatch.setattr(runner,'short_command',command)
    if wrong_sdk:
        with pytest.raises(ValueError,match='SDK differs'):runner.snapshot(a,'quactlize',build,tmp_path/'snapshot')
    else:
        receipt=runner.snapshot(a,'quactlize',build,tmp_path/'snapshot')
        assert {str(a.quactlize_package/name) for name in files}<=receipt.keys()
        runner.check_unchanged(receipt)
        (a.quactlize_package/files[1]).write_bytes(b'changed helper')
        with pytest.raises(ValueError,match='changed during'):runner.check_unchanged(receipt)


def test_route_requires_prefill_and_decode_and_leaves_moe_native():
    text = ('[quactlize-dense] tensor=blk.0.attn.weight m=1 q=8 config=simt\n'
            '[quactlize-dense] tensor=blk.0.attn.weight m=2048 q=8 config=tc\n')
    out = runner.check_routes(text, 'quactlize', 2048)
    assert len(out['plans']) == 2 and not out['no_regression_admitted']
    assert out['scope'] == 'DENSE_ONLY_MOE_REMAINS_NATIVE'
    with pytest.raises(ValueError):
        runner.check_routes(text, 'reference', 2048)
    with pytest.raises(ValueError):
        runner.check_routes(text.splitlines()[0], 'quactlize', 2048)
    runner.check_routes('mul_mat_vec_q', 'reference', 2048)


def test_sqlite_requires_gpu_not_host_timers(tmp_path):
    path = tmp_path / 'trace.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end)')
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,2)')
    with pytest.raises(ValueError, match='no GPU kernel activities'):
        runner.trace_inventory(path)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,demangledName)')
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,2,1)')
    result = runner.trace_inventory(path)
    assert result['kernel_tables']['CUPTI_ACTIVITY_KIND_KERNEL'] == 1
    assert result['exclude_first_complete_pp_tg_pass']
    assert not result['no_regression_admitted']


def test_review_only_includes_needed_evidence(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    for name in ('trace.sqlite', 'trace.asysrep', 'run.log', 'summary.json', 'weights.bin',
                 'libquactlize_ppu.so', 'core.123'):
        (run / name).write_text(name)
    (run / 'link.log').symlink_to(run / 'weights.bin')
    archive = runner.pack_review(run)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == {'trace.sqlite', 'run.log', 'summary.json'}


def test_changed_binary_is_not_a_valid_comparison(tmp_path):
    binary = tmp_path / 'binary'
    binary.write_bytes(b'one')
    receipt = {'binary': dict(realpath=str(binary), bytes=3, sha256=runner.digest(binary))}
    runner.check_unchanged(receipt)
    binary.write_bytes(b'two')
    with pytest.raises(ValueError, match='changed during comparison'):
        runner.check_unchanged(receipt)


def test_subprocess_failure_keeps_log_and_return_code(tmp_path):
    with pytest.raises(ValueError, match='rc=2'):
        runner.execute([sys.executable, '-c', 'print("intentional failure");raise SystemExit(2)'],
                       {}, tmp_path, 'host-test')
    assert 'intentional failure' in (tmp_path / 'run.log').read_text()
    assert json.loads((tmp_path / 'process.json').read_text())['rc'] == 2


@pytest.mark.parametrize('fail_arm', [None, 'quactlize'])
def test_three_model_orchestrator_continues_independent_arms(tmp_path, monkeypatch, fail_arm):
    llama, sdk, models, output = [tmp_path / s for s in ('llama', 'sdk', 'models', 'output')]
    for path in (llama, sdk, models, output, llama / 'build-ci-ppu-dev',
                 llama / 'build-ci-quactlize-dense-gemm'):
        path.mkdir(exist_ok=True)
    asys = sdk / 'asight/bin/asys'
    asys.parent.mkdir(parents=True)
    asys.write_text('#!/bin/sh\nexit 0\n')
    asys.chmod(0o700)
    for relative in runner.MODELS.values():
        path = models / relative
        path.parent.mkdir()
        path.write_bytes(b'not read by the header stub')
    monkeypatch.setattr(runner, 'read_gguf_header', lambda *a: dict(metadata={}, tensors=[]))
    monkeypatch.setattr(runner, 'snapshot', lambda *a: {})
    calls = []

    def run(a, model, arm, phase, build, directory):
        directory.mkdir(parents=True)
        calls.append((model.name, arm, phase, build.name))
        if arm == fail_arm:
            raise ValueError('injected failure')
        return dict(pp_total_ms=123, tpot_ms=4, profiler_attached=phase == 'trace')

    monkeypatch.setattr(runner, 'run_arm', run)
    monkeypatch.setattr(sys, 'argv', ['run_dense_model_ab.py', '--llama-dir', str(llama),
        '--sdk', str(sdk), '--model-root', str(models), '--output-root', str(output), '--idle-confirmed'])
    assert runner.main() == int(bool(fail_arm))
    assert len(calls) == 12
    assert [(r[1], r[2]) for r in calls[:4]] == [
        ('reference', 'perf'), ('quactlize', 'perf'), ('quactlize', 'trace'), ('reference', 'trace')]
    archives = list(output.glob('*.review.tgz'))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as tar:
        summary = json.load(tar.extractfile('summary.json'))
    assert len(summary['failures']) == (6 if fail_arm else 0)
    assert not summary['no_regression_admitted']
    assert len(summary['results']) == 3
