"""Host-only tests: no claim of model execution, accuracy or device performance."""
import json
from pathlib import Path
import sqlite3
import struct
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


def test_two_complete_warmups_are_kept_in_raw_rows_not_steady_median(tmp_path):
    a=args(tmp_path);a.warmup_passes=2;a.trace_repeats=3
    rows=[record(t_pp=t) for t in (100,10,.4,.5,.6)]
    result=runner.parse_rows('\n'.join(map(json.dumps,rows)),2048,128,3,2)
    assert result['pp_total_ms']==500 and result['excluded_complete_passes']==2
    assert [r['excluded'] for r in result['rows']]==[True,True,False,False,False]
    assert result['rows'][0]['pp_total_ms']==100000
    for phase in ('perf','trace'):
        cmd=runner.command(a,tmp_path/'build',tmp_path/'model.gguf',phase)
        assert cmd[cmd.index('-npp')+1]==','.join(['2048']*5)
    with pytest.raises(ValueError,match='warmup'):
        runner.parse_rows('',2048,128,3,0)


def bf16_files(root, *, qtypes=(30,0), emit=2, duplicate=False, stem='model-BF16'):
    """Real GGUF headers only, not a runnable model or numeric evidence."""
    root.mkdir(exist_ok=True)
    def string(text):
        data=text.encode();return struct.pack('<Q',len(data))+data
    files=[]
    for index,qtype in enumerate(qtypes[:emit]):
        path=root/f'{stem}-{index+1:05d}-of-{len(qtypes):05d}.gguf'
        metadata={'split.no':index,'split.count':len(qtypes),'split.tensors.count':len(qtypes)}
        data=struct.pack('<4sIQQ',b'GGUF',3,1,len(metadata))
        for key,value in metadata.items():data+=string(key)+struct.pack('<II',4,value)
        data+=string('tensor' if duplicate else f'tensor-{index}')+struct.pack('<IQQIQ',2,32,64,qtype,0)
        path.write_bytes(data);files.append(path)
    return files


def test_explicit_bf16_reads_both_shards_without_hashing_weights(tmp_path):
    files=bf16_files(tmp_path)
    row=runner.bf16_inventory(files[0])
    assert [s['path'] for s in row['shards']]==list(map(str,files))
    assert row['tensor_count']==2 and row['tensor_types']=={30:1,0:1}
    assert not row['weights_hashed'] and not row['expected_quactlize_compute']
    with pytest.raises(ValueError,match='first-shard'):runner.bf16_inventory(files[1])


def test_explicit_quantized_model_validates_all_three_shards(tmp_path):
    files=bf16_files(tmp_path,qtypes=(12,14,8),emit=3,stem='Qwen3.5-122B-A10B-Q4_K_M')
    row=runner.model_inventory(files[0])
    assert [s['path'] for s in row['shards']]==list(map(str,files))
    assert row['tensor_count']==3 and row['tensor_types']=={12:1,14:1,8:1}
    assert not row['weights_hashed'] and row['expected_quactlize_compute']
    with pytest.raises(ValueError,match='first-shard'):runner.model_inventory(files[1])
    files[2].unlink()
    with pytest.raises(ValueError):runner.model_inventory(files[0])


@pytest.mark.parametrize('fault',['missing','quantized','no-bf16','duplicate'])
def test_bf16_control_rejects_incomplete_or_quantized_inventory(tmp_path,fault):
    files=bf16_files(tmp_path,emit=1 if fault=='missing' else 2,
        qtypes=(30,8) if fault=='quantized' else (1,0) if fault=='no-bf16' else (30,0),
        duplicate=fault=='duplicate')
    with pytest.raises(ValueError):runner.bf16_inventory(files[0])


@pytest.mark.parametrize('arm',['reference','quactlize'])
def test_bf16_does_not_require_quantized_plans_but_rejects_them(arm):
    result=runner.check_routes('ordinary native BF16 GEMM',arm,2048,require_kpack=False)
    assert result['plans']==[] and not result['no_regression_admitted']
    for marker in ('[quactlize-dense] m=2048','[quactlize-plan]','KPACK_JIT_RESOLVE parent=x'):
        with pytest.raises(ValueError,match='BF16 control'):
            runner.check_routes(marker,arm,2048,require_kpack=False)


def test_bf16_trace_checks_kernel_names_not_unused_host_strings(tmp_path):
    path=tmp_path/'bf16.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE StringIds(id,value)')
        db.executemany('INSERT INTO StringIds VALUES(?,?)',[(1,'native_bf16'),(2,'quactlize::gemm')])
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,demangledName)')
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,2,1)')
    row=runner.trace_inventory(path,native_only=True,warmup_passes=2)
    assert row['quactlize_kernel_names']==[] and row['excluded_complete_passes']==2
    with sqlite3.connect(path) as db:db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(3,4,2)')
    with pytest.raises(ValueError,match='Quactlize kernels'):runner.trace_inventory(path,native_only=True)


def test_bf16_run_accepts_no_loaded_quactlize_but_quantized_stays_strict(tmp_path,monkeypatch):
    a=args(tmp_path);a.bf16_model=tmp_path/'bf16.gguf';a.warmup_passes=2
    build=tmp_path/'build-ci';(build/'bin').mkdir(parents=True)
    images=[]
    for name in ('libggml-cuda.so','libllama.so'):
        path=build/'bin'/name;path.write_text('host fixture');images.append(str(path))
    monkeypatch.setattr(runner.shutil,'which',lambda _:None)
    def execute(argv,env,directory,label):
        rows='\n'.join(json.dumps(record()) for _ in range(5))
        plans='' if a.bf16_model else '[quactlize-dense] m=1\n[quactlize-dense] m=2048\n'
        (directory/'run.log').write_text(plans+rows)
        runner.save(directory/'process.json',dict(loaded_images=images))
    monkeypatch.setattr(runner,'execute',execute)
    result=runner.run_arm(a,a.bf16_model,'quactlize','perf',build,tmp_path/'bf16-run')
    assert result['excluded_complete_passes']==2 and result['measured_passes']==3
    a.bf16_model=None
    with pytest.raises(ValueError,match='public Quactlize library'):
        runner.run_arm(a,tmp_path/'q4.gguf','quactlize','perf',build,tmp_path/'q4-run')


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
        for flag,value in [('-c','2176'),('-t','16'),('-tb','32'),('--fit','off')]:
            assert cmd[cmd.index(flag)+1]==value
        assert '--mmap' in cmd
        assert '-ot' not in cmd and '--kpack-cache' not in cmd


def test_caller_defaults_keep_workload_but_do_not_override_runtime_defaults(tmp_path):
    a=args(tmp_path);a.caller_defaults=True;a.pp=5120;a.tg=512;a.warmup_passes=2
    for phase,count in [('perf',5),('trace',4)]:
        cmd=runner.command(a,tmp_path/'build-ci',tmp_path/'model.gguf',phase)
        assert cmd[cmd.index('-npp')+1]==','.join(['5120']*count)
        for flag,value in [('-ntg','512'),('-b','5120'),('-ub','5120'),('-ngl','all'),('-fa','1')]:
            assert cmd[cmd.index(flag)+1]==value
        assert not {'-c','-t','-tb','--fit','--mmap','-sm','-ot'} & set(cmd)
    a.tp=2
    cmd=runner.command(a,tmp_path/'build-ci',tmp_path/'model.gguf','perf')
    assert cmd[cmd.index('-sm')+1]=='tensor' and cmd[cmd.index('-ts')+1]=='1,1'


@pytest.mark.parametrize('device,tp', [('0,1',2),('7,5',2),('0,1,2,3',4),('7',1)])
def test_tp_device_mapping_is_explicit_and_ordered(device,tp):
    assert runner.device_ordinals(device,tp)==[int(v) for v in device.split(',')]


@pytest.mark.parametrize('device,tp', [('0',2),('0,1',1),('0,0',2),('',1),('0, 1',2),
                                      ('-1,0',2),('0,01',2),('0,',2),('0',0)])
def test_tp_device_mapping_rejects_wrong_count_and_duplicates(device,tp):
    with pytest.raises(ValueError):runner.device_ordinals(device,tp)


def test_tp2_command_and_both_environments_use_same_cards_and_tensor_split(tmp_path):
    a=args(tmp_path);a.tp=2;a.device='7,5';a.warmup_passes=2;a.tg=1
    for arm in ('reference','quactlize'):
        cmd=runner.command(a,tmp_path/arm,tmp_path/'first-shard.gguf','perf')
        assert cmd[cmd.index('-sm')+1]=='tensor' and cmd[cmd.index('-ts')+1]=='1,1'
        assert cmd[cmd.index('-npp')+1]==','.join(['2048']*5)
        assert cmd[cmd.index('-ntg')+1]=='1'
        assert runner.environment(a,tmp_path/arm,arm)['CUDA_VISIBLE_DEVICES']=='7,5'
        assert '--no-warmup' not in cmd and '-ot' not in cmd


@pytest.mark.parametrize('tp_supported',[False,True])
def test_tp2_snapshot_requires_split_flags_not_no_warmup(tmp_path,monkeypatch,tp_supported):
    a=args(tmp_path);a.tp=2
    build=tmp_path/'build';(build/'bin').mkdir(parents=True)
    (build/'bin/llama-batched-bench').write_bytes(b'host fixture only')
    def command(argv,env,log):
        log.write_text('--output-format -npp -ntg --fit'+
                       (' --split-mode --tensor-split' if tp_supported else ''))
        return 0
    monkeypatch.setattr(runner,'short_command',command)
    if tp_supported:
        assert runner.snapshot(a,'reference',build,tmp_path/'snapshot')
    else:
        with pytest.raises(ValueError,match='required flags'):
            runner.snapshot(a,'reference',build,tmp_path/'snapshot')


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
    old_policy=build/'bin/quactlize-runtime/dense-prefill.json'
    old_policy.parent.mkdir();old_policy.write_text('{"old_install":true}')
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
        assert (tmp_path/'snapshot/dense-prefill.json').read_text()=='{}'
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


def test_tp2_requires_decode_and_prefill_plans_on_each_logical_rank():
    lines=[f'[quactlize-dense] device={rank} m={m} q=8'
           for rank in (0,1) for m in (1,2048)]
    assert runner.check_routes('\n'.join(lines),'quactlize',2048,tp=2)['tp']==2
    for omitted in range(len(lines)):
        with pytest.raises(ValueError,match='TP logical rank'):
            runner.check_routes('\n'.join(lines[:omitted]+lines[omitted+1:]),'quactlize',2048,tp=2)
    runner.check_routes('ordinary native TP compute','reference',2048,tp=2)


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


def test_tp_trace_requires_both_gpus_but_does_not_assume_device_id_numbering(tmp_path):
    path=tmp_path/'tp.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,name)')
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,2,5,'native')")
    with pytest.raises(ValueError,match='TP GPU coverage'):runner.trace_inventory(path,tp=2)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(2,3,7,'quactlize::dense')")
    result=runner.trace_inventory(path,tp=2,warmup_passes=2)
    assert result['gpu_device_ids']==[5,7] and result['tp']==2
    assert result['quactlize_kernel_names']==['quactlize::dense']
    assert result['excluded_complete_passes']==2 and not result['no_regression_admitted']


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
                 llama / 'build-ci'):
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


def test_bf16_orchestrator_uses_requested_builds_and_model_only(tmp_path,monkeypatch):
    llama,sdk,models,output=[tmp_path/s for s in ('llama','sdk','models','output')]
    for path in (llama,sdk,models,output,llama/'build-ci-ppu-dev',llama/'build-ci'):
        path.mkdir(exist_ok=True)
    model=bf16_files(models)[0]
    asys=sdk/'asight/bin/asys';asys.parent.mkdir(parents=True);asys.touch();asys.chmod(0o700)
    monkeypatch.setattr(runner,'snapshot',lambda *a:{})
    calls=[]
    def run(a,model,arm,phase,build,directory):
        directory.mkdir(parents=True);env=runner.environment(a,build,arm)
        calls.append((model,arm,phase,build.name,env['GGML_QUACTLIZE_DENSE']))
        assert a.warmup_passes==2 and a.pp==2048 and a.tg==1 and a.bf16_model
        assert env['CUDA_VISIBLE_DEVICES']=='0'
        return dict(pp_total_ms=123,tpot_ms=4,profiler_attached=phase=='trace')
    monkeypatch.setattr(runner,'run_arm',run)
    monkeypatch.setattr(sys,'argv',['run_dense_model_ab.py','--llama-dir',str(llama),
        '--sdk',str(sdk),'--bf16-model',str(model),'--quactlize-build',str(llama/'build-ci'),
        '--output-root',str(output),'--device','0','--idle-confirmed',
        '--warmup-passes','2','--tg','1','--trace-repeats','3'])
    assert runner.main()==0
    assert len(calls)==4 and {v[0] for v in calls}=={model}
    assert [(c[1],c[2],c[3],c[4]) for c in calls]==[
        ('reference','perf','build-ci-ppu-dev','0'),('quactlize','perf','build-ci','1'),
        ('quactlize','trace','build-ci','1'),('reference','trace','build-ci-ppu-dev','0')]
    with tarfile.open(next(output.glob('*.review.tgz'))) as tar:
        summary=json.load(tar.extractfile('summary.json'))
    assert list(summary['results'])==['bf16-control'] and summary['warmup_passes']==2
    assert summary['scope']=='BF16_NATIVE_COMPUTE_BUILD_CONTROL_NOT_QUANTIZED_KERNEL_AB'


@pytest.mark.parametrize('compare_packages',[False,True])
def test_same_build_control_runs_one_quantized_model(tmp_path,monkeypatch,compare_packages):
    llama,sdk,models,output=[tmp_path/s for s in ('llama','sdk','models','output')]
    for path in (llama,sdk,models,output,llama/'build-ci'):
        path.mkdir(exist_ok=True)
    model=bf16_files(models,qtypes=(12,),emit=1,stem='Qwen3-32B-Q4_K_M')[0]
    package=tmp_path/'fresh';package.mkdir()
    asys=sdk/'asight/bin/asys';asys.parent.mkdir(parents=True);asys.touch();asys.chmod(0o700)
    monkeypatch.setenv('LLAMA_ARG_CTX_SIZE','64')
    monkeypatch.setattr(runner,'snapshot',lambda *a:{})
    calls=[]
    def run(a,requested,arm,phase,build,directory):
        directory.mkdir(parents=True)
        env=runner.environment(a,build,arm);cmd=runner.command(a,build,requested,phase)
        assert requested==model and a.pp==5120 and a.tg==512 and a.caller_defaults
        assert a.warmup_passes==2 and a.device=='7' and build==llama/'build-ci'
        assert not {'-c','-t','-tb','--fit','--mmap'} & set(cmd)
        assert 'LLAMA_ARG_CTX_SIZE' not in env
        if compare_packages:
            assert env['GGML_QUACTLIZE_DENSE']=='1'
            expected=package if arm=='quactlize' else build/'bin'
            assert env['QUACTLIZE_PPU_PACK_LIBRARY']==str(expected/'libquactlize_ppu.so')
        calls.append((arm,phase,cmd[0],env['GGML_QUACTLIZE_DENSE'],env['LD_LIBRARY_PATH']))
        return dict(pp_total_ms=123,tpot_ms=4,profiler_attached=phase=='trace')
    monkeypatch.setattr(runner,'run_arm',run)
    argv=['run_dense_model_ab.py','--llama-dir',str(llama),
        '--reference-build',str(llama/'build-ci'),'--quactlize-build',str(llama/'build-ci'),
        '--sdk',str(sdk),'--model-file',str(model),'--output-root',str(output),
        '--device','7','--idle-confirmed','--warmup-passes','2','--pp','5120','--tg','512',
        '--caller-defaults']
    if compare_packages:argv+=['--compare-quactlize-packages','--quactlize-package',str(package)]
    monkeypatch.setattr(sys,'argv',argv)
    assert runner.main()==0
    assert [(v[0],v[1],v[3]) for v in calls]==[
        ('reference','perf','1' if compare_packages else '0'),('quactlize','perf','1'),
        ('quactlize','trace','1'),('reference','trace','1' if compare_packages else '0')]
    assert len({v[2] for v in calls})==1
    assert len({v[4] for v in calls})==(2 if compare_packages else 1)
    with tarfile.open(next(output.glob('*.review.tgz'))) as tar:
        summary=json.load(tar.extractfile('summary.json'))
    assert list(summary['results'])==['explicit-model']
    assert summary['caller_control']==('SAME_BUILD_TWO_QUACTLIZE_PACKAGES' if compare_packages else 'SAME_BUILD_DENSE_ON_OFF')
    assert summary['caller_defaults'] and summary['compare_quactlize_packages']==compare_packages
    if compare_packages:assert summary['scope']=='QUACTLIZE_PACKAGE_AB_NOT_NATIVE_AB'
    assert summary['split_mode']=='binary-default'
    assert not summary['no_regression_admitted'] and not summary['correctness_admitted']


@pytest.mark.parametrize('arm',['reference','quactlize'])
@pytest.mark.parametrize('fault',[None,'wrong-library','mixed-runtime'])
def test_package_comparison_checks_both_arms_plans_and_loaded_images(tmp_path,monkeypatch,arm,fault):
    a=args(tmp_path);a.compare_quactlize_packages=True;a.warmup_passes=2
    a.pp=5120;a.tg=512;a.quactlize_package=tmp_path/'fresh'
    build=tmp_path/'build-ci';(build/'bin').mkdir(parents=True)
    images=[]
    for name in ('libggml-cuda.so','libllama.so'):
        path=build/'bin'/name;path.write_text('host fixture');images.append(str(path))
    installed=build/'bin';fresh=a.quactlize_package;fresh.mkdir()
    for root in (installed,fresh):
        (root/'libquactlize_ppu.so').write_text('public fixture')
        (root/'quactlize-runtime').mkdir()
        (root/'quactlize-runtime/libquactlize_ppu_prefill.so').write_text('private fixture')
    expected=installed if arm=='reference' else fresh
    other=fresh if arm=='reference' else installed
    images.append(str((other if fault=='wrong-library' else expected)/'libquactlize_ppu.so'))
    images.append(str((other if fault=='mixed-runtime' else expected)/'quactlize-runtime/libquactlize_ppu_prefill.so'))
    monkeypatch.setattr(runner.shutil,'which',lambda _:None)
    def execute(argv,env,directory,label):
        assert env['GGML_QUACTLIZE_DENSE']=='1'
        plans='[quactlize-dense] m=1\n[quactlize-dense] m=5120\n'
        row=record(pp=5120,tg=512)|dict(n_kv_max=5632)
        (directory/'run.log').write_text(plans+'\n'.join([json.dumps(row)]*5))
        runner.save(directory/'process.json',dict(loaded_images=images))
    monkeypatch.setattr(runner,'execute',execute)
    if fault:
        with pytest.raises(ValueError,match='public Quactlize library|mixed old/candidate'):
            runner.run_arm(a,tmp_path/'model.gguf',arm,'perf',build,tmp_path/'result')
    else:
        result=runner.run_arm(a,tmp_path/'model.gguf',arm,'perf',build,tmp_path/'result')
        assert result['measured_passes']==3 and result['excluded_complete_passes']==2


@pytest.mark.parametrize('fault',['missing-package','different-build','bf16'])
def test_package_comparison_rejects_invalid_control(tmp_path,monkeypatch,fault):
    for name in ('llama','sdk','package','reference','candidate'):(tmp_path/name).mkdir()
    argv=['run_dense_model_ab.py','--llama-dir',str(tmp_path/'llama'),'--sdk',str(tmp_path/'sdk'),
          '--reference-build',str(tmp_path/'reference'),
          '--quactlize-build',str(tmp_path/('candidate' if fault=='different-build' else 'reference')),
          '--compare-quactlize-packages','--idle-confirmed','--phases','perf']
    if fault!='missing-package':argv+=['--quactlize-package',str(tmp_path/'package')]
    if fault=='bf16':argv+=['--bf16-model',str(tmp_path/'bf16.gguf')]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(SystemExit) as exc:runner.main()
    assert exc.value.code==2


def test_tp2_orchestrator_uses_only_explicit_three_shard_model(tmp_path,monkeypatch):
    llama,sdk,models,output=[tmp_path/s for s in ('llama','sdk','models','output')]
    for path in (llama,sdk,models,output,llama/'build-ci-ppu-dev',llama/'build-ci'):
        path.mkdir(exist_ok=True)
    files=bf16_files(models,qtypes=(12,14,8),emit=3,stem='Qwen3.5-122B-A10B-Q4_K_M')
    asys=sdk/'asight/bin/asys';asys.parent.mkdir(parents=True);asys.touch();asys.chmod(0o700)
    monkeypatch.setattr(runner,'snapshot',lambda *a:{})
    calls=[]
    def run(a,model,arm,phase,build,directory):
        directory.mkdir(parents=True)
        assert a.tp==2 and a.device=='0,1' and not a.bf16_model and a.warmup_passes==2
        assert a.model_file==model==files[0] and a.pp==2048 and a.tg==1
        assert runner.command(a,build,model,phase)[runner.command(a,build,model,phase).index('-sm')+1]=='tensor'
        calls.append((arm,phase,build.name))
        return dict(pp_total_ms=123,tpot_ms=4,profiler_attached=phase=='trace')
    monkeypatch.setattr(runner,'run_arm',run)
    monkeypatch.setattr(sys,'argv',['run_dense_model_ab.py','--llama-dir',str(llama),
        '--sdk',str(sdk),'--model-file',str(files[0]),'--output-root',str(output),
        '--device','0,1','--tp','2','--idle-confirmed','--warmup-passes','2',
        '--tg','1','--trace-repeats','3'])
    assert runner.main()==0
    assert calls==[('reference','perf','build-ci-ppu-dev'),('quactlize','perf','build-ci'),
                   ('quactlize','trace','build-ci'),('reference','trace','build-ci-ppu-dev')]
    with tarfile.open(next(output.glob('*.review.tgz'))) as tar:
        summary=json.load(tar.extractfile('summary.json'))
        inventory=json.load(tar.extractfile('explicit-model.inventory.json'))
    assert list(summary['results'])==['explicit-model']
    assert summary['tp']==2 and summary['physical_devices']==[0,1]
    assert summary['split_mode']=='tensor' and summary['tensor_split']==[1,1]
    assert summary['warmup_passes']==2 and len(inventory['shards'])==3
    assert not summary['no_regression_admitted'] and not summary['correctness_admitted']
