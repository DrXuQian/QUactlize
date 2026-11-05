"""Host-only acceptance-runner tests; mocked execution is NOT GPU evidence."""
import json
from pathlib import Path
import tarfile

import pytest
from ci import run as ci_run
from tools import run_flashnext_box as box


def test_commands_cover_both_abis_and_real_graph_geometry(tmp_path):
    cmds = dict(box.commands(tmp_path/'source', tmp_path/'package', tmp_path/'results', Path('/sdk'), 3))
    assert tuple(cmds) == ('precheck','build','compat','serving','graph-m1','graph-m4','graph-m16')
    assert cmds['build'][:3] == ['bash',str(tmp_path/'source/build.sh'),'--library']
    assert 'tests/ppu/test_compat_library.py' in cmds['compat']
    assert str(tmp_path/'source/tests/serving/test_serving_device.py') in cmds['serving']
    for m in (1,4,16):
        command=cmds[f'graph-m{m}']
        assert command[command.index('--m')+1] == str(m)
        assert command[command.index('--k')+1] == '2560'
        assert command[command.index('--hidden')+1] == '640'
        assert command[command.index('--experts')+1] == '512'


@pytest.mark.parametrize('fault', ['empty','skipped','errors','failures'])
def test_skipped_and_partial_coverage_cannot_pass(fault):
    counts=dict(tests=1,passed=1,failures=0,errors=0,skipped=0)
    if fault=='empty': counts['tests']=0
    else: counts[fault]=1
    with pytest.raises(ValueError,match='incomplete'):box.check_counts(counts)


@pytest.mark.parametrize('failure', [None,'checkout','precheck','build','compat','serving','graph-m4'])
def test_fail_fast_and_always_pack_only_results(tmp_path,monkeypatch,failure):
    output=tmp_path/'run';output.mkdir();repo=tmp_path/'repo';repo.mkdir()
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.setattr(box.tempfile,'mkdtemp',lambda **kw:str(output))
    monkeypatch.setattr(box,'git',lambda root,*a: str(repo) if '--show-toplevel' in a else 'a'*40)
    def checkout(repo,source,revision,log,env):
        log.write_text('checkout log')
        if failure=='checkout':raise ValueError('checkout failed')
        source.mkdir();(source/'private-source').write_text('do not pack')
        return 'b'*40
    monkeypatch.setattr(box,'checkout',checkout)
    stages=[]
    def run(command,log,env,timeout):
        stage=log.stem;stages.append(stage);log.write_text(f'{stage} diagnostics\n')
        assert env['CUDA_VISIBLE_DEVICES']=='2'
        assert 'PYTEST_ADDOPTS' not in env
        if stage=='build':
            p=output/'package';p.mkdir();(p/'manifest.json').write_text('{}')
            (p/'libncp_quactlize.so').write_bytes(b'host fixture only, not a library')
            p=output/'package.build';p.mkdir()
            (p/'compile.log').write_text('compiler diagnostics')
            (p/'compile.o').write_bytes(b'never pack')
        if stage in ('compat','serving'):
            path=output/'results'/(stage+'.xml')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('<testsuite><testcase name="host-runner-fixture"/></testsuite>')
        return 9 if stage==failure else 0
    monkeypatch.setattr(ci_run,'run_command',run)
    monkeypatch.setenv('PYTEST_ADDOPTS','-k skip_everything')
    assert box.main(['--ref','HEAD','--device','2'])==int(failure is not None)
    summary=json.loads((output/'results/summary.json').read_text())
    assert summary['status']==('FAIL' if failure else 'PASS')
    if failure: assert summary['failed_stage']==failure
    assert not failure or failure=='checkout' or stages[-1]==failure
    with tarfile.open(tmp_path/'run.review.tgz') as tar:
        names=tar.getnames()
    assert 'results/summary.json' in names
    assert all(name=='results' or name.startswith('results/') for name in names)
    assert not any(name.endswith(('.so','.o')) or 'private-source' in name for name in names)
    assert ('results/compiler/compile.log' in names)==(failure=='build')
