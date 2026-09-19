import json
from pathlib import Path

import pytest

from tools.run_selected_decode_box import compatibility, environment


def fixture(tmp_path):
    previous=tmp_path/'old-run';(previous/'results').mkdir(parents=True)
    receipt={'sdk':str(tmp_path/'sdk'),'libraries':['unchanged']}
    text=json.dumps(receipt)
    (previous/'results/compatibility-bundle-manifest.json').write_text(text)
    legacy=tmp_path/'q4-overlay'/'bundle';legacy.mkdir(parents=True)
    (legacy/'manifest.json').write_text(text)
    for name,file in (('llama','.aoneci/scripts/build.sh'),('caller','CMakeCache.txt'),
                      ('ncp','build/CMakeCache.txt'),('ncp','CMakeLists.txt')):
        path=tmp_path/name/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    build=dict(llama_worktree={'directory':str(tmp_path/'llama')},build=str(tmp_path/'caller'),
               ncp_directory=str(tmp_path/'ncp'))
    (previous/'results/caller-ci-build.json').write_text(json.dumps(build))
    return previous,legacy


def test_reuse_uses_receipts_and_exact_manifest_not_a_guessed_latest(tmp_path):
    previous,legacy=fixture(tmp_path)
    env,keys=environment(previous,{'NCP_LIB_DIR':str(tmp_path/'ncp')})
    assert env['QUACTLIZE_PPU_BUNDLE']==str(legacy)
    assert env['LLAMA_CI_BUILD_DIR']==str(tmp_path/'caller')
    assert env['PPU_SDK']==str(tmp_path/'sdk')
    assert env['CUDA_VISIBLE_DEVICES']=='0,1' and env['JOBS']=='192'
    assert 'PPU_SDK' in keys
    env,_=environment(previous,{'NCP_LIB_DIR':str(tmp_path/'ncp'),'PPU_SDK':'/explicit/sdk','JOBS':'64'})
    assert env['JOBS']=='64' and env['PPU_SDK']=='/explicit/sdk'


def test_missing_wrong_or_symlink_discovery_does_not_select_other_bundles(tmp_path):
    previous,legacy=fixture(tmp_path)
    bad=tmp_path/'wrong';bad.mkdir();(bad/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='differs'):compatibility(previous,str(bad))
    elsewhere=tmp_path/'results';legacy.rename(elsewhere)  # Excluded archived results.
    (tmp_path/'overlay-link').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError,match='not found'):compatibility(previous)
    assert compatibility(previous,str(elsewhere))==elsewhere


def test_absent_previous_build_requires_an_explicit_override(tmp_path):
    previous,legacy=fixture(tmp_path)
    with pytest.raises(ValueError,match='LLAMA_CI_BUILD_DIR'):
        environment(previous,{'LLAMA_CI_BUILD_DIR':str(tmp_path/'not-a-build')})
