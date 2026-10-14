import pytest

from dev.strategy.provenance import source_identity, verify_unchanged, harness_identity


def test_source_guard_covers_uncommitted_new_files_and_evidence(tmp_path):
    directory=tmp_path/'dev/strategy'
    directory.mkdir(parents=True)
    first=directory/'rule.py'
    first.write_text('rule = 1\n')
    before=source_identity(tmp_path)
    assert verify_unchanged(before,tmp_path)==before
    first.write_text('rule = 2\n')
    with pytest.raises(ValueError,match='rule.py'):
        verify_unchanged(before,tmp_path)
    before=source_identity(tmp_path)
    (directory/'new.py').write_text('changed = True\n')
    with pytest.raises(ValueError,match='new.py'):
        verify_unchanged(before,tmp_path)
    before=source_identity(tmp_path)
    evidence=tmp_path/'policies'
    evidence.mkdir()
    (evidence/'board.json').write_text('{}')
    with pytest.raises(ValueError,match='board.json'):
        verify_unchanged(before,tmp_path)


def test_other_user_logs_are_not_rule_input(tmp_path):
    before=source_identity(tmp_path)
    (tmp_path/'unrelated.log').write_text('another job\n')
    assert verify_unchanged(before,tmp_path)==before


def test_resource_resume_cannot_hide_changed_actlize_inline_header(tmp_path):
    before=source_identity(tmp_path)
    path=tmp_path/'third_party/actlize/include/cute/traits.inl'
    path.parent.mkdir(parents=True);path.write_text('// different atom\n')
    with pytest.raises(ValueError,match='traits.inl'):verify_unchanged(before,tmp_path)


def test_harness_closure_includes_nested_fixture_and_oracle(tmp_path):
    before=harness_identity(tmp_path)
    for name in ('dev/smallm_closure/fixture.py','dev/bf16_compute/native.py','reference/gguf_kpack.py'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('changed = True\n')
        after=harness_identity(tmp_path)
        assert name in after['files'] and after!=before
        before=after
    (tmp_path/'another-job.log').write_text('unrelated\n')
    assert harness_identity(tmp_path)==before


def test_box_orchestrator_is_part_of_sealed_source(tmp_path):
    from dev.strategy.provenance import source_identity, verify_unchanged
    import pytest
    script=tmp_path/'tools/run_probe.sh'
    script.parent.mkdir()
    script.write_text('python probe.py --max-modules 64\n')
    before=source_identity(tmp_path)
    assert 'tools/run_probe.sh' in before['files']
    script.write_text('python probe.py --max-modules 6400\n')
    with pytest.raises(ValueError,match='run_probe.sh'):
        verify_unchanged(before,tmp_path)
