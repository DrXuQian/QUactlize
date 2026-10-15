"""One generated paired decision and exact implementation, without device claims."""
import copy
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import paired_decode_selection as generate
from quactlize.fusion.native import Request, Recipe, Config, FusionCall, MappedCall, Layout, Sizes, Call
from quactlize.execution.native import arrangement

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads(generate.EVIDENCE.read_text())
RECIPES = {r['name']: r for r in CATALOG['recipes']}


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    out = tmp_path_factory.mktemp('paired-recipes')
    source = out / 'host.cpp'
    source.write_text('''#include "quactlize/fusion/validation.hpp"
extern "C" int select_recipe(qkg_gate_up_request_v2 const* r,qkg_gate_up_recipe_v2* out) {
    return quactlize::fusion::select(*r,out);
}
extern "C" int select_old(int q,int n,int k,int e,int t,int compute,qkg_gate_up_config_v1* out) {
    return quactlize::fusion::select(q,n,k,e,t,compute,out);
}
extern "C" uint64_t old_body(qkg_gate_up_call_v1 const* c,qkg_gate_up_config_v1 const* f) {
    auto r=quactlize::fusion::legacy_recipe(*c,*f);return r?r->recipe_id:0;
}
extern "C" int query_recipe(qkg_gate_up_call_v2 const* d,qkg_gate_up_recipe_v2 const* r,
    qkg_gate_up_layout_v1 const* a,qkg_sizes_v1* s) {
    return quactlize::fusion::query_recipe(*d,*r,*a,*s);
}
''')
    library = out / 'host.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I'+str(ROOT),
                    str(source), '-o', str(library)], check=True)
    result = C.CDLL(str(library))
    result.select_recipe.argtypes = [C.POINTER(Request), C.POINTER(Recipe)]
    result.select_old.argtypes = [C.c_int] * 6 + [C.POINTER(Config)]
    result.old_body.argtypes = [C.POINTER(FusionCall), C.POINTER(Config)]
    result.old_body.restype = C.c_uint64
    result.query_recipe.argtypes = [C.POINTER(MappedCall), C.POINTER(Recipe), C.POINTER(Layout), C.POINTER(Sizes)]
    return result


def get(host, row):
    request = Request(**row['request'])
    recipe = Recipe()
    assert host.select_recipe(C.byref(request), C.byref(recipe)) == 0
    return request, recipe


def call(request):
    c = Call(version=1, size=C.sizeof(Call), qtype=request.qtype, n=request.n, k=request.k,
        experts=request.experts, rows=request.tokens*request.topk, mode=request.mode,
        input_type=request.input_type, channels=request.channels, topk=request.topk,
        a_row_stride=request.k, a_token_stride=request.k*request.channels,
        ids_stride=request.topk, out_row_stride=request.n)
    d = MappedCall(FusionCall(c, request.compute_type, request.output_type, request.round_projection))
    a = Layout(1, C.sizeof(Layout), 0x47554e3400000001, arrangement(request.qtype))
    return d, a


def test_generated_catalog_and_ordinary_decisions_are_frozen():
    assert generate.header(CATALOG) == generate.OUTPUT.read_text()
    for path in ('policies/kpack_smallm_effective_v1.hpp', 'quactlize/execution/measured_decode.inc'):
        before = subprocess.check_output(['git', 'show', generate.BASE_SOURCE+':'+path], cwd=ROOT)
        if path.endswith('.hpp'):
            from tools.generate_decode_selection import EVIDENCE, effective, header
            # Paired work did not alter ordinary selection. Later component
            # promotions have their own replay; recover the frozen base here.
            original = header(effective(json.loads(EVIDENCE.read_text()), include_components=False))
            original = ''.join(line for line in original.splitlines(keepends=True)
                               if not line.startswith('inline constexpr int kDequant[]='))
            assert before == original.encode()
        else:
            assert before == (ROOT/path).read_bytes()
    assert len(CATALOG['rows']) == 26 and len(CATALOG['legacy']) == 17


@pytest.mark.parametrize('row', CATALOG['rows'], ids=lambda r: r['recipe']+'-'+str(r['request']['tokens']))
def test_every_catalog_choice_and_explicit_query(host, row):
    request, selected = get(host, row)
    expected = RECIPES[row['recipe']]
    assert selected.recipe_id == int(expected['id'], 16)
    assert selected.catalog_id == int(CATALOG['sha256'][:16], 16)
    for field in generate.RECIPE_FIELDS:
        actual = getattr(selected.config if field in ('backend', 'split', 'tile_m', 'warps') else selected, field)
        assert actual == expected[field]
    d, a = call(request)
    assert host.query_recipe(C.byref(d), C.byref(selected), C.byref(a), C.byref(Sizes())) == 0


@pytest.mark.parametrize('field,value', [('qtype', 13), ('n', 1024), ('k', 2048), ('experts', 128),
    ('tokens', 2), ('mode', 0), ('topk', 4), ('channels', 8), ('input_type', 0),
    ('compute_type', 0), ('output_type', 2), ('round_projection', 0), ('row_order', 1),
    ('consumer', 2), ('consumer_qtype', 14), ('consumer_n', 2048), ('layout_id', 0)])
def test_new_winner_does_not_leak_to_another_key(host, field, value):
    request, expected = get(host, CATALOG['rows'][-1])
    setattr(request, field, value)
    selected = Recipe()
    rc = host.select_recipe(C.byref(request), C.byref(selected))
    assert rc != 0 or selected.recipe_id != expected.recipe_id


def test_capability_is_not_a_performance_promotion(host):
    request, selected = get(host, CATALOG['rows'][-1])
    for tokens in range(2, 9):
        request.tokens = tokens
        assert host.select_recipe(C.byref(request), C.byref(Recipe())) == 24
        d, a = call(request)
        assert host.query_recipe(C.byref(d), C.byref(selected), C.byref(a), C.byref(Sizes())) == 0


def test_legacy_selector_and_hidden_bodies_are_preserved(host):
    for q in (8, 10, 11, 12, 13, 14):
        for n, k in ((512, 2048), (512, 3072), (1024, 3072), (256, 512)):
            for t in range(1, 10):
                f = Config()
                e, compute = (1, 0) if q == 8 else (256, 1)
                rc = host.select_old(q, n, k, e, t, compute, C.byref(f))
                admitted = (q in (8, 12) and n == 512 and k == 2048 and t <= 8) or (q == 8 and n == 1024 and k == 3072 and t == 1)
                assert (rc == 0) == admitted
                if not admitted:
                    assert f.version == 0
                    continue
                expected = (0, 1, 0, 8) if q == 8 or t in (1, 4) else (0, 1, 0, 4) if t == 2 else (1, 2 if t == 3 else 1, 16 if t <= 6 else 8, 0)
                assert (f.backend, f.split, f.tile_m, f.warps) == expected
    for row in CATALOG['overrides']:
        r = Request(**row['request']);d, _ = call(r)
        expected = RECIPES[row['recipe']]
        assert host.old_body(C.byref(d.call), C.byref(Config(warps=8))) == int(expected['id'], 16)
        d.call.output_type = 2
        assert host.old_body(C.byref(d.call), C.byref(Config(warps=8))) == 0


@pytest.mark.parametrize('field', ['recipe_id', 'catalog_id', 'reader', 'values', 'changes', 'fixed_k'])
def test_recipe_edits_and_fixed_dimension_mismatch_decline(host, field):
    request, recipe = get(host, CATALOG['rows'][-1]);d, a = call(request)
    setattr(recipe, field, getattr(recipe, field)^1)
    assert host.query_recipe(C.byref(d), C.byref(recipe), C.byref(a), C.byref(Sizes())) != 0


def test_no_compact_or_fp16_alias_for_new_reader(host):
    request, recipe = get(host, CATALOG['rows'][-1]);d, a = call(request)
    for name, value in (('n', 1024), ('k', 2048), ('input_type', 0)):
        changed = MappedCall.from_buffer_copy(d)
        setattr(changed.call.input.call, name, value)
        assert host.query_recipe(C.byref(changed), C.byref(recipe), C.byref(a), C.byref(Sizes())) != 0
    d.input_rows = 0x100000
    assert host.query_recipe(C.byref(d), C.byref(recipe), C.byref(a), C.byref(Sizes())) != 0


def test_evidence_cannot_be_silently_edited():
    changed = copy.deepcopy(CATALOG)
    changed['recipes'][-1]['values'] = 8
    with pytest.raises(ValueError):
        generate.header(changed)


def test_caller_trace_binds_recipe_body_and_scope():
    llama = Path(os.environ['LLAMA_CI_DIR'])
    sys.path.insert(0, str(llama/'tests'))
    import quactlize_native as native
    row = CATALOG['rows'][-1];r = RECIPES[row['recipe']]
    record = dict(q='12',n='512',k='3072',tokens='1',experts='256',op='grouped',activation='BF16',
        backend='simt',recipe_version='2',recipe='0x'+r['id'],catalog='0x'+CATALOG['sha256'][:16],
        consumer='1',consumer_qtype='13',consumer_n='3072',
        **{f:str(r[f]) for f in generate.RECIPE_FIELDS if f!='backend'})
    receipt = dict(selection=CATALOG)
    assert native.paired_catalog_selection(record,receipt) == record
    symbol = native.paired_symbol_recipe('quactlize::fusion::simt_gate_up_paired16<4,4,3,1024,3072>(DeviceCall)')
    assert native.paired_matches_plan(symbol,record)
    for bad in (dict(values='8'),dict(reader='2'),dict(tokens='2'),dict(consumer='2'),dict(consumer_qtype='14')):
        with pytest.raises(ValueError):
            native.paired_catalog_selection(record|bad,receipt)
    assert not native.paired_matches_plan(symbol,record|dict(values='8'))
    assert not native.paired_matches_plan(symbol,record|dict(k='2048'))
