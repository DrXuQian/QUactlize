"""One Q6 MoE regression point, not a new sweep or production selector."""
import csv
import io
import re

from quactlize.dispatch.table import Recipe, Workload

POINT = 'aae88001ab9fd8ed6769b58237bd58c10b94e5758a296cd73ffb650fbd43ec0e'
CONFIG = dict(kind='simt', variant=3, columns=4, warps=8, values=8, split=1)
WORKLOAD = dict(qtype=14, n=512, k=2048, m=8, operation='indexed', compute='bf16',
                storage='f32', experts=256, topk=8, channels=8, input_stride=2048,
                output_stride=512, router='cluster', fusion='none')
NATIVE = [
    dict(role='adapter', symbol='_ZL13quantize_q8_1PKfPvlllllj5uint3',
         grid=[8, 8, 8], block=[256, 1, 1]),
    dict(role='producer', symbol='_ZL17mul_mat_vec_q_moeIL9ggml_type14ELi2EEvPKvS2_PKiPfj5uint3jjjjjjjjj',
         grid=[256, 8, 1], block=[32, 8, 1]),
]


def selected(rows):
    matches = [r for r in rows if r['id'] == POINT]
    if len(matches) != 1:
        raise ValueError('missing/duplicate frozen Q6 profile point')
    row = matches[0]
    if row['workload'] != WORKLOAD or row['selected_recipe'] != CONFIG or row['pending']:
        raise ValueError('Q6 workload/selected recipe/helper obligations changed')
    recipe = Recipe.make('simt', 'fq', **{k: v for k, v in CONFIG.items() if k != 'kind'})
    if not recipe.eligible(Workload(**WORKLOAD)):
        raise ValueError('frozen Q6 recipe is ineligible')
    return row, recipe


def check_native(nodes):
    actual = [{k: n[k] for k in ('role', 'symbol', 'grid', 'block')} for n in nodes]
    if actual != NATIVE:
        raise ValueError('native dispatcher changed its Q6 kernel/launch inventory')
    return actual


def acu_command(acu, report, child, *, symbol=None, graph_profiling='node'):
    # SDK 2.1.1: launch-skip counts MATCHING kernels, not all launches.
    # Native: first linked call + three adapter warmups; collect warmup #3.
    # Candidate: only one ordinary call inside its explicit profiler range.
    if graph_profiling not in ('none', 'node', 'graph'):
        raise ValueError('unsupported graph profiling mode')
    result = [str(acu), '--set', 'full', '--profile-from-start', 'yes' if symbol else 'no',
              '--graph-profiling', graph_profiling, '--replay-mode', 'kernel', '--cache-control', 'all',
              '--kill', 'no', '--check-exit-code', 'yes', '--launch-count', '1',
              '--export', str(report)]
    if symbol:
        result += ['--kernel-name-base', 'mangled', '--kernel-name', symbol, '--launch-skip', '3']
    return result + list(map(str, child))


def check_report(text, expected):
    """Accept ACU's wide raw table or per-metric rows; exactly ONE launch."""
    start = text.find('"ID"')
    if start < 0:
        raise ValueError('ACU raw export has no kernel table')
    rows = [r for r in csv.DictReader(io.StringIO(text[start:])) if r.get('Kernel Name')]
    launches = {(r['ID'], r.get('Kernel Mangled Name', r['Kernel Name'])) for r in rows}
    if len(launches) != 1:
        raise ValueError('ACU must contain exactly one selected kernel launch')
    normalize = lambda v: re.sub(r'\s+', '', v)
    geometry = lambda v: '(' + ','.join(map(str, v)) + ')'
    for row in rows:
        # Some versions always print a demangled name even with the import
        # option set to mangled. Require either exact symbol or exact name.
        if (expected['symbol'] not in (row['Kernel Name'], row.get('Kernel Mangled Name')) and
                normalize(row['Kernel Name']) != normalize(expected.get('name', expected['symbol']))):
            raise ValueError('ACU kernel differs from the independently timed entry')
        for key, field in (('grid', 'Grid Size'), ('block', 'Block Size')):
            if normalize(row.get(field, '')) != geometry(expected[key]):
                raise ValueError('ACU launch geometry differs: ' + field)
    return dict(symbol=expected['symbol'], grid=expected['grid'], block=expected['block'],
                launches=1, raw_rows=len(rows), cache_control='all',
                scope='FORCED_COLD_PROFILER_COUNTERS_NOT_UNPROFILED_EVENT_LATENCY')
