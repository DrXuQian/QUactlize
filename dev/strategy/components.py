"""Semantic component identities and conservative, host-only cost composition."""
from dataclasses import asdict
import json
import math
import re
import statistics

from quactlize.runtime.tuning import digest


def kernel_entry(name):
    """Ignore template arguments and parameter types when naming an entry.

    A GEMM may contain Directory/Metadata/Dequant/SwiGLU types without being
    one of those standalone helpers. Anonymous namespaces are not arguments.
    """
    head = re.split(r'[<(]', name.replace('(anonymous namespace)', '__anonymous'), maxsplit=1)[0].strip()
    return head.rsplit(None, 1)[-1] if head else ''


def role(name):
    n = kernel_entry(name).lower()
    if n in ('cutlass::device_kernel', 'cutlass::kernel'):
        # CUTLASS's outer callable is generic. Classify its first kernel TYPE,
        # never the nested scheduler, collective, epilogue or argument types.
        if '<' not in name:
            return 'unclassified'
        n = kernel_entry(name.split('<', 1)[1].split('>', 1)[0]).lower()
    if any(s in n for s in ('reduction', 'reduce_decode', 'reduce_bf16_grouped', 'register_reuse_reduce')):
        return 'reducer'
    if any(s in n for s in ('dequant', 'sf_prepare', 'expand_scale', 'expand_metadata',
                             'prepass_unit_kernel', 'sf_columns', 'sf_unit16', 'sf_bfloat_units')):
        return 'expansion'
    if any(s in n for s in ('convert_rows', 'convert_flat', 'copy_rows', 'gather', 'scatter')):
        return 'adapter'
    if any(s in n for s in ('metadata', 'moe_directory::build_kernel', 'moe_chain_prepare', 'mm_ids_helper', 'topk_moe', 'indexed_prepare', 'active_experts', 'computeblockinfokernel')):
        return 'prepare'
    if any(s in n for s in ('swiglu', 'weighted_finish', 'indexed_finish', 'finish_bf16')):
        return 'postprocess'
    if any(s in n for s in ('gemm', 'gemv', 'register_reuse', 'q4_decode::kernel',
                            'q4_s1::', 'model_gemv::', 'simt::')):
        return 'producer'
    # Unknown kernels remain visible and timed, never silently omitted.
    return 'unclassified'


def stage_role(name, split):
    if kernel_entry(name).rsplit('::', 1)[-1] == 'indexed_finish':
        return 'reducer_finish' if split > 1 else 'postprocess'
    return role(name)


def component_key(workload, recipe, node, *, authority, source, cache):
    """Dedup helper work, not superficially similar kernel names.

    Source identifies the relevant implementation closure. Node includes exact
    mangled symbol and launch geometry; precision/layout/domain/stride remain
    part of the semantic key. Different cold/hot protocols never alias.
    """
    w = asdict(workload)
    p = dict(recipe.parameters) if recipe is not None else {}
    split = p.get('split', node.get('split', 1))
    kind = stage_role(node['name'], split)
    contract = dict(workload=w, recipe=asdict(recipe) if recipe else 'ORIGINAL', split=split)
    if kind in ('reducer', 'reducer_finish'):
        contract = dict(operation=w['operation'], rows=w['m'] * w['topk'], n=w['n'], split=split,
                        compute=w['compute'], storage=w['storage'], fusion=w['fusion'],
                        output_stride=w['output_stride'], router=w['router'],
                        channels=w['channels'], experts=w['experts'],
                        partial_layout=node.get('partial_layout', 'EXACT_SYMBOL_CONTRACT'))
    elif kind == 'expansion':
        # Active-only expansion retains M/router unless the caller binds its
        # exact resident ID set. Dense and all-expert SF are weight-only.
        domain = node.get('active_expert_ids')
        if domain is not None and (not isinstance(domain, list) or not domain or
                any(type(e) is not int or not 0<=e<w['experts'] for e in domain) or domain!=sorted(set(domain))):
            raise ValueError('invalid explicit expansion expert domain')
        contract = dict(qtype=w['qtype'], n=w['n'], k=w['k'], experts=w['experts'],
                        compute=w['compute'], mapping=workload.mapping,
                        domain='ALL_DENSE_WEIGHTS' if w['operation'] == 'dense' else
                               'ALL_EXPERTS' if node.get('expert_domain')=='ALL' else
                               dict(active_expert_ids=domain) if domain is not None else w)
    elif kind == 'adapter' and w['operation'] == 'dense':
        # Kernel parameter columns/strides are not encoded in the symbol.
        # Conservatively retain both N/K; remove only the producer config.
        contract = w
    elif kind in ('prepare', 'postprocess'):
        # Exact entry/type/geometry plus all row/stride/domain fields remain.
        # A different GEMM tile must not re-time identical indexed helpers.
        # indexed_prepare has a runtime splits loop when writing pointers;
        # identical symbol/geometry at different S is NOT identical work.
        contract = dict(workload=w, split=split) if kind == 'prepare' else w
    # Source identity, not ASLR addresses or unrelated parent module hashes.
    # Split is reducer semantics, not a property of an otherwise identical
    # conversion/expansion kernel. Do not let a producer's S duplicate helpers.
    kernel_node = {k:v for k,v in node.items() if k not in ('split','partial_layout')}
    payload = dict(schema='quactlize.isolated-component.v1', authority=authority,
                   implementation=source, node=kernel_node, kind=kind, cache=cache, contract=contract)
    payload=json.loads(json.dumps(payload))
    return digest(payload), payload


def statistics_checked(rounds):
    if len(rounds) < 3 or any(len(r) < 3 for r in rounds):
        raise ValueError('at least 3 rounds x 3 samples are required')
    if any(type(t) not in (float, int) or not math.isfinite(t) or t <= 0 for r in rounds for t in r):
        raise ValueError('component samples must be finite positive times')
    medians = [statistics.median(r) for r in rounds]
    center = statistics.median(medians)
    return dict(median_us=center, round_medians_us=medians,
                spread_pct=100 * (max(medians) / min(medians) - 1),
                samples_us=rounds, status='PASS', scope='ISOLATED_SINGLE_KERNEL')


def component_statistics(row):
    """Keep profiler device activity distinct from isolated graph events."""
    result = statistics_checked(row['samples_us'])
    if row.get('scope') == 'PROFILED_SINGLE_KERNEL':
        identity = row.get('identity', {})
        if (row.get('timing_method') != 'ASYS_GPU_ACTIVITY' or
                identity.get('schema') != 'quactlize.profiled-component.v1' or
                identity.get('cuda_graph') is not False or not identity.get('trace_sha256')):
            raise ValueError('profiled component lacks its no-Graph trace identity')
        result['scope'] = 'PROFILED_SINGLE_KERNEL'
    return result


def recommendation(arms, records, baseline='original', *, roles=None):
    """No theoretical launch penalty and no double charging of reductions.

    This emits a recommendation, NOT a production admission. Original wiring
    and warmed model prefill/TPOT remain the final no-regression gate.
    """
    if baseline not in arms:
        raise ValueError('original production baseline is mandatory')
    costs, missing, noisy = {}, {}, []
    for arm, keys in arms.items():
        if not keys:
            missing[arm] = ['EMPTY_CALL']
            continue
        bad = [k for k in keys if k not in records or records[k].get('status') != 'PASS' or
               records[k].get('scope') not in ('ISOLATED_SINGLE_KERNEL', 'PROFILED_SINGLE_KERNEL')]
        if bad:
            missing[arm] = bad
            continue
        # Multiplicity is deliberate: two real invocations cost twice, though
        # their shared component was profiled once.
        costs[arm] = sum(records[k]['median_us'] for k in keys)
        if any(records[k]['spread_pct'] > 5 for k in keys):
            noisy.append(arm)
    if roles is None:
        roles={a:['original' if a==baseline else 'rule'] for a in arms}
    if (set(roles)!=set(arms) or 'original' not in roles[baseline] or
            any(not isinstance(v,list) or not v or set(v)-{'original','rule','control'} for v in roles.values()) or
            any('original' in v and a!=baseline for a,v in roles.items())):
        raise ValueError('arm roles must cover the original and all declared recipes')
    rule=[a for a in costs if 'rule' in roles[a]]
    control=[a for a in costs if 'control' in roles[a] or a==baseline]
    best_rule=min(rule,key=lambda a:costs[a],default=None)
    best_control=min(control,key=lambda a:costs[a],default=None)
    eligible=[a for a in costs if a==baseline or 'rule' in roles[a]]
    winner = min(eligible, key=lambda a: (costs[a], a != baseline)) if baseline in costs else baseline
    regret=(100*(costs[best_rule]/costs[best_control]-1)
            if best_rule is not None and best_control is not None else None)
    control_only_wins=(best_control is not None and winner in costs and costs[best_control]<costs[winner])
    if missing or noisy or control_only_wins:
        winner = baseline
    profiled = any(r.get('scope') == 'PROFILED_SINGLE_KERNEL' for r in records.values())
    return dict(recommended=winner, baseline=baseline, costs_us=costs, missing=missing, noisy=noisy,
                best_rule=best_rule,best_control=best_control,rule_regret_pct=regret,
                control_only_wins=control_only_wins,roles=roles,
                scope=('SUM_OF_COMPONENT_GPU_TIMES_MIXED_METHODS_NOT_MEASURED_E2E' if profiled else
                       'SUM_OF_ISOLATED_KERNELS_NOT_MEASURED_E2E'),
                host_launch_cost_added=False, production_changed=False,
                admission='ORIGINAL_WIRING_MODEL_AB_REQUIRED',
                regression_action='RETAIN_ORIGINAL')
