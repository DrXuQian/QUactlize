"""Sealed production catalog validation/generation; no device experiments."""
from quactlize.runtime.tuning import digest


ARCHIVES = {
    'screen': '4d78e5c015b790e96cc6a204eb86c542437a44b5e959823d5f5605f36b928dab',
    'retry': '0fce9389c80ebb4412e714e5e4f8f75dd12d5dc90ced59fe99eebdf2b6c6d782',
    'q8': '585b35284c6c0a1c09d37520dc73b5d15dda49ebacbd267e0cf24c52766af902',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(w):
    return (w['qtype'], 2 if w['operation'] == 'indexed' else 0, w['n'], w['k'],
            w['experts'], w['topk'], w['channels'], w['m'], int(w['compute'] == 'bf16'))


def semantic(c):
    # Parent names are aliases, not a reason to compare different bodies.
    # All codegen-affecting geometry/precision/operator fields remain in key.
    return digest({k: v for k, v in c.items() if k not in ('symbol', 'persistent')})


def core_semantic(c):
    return semantic({k: v for k, v in c.items() if k != 'dequant_config'})


def apply(evidence, baseline):
    """Flatten admitted replacements into the one existing runtime catalog."""
    import copy
    require(evidence['schema'] == 'quactlize.component-decode.v1' and
            evidence['archives'] == ARCHIVES, 'component evidence authority differs')
    require(evidence['evidence_sha256'] == digest({k: v for k, v in evidence.items()
            if k != 'evidence_sha256'}), 'component evidence was edited; recapture the reviewed archives')
    require(evidence['baseline_sha256'] == digest(baseline), 'component incumbent catalog changed')
    require(evidence['scope'] == 'ALTERNATING_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT' and
            not evidence['runtime_router_readback'] and not evidence['small_m_full_dequant'],
            'component admission scope differs')
    result = copy.deepcopy(baseline)
    by_key = {tuple(r['key']): r for r in result['exact']}
    seen = set()
    for r in evidence['rows']:
        k = tuple(r['key'])
        require(k not in seen and k in by_key, 'duplicate or unknown component host key')
        seen.add(k)
        row = by_key[k]
        require(core_semantic(row['config']) == core_semantic(r['original']), 'component incumbent differs')
        if not r['changed']:
            require(r['dequant_config'] == -1 and
                    core_semantic(r['config']) == core_semantic(r['original']), 'changed retained row')
            continue
        require(r['status'] == 'CONFIRMED_COMPONENT_REPLACEMENT' and r['profiles'] and
                all(p['selected_arm'] != 'original' and p['selected_us'] <= p['original_us'] and
                    p['spread_pct'] <= 5 and 0 <= p['error'] < .005 for p in r['profiles']),
                'replacement lacks confirmed no-regression evidence')
        dq = r['dequant_config']
        require(dq == -1 or (r['config']['kind'] == 'tc' and r['config']['route'] == 1 and
                            k[0] in (12, 13) and dq in range(6)), 'unsupported measured SF helper')
        row.update(config=r['config'], policy=18, dequant_config=dq, status=r['status'])
        # Buckets are proposals on unseen shapes. A Q4 S1 body bakes in N/K;
        # keep its old generic guard as bucket donor, never launch it elsewhere.
        if r['config']['kind'] != 'q4':
            row['guard'] = r['config']
    return result
