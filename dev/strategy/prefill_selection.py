"""Host-only prefill composition. The legacy selector is a control, not a fallback."""
from dev.strategy.components import recommendation
from quactlize.dispatch.table import Recipe, Workload


def control_helper_config(w, recipe):
    """Mirror the legacy caller's typed SF binding, not its FP16 cost-table ID.

    llama.cpp quactlize-execution.cu normalizes BF16 SF IDs other than 4/5 to
    zero. Dense prefill in this harness has F16 metadata. Full dequant IDs are
    a separate namespace and must never undergo the SF normalization.
    """
    config = max(0, dict(recipe.parameters).get('dequant_config', -1))
    if recipe.route == 'sf' and w.operation == 'indexed' and w.compute == 'bf16' and config not in (4, 5):
        return 0
    return config


def rule_selection(arms, records, roles):
    result = recommendation(arms, records, roles=roles)
    best, control = result['best_rule'], result['best_control']
    costs = result['costs_us']
    critical = {best, control, 'original'}
    if result['missing']:
        decision = 'INCOMPLETE'
    elif best is None:
        decision = 'NO_RULE_CANDIDATE'
    elif critical.intersection(result['noisy']):
        decision = 'UNSTABLE_COMPARISON'
    elif costs[best] > costs[control]:
        decision = 'RULE_GAP'
    else:
        decision = 'CONFIRM_REQUIRED'
    result.update(recommended=best if decision == 'CONFIRM_REQUIRED' else None,
        decision=decision, baseline_role='REFERENCE_ONLY', legacy_fallback=False,
        regression_action='REFINE_RULE_NO_LEGACY_SELECTOR', performance_admitted=False,
        admission='ALTERNATING_RULE_ONLY_FINALISTS_AND_ORIGINAL_MODEL_AB_REQUIRED')
    return result


def compose(point, data, helpers, provider, records, missing):
    """Recompose existing samples without mutating recipes, receipts or timers."""
    w = Workload(**point['workload'])
    best = {role:min(pool, key=lambda c:sum(records[k]['median_us'] for k in pool[c]), default=None)
            for role, pool in helpers.items()}
    arms, roles, bindings = {}, {}, {}
    for arm in point['arms']:
        r = Recipe.from_dict(arm['recipe']); aid = arm['id']; roles[aid] = arm['roles']
        components = list(provider if r.kind == 'full' else data.get(aid, []))
        role = ('full_dequant' if r.kind == 'full' else
                'sf' if r.route == 'sf' and w.qtype != 8 else None)
        if role:
            config = best[role] if 'rule' in roles[aid] else control_helper_config(w, r)
            if 'rule' not in roles[aid]:
                bindings[aid] = dict(requested=dict(r.parameters).get('dequant_config', -1),
                                    effective=config, route=r.route)
            components = components+helpers[role][config] if components and config in helpers[role] else []
        arms[aid] = components
    selection = rule_selection(arms, records, roles)
    complete = not missing and not selection['missing'] and selection['best_rule'] is not None
    if not complete:
        selection.update(recommended=None, decision='INCOMPLETE')
    return dict(id=point['id'], workload=point['workload'], status='PASS' if complete else 'INCOMPLETE',
        missing=missing, selection=selection, best_helpers=best, components=arms,
        control_helper_bindings=bindings,
        admission='SCREEN_ONLY_RULE_ONLY_FINALISTS_AND_ORIGINAL_MODEL_AB_REQUIRED')
