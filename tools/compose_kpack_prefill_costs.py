#!/usr/bin/env python3
"""Join measured expansion and BF16 GEMM costs without running GPU work.

The original receipts are read-only. Replacing a dequant measurement is
allowed only when its canonical input and independently checked BF16 output
match the provider's original weights. Component sums are never E2E timings
or permission to select an unmeasured FQ/SF route.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quactlize.runtime.compiler import sha
from tools.run_kpack_bf16_gate import cost_record, prefill_candidates, validate_gemm_result
from tools.run_kpack_dequant_gate import validate_result


class Results:
    """A checksummed measurement directory; an unrelated failed case is not fatal."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve(strict=True)
        self.summary = json.loads((self.folder / 'result.json').read_text())
        self.files = self.summary['files']
        for name, digest in self.files.items():
            path = self.folder / name
            if (Path(name).name != name or path.is_symlink() or not path.is_file()
                    or sha(path) != digest):
                raise ValueError(f'measurement file checksum/path differs: {name}')
        self.authority = self.read('authority.json')

    def read(self, name):
        if name not in self.files:
            raise ValueError(f'missing checksummed evidence: {name}')
        return json.loads((self.folder / name).read_text())

    def reference(self, name):
        return dict(file=name, sha256=self.files[name])

    def receipt(self):
        return dict(directory=str(self.folder), result_sha256=sha(self.folder / 'result.json'),
                    authority_sha256=self.files['authority.json'],
                    manifest_sha256=self.authority['manifest_sha256'],
                    original_status=self.summary['status'],
                    original_failures=self.summary.get('failed', []),
                    verified_files=len(self.files))


def best_dequant(study, workload):
    if workload not in study.authority['workloads']:
        raise ValueError('dequant workload absent from authority')
    record = study.read(workload['id'] + '.json')
    validate_result(record, workload, study.authority['device'], study.authority['peak_gbps'])
    if workload['smoke']:
        raise ValueError('untimed smoke is not a cost')
    best = min(record['rows'], key=lambda x: x['median_us'])
    if best['proof']['signed_zero_differences']:
        raise ValueError('replacement dequant output is not bit exact')
    return record, best


def dequant_component(study, workload, row):
    return dict(config=row['config'], us=row['median_us'],
                effective_gbps=row['effective_gbps'], effective_pct=row['effective_pct'],
                evidence=study.reference(workload['id'] + '.json'))


def same_environment(left, right):
    for key in ('runtime', 'python_packages'):
        if left.authority[key] != right.authority[key]:
            raise ValueError('component environments differ: ' + key)


def combine_cell(reference, replacement, provider, gemm):
    """Return a labelled cost, retaining both original and replacement evidence."""
    w, tokens = gemm['workload'], gemm['tokens']
    validate_gemm_result(gemm, w, tokens)
    if gemm['active_experts'] != gemm['expanded_experts']:
        raise ValueError('sparse expert domain needs measured active-only dequant, not proportional scaling')
    same_environment(reference, replacement)
    same_environment(reference, provider)
    if reference.authority['peak_gbps'] != replacement.authority['peak_gbps']:
        raise ValueError('component bandwidth denominators differ')
    if (provider.authority['dequant_authority_sha256'] != reference.files['authority.json']
            or provider.authority['manifest_sha256'] != reference.authority['manifest_sha256']):
        raise ValueError('provider is not bound to the original dequant package/results')
    old, old_best = best_dequant(reference, w)
    new, new_best = best_dequant(replacement, w)
    identity = gemm['identity']
    expected = (dict(provider='CUBLAS_PPU_SDK', entry='cublasGemmEx', a='BF16_M_K',
                     b='BF16_N_K', output='BF16_M_N', compute='CUBLAS_COMPUTE_32F')
        if w['experts'] == 1 else
        dict(provider='DEEPGEMM_INSTALLED', entry='m_grouped_gemm_bf16_bf16_bf16_nt_nopad',
             a='BF16_SORTED_ROWS_K', b='BF16_E_N_K', output='BF16_SORTED_ROWS_N',
             benchmark_stream='TORCH_CURRENT_NONBLOCKING_STREAM', entry_kind='function'))
    if any(identity['provider'].get(k) != v for k, v in expected.items()):
        raise ValueError('provider precision/entry contract differs')
    if (identity['device'] != reference.authority['device']
            or identity['device'] != replacement.authority['device']):
        raise ValueError('component physical device differs')
    if identity['dequant_result_sha256'] != reference.files[w['id'] + '.json']:
        raise ValueError('provider original dequant result differs')
    if identity['dequant_config'] != old_best['config'] or gemm['cost']['full_dequant_us'] != old_best['median_us']:
        raise ValueError('provider original dequant selection differs')
    if old['fixture_hashes'] != new['fixture_hashes']:
        raise ValueError('replacement canonical input bytes differ')
    if not old['golden_sha256'] == new['golden_sha256'] == identity['golden_sha256']:
        raise ValueError('replacement BF16 weight bytes differ')
    sf_work = w | dict(operation=0, id=w['id'].removesuffix('-full') + '-sf')
    sf, sf_best = best_dequant(reference, sf_work)
    if sf['fixture_hashes']['units'] != old['fixture_hashes']['units']:
        raise ValueError('SF metadata and full-dequant input units differ')
    if gemm['sf_dequant_us'] != sf_best['median_us']:
        raise ValueError('provider original SF measurement differs')
    cost = cost_record(gemm['median_us'], new_best['median_us'])
    previous = gemm['cost']
    return dict(
        workload=w, tokens=tokens, total_rows=gemm['total_rows'], topk=gemm['topk'],
        active_experts=gemm['active_experts'], max_rows=gemm['max_rows'],
        routes_sha256=gemm['routes_sha256'], golden_sha256=new['golden_sha256'],
        full_dequant=dequant_component(replacement, w, new_best),
        previous_full_dequant=dequant_component(reference, w, old_best),
        sf_dequant=dequant_component(reference, sf_work, sf_best),
        bf16_gemm=dict(provider=identity['provider']['provider'], us=gemm['median_us'],
            first_use_excluded_seconds=gemm['first_use_excluded_seconds'],
            provider_identity_sha256=hashlib.sha256(json.dumps(identity['provider'], sort_keys=True).encode()).hexdigest(),
            loaded_images=gemm['loaded_images'],
            evidence=provider.reference(w['id'] + f'-m{tokens}.json')),
        cost=cost, previous_cost=previous,
        sum_delta_pct=100 * (cost['sum_estimate_us'] / previous['sum_estimate_us'] - 1),
        candidates=prefill_candidates(sf_dequant=sf_best['median_us'],
            full_dequant=new_best['median_us'], bf16_gemm=gemm['median_us']),
        break_even_us=dict(fq_gemm=cost['sum_estimate_us'],
            sf_gemm=cost['sum_estimate_us'] - sf_best['median_us'],
            scope='DERIVED_THRESHOLDS_NOT_MEASURED_FQ_OR_SF_GEMM'),
        selection=None, selection_admission='PENDING_MATCHED_FQ_SF_GEMM')


def compose(reference, replacement, providers):
    rows, planned, seen, missing = [], set(), set(), []
    for index, study in enumerate(providers):
        for w in study.authority['workloads']:
            for tokens in study.authority['ms']:
                key = (w['id'], tokens)
                planned.add(key)
                name = w['id'] + f'-m{tokens}.json'
                if name not in study.files:
                    continue
                if key in seen:
                    raise ValueError(f'duplicate provider cost: {key}')
                gemm = study.read(name)
                if gemm['workload'] != w or gemm['tokens'] != tokens:
                    raise ValueError('provider record differs from its declared workload')
                row = combine_cell(reference, replacement, study, gemm)
                row['provider_source'] = index
                rows.append(row)
                seen.add(key)
    for case, tokens in sorted(planned - seen):
        missing.append(dict(case=case, tokens=tokens, reason='NO_VALID_BF16_PROVIDER_RECORD'))
    rows.sort(key=lambda x: (x['workload']['q'], x['workload']['experts'],
                              x['workload']['n'], x['workload']['k'], x['tokens']))
    return dict(schema='quactlize.prefill-component-costs.v1',
        status='COMPONENTS_COMPLETE' if rows and not missing else 'INCOMPLETE',
        scope='SUM_OF_ISOLATED_MEASUREMENTS_NOT_MEASURED_E2E',
        expected=len(planned), complete=len(rows), missing=missing,
        device=reference.authority['device'], peak_gbps=reference.authority['peak_gbps'],
        sources=dict(reference_dequant=reference.receipt(), full_dequant=replacement.receipt(),
                     bf16=[study.receipt() for study in providers]),
        composition_source_sha256=sha(Path(__file__)), rows=rows,
        fq_gemm_measured=0, sf_gemm_measured=0,
        selection_admission='PENDING_MATCHED_FQ_SF_GEMM', production_changed=False,
        excludes=cost_record(1., 1.)['excludes'],
        reuse_assumption='NO_SF_OR_FULL_WEIGHT_CACHE_REUSE_ACROSS_CALLS')


def write_outputs(output, report):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    columns = ['qtype', 'n', 'k', 'experts', 'tokens', 'total_rows', 'provider',
               'sf_dequant_config', 'sf_dequant_us', 'full_dequant_config', 'full_dequant_us',
               'bf16_gemm_us', 'previous_sum_us', 'sum_estimate_us', 'sum_delta_pct',
               'fq_gemm_us', 'sf_gemm_us', 'fq_break_even_us', 'sf_gemm_break_even_us', 'selection']
    with (output / 'summary.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(columns)
        for r in report['rows']:
            w = r['workload']
            values = [w['q'], w['n'], w['k'], w['experts'], r['tokens'], r['total_rows'],
                r['bf16_gemm']['provider'], r['sf_dequant']['config'], r['sf_dequant']['us'],
                r['full_dequant']['config'], r['full_dequant']['us'], r['bf16_gemm']['us'],
                r['previous_cost']['sum_estimate_us'], r['cost']['sum_estimate_us'], r['sum_delta_pct'],
                None, None, r['break_even_us']['fq_gemm'], r['break_even_us']['sf_gemm'], 'PENDING']
            writer.writerow([f'{v:.6f}' if isinstance(v, float) else v for v in values])
    # This is the exact missing denominator, not a proposal to resweep BF16.
    tasks = []
    for r in report['rows']:
        w = r['workload']
        tasks.append(dict(workload=w, tokens=r['tokens'], total_rows=r['total_rows'],
            max_rows=r['max_rows'], active_experts=r['active_experts'], topk=r['topk'],
            routes_sha256=r['routes_sha256'], required=['FQ_GEMM_ONLY', 'SF_GEMM_ONLY'],
            timing='ROTATING_CANONICAL_WEIGHTS_COMPLETE_RING_EVENTS_FIRST_USE_EXCLUDED',
            include_splitk_reducer=True, sf_prepass_inside_gemm_timing=False))
    pending = dict(status='PLAN_ONLY_NOT_EXECUTABLE', cells=len(tasks)*2, tasks=tasks,
        requirements=['Same physical PPU and original GGUF bytes',
            'Same exact expert rows; no E256-to-active proportional scaling',
            'Use selected tactics and retain historical winning configurations as challenges',
            'Independent numerical checks for each route precision contract',
            'Measure FQ/SF GEMM separately; do not double-count SF expansion',
            'Record routing, activation/output adapters and precision conversion costs separately'])
    (output / 'missing-gemm-plan.json').write_text(json.dumps(pending, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dequant', type=Path, required=True)
    p.add_argument('--full-dequant', type=Path, required=True)
    p.add_argument('--bf16-results', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    inputs = [a.reference_dequant, a.full_dequant, *a.bf16_results]
    if any(a.output.resolve().is_relative_to(x.resolve()) for x in inputs):
        raise ValueError('output must not be inside a source results directory')
    report = compose(Results(a.reference_dequant), Results(a.full_dequant),
                     [Results(x) for x in a.bf16_results])
    write_outputs(a.output, report)
    print(f'KPACK_PREFILL_COMPOSE status={report["status"]} '
          f'cells={report["complete"]}/{report["expected"]} '
          f'fq=UNMEASURED sf_gemm=UNMEASURED selection=PENDING output={a.output}')
    return 0 if report['status'] == 'COMPONENTS_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
