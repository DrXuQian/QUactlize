"""Independent Machete/Stream-K gate; hybrid/Marlin wins cannot satisfy it."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from compare import CONTRACT, sha, validate
from families import filter_rows
from reconfirm import selected_configs

POINTS = {(0, m, n, k) for m in (1, 8, 128) for n, k in ((512, 2048), (4096, 4096))}
INVENTORY = {'machete': set(range(9)), 'quactlize': {0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16}}


def admit(path, point):
    receipt = json.loads(path.read_text())
    if receipt['status'] != 'PASS' or len(receipt['result']) != 1:
        raise ValueError('unadmitted point')
    data = receipt['result'][0]
    validate(data)
    if (data.get('family') != 'machete-streamk' or data['shape'] != list(point[1:])
            or data['dtype'] != point[0] or data['cache'] != 'rotating'):
        raise ValueError('point identity/family differs')
    rows = data['rows']
    if len(filter_rows(rows, 'machete-streamk')) != len(rows):
        raise ValueError('hybrid/Marlin candidate in independent Machete evidence')
    seen = set()
    for row in rows:
        key = (row['arm'], row['config'])
        if key in seen:
            raise ValueError('duplicate configuration')
        seen.add(key)
        if (row['status'] != 'PASS' or row.get('warmup_min_device_ms') != 250
                or row['clocks_before'] != row['clocks_after']):
            raise ValueError('steady-clock correctness/timing evidence missing')
        if len(row['correctness']) != 5 or any(p['guards'] != 'PASS' for p in row['correctness']):
            raise ValueError('correctness/guard coverage missing')
        if 'same_math_parent' in row and row.get('same_math_raw_equality') != 'PASS':
            raise ValueError('stage candidate lacks raw control equality')
    devices = {(x['uuid'], x['driver']) for x in receipt['telemetry']}
    if len(devices) != 1:
        raise ValueError('device/driver identity differs')
    authority = (receipt['reference_manifest_sha256'], receipt['importer_sha256'],
                 json.dumps(receipt['source_sha256'], sort_keys=True), data['l2_bytes'], data['sm_count'],
                 tuple(sorted(devices)))
    fixture = (data['logical_codes_sha256'], data['scale_bits_sha256'])
    return rows, authority, fixture


def paired_winner(rounds):
    """Select one config per arm by median of rounds, never the fastest round."""
    if len(rounds) != 3:
        raise ValueError('three independent rounds required')
    inventories = [{(r['arm'], r['config']) for r in rows} for rows in rounds]
    if any(items != inventories[0] for items in inventories[1:]):
        raise ValueError('round inventories differ')
    if any(len(filter_rows(rows, 'machete-streamk')) != len(rows) for rows in rounds):
        raise ValueError('hybrid/Marlin candidate cannot win this gate')
    best = {}
    for arm in ('machete', 'quactlize'):
        candidates = []
        for _, config in sorted(key for key in inventories[0] if key[0] == arm):
            rows = [next(r for r in rows if r['arm'] == arm and r['config'] == config) for rows in rounds]
            medians = [r['median_us'] for r in rows]
            candidates.append(dict(config=config, median_us=statistics.median(medians),
                round_medians_us=medians, clocks=[r['clocks_after'] for r in rows],
                config_detail=rows[0]['config_detail']))
        best[arm] = min(candidates, key=lambda row: row['median_us'])
    reference, candidate = best['machete'], best['quactlize']
    if reference['clocks'] != candidate['clocks']:
        raise ValueError('selected candidate/reference clocks differ')
    deltas = [100*(q/r-1) for q,r in zip(candidate['round_medians_us'], reference['round_medians_us'])]
    delta = 100*(candidate['median_us']/reference['median_us']-1)
    return dict(best=best, delta_pct=delta, paired_round_delta_pct=deltas,
                within_5pct=delta <= 5 and max(deltas) <= 5,
                strict_no_slower=delta <= 0 and max(deltas) <= 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('use a fresh report directory')
    screen = json.loads((args.screen/'summary.json').read_text())
    confirm = json.loads((args.confirmation/'summary.json').read_text())
    for run in (screen, confirm):
        if not run['complete'] or run.get('family') != 'machete-streamk' or run['contract'] != CONTRACT:
            raise ValueError('complete independent Machete evidence required')
    if len(screen['results']) != 6 or len(confirm['results']) != 18 or confirm['rounds'] != 3:
        raise ValueError('six shapes x three rounds required')
    authorities, fixtures, selected, per_point = set(), {}, {}, {}
    for case in screen['results']:
        point = tuple(case['point'])
        if point in selected:
            raise ValueError('duplicate screen point')
        dtype,m,n,k = point
        rows, authority, fixture = admit(args.screen/f'd{dtype}-m{m}-n{n}-k{k}/summary.json', point)
        if {arm: {r['config'] for r in rows if r['arm'] == arm} for arm in INVENTORY} != INVENTORY:
            raise ValueError('full reference/candidate screen inventory required')
        authorities.add(authority)
        fixtures[point] = fixture
        selected[point] = set(selected_configs(rows, 'machete-streamk'))
        per_point[point] = {}
    if set(selected) != POINTS:
        raise ValueError('shape cohort differs')
    for case in confirm['results']:
        point, repeat = tuple(case['point']), case['round']
        if point not in POINTS or repeat not in range(3) or repeat in per_point[point] or case['returncode'] != 0:
            raise ValueError('invalid/duplicate confirmation')
        dtype,m,n,k = point
        rows, authority, fixture = admit(args.confirmation/f'r{repeat}-d{dtype}-m{m}-n{n}-k{k}/summary.json', point)
        if fixture != fixtures[point] or {f"{r['arm']}:{r['config']}" for r in rows} != selected[point]:
            raise ValueError('confirmation fixture/leader coverage differs')
        authorities.add(authority)
        per_point[point][repeat] = rows
    if len(authorities) != 1:
        raise ValueError('source/binary/device geometry differs')
    points = [dict(point=point, **paired_winner([per_point[point][r] for r in range(3)])) for point in sorted(POINTS)]
    verdict = 'PARITY_WITHIN_5PCT' if all(p['within_5pct'] for p in points) else 'TARGET_OPEN'
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(verdict=verdict, strict_no_slower=all(p['strict_no_slower'] for p in points), points=points,
                  scope='H800_GPTQ_MACHETE_STREAMK_ONLY_FULL_PREPARED_CALL_NOT_PPU17_ADMISSION',
                  screen_sha256=sha(args.screen/'summary.json'), confirmation_sha256=sha(args.confirmation/'summary.json'))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/'benchmark.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('m','n','k','reference_config','candidate_config',
            'reference_us','candidate_us','delta_pct','worst_round_delta_pct','within_5pct'))
        writer.writeheader()
        for row in points:
            _,m,n,k = row['point']
            ref, candidate = row['best']['machete'], row['best']['quactlize']
            writer.writerow(dict(m=m,n=n,k=k,reference_config=ref['config'],candidate_config=candidate['config'],
                reference_us=ref['median_us'],candidate_us=candidate['median_us'],delta_pct=row['delta_pct'],
                worst_round_delta_pct=max(row['paired_round_delta_pct']),within_5pct=row['within_5pct']))
    print('PPU17_MACHETE_PARITY '+verdict, flush=True)


if __name__ == '__main__':
    main()
