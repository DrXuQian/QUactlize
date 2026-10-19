"""Frequency-aware cohort audit; performance score is measured CE cycles only."""
import csv
import io
import math
from pathlib import Path
import statistics

CYCLES=('ce__cycles_elapsed.avg','ce__cycles_active.avg','cu__cycles_active.avg')
RATES=('ce__cycles_elapsed.avg.per_second','cu__cycles_elapsed.avg.per_second',
       'pu__cycles_elapsed.avg.per_second','device__attribute_memory_clock_rate')


def metrics(text):
    start=text.find('"ID"')
    if start<0:raise ValueError('ACU cycle table missing')
    rows=list(csv.DictReader(io.StringIO(text[start:])))
    rows=[r for r in rows if r.get('Kernel Name')]
    if not rows or len({(r['ID'],r['Kernel Name']) for r in rows})!=1:
        raise ValueError('cycles need exactly one kernel launch')
    if 'Metric Name' in rows[0]:
        values={r['Metric Name']:r['Metric Value'] for r in rows}
        if len(values)!=len(rows):raise ValueError('duplicate cycle metrics')
    else:
        if len(rows)!=1:raise ValueError('duplicate cycle rows')
        values=rows[0]
    result={}
    for key in CYCLES+RATES:
        try:value=float(values[key].replace(',',''))
        except (KeyError,ValueError):raise ValueError('missing/invalid ACU cycle metric: '+key) from None
        if not math.isfinite(value) or value<=0:raise ValueError('nonpositive/nonfinite cycle metric: '+key)
        result[key]=value
    for key in ('launch__registers_per_thread','launch__shared_mem_per_block',
                'pu__inst_executed.sum','dram__bytes_read.sum','derived__l2_bytes_pipe_l1_total'):
        if key in values:result[key]=float(values[key].replace(',',''))
    return result


def compare(reports,rounds=3):
    expected={(r,a) for r in range(rounds) for a in (0,1)}
    if (len(reports)!=len(expected) or {(r.get('round'),r.get('arm')) for r in reports}!=expected or
            any(r.get('status')!='PASS' for r in reports)):
        raise ValueError('incomplete/duplicate paired cycle inventory')
    if len({(tuple(r['grid']),tuple(r['block'])) for r in reports})!=1:
        raise ValueError('cycle cohort launch geometry differs')
    if any(len({r['symbol'] for r in reports if r['arm']==a})!=1 for a in (0,1)):
        raise ValueError('cycle cohort kernel identity changed')
    data={(r['round'],r['arm']):metrics(Path(r['raw_csv']).read_text(errors='replace')) for r in reports}
    samples=[[data[r,a][CYCLES[0]] for r in range(rounds)] for a in (0,1)]
    medians=[statistics.median(x) for x in samples]
    deltas=[100*(samples[1][r]/samples[0][r]-1) for r in range(rounds)]
    clocks={key:sorted({row[key] for row in data.values()}) for key in RATES}
    matched=all(max(v)/min(v)-1<=.001 for v in clocks.values())
    if not matched:verdict='CLOCK_CONDITIONS_DIFFER_NO_SPEED_ADMISSION'
    elif rounds<3:verdict='REPEATS_REQUIRED'
    elif all(d<0 for d in deltas):verdict='LOWER_CYCLES_ALL_PAIRS'
    elif all(d>0 for d in deltas):verdict='HIGHER_CYCLES_REJECT'
    else:verdict='MIXED_CYCLES_NO_SPEED_ADMISSION'
    return dict(metric=CYCLES[0],unit='cycles',samples=samples,median=medians,
                paired_delta_pct=deltas,median_delta_pct=100*(medians[1]/medians[0]-1),
                spread_pct=[100*(max(x)-min(x))/statistics.median(x) for x in samples],
                clock_conditions=clocks,clock_conditions_match=matched,verdict=verdict,
                reports=[dict(round=r,arm=a,metrics=data[r,a]) for r,a in sorted(data)],
                selection_score='CE_CYCLES_NOT_EVENT_MICROSECONDS_OR_OPCODE_COUNT',
                production_changed=False,performance_admitted=False,
                scope='SAME_POINT_SAME_BUILD_PAIRED_FORCED_COLD_NOT_MODEL')
