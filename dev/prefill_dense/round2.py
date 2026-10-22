"""Round2: coalescing, unsigned coordinates, and geometry, independently.

No provider tuning or production admission. Keep round1 kernels immutable;
compare same-dtype controls at the current clock, not historical wall times.
"""
import ctypes as C
import json
import math
import statistics
from .spec import STAGES,workloads
from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest


def baseline_variant(point,storage):
    # Prior component screen, not an assertion of a confirmed global winner.
    return ([2,1,1,2,2,1,2]if storage else[1,1,1,2,2,1,2])[point]


def pool(point):
    result=[]
    for dtype,storage in [('bf16',1),('fp16',0)]:
        for variant in (1,2,4,5,6,7):
            result.append(dict(name=f'r2-{dtype}-dq{variant}',storage=storage,vector_io=1,
                dequant_variant=variant,math_mode=16,algorithm=-1))
        result.append(dict(name=f'r2-{dtype}-io2',storage=storage,vector_io=2,
            dequant_variant=baseline_variant(point,storage),math_mode=16,algorithm=-1))
    return result


def stages(point,arm):
    if arm['vector_io']==2:return ('input','output')
    if arm['dequant_variant']==baseline_variant(point,arm['storage']):return ('dequant','input','output')
    return ('dequant',)


def composed(point,rows):
    result=[]
    for storage in (1,0):
        candidates=[r for r in rows if r['arm']['storage']==storage]
        control=next(r for r in candidates if r['arm']['vector_io']==1 and
                     r['arm']['dequant_variant']==baseline_variant(point,storage))
        dq=min((r for r in candidates if 'dequant'in r['median_us']),key=lambda r:r['median_us']['dequant'])
        io=min((r for r in candidates if 'input'in r['median_us']),
               key=lambda r:r['median_us']['input']+r['median_us']['output'])
        result.append(control['arm']|dict(name=f'composed-{storage}-control'))
        result.append(dq['arm']|dict(name=f'composed-{storage}-screen-choice',vector_io=io['arm']['vector_io']))
    return result


def profile_jobs(results):
    # Six pairs, each AB then BA. No GEMM/config sweep and no cold ring in ACU.
    # F16 IO has the same mapping; both dtypes still pass full bit/timing gates.
    pairs=[(0,3,'r2-bf16-dq2','r2-bf16-io2'),
           (5,0,'r2-bf16-dq1','r2-bf16-io2'),
           (0,1,'r2-fp16-dq1','r2-fp16-dq4'),
           (5,1,'r2-bf16-dq1','r2-bf16-dq4'),
           (5,1,'r2-fp16-dq1','r2-fp16-dq4')]
    small=next(r for r in results if r['point']==6 and r['status']=='PASS')
    geometry=min((r for r in small['arms']if r['arm']['storage']==1 and r['arm']['dequant_variant']in (6,7)),
                 key=lambda r:r['median_us']['dequant'])['arm']['name']
    pairs.append((6,1,'r2-bf16-dq5',geometry))
    jobs=[]
    for pair,(point,stage,control,candidate)in enumerate(pairs):
        for repeat,order in enumerate(((control,candidate),(candidate,control))):
            for arm in order:jobs.append(dict(point=point,stage=stage,arm=arm,pair=pair,repeat=repeat,
                                              role='control'if arm==control else'candidate'))
    return jobs


def check_profile(path,job):
    import csv,io
    from quactlize.runtime.compiler import sha
    text=path.read_text(errors='replace');start=text.find('"ID"')
    if start<0:raise ValueError('ACU export has no kernel table')
    rows=[r for r in csv.DictReader(io.StringIO(text[start:]))if r.get('Kernel Name')]
    launches=sorted({(r['ID'],r['Kernel Name'])for r in rows})
    if len(launches)!=1:raise ValueError('round2 must capture exactly one component kernel')
    if job['stage']==1:
        expected='dense_dequant_unsigned'if int(job['arm'][-1])>=4 else'dense_dequant_packed'
    else:expected='dense_convert_stripe4'if job['arm'].endswith('io2')else'dense_convert_vec8'
    if expected not in launches[0][1]:raise ValueError('ACU component differs from selected candidate')
    row=next(r for r in rows if r['ID']==launches[0][0])
    def value(key):
        x=float(row[key].replace(',',''))
        if not math.isfinite(x):raise ValueError('nonfinite ACU counter: '+key)
        return x
    w=workloads()[job['point']]
    if job['stage']==1:
        variant=int(job['arm'][-1]);tn=16 if variant==6 else 32;tk=128 if variant in (1,4)else 256
        grid=[w.n//tn,w.k//tk,1];block=256 if variant==7 else 128
    else:grid=[(w.m*(w.k if job['stage']==0 else w.n)+2047)//2048,1,1];block=256
    if [value('launch__grid_dim_'+axis)for axis in 'xyz']!=grid or value('launch__block_size')!=block:
        raise ValueError('ACU launch geometry differs from selected candidate')
    cycles=value('ce__cycles_elapsed.avg');frequency=value('ce__cycles_elapsed.avg.per_second')
    if cycles<=0 or frequency<=0:raise ValueError('ACU cycles/frequency unavailable')
    counters={key:value(key)for key in row if key.startswith(('kvd__transactions_pipe_lsu_mem_global_',
        'kvd__transactions_pipe_l2_mem_global_','tsm__bank_conflicts','cu__we_pipe_ialu_cycles_active',
        'launch__occupancy','launch__registers_per_thread','pu__we_warps_eligible'))and row[key]}
    return dict(launches=launches,raw_sha256=sha(path),cycles=cycles,frequency_hz=frequency,
        grid=grid,block=block,counters=counters,scope='FORCED_COLD_ACU_ONE_COMPONENT_ABBA')


def cycle_pairs(reports):
    result=[]
    for pair in range(6):
        rows=[r for r in reports if r['pair']==pair and r['status']=='PASS']
        sides={side:[r['evidence']['cycles']for r in rows if r['role']==side]for side in ('control','candidate')}
        if any(len(v)!=2 for v in sides.values()):continue
        med={s:statistics.median(v)for s,v in sides.items()}
        result.append(dict(pair=pair,cycles=sides,median_cycles=med,
            cycle_delta_pct=100*(med['candidate']/med['control']-1),
            frequency_hz=[r['evidence']['frequency_hz']for r in rows],
            scope='SAME_COMPONENT_ABBA_CYCLES_NOT_FREQUENCY_NORMALIZED_LATENCY',performance_admitted=False))
    return result


def measure(a,sdk,w,lib,timer,manifest,authority):
    from .bench import Bench,io_gate,dequant_gate,dynamic_gate
    from quactlize.runtime.native import checked
    profile=a.arm is not None
    if not profile and a.point==0:
        atomic_json(a.output/'io-gate.json',io_gate(sdk,a.sdk,lib,vectors=(0,1,2)))
        atomic_json(a.output/'dequant-gate.json',dequant_gate(sdk,lib,variants=range(8)))
        atomic_json(a.output/'dynamic-gate.json',dynamic_gate(sdk,a.sdk,lib,timer,a.l2_bytes,round_two=True))
    candidates=pool(a.point)
    if profile:
        prior=json.loads((a.output/f'point-{a.point}.json').read_text())
        if prior['status']!='PASS'or prior['library_sha256']!=manifest['sha256']or prior['authority']!=digest(authority):
            raise ValueError('round2 prior numerical/timing admission missing')
        if a.arm not in {r['arm']['name']for r in prior['arms']}:raise ValueError('profile arm not admitted')
        candidates=[arm for arm in candidates if arm['name']==a.arm]
    b=None
    try:
        b=Bench(sdk,a.sdk,lib,timer,w,a.l2_bytes,profile=profile)
        print(f'DENSE_R2_FIXTURE point={a.point} copies={b.copies} profile={int(profile)}',flush=True)
        handles={arm['name']:b.handle(arm)for arm in candidates};proofs={}
        def check(arm,h):
            proof=[]
            for repeat in (0,1):
                b.update(repeat)
                for i in sorted({0,b.copies-1}):b.launch(h,i);proof.append(b.check())
            bits=b.check_weights(arm['storage'],b.copies-1)
            b.update(0,zero=True);b.launch(h);b.check(zero=True)
            b.update(0);b.launch(h);proof.append(b.check())
            print(f'DENSE_R2_NUMERIC point={a.point} arm={arm["name"]} status=PASS',flush=True)
            return dict(dot=proof,weights=bits)
        for arm in candidates:proofs[arm['name']]=check(arm,handles[arm['name']])
        if profile:
            h=handles[a.arm]
            for _ in range(3):b.launch(h)
            sdk.synchronize(b.r.stream)
            start,stop=sdk.lib.hggcProfilerStart,sdk.lib.hggcProfilerStop
            for fn in (start,stop):fn.argtypes=[];fn.restype=C.c_int
            print(f'DENSE_R2_ACU_BEGIN point={a.point} arm={a.arm} stage={a.stage} calls=1 cold_ring=0',flush=True)
            checked(start(),'profiler start')
            try:b.launch(h,stage=a.stage);sdk.synchronize(b.r.stream)
            finally:checked(stop(),'profiler stop')
            b.launch(h);b.check()
            result=dict(status='PASS',point=a.point,arm=a.arm,stage=a.stage,calls=1,cold_ring=False,profiled=True)
        else:
            timings={arm['name']:{s:[]for s in stages(a.point,arm)}for arm in candidates};rounds=[]
            inverse={name:stage for stage,name in STAGES.items()}
            for rnd in range(4):
                order=candidates if rnd%2==0 else list(reversed(candidates))
                for arm in order:
                    samples={s:b.samples(handles[arm['name']],inverse[s])for s in stages(a.point,arm)}
                    for s,values in samples.items():timings[arm['name']][s]+=values
                    rounds.append(dict(round=rnd,arm=arm['name'],samples_us=samples))
                    print(f'DENSE_R2_PROGRESS point={a.point} round={rnd+1}/4 arm={arm["name"]}',flush=True)
            rows=[dict(arm=arm,proof=proofs[arm['name']],samples_us=timings[arm['name']],
                       median_us={s:statistics.median(v)for s,v in timings[arm['name']].items()})for arm in candidates]
            complete=[];selected=composed(a.point,rows)
            for arm in selected:
                h=b.handle(arm);handles[arm['name']]=h;proofs[arm['name']]=check(arm,h)
            for rnd in range(4):
                for arm in (selected if rnd%2==0 else list(reversed(selected))):
                    values=b.samples(handles[arm['name']],-1)
                    complete.append(dict(round=rnd,arm=arm,samples_us=values,median_us=statistics.median(values)))
            result=dict(status='PASS',point=a.point,receipt=b.receipt,arms=rows,rounds=rounds,
                complete=complete,complete_scope='SEPARATE_MEASURED_FULL_CALL_NOT_COMPONENT_SUM',
                timing='ORDINARY_EVENT_COMPONENT_INTERVALS_NOT_ACU_CYCLES',
                selection_scope='LOCAL_COMPONENT_SCREEN_NOT_PRODUCTION_ADMISSION')
        result.update(library_sha256=manifest['sha256'],authority=digest(authority),numerical=proofs,
                      production_changed=False,performance_admitted=False,round_two=True)
        atomic_json(a.result,result);return 0
    finally:
        if b:b.close()
