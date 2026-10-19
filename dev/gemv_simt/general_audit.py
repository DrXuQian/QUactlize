"""Every compiled mode/precision body is checked; static counts are not speed."""
from collections import Counter
import hashlib
import re
from dev.gemv_simt.q6_schedule_audit import instructions


def audit(directory,compute,configs):
    bodies=instructions((directory/(compute+'.isa.txt')).read_text())
    resources=(directory/(compute+'.resources.txt')).read_text();rows=[]
    for i,f in enumerate(configs):
        for arm in (0,1):
            for mode in ((None,) if arm==0 else (0,1,2)):
                name=f'general_control_{i}' if arm==0 else f'general_candidate_{i}_{mode}'
                symbol,=[s for s in bodies if f'{len(name)}{name}E' in s]
                code=bodies[symbol];counts=Counter(r[2].split()[0] for r in code if r[2]!='s.nop')
                resource=resources.split(symbol+' RESOURCE INFO:',1)[1].split('.ARGUMENT:',1)[0]
                fields={k:int(re.search(k+r':(\d+)',resource)[1]) for k in
                        ('vreg_number','sreg_number','shared_memory_size','STACK SIZE')}
                if not any(k.startswith('v.fma.f32') for k in counts):raise ValueError('missing actual FP32 dot body')
                if arm==1 and fields['STACK SIZE']:raise ValueError('candidate stack spill; review before Box')
                rows.append(dict(config=i,arm=arm,mode=mode,symbol=symbol,resources_raw=fields,
                                 source_shared_bytes=f['warps']*f['columns']*f['values']*4,
                                 native_instruction_bytes_sha256=hashlib.sha256(b''.join(r[1] for r in code)).hexdigest(),
                                 static_non_nop_sites=sum(counts.values()),opcode_sites=dict(counts)))
    return dict(compute=compute,records=rows,scope='STATIC_NATIVE_CODEGEN_NOT_SPEED_OR_DYNAMIC_WORK',
                production_changed=False,device_admission='NOT_RUN')
