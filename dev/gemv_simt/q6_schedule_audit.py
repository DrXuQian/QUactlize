"""Static native evidence for six scheduling bodies; never a timing verdict."""
from collections import Counter
import hashlib
import re

from dev.gemv_simt.q6_schedule import KERNELS, WARPS, EQUAL_GROUPS


def instructions(text):
    found = {}
    for section in text.split('Disassembly of section .text.kernel.')[1:]:
        symbol = section.split(':',1)[0]
        rows = []
        for line in section.splitlines():
            match = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){7}[0-9a-f]{2})\s+(.+)', line)
            if match:
                rows.append((int(match[1],16), bytes.fromhex(match[2]), match[3]))
        if not rows or symbol in found:
            raise ValueError('missing/duplicate native kernel body')
        found[symbol] = rows
    return found


def audit(build, compute, control_build=None):
    bodies = instructions((build/(compute+'.isa.txt')).read_text())
    resources = (build/(compute+'.resources.txt')).read_text()
    records = []
    for arm, name in enumerate(KERNELS):
        symbol, = [s for s in bodies if f'{len(name)}{name}E' in s]
        instr = bodies[symbol]
        counts = Counter(row[2].split()[0] for row in instr if row[2] != 's.nop')
        resource = resources.split(symbol+' RESOURCE INFO:',1)[1].split('.ARGUMENT:',1)[0]
        values = {key:int(re.search(key+r':(\d+)',resource)[1]) for key in
                  ('vreg_number','sreg_number','shared_memory_size','STACK SIZE')}
        first_fma = next(i for i,r in enumerate(instr) if r[2].startswith('v.fma.f32'))
        loads = [(i,r[0]) for i,r in enumerate(instr) if r[2].startswith('vmem.ld.b32x4')]
        records.append(dict(arm=arm, kernel=name, symbol=symbol, resources=values,
                            static_non_nop_sites=sum(counts.values()), opcode_sites=dict(counts),
                            source_shared_bytes=WARPS[arm]*32*4,
                            native_instruction_bytes_sha256=hashlib.sha256(b''.join(r[1] for r in instr)).hexdigest(),
                            first_fma_instruction_index=first_fma,
                            vector128_loads_before_first_fma=sum(i<first_fma for i,pc in loads),
                            vector128_load_sites=[dict(index=i,pc=hex(pc)) for i,pc in loads]))
    comparisons = []
    for left,right in EQUAL_GROUPS:
        a,b = records[left],records[right]
        def relevant(row):
            return {k:v for k,v in row['opcode_sites'].items() if k.startswith(
                ('vmem.ld.', 'vmem.st.', 'v.fma.', 'v.cnvt.', 'tsm.ld.', 'tsm.st.', 'v.shuffle.'))}
        comparisons.append(dict(arms=[left,right], same_math_load_shuffle_sites=relevant(a)==relevant(b),
                                same_register_allocation=a['resources']==b['resources'],
                                native_schedule_differs=a['native_instruction_bytes_sha256']!=b['native_instruction_bytes_sha256']))
    result = dict(compute=compute, arms=records, pairs=comparisons,
                  no_stack=all(r['resources']['STACK SIZE']==0 for r in records),
                  scope='STATIC_NATIVE_SITES_NOT_DYNAMIC_WORK_OR_PERFORMANCE', device_admission='NOT_RUN')
    if control_build is not None:
        control = instructions((control_build/(compute+'.isa.txt')).read_text())
        result['a3_control_bytes_unchanged'] = (
            b''.join(r[1] for r in control[records[0]['symbol']]) ==
            b''.join(r[1] for r in bodies[records[0]['symbol']]))
    return result
