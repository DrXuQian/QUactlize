"""Static per-PC shared reader inventory, NOT measured conflict attribution."""
import hashlib
import re


def bodies(isa):
    entries = list(re.finditer(r'Disassembly of section \.text\.kernel\.([^\n]+):', isa))
    return {e[1]: isa[e.end():entries[i + 1].start() if i + 1 < len(entries) else None]
            for i, e in enumerate(entries)}


def encoded(body):
    return re.findall(r'^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} ){7}[0-9a-f]{2})', body, re.M)


def shared_sites(isa):
    result = []
    for symbol, body in bodies(isa).items():
        if 'dense_dequant_unique_high' not in symbol and not (
                'dense_dequant_cooperative' in symbol and 'ELj64E' in symbol):
            continue
        words = encoded(body)
        if not words:
            raise ValueError('high-plane kernel ISA missing')
        sites = []
        source = None
        for line in body.splitlines():
            if re.search(r'\.(?:cu|cuh|hpp):\d+', line):
                source = line.strip()
            instruction = re.match(r'\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*(tsm\.(?:ld|st)\S*)\s+(.*)', line)
            if instruction:
                sites.append(dict(pc='0x' + instruction[1], opcode=instruction[2], operands=instruction[3], source=source))
        if not sites:
            raise ValueError('high-plane shared accesses missing')
        result.append(dict(symbol=symbol, instruction_sha256=hashlib.sha256('\n'.join(words).encode()).hexdigest(),
                           shared_sites=sites))
    if len(result) != 20:  # 12 existing Q4/Q5/Q6 N64 + 8 Q5/Q6 candidates.
        raise ValueError('high-plane static inventory differs: ' + str(len(result)))
    return dict(scope='STATIC_SHARED_PC_AND_SOURCE_NOT_BANK_CONFLICT_COUNTS', kernels=result,
                performance_admitted=False)
