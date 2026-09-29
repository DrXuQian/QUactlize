"""Verify full Hopper Marlin body reuse, not native PPU1.7 admission."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def bodies(text):
    result = {}
    current = None
    for line in text.splitlines():
        if 'Function :' in line:
            current = line.split('Function :', 1)[1].strip()
            if current in result:
                raise ValueError('duplicate device body')
            result[current] = []
        elif current is not None and re.match(r'\s*/\*[0-9a-f]+\*/', line):
            # Ignore only listing PCs. Preserve instruction encoding AND the
            # following control word, not merely an equal mnemonic sequence.
            result[current].append(re.sub(r'^\s*/\*[0-9a-f]+\*/\s*', '', line).strip())
        elif current is not None and re.fullmatch(r'\s*/\* 0x[0-9a-f]+ \*/\s*', line):
            result[current].append(line.strip())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--cuobjdump', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.bundle/'manifest.json').read_text())
    measured = []
    for arm in ('marlin','quactlize_marlin'):
        row = manifest['libraries'][arm]
        library = a.bundle/row['file']
        if hashlib.sha256(library.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('library hash mismatch')
        text = subprocess.check_output([str(a.cuobjdump),'--dump-sass',str(library)],text=True)
        measured.append(bodies(text))
    reference, candidate = measured
    if len(reference) != 6 or reference != candidate:
        raise ValueError('six complete Marlin instruction bodies must match')
    for instructions in candidate.values():
        if (not any('HMMA.' in x and '.F32' in x for x in instructions)
                or not any('LDGSTS' in x for x in instructions)):
            raise ValueError('missing live FP32 MMA or async load')
    print(json.dumps({'status':'PASS', 'scope':'CUDA_SM90_BODY_REUSE_NOT_NATIVE_PPU17',
        'bodies':{name:{'instructions':sum(not x.startswith('/*') for x in value),
                       'sha256':hashlib.sha256('\n'.join(value).encode()).hexdigest()}
                  for name,value in candidate.items()}},indent=2))


if __name__ == '__main__':
    main()
