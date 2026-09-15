#!/usr/bin/env python3
"""Copy the public C ABI into a private llama checkout, rewriting includes only."""
import argparse
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES={
    'quactlize/integrations/llama/indexed.h':'kpack_indexed.h',
    'quactlize/integrations/llama/moe_graph.hpp':'moe_graph.hpp',
    'quactlize/runtime/abi.h':'kpack_module.h',
    'quactlize/execution/api.h':'kpack_execution.h',
    'quactlize/dispatch/api.h':'kpack_dispatch.h',
    'quactlize/decode/api.h':'kpack_decode_io.h',
    'quactlize/dequant/api.h':'kpack_dequant.h',
    'quactlize/prefill/api.h':'kpack_prefill.h',
    'quactlize/execution/q4_decode.h':'kpack_q4_decode.h',
}
INCLUDES={
    '../runtime/abi.h':'kpack_module.h',
    '../execution/api.h':'kpack_execution.h',
    '../integrations/llama/indexed.h':'kpack_indexed.h',
    '../decode/api.h':'kpack_decode_io.h',
    '../dequant/api.h':'kpack_dequant.h',
    '../execution/q4_decode.h':'kpack_q4_decode.h',
    '../include/quactlize_ppu_config.h':'quactlize_ppu_config.h',
}


def sync(llama):
    folder=llama/'ggml/src/ggml-cuda/quactlize'
    if not folder.is_dir():raise ValueError('missing private integration headers')
    for source,target in FILES.items():
        text=(ROOT/source).read_text()
        if source=='quactlize/execution/q4_decode.h':
            text=text.replace('"api.h"','"kpack_execution.h"')
        for before,after in INCLUDES.items():text=text.replace('"'+before+'"','"'+after+'"')
        (folder/target).write_text(text)
    sums=folder/'ABI_SHA256'
    names={line.split()[-1] for line in sums.read_text().splitlines() if line.strip()}|set(FILES.values())
    sums.write_text(''.join(hashlib.sha256((folder/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in sorted(names)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('llama',type=Path);sync(p.parse_args().llama.resolve(strict=True))
