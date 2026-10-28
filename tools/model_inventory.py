"""GGUF header/model metadata utilities; no benchmark execution."""
from collections import Counter
import json
from quactlize.gguf_roles import match_role
from tools.gguf_header import read_gguf_header
from tools.model_files import resolve_binding
def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(path):
    files = resolve_binding(path)
    tensors, seen = [], set()
    for index, file in enumerate(files):
        with file.open("rb") as stream:
            header = read_gguf_header(stream, str(file))
        metadata = header['metadata']
        if (metadata.get('split.count', 1) != len(files) or
                metadata.get('split.no', 0) != index):
            raise ValueError('GGUF split header disagrees with filename set: ' + str(file))
        for tensor in header['tensors']:
            if tensor['name'] in seen:
                raise ValueError('duplicate tensor across GGUF shards: ' + tensor['name'])
            seen.add(tensor['name'])
            tensors.append(tensor)
    eligible, omitted, q8, operators = [], [], [], set()
    for t in tensors:
        dims, q = t["dims_gguf"], t["qtype"]
        role = match_role(t["name"], len(dims)) if q in (8,10,11,12,13,14) else None
        if role and role[0].route_class in ("dense", "grouped") and dims[0] % (512 if q in (11,14) else 256) == 0 and dims[1] % 256 == 0:
            eligible.append(t["name"])
            operators.add(role[0].route_class)
            if q == 8: q8.append(t["name"])
        else:
            omitted.append(dict(name=t["name"], qtype=q))
    return dict(eligible=eligible, q8=q8, omitted=omitted, operators=sorted(operators),
                qtypes=dict(Counter(str(t["qtype"]) for t in tensors)),
                shards=len(files), files=list(map(str, files)), header_only=True)
