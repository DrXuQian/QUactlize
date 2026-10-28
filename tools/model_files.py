"""GGUF header/model metadata utilities; no benchmark execution."""
import pathlib
import re
from typing import Iterable
SPLIT_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I)


class ResolveError(ValueError):
    pass


def split_group(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    """Validate one plain GGUF or one complete standard split filename set."""
    # The public filename owns the shard index. A file symlink may point to a
    # download-cache blob with a different name (or no extension). Resolve only
    # the directory: llama loads subsequent shards beside the public first file.
    resolved = sorted({path.parent.resolve() / path.name for path in paths}, key=lambda item: item.name)
    if not resolved:
        raise ResolveError("binding resolved to no .gguf files")
    for path in resolved:
        if not path.is_file():
            raise ResolveError(f"GGUF path is not a regular file: {path}")
        if path.suffix.lower() != ".gguf":
            raise ResolveError(f"bound file does not end in .gguf: {path}")
    matches = [SPLIT_RE.fullmatch(path.name) for path in resolved]
    if not any(matches):
        if len(resolved) != 1:
            raise ResolveError(
                "directory contains multiple unsplit GGUF files; bind the exact file or one split directory; "
                f"found={[path.name for path in resolved[:8]]}")
        return resolved
    if not all(matches):
        raise ResolveError("binding mixes split and unsplit GGUF filenames; "
                           f"found={[path.name for path in resolved[:8]]}")
    groups = {(match.group("prefix"), int(match.group("count"))) for match in matches if match}
    if len(groups) != 1:
        raise ResolveError(f"binding contains multiple GGUF split groups: {sorted(groups)}")
    prefix, count = next(iter(groups))
    indices = [int(match.group("index")) for match in matches if match]
    # GGUF filenames are one-based while split.no inside the header is
    # zero-based.  Header agreement is checked by the inventory.
    expected = list(range(1, count + 1))
    if indices != expected:
        raise ResolveError(
            f"incomplete or duplicated split set {prefix}: got={indices} expected={expected}")
    if len(resolved) != count:
        raise ResolveError(f"split count {count} disagrees with files={len(resolved)}")
    return resolved


def resolve_binding(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        match = SPLIT_RE.fullmatch(path.name)
        if match:
            siblings = []
            for sibling in path.parent.iterdir():
                sibling_match = SPLIT_RE.fullmatch(sibling.name)
                if (sibling.is_file() and sibling_match and
                        sibling_match.group("prefix") == match.group("prefix") and
                        sibling_match.group("count") == match.group("count")):
                    siblings.append(sibling)
            return split_group(siblings)
        return split_group([path])
    if path.is_dir():
        return split_group(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == ".gguf")
    raise ResolveError(f"binding path does not exist: {path}")
