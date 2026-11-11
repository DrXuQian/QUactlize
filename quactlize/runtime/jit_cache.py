"""Fast selected-parent lookup shared with dispatch/jit_cache.hpp.

The dispatcher already embeds the kernel source contract. Like DeepGEMM's
runtime cache, a hit loads that build directly, without starting a compiler
or hashing its source tree. Compiler receipts remain the authoritative build
records; this disposable index only avoids recomputing their keys.

SDK/source installations are immutable during a runtime's lifetime. File
stamps invalidate disk entries across SDK replacements or cache edits. The
native loader still checks the exported parent/build identity and ABI.
"""

import os
from pathlib import Path
import shutil
import tempfile

from .compiler import LIBRARIES
from .tuning import ROUTES, digest


def stamp(path):
    s = Path(path).stat()
    return f'{s.st_dev} {s.st_ino} {s.st_size} {s.st_mtime_ns} {s.st_ctime_ns}'


def name(value):
    # FNV-1a names the lookup file, not the compiled module. The full request
    # is compared on read, so even a collision cannot select another parent.
    h = 14695981039346656037
    for byte in value.encode():
        h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f'{h:016x}'


def is_digest(value):
    return len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class ParentCache:
    def __init__(self, sdk, cache, source):
        sdk = Path(sdk).resolve(strict=True)
        host = Path(shutil.which('g++') or 'g++').resolve(strict=True)
        paths = [sdk / 'bin/hgcc', sdk / 'bin/hgobjdump'] + [
            sdk / 'lib' / f'lib{x}.so' for x in LIBRARIES] + [host]
        self.cache = Path(cache).resolve()
        self.directory = self.cache / '.jit-v1'
        self.source = source
        if not is_digest(source):
            raise ValueError('invalid JIT source contract')
        self.environment = 'QK_PARENT_CACHE_V1\n' + '\n'.join(
            [source, str(sdk), str(host), *map(stamp, paths)]) + '\n'

    def request(self, parent, dense_io, compute_type):
        fields = [parent['qtype'], ROUTES.index(parent['route'])] + [parent[f] for f in
                  ('tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn', 'persistent')]
        fields += [int(dense_io), int(compute_type == 'bf16')]
        return self.environment + parent['symbol'] + '\n' + ' '.join(map(str, fields)) + '\n'

    def read(self, parent, dense_io, compute_type):
        request = self.request(parent, dense_io, compute_type)
        path = self.directory / (name(request) + '.txt')
        try:
            if self.directory.is_symlink() or path.is_symlink():
                return None
            with path.open() as f:
                data = f.read(16385)
            if len(data) > 16384 or not data.startswith(request):
                return None
            library_stamp, receipt_stamp, receipt = data[len(request):].split('\n', 2)
            tag, key, contract, source = receipt.split()
            if tag != 'QK_JIT_V1' or not is_digest(key) or not is_digest(contract) or source != self.source:
                return None
            module = self.cache / key
            library, manifest = module / 'kernel.so', module / 'manifest.json'
            if any(p.is_symlink() for p in (module, library, manifest)):
                return None
            if stamp(library) != library_stamp or stamp(manifest) != receipt_stamp:
                return None
            return receipt
        except (OSError, ValueError):
            return None

    def publish(self, record):
        identity = record['identity']
        request = self.request(record['parent'],
                               identity.get('endpoints') == 'decode-m1-8-f32-bf16-v1',
                               identity.get('compute_type', 'f16'))
        key = record['key']
        if not is_digest(key):
            raise ValueError('invalid module key')
        module = self.cache / key
        library, manifest = module / 'kernel.so', module / 'manifest.json'
        if any(p.is_symlink() for p in (self.directory, module, library, manifest)):
            raise ValueError('JIT cache index/module must not be a symlink')
        data = (request + stamp(library) + '\n' + stamp(manifest) + '\n' +
                f'QK_JIT_V1 {key} {digest(identity)} {self.source}\n')
        self.directory.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=self.directory)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(data)
            os.replace(temporary, self.directory / (name(request) + '.txt'))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
