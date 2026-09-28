"""CPU-only identity checks; no GPU or third-party source mutations."""
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'quactlize/backends/ppu17'))
import dependency
from native import Backend


class Dependency(unittest.TestCase):
    def setUp(self):
        root = os.environ.get('Q17_TEST_OUTPUT')
        if not root:
            self.skipTest('set Q17_TEST_OUTPUT to an explicit external test directory')
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = self.root / 'include/cutlass/version.h'
        self.version.parent.mkdir(parents=True)
        self.version.write_text('#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 3\n#define CUTLASS_PATCH 0\n')

    def test_exact_tree_and_version(self):
        digest = dependency.tree_sha256(self.root)
        with patch.object(dependency, 'TREE_SHA256', digest):
            self.assertEqual(dependency.require_cutlass(self.root)['version'], '4.3.0')
            self.version.write_text(self.version.read_text() + '// modified\n')
            with self.assertRaisesRegex(ValueError, 'source tree differs'):
                dependency.require_cutlass(self.root)

    def test_old_version_rejected(self):
        self.version.write_text('#define CUTLASS_MAJOR 3\n#define CUTLASS_MINOR 6\n#define CUTLASS_PATCH 0\n')
        with self.assertRaisesRegex(ValueError, 'requires.*4.3.0'):
            dependency.require_cutlass(self.root)

    def test_internal_link_is_bound(self):
        link = self.root / 'version-link'
        link.symlink_to('include/cutlass/version.h')
        first = dependency.tree_sha256(self.root)
        link.unlink()
        self.assertNotEqual(dependency.tree_sha256(self.root), first)

    def test_external_link_rejected(self):
        (self.root / 'outside').symlink_to(Path(__file__).resolve())
        with self.assertRaisesRegex(ValueError, 'symlink escapes'):
            dependency.tree_sha256(self.root)

    def test_git_metadata_is_not_snapshot_content(self):
        first = dependency.tree_sha256(self.root)
        (self.root / '.git').mkdir()
        (self.root / '.git/HEAD').write_text('local-checkout-metadata\n')
        self.assertEqual(dependency.tree_sha256(self.root), first)

    def test_old_bundle_rejected_before_library_load(self):
        (self.root / 'manifest.json').write_text(json.dumps(
            {'schema': 1, 'backend': 'ppu17-cutlass360', 'target': 'cuda-source'}))
        with self.assertRaisesRegex(ValueError, 'not a PPU1.7 backend bundle'):
            Backend(self.root, allow_source_validation=True)


if __name__ == '__main__':
    unittest.main()
