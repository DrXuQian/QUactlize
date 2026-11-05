"""Build-machine checks shared by the public builder and setuptools, never imported at runtime."""
import json
from pathlib import Path
import platform
import re
import subprocess

from quactlize._library import PUBLIC_EXPORTS, PUBLIC_LIBRARY, file_sha256, verify_payloads
from quactlize._version import __version__


def source_files(root):
    suffixes = {'.py', '.cu', '.cpp', '.h', '.hpp', '.cuh', '.inc', '.inl', '.json', '.toml'}
    paths = [root / name for name in ('setup.py', 'pyproject.toml', 'build.sh')]
    for directory in ('quactlize', 'policies'):
        paths += [p for p in (root / directory).rglob('*') if p.is_file() and p.suffix in suffixes
                  and '_native' not in p.relative_to(root).parts and '__pycache__' not in p.parts]
    paths += [p for p in (root / 'tools').glob('*.py') if p.name.startswith(
        ('build_', 'generate_', 'fit_prefill_refresh', 'kpack_jit', 'gguf_header', 'wheel_support'))]
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}


def build_info(root, sdk):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    vendor = subprocess.check_output(['git', '-C', str(root / 'third_party/actlize'),
                                      'rev-parse', 'HEAD'], text=True).strip()
    return dict(version=__version__, source_revision=revision, actlize_revision=vendor,
                source_tree_dirty=bool(subprocess.check_output(
                    ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=normal'], text=True)),
                source_hashes=source_files(root), supported_targets=['ppu10'], default_target='ppu10',
                device_arch='ppu_10', sdk_release=sdk_release(sdk),
                runtime_api=(sdk / 'VERSION.txt').read_text().strip(),
                platform_abi_versions=sdk_abi_versions(sdk),
                device_validation='NOT_RUN_BY_BUILDER')


def sdk_release(sdk):
    # VERSION.txt identifies the runtime API (v3), not the SDK release.
    text = (sdk / 'release.yaml').read_text()
    match = re.search(r'^version:\s*([\w.+-]+)\s*$', text, re.M)
    if not match:
        raise ValueError('invalid SDK release metadata: ' + str(sdk / 'release.yaml'))
    return match[1]


def sdk_abi_versions(sdk):
    result = set()
    for name in ('hggc_wrapper', 'hggcrt1', 'hggc', 'hg_wrapper'):
        data = subprocess.check_output(['readelf', '--version-info',
                                        str(sdk / 'lib' / f'lib{name}.so')], text=True)
        result.update(re.findall(r'Name:\s+((?:GLIBC|GLIBCXX|CXXABI)_[\d.]+)', data))
    return sorted(result)


def elf_dependencies(bundle):
    """Inspect ELF data without dlopen; system SDK/driver libraries stay outside the wheel."""
    libraries = [bundle / PUBLIC_LIBRARY, *sorted((bundle / 'quactlize-runtime').glob('*.so'))]
    private = {p.name for p in libraries}
    allowed = {PUBLIC_LIBRARY, 'libquactlize_kpack_dispatch.so',
               'libquactlize_ppu_execution.so', 'libquactlize_ppu_prefill.so'}
    if private != allowed:
        raise ValueError(f'unexpected/missing private libraries: {private ^ allowed}')
    dependencies = {}
    for library in libraries:
        header = subprocess.check_output(['readelf', '-h', str(library)], text=True)
        if 'ELF64' not in header or 'Advanced Micro Devices X86-64' not in header:
            raise ValueError(f'unsupported ELF platform: {library}')
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        for paths in re.findall(r'\((?:RPATH|RUNPATH)\).*?\[(.*?)\]', dynamic):
            for path in paths.split(':'):
                if path not in ('$ORIGIN', '$ORIGIN/quactlize-runtime'):
                    raise ValueError(f'non-relocatable library search path {path}: {library}')
        needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
        if any('/' in name for name in needed):
            raise ValueError(f'absolute DT_NEEDED: {library}')
        dependencies[library.name] = needed
    return dict(elf_needed=dependencies,
                platform_libraries=sorted({name for values in dependencies.values() for name in values} - private),
                optional_preparation_provider='PPU SDK libcublas.so (not bundled)',
                policy='Use the platform SDK and Torch. No system driver/runtime is bundled.')


def wheel_metadata(bundle, root):
    # This is the existing builder's export/selector/JIT verification, not a second build chain.
    from tools.build_ppu_library import verify
    manifest = verify(bundle)
    verify_payloads(bundle, manifest)
    info = manifest.get('build_info', {})
    if info.get('version') != __version__ or info.get('source_hashes') != source_files(root):
        raise ValueError('bundle/Python sources or package version differ; rebuild the complete public bundle')
    if info.get('device_arch') != 'ppu_10' or info.get('supported_targets') != ['ppu10']:
        raise ValueError('this builder only produces ppu10; ppu15/ppu17 require separate validated builds')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('the PPU wheel builder currently supports Linux x86_64 only')
    exports = {line.split()[-1] for line in subprocess.check_output(
        ['nm', '-D', '--defined-only', str(bundle / PUBLIC_LIBRARY)], text=True).splitlines() if line.split()}
    # Every shipped file must be sealed; never sweep stray SDK DSOs or build objects into the wheel.
    extras = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts} - set(manifest['payloads']) - {'manifest.json'}
    if extras:
        raise ValueError('unsealed files in native bundle: ' + repr(sorted(extras)))
    for name in manifest['payloads']:
        if '.so' in Path(name).name and name not in (
                PUBLIC_LIBRARY, 'quactlize-runtime/libquactlize_kpack_dispatch.so',
                'quactlize-runtime/libquactlize_ppu_execution.so', 'quactlize-runtime/libquactlize_ppu_prefill.so'):
            raise ValueError('system/unexpected library must not be bundled: ' + name)
    return dict(schema='quactlize.ppu-wheel.v1', version=__version__,
                source_revision=info['source_revision'], supported_targets=info['supported_targets'],
                default_target=info['default_target'], build_info=info,
                manifest_sha256=file_sha256(bundle / 'manifest.json'), verified_exports=sorted(exports),
                sdk_requirements=dict(build_sdk=info['sdk_release'], sdk_identity=manifest['sdk'],
                                      platform_abi_versions=info.get('platform_abi_versions', []),
                                      **elf_dependencies(bundle)),
                abi='ctypes C ABI; optional host extension uses the build platform Torch ABI')
