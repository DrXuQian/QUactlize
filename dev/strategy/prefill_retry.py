"""Parser-only prefill retry: retain old PASS evidence, never re-label it.

The new run owns new attempts. Copied old receipts keep their original
authority and samples; a changed kernel, fixture, timer or device is rejected.
This is not a general cross-version/cross-device measurement cache.
"""
import hashlib
import json
from pathlib import Path

from dev.strategy.components import stage_role, component_statistics
from dev.strategy.journal import ArmJournal, atomic_json
from dev.strategy.resource_pilot import load_probe, verified_plan
from quactlize.runtime.tuning import digest


CLASSIFIER_BEFORE = 'cd2f7541a029b186f8851a43e80b7cb3f66d6f7c66fdeac7cc6587bc6b61ecb4'
ORCHESTRATION = {
    'dev/strategy/prefill_retry.py',
    'tools/run_kpack_strategy_prefill.py',
    'tools/run_kpack_strategy_prefill_box.sh',
    'tools/pack_kpack_strategy_review.py',
}


def read(root, relative):
    path = root/relative
    if not path.resolve(strict=True).is_relative_to(root) or any(
            p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError('retry evidence must be a regular file inside its run: '+str(path))
    raw = path.read_bytes()
    value = json.loads(raw)
    if 'record_sha256' in value and value['record_sha256'] != digest(
            {k: v for k, v in value.items() if k != 'record_sha256'}):
        raise ValueError('retry evidence digest differs: '+str(path))
    return value, raw


def prior_root(path):
    root = Path(path).resolve(strict=True)
    if (root/'measurement').is_dir(): root = root/'measurement'
    if (root/'prior/index.json').exists():
        raise ValueError('resume the existing retry; chained parser migrations are not supported')
    # A stopped campaign may have no final summary. The caller acquires the
    # SAME physical-device lock before import, so an old live child cannot
    # race with the retry even when its last receipt still says RUNNING.
    for name in ('authority.json', 'execution-plan.json', 'plan/plan.json'):
        read(root, name)
    return root


def same_inputs(previous, current, *, classifier_source):
    for value in (previous, current):
        if value.get('sha256') != digest(value['files']):
            raise ValueError('retry source closure digest differs')
    old, new = previous['files'], current['files']
    changed = {p for p in old.keys() | new.keys() if old.get(p) != new.get(p)}
    classifier = 'dev/strategy/components.py'
    if classifier in changed:
        # Admit precisely this parser repair, not arbitrary cost/timer edits
        # hidden in the same module. Removing its one added name reconstructs
        # the reviewed old file byte-for-byte.
        before = classifier_source.replace(b"'reduce_bf16_grouped', ", b'', 1)
        if (old.get(classifier) != CLASSIFIER_BEFORE or
                hashlib.sha256(classifier_source).hexdigest() != new.get(classifier) or
                hashlib.sha256(before).hexdigest() != CLASSIFIER_BEFORE):
            raise ValueError('retry allows only the reviewed BF16 reducer classifier repair')
        changed.remove(classifier)
    if changed-ORCHESTRATION:
        raise ValueError('retry measurement inputs changed: '+', '.join(sorted(changed-ORCHESTRATION)))


def previous_probe(root, frozen, current_harness, source_root, *, provider_upgrade=False):
    probe = load_probe(root, pilot=False)
    old = probe['plan']
    omit = {'plan_sha256', 'source_identity'}
    # The catalog seal also covers small-M policy, but the actual prefill
    # points/configs/incumbents below must still compare byte-for-byte.
    if provider_upgrade: omit.add('catalog')
    if {k:v for k,v in old.items() if k not in omit} != {
            k:v for k,v in frozen.items() if k not in omit}:
        raise ValueError('retry frozen workload/config inventory changed')
    authority, _ = read(root, 'authority.json')
    if authority['resource'] != probe['authority']:
        raise ValueError('retry resource authority differs')
    if provider_upgrade:
        from dev.strategy.prefill_upgrade import source_delta
        source_delta(old['source_identity'], frozen['source_identity'], 'source')
        source_delta(authority['harness'], current_harness, 'harness')
    else:
        classifier = (source_root/'dev/strategy/components.py').read_bytes()
        same_inputs(old['source_identity'], frozen['source_identity'], classifier_source=classifier)
        same_inputs(authority['harness'], current_harness, classifier_source=classifier)
    return probe


def copy_exact(destination, raw):
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != raw:
            raise ValueError('retry destination differs: '+str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only immutable evidence in a new run is copied. Never update its seals.
    with destination.open('xb') as stream:
        stream.write(raw)


def reuse_build(root, output, name, sdk, verify):
    files = {'dequant': ('libquactlize_ppu_dequant.so', 'manifest.json', 'native.json'),
             'helpers': ('libstrategy_prefill.so', 'manifest.json')}[name]
    old = verify(root/name, sdk)
    for filename in files:
        source = root/name/filename
        if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root):
            raise ValueError('retry library path escapes its run')
        destination = output/name/filename
        # DSOs are small helper libraries, not the TC compile cache.
        copy_exact(destination, source.read_bytes())
    if verify(output/name, sdk) != old:
        raise ValueError('copied helper manifest changed')


def checked_component(root, cid, authority):
    value, raw = read(root, Path('components')/(cid+'.json'))
    identity = value['identity']
    provider_event = value.get('scope') == 'PROVIDER_CALL_EVENTS'
    kind = 'provider' if provider_event else stage_role(identity['node']['name'], identity['contract'].get('split', 1))
    if (digest(identity) != cid or identity['authority'] != authority or
            value.get('kind') != identity['kind'] or
            identity['kind'] != kind or
            value.get('timed_kernel_types') != (None if provider_event else 1) or value.get('first_use_excluded') is not True or
            value.get('host_launch_cost_added') is not False):
        raise ValueError('retry component identity/timing scope changed')
    if any(value.get(k) != v for k,v in component_statistics(value).items()):
        raise ValueError('retry component samples/statistics differ')
    return value, raw


def import_passes(root, output, plan, authority, jobs, *, provider_upgrade=False):
    """Called by the parent while it owns the original physical-device lock."""
    old_plan, _ = read(root, 'execution-plan.json')
    verified_plan(old_plan)
    old, old_raw = read(root, 'authority.json')
    if old_plan != plan or old['plan'] != plan['plan_sha256']:
        raise ValueError('retry resolved execution plan changed')
    upgrade = None
    if provider_upgrade:
        from dev.strategy.prefill_upgrade import authority_check, replaces
        upgrade = authority_check(old, authority)
    elif {k:v for k,v in old.items() if k != 'harness'} != {
            k:v for k,v in authority.items() if k != 'harness'}:
        raise ValueError('retry SDK/device/binary/oracle/protocol authority changed')
    index_path = output/'prior/index.json'
    if index_path.exists():
        index, _ = read(output, 'prior/index.json')
        if index['previous_output'] != str(root) or index['authority_sha256'] != digest(old):
            raise ValueError('retry parent changed')
        if index.get('provider_upgrade') != upgrade:
            raise ValueError('retry provider migration changed')
        for cid, expected in index['components'].items():
            value, _ = checked_component(output, cid, dict(sha256=digest(old)))
            if value['record_sha256'] != expected:
                raise ValueError('imported component changed on resume')
        return index
    previous = ArmJournal(root/'arms', dict(sha256=digest(old)), '0'*32)
    receipts, components, replaced = {}, {}, []
    for point in plan['points']:
        for job in jobs(point):
            if provider_upgrade and replaces(point, job):
                replaced.append(point['id']+'/'+job['id'])
                continue
            row = previous.read(point, job)
            if not row or row['status'] != 'PASS': continue
            relative = Path('arms')/point['id']/job['id']/'current.json'
            checked, raw = read(root, relative)
            if checked != row: raise ValueError('retry parent changed while reading')
            for cid in row['data']['components']:
                if cid in components: continue
                path = Path('components')/(cid+'.json')
                value, component_raw = checked_component(root, cid, previous.authority)
                copy_exact(output/path, component_raw)
                components[cid] = value['record_sha256']
            copy_exact(output/'prior'/relative, raw)
            receipts[point['id']+'/'+job['id']] = row['record_sha256']
    # Publish the index last: partial imports can be checked and continued.
    copy_exact(output/'prior/authority.json', old_raw)
    index = dict(schema='quactlize.prefill-parser-retry.v1', previous_output=str(root),
        authority_sha256=digest(old), receipts=receipts, components=components,
        passed_jobs=len(receipts), scope='PRIOR_PASS_UNCHANGED_NEW_FAILED_OR_PENDING_ATTEMPTS_ONLY')
    if upgrade:
        index.update(schema='quactlize.prefill-provider-retry.v1', provider_upgrade=upgrade,
                     replaced_jobs=replaced)
    index['record_sha256'] = digest(index)
    atomic_json(index_path, index)
    return index


class PrefillJournal(ArmJournal):
    def __init__(self, directory, authority, invocation):
        super().__init__(directory, authority, invocation)
        self.index = None
        if (self.root.parent/'prior/index.json').exists():
            self.index, _ = read(self.root.parent, 'prior/index.json')
            old, _ = read(self.root.parent, 'prior/authority.json')
            if digest(old) != self.index['authority_sha256']:
                raise ValueError('retry imported authority changed')
            self.previous = ArmJournal(self.root.parent/'prior/arms',
                                       dict(sha256=digest(old)), invocation)

    def read(self, point, arm):
        row = super().read(point, arm)
        key = point['id']+'/'+arm['id']
        if not self.index or key not in self.index['receipts']: return row
        if row is not None: raise ValueError('retry remeasured an imported PASS')
        old = self.previous.read(point, arm)
        if not old or old['status'] != 'PASS' or old['record_sha256'] != self.index['receipts'][key]:
            raise ValueError('retry imported PASS receipt changed')
        return old
