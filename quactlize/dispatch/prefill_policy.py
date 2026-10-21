"""One active prefill catalog; old catalogs remain immutable evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = 'policies/kpack_prefill_refresh_v1'
LEGACY = 'policies/kpack_prefill_confirmation_v1'
EVIDENCE = ROOT / (STEM + '.json')
HEADER = ROOT / (STEM + '.hpp')
ADMISSION = '424_REFRESH_CONFIRMED_6_CURRENT_RETAINED_MODEL_AB_PENDING'


def receipt_fields():
    return dict(source_path=STEM + '.json', header_path=STEM + '.hpp', admission=ADMISSION)


def check_receipt(receipt, hashes, evidence_sha):
    # Packages made before refresh did not record the paths. Never accept an
    # arbitrary manifest path, or silently verify a new payload as the old one.
    source = receipt.get('source_path', LEGACY + '.json')
    header = receipt.get('header_path', LEGACY + '.hpp')
    if ((source, header) not in {(s + '.json', s + '.hpp') for s in (STEM, LEGACY)} or
            not evidence_sha or not receipt.get('header_sha256') or
            hashes.get(source) != evidence_sha or hashes.get(header) != receipt['header_sha256']):
        raise ValueError('confirmed prefill selector identity differs')
