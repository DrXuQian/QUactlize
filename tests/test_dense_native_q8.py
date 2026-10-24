import copy
import csv
import ctypes as C
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dev.dense_helpers.native_q8 import NativeQ8,describe,verify,check_profile
from tools import run_dense_dequant as run


class NativeQ8Tests(unittest.TestCase):
    def test_existing_library_getter_and_callback_abi(self):
        with tempfile.TemporaryDirectory() as temp:
            library=Path(temp)/'libnative.so'
            subprocess.run(['g++','-shared','-fPIC','-O2','-std=c++17',
                            str(Path(__file__).with_name('native_q8_stub.cpp')),'-o',str(library)],check=True)
            receipt=describe(library);verify(receipt)
            ref=NativeQ8(receipt)
            data=(C.c_uint16*3)(0x517a,0,0);out=(C.c_uint16*3)()
            ref.launch(C.addressof(data),C.addressof(out),0x32100000001,C.c_void_p(0xabc))
            self.assertEqual(list(out),[0x517a,0x321,0xabc])
            for key,value in (('entry','wrong'),('qtype',12),('library_sha256','0'*64)):
                with self.assertRaises(ValueError):verify(dict(receipt,**{key:value}))
            missing=Path(temp)/'missing.so'
            subprocess.run(['g++','-shared','-fPIC','-DOMIT_GETTER',
                            str(Path(__file__).with_name('native_q8_stub.cpp')),'-o',str(missing)],check=True)
            with self.assertRaisesRegex(ValueError,'does not export'):describe(missing)

    def test_product_and_native_plan_and_profile_selection(self):
        p=run.plan(list(range(2,8)),67108864,q8=True,native=True)
        self.assertEqual(p['variants'],[-1,0,22,23,38,41])
        self.assertEqual(p['order'],[-1,0,22,23,38,41,41,38,23,22,0,-1])
        self.assertEqual(p['experiment'],'Q8_PRODUCT_NATIVE_F16_AB')
        self.assertFalse(p['native_recompiled'])
        with self.assertRaises(ValueError):run.plan([2],67108864,native=True)
        for control in (0,22,23):
            medians={str(v):10. for v in p['variants']};medians[str(control)]=8
            self.assertEqual(run.profile_variants(medians,q8=True,native=True),(-1,control,38,41))
        with tempfile.TemporaryDirectory() as temp:
            lib=Path(temp)/'lib.so';lib.touch()
            out=Path(temp)/'plan'
            subprocess.run([sys.executable,run.__file__,'--output',str(out),'--q8',
                            '--native-library',str(lib),'--l2-bytes','67108864','--plan-only'],check=True,capture_output=True)
            self.assertEqual(json.loads((out/'plan.json').read_text()),p)

    def test_native_profile_rejects_other_dtype_tail_geometry_and_extra_launch(self):
        def text(name='void dequantize_block_q8_0_f16<false>(void const*, __half*, long)',grid='(4096,1,1)',extra=False):
            out=io.StringIO();w=csv.writer(out,quoting=csv.QUOTE_ALL)
            w.writerow(['ID','Kernel Name','Kernel Mangled Name','Grid Size','Block Size'])
            w.writerow(['0',name,'mangled-native',grid,'(32,1,1)'])
            if extra:w.writerow(['1',name,'mangled-native',grid,'(32,1,1)'])
            return out.getvalue()
        self.assertEqual(check_profile(text(),4096,2048)['launches'],1)
        for value in (text(name='dequantize_block_q8_0_f16<true>(...)'),text(name='dequantize_q8_0<float>(...)'),
                      text(grid='(1024,1,1)'),text(extra=True)):
            with self.assertRaises(ValueError):check_profile(value,4096,2048)

    def test_raw_fixture_is_exact_source_of_canonical_and_equal_bytes(self):
        import hashlib
        from dev.bf16_compute.fixture import planes
        packed,gold,receipt,raw=run.fixture(8,256,512,include_raw=True)
        self.assertEqual(raw.nbytes,sum(x.nbytes for x in packed.values()))
        self.assertEqual(hashlib.sha256(memoryview(raw)).hexdigest(),receipt['raw_sha256'])
        again=planes(raw,8)
        for key,value in packed.items():self.assertEqual(value.tobytes(),again[key].tobytes())
        self.assertEqual(gold[0].nbytes,256*512*2)

    def test_native_receipt_requires_identical_raw_and_packed_workload(self):
        q,n,k=run.POINTS[4];vs=run.variants_for(q8=True,native=True)
        proof=dict(bad=0,compared=n*k,guard='PASS',control_raw_bits_equal=True)
        data=dict(status='PASS',point=4,q=q,n=n,k=k,scope=run.SCOPE,experiment='Q8_PRODUCT_NATIVE_F16_AB',
            production_changed=False,performance_admitted=False,negative_bad=100,
            traffic=dict(reads=3*67108864,writes=3*67108864),ring_copies=2,calls_per_graph=4,
            samples_us={str(v):[10.]*30 for v in vs},median_us={str(v):10. for v in vs},
            round_medians_us={str(v):[10.]*3 for v in vs},proof={'0':{str(v):proof.copy() for v in vs}},
            memory_alignment={p:{str(w):[0] for w in (16,32,64,128)} for p in ('low','units','output','raw')},
            bf16_timing=None,half2_gate=dict(pairs=256*63488,lanes=2,bad_ordinary=0,bad_fast=0,
                                           negative_bad=1,bf16_rejected=[38]),paired_gate=run.paired_gate_receipt(),
            fixture=dict(raw_sha256='a'*64),native_raw_sha256='a'*64,native_bytes=3*67108864,
            changed_address_copies_checked=2)
        self.assertIs(run.validate_point(data,4,67108864,q8=True,native=True),data)
        for key,value in (('native_bytes',1),('native_raw_sha256','b'*64),('changed_address_copies_checked',1)):
            bad=copy.deepcopy(data);bad[key]=value
            with self.assertRaises(ValueError):run.validate_point(bad,4,67108864,q8=True,native=True)
