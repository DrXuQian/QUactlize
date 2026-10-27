"""Paired Q8 helper: actual arithmetic/maps, independent GGUF and runner gates."""
from collections import Counter
import copy
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from tools import run_dense_dequant as run


class PairedHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();out=Path(cls.temp.name)/'host.so'
        root=Path(__file__).resolve().parents[1]
        sdk=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
        subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-ffp-contract=off',
                        '-I'+str(root),'-I'+str(root/'quactlize/include'),
                        '-I'+str(root/'third_party/actlize/include'),'-I'+str(sdk/'include'),
                        str(root/'tests/dense_q8_paired_host.cpp'),'-o',str(out)],check=True)
        cls.lib=C.CDLL(str(out))
        cls.lib.dense_q8_paired_exhaustive.argtypes=[C.POINTER(C.c_uint64)]
        cls.lib.dense_q8_byteperm_encoded.argtypes=[C.POINTER(C.c_uint64),C.c_int]
        cls.lib.dense_q8_paired_map.argtypes=[C.c_uint]*3
        cls.lib.dense_q8_paired_map.restype=C.c_uint
        cls.lib.dense_q8_paired_host.argtypes=[C.c_int,C.c_uint]+[C.c_void_p]*3+[C.c_uint]*2+[C.c_int]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_all_codes_scales_both_phases_distinct_n_scales(self):
        count=C.c_uint64()
        self.assertEqual(self.lib.dense_q8_paired_exhaustive(C.byref(count)),0)
        self.assertEqual(count.value,2*256*63488)

    def test_byteperm_all_retained_pairs_discarded_bytes_and_wrong_phase(self):
        count=C.c_uint64()
        self.assertEqual(self.lib.dense_q8_byteperm_encoded(C.byref(count),0),0)
        self.assertEqual(count.value,2*256*256*4)
        self.assertNotEqual(self.lib.dense_q8_byteperm_encoded(C.byref(count),1),0)

    def test_actual_layout_matches_gguf_all_candidates_and_fault(self):
        import numpy as np
        for n,k in ((256,256),(512,512),(256,1024)):
            packed,gold,_=run.fixture(8,n,k)
            for signed in (0,1,2):
                for threads in (128,256):
                    for fault in (0,1):
                        with self.subTest(n=n,k=k,signed=signed,threads=threads,fault=fault):
                            guarded=np.full(n*k+128,0xa5a5,dtype='<u2');out=guarded[64:-64].reshape(n,k)
                            rc=self.lib.dense_q8_paired_host(signed,threads,packed['low'].ctypes.data,
                                packed['units'].ctypes.data,out.ctypes.data,n,k,fault)
                            self.assertEqual(rc,0)
                            self.assertTrue(np.all(guarded[:64]==0xa5a5) and np.all(guarded[-64:]==0xa5a5))
                            self.assertEqual(bool(np.any((out!=gold[0])&(((out|gold[0])&0x7fff)!=0))),bool(fault))

    def test_maps_ownership_and_per_instruction_global_footprints(self):
        scale=lambda pair,g:self.lib.dense_q8_paired_map(0,pair,g)
        self.assertEqual(sorted(scale(p,g) for p in range(32) for g in range(4)),list(range(128)))
        for threads in (128,256):
            self.assertEqual(sorted(t+p*threads for t in range(threads) for p in range(512//threads)),list(range(512)))
            for first in range(0,threads,32):
                for part in range(512//threads):
                    addresses=[scale(t//16+part*(threads//16),(t%16*8)//32) for t in range(first,first+32)]
                    # Model32 banks,4B each, broadcast allowed. Not an ACU claim.
                    per_bank={}
                    for a in addresses:per_bank.setdefault(a%32,set()).add(a)
                    self.assertTrue(all(len(v)==1 for v in per_bank.values()))
        for _,n,k in run.POINTS[2:]:
            for n0 in (0,n-64):
                for k0 in (0,k-128):
                    for tasks,width,address in (
                        (128,4,lambda t:2*((k0//32+t//32)*n+n0+2*(t%32))),
                        (512,16,lambda t:16*((k0//2+t//8)*(n//8)+n0//8+t%8)),
                        (512,16,lambda t:16*((n0+2*(t//16))*(k//8)+k0//8+t%16)),
                        (512,16,lambda t:16*((n0+2*(t//16)+1)*(k//8)+k0//8+t%16))):
                        unique=Counter()
                        for first in range(0,tasks,32):
                            addrs=[address(t) for t in range(first,first+32)]
                            self.assertTrue(all(a%width==0 for a in addrs))
                            unique.update(a+b for a in addrs for b in range(width))
                            for sector in (32,64,128):
                                histogram=Counter((a+b)//sector for a in addrs for b in range(width))
                                self.assertEqual(set(histogram.values()),{sector})
                        self.assertEqual(set(unique.values()),{1})


class PairedRunner(unittest.TestCase):
    def test_inventory_retains_native_and_all_previous_winners(self):
        p=run.plan(list(range(2,8)),67108864,q8=True,native=True)
        self.assertEqual(p['variants'],[-1,0,22,23,38,41])
        self.assertEqual(p['order'],p['variants']+p['variants'][::-1])
        self.assertEqual(p['correctness_storage'],['F16'])
        self.assertFalse(p['production_changed'])
        self.assertEqual(p['experiment'],'Q8_PRODUCT_NATIVE_F16_AB')
        for control in (0,22,23):
            medians={str(v):10. for v in p['variants']};medians[str(control)]=5.
            self.assertEqual(run.profile_variants(medians,q8=True,native=True),(-1,control,38,41))
        with self.assertRaises(ValueError):run.plan([2],0,native=True)
        with tempfile.TemporaryDirectory() as temp:
            lib=Path(temp)/'lib.so';lib.touch();out=Path(temp)/'plan'
            subprocess.run([sys.executable,run.__file__,'--output',str(out),'--q8',
                            '--native-library',str(lib),'--l2-bytes','67108864','--plan-only'],check=True,capture_output=True)
            self.assertEqual(json.loads((out/'plan.json').read_text()),p)

    def test_profile_identity_and_bf16_rejection(self):
        for variant,c in run.q8_paired(True).items():
            symbol=f'_dense_dequant_q8_pairedILb{c["signed"]}ELj{c["threads"]}ELb{c["byteperm"]}E'
            native=dict(kernels={symbol:{}})
            self.assertEqual(run.profile_entry(native,5,variant,0),
                             dict(symbol=symbol,grid=[128,16,1],block=[c['threads'],1,1]))
            with self.assertRaises(ValueError):run.profile_entry(native,5,variant,1)

    def test_device_gate_observed_counts_negative_and_bf16(self):
        import numpy as np
        def bench(coverage=2*256*63488,negative=1,reject=24):
            values=[np.array(row,dtype='<u4').tobytes() for row in
                    ([0,0,0xffffffff,coverage],[0,negative,0,coverage])]
            return SimpleNamespace(lib=SimpleNamespace(quactlize_q8_paired_gate=Mock(return_value=0),
                quactlize_q8_byteperm_gate=Mock(return_value=0)),
                r=SimpleNamespace(alloc=Mock(return_value=4096),stream=C.c_void_p(16)),
                sdk=SimpleNamespace(lib=SimpleNamespace(hggcMemcpy=Mock(return_value=0)),
                    synchronize=Mock(),download=Mock(side_effect=values)),launch=Mock(return_value=reject))
        self.assertEqual(run.paired_gate(bench()),run.paired_gate_receipt())
        self.assertEqual(run.paired_gate(bench(),True),run.paired_gate_receipt(True))
        for options in (dict(coverage=256),dict(negative=0),dict(reject=0),dict(reject=1)):
            with self.assertRaises(ValueError):run.paired_gate(bench(**options))
            with self.assertRaises(ValueError):run.paired_gate(bench(**options),True)

    def test_byteperm_opt_in_inventory_plan_and_precise_profile_identity(self):
        p=run.plan(list(range(2,8)),67108864,q8=True,native=True,byteperm=True)
        self.assertEqual(p['variants'],[-1,38,41,43])
        self.assertEqual(p['order'],[-1,38,41,43,43,41,38,-1])
        self.assertEqual(p['candidate'],43)
        self.assertEqual(p['matched_control'],41)
        self.assertEqual(p['numeric_anchor'],0)
        self.assertFalse(p['production_changed'])
        self.assertEqual(run.profile_variants({},True,True,True),(-1,38,41,43))
        self.assertNotIn(43,run.variants_for(q8=True,native=True))
        with self.assertRaises(ValueError):run.variants_for(byteperm=True)
        old='_dense_dequant_q8_pairedILb0ELj128ELb0E'
        new='_dense_dequant_q8_pairedILb0ELj128ELb1E'
        native=dict(kernels={old:{},new:{}})
        self.assertEqual(run.profile_entry(native,7,41,0)['symbol'],old)
        self.assertEqual(run.profile_entry(native,7,43,0)['symbol'],new)
        with self.assertRaises(ValueError):run.profile_entry(dict(kernels={old:{}}),7,43,0)
        with tempfile.TemporaryDirectory() as temp:
            lib=Path(temp)/'lib.so';lib.touch();out=Path(temp)/'plan'
            subprocess.run([sys.executable,run.__file__,'--output',str(out),'--q8','--q8-byteperm',
                '--native-library',str(lib),'--l2-bytes','67108864','--plan-only'],check=True,capture_output=True)
            self.assertEqual(json.loads((out/'plan.json').read_text()),p)

    def test_resume_requires_new_exhaustive_gate_and_candidate_rows(self):
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
        for key,value in (('pairs',0),('distinct_scales',False),('bad_signed',1),('negative_bad',0)):
            bad=copy.deepcopy(data);bad['paired_gate'][key]=value
            with self.assertRaises(ValueError):run.validate_point(bad,4,67108864,q8=True,native=True)
        with self.assertRaises(ValueError):run.validate_point(data,4,67108864,q8=True)
        candidate=copy.deepcopy(data)
        candidate['experiment']='Q8_BYTEPERM_NATIVE_F16_AB'
        for name in ('samples_us','median_us','round_medians_us'):
            candidate[name]['43']=copy.deepcopy(candidate[name]['41'])
        candidate['proof']['0']['43']=proof.copy()
        with self.assertRaises(ValueError):run.validate_point(candidate,4,67108864,q8=True,native=True,byteperm=True)
        candidate['byteperm_gate']=run.paired_gate_receipt(True)
        self.assertIs(run.validate_point(candidate,4,67108864,q8=True,native=True,byteperm=True),candidate)
        candidate['byteperm_gate']['negative_bad']=0
        with self.assertRaises(ValueError):run.validate_point(candidate,4,67108864,q8=True,native=True,byteperm=True)
