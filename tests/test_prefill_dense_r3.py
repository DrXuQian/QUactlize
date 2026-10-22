import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))

class RoundThreeHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();so=Path(cls.temp.name)/'host.so'
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off',
            '-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
            '-I'+str(SDK/'include'),str(ROOT/'tests/prefill_dense_r3_host.cpp'),'-o',str(so)],check=True)
        cls.lib=C.CDLL(str(so))
        cls.lib.dense_r3_host.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
        cls.lib.dense_r3_fp32.argtypes=[C.c_uint,C.c_uint];cls.lib.dense_r3_fp32.restype=C.c_float

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_direct_fp32_every_integer_code(self):
        import struct
        for q,bits,bias in ((12,4,0),(13,5,0),(14,6,32)):
            for raw in range(1<<bits):
                got=self.lib.dense_r3_fp32(q,raw)
                self.assertEqual(struct.pack('f',got),struct.pack('f',raw-bias))

    def test_pair_geometry_independent_gguf_bits_and_faults(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference import gguf_kpack as ref
        from tools.kpack_warmup_fixture import prepare_expert
        from dev.bf16_compute.fixture import bf16_bits
        for q in (12,13,14):
            for n,k in ((256,512),(256,1024),(512,512)):
                s=ref.SPECS[q];rng=np.random.default_rng(q+n+k)
                raw=rng.integers(0,256,(n*k//256,s.raw_bytes),dtype='u1')
                for offset in (s.d_offset,s.dmin_offset):
                    if offset>=0:
                        raw[:,offset:offset+2]=rng.uniform(-.009,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                        # Exercise signed zero, subnormal, and non-dyadic scales.
                        raw[:8,offset:offset+2]=np.array([0,0x8000,1,0x8001,0x3555,0xb555,0x0400,0x8400],dtype='<u2').view('u1').reshape(-1,2)
                packed=prepare_expert(raw,q,n,k)
                gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
                for bf in (0,1):
                    want=bf16_bits(gold)if bf else gold.astype('<f2').view('<u2')
                    for variant in range(8,14):
                        with self.subTest(q=q,n=n,k=k,bf=bf,variant=variant):
                            out=np.full_like(want,0x7fff)
                            args=(q,bf,variant,*[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                            self.assertEqual(self.lib.dense_r3_host(*args,0),0)
                            self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            for fault in (1,3):
                                self.assertEqual(self.lib.dense_r3_host(*args,fault),0)
                                self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            if variant in (10,11):self.assertNotEqual(self.lib.dense_r3_host(*args,2),0)

    def test_each_output_instruction_is_coalesced(self):
        # A lane owns K8 in each of two N rows. Count each 16-byte store
        # separately: do not hide holes by combining two instructions.
        for tn,tk in ((32,128),(32,256),(16,256)):
            for base in range(0,tn//2*(tk//8),32):
                for side in (0,1):
                    addresses=[((2*(i//(tk//8))+side)*1024+(i%(tk//8))*8)*2
                               for i in range(base,base+32)]
                    self.assertTrue(all(a%16==0 for a in addresses))
                    for sector in (32,64,128):
                        footprint={(a+b)//sector for a in addresses for b in range(16)}
                        self.assertEqual(len(footprint),512//sector)

    def test_each_input_instruction_sector_coverage_and_existing_high_reuse(self):
        from collections import Counter
        for q in (12,13,14):
            for tn,tk in ((32,128),(32,256),(16,256)):
                vectors=tn//8;extent=512
                for n0 in (0,tn,256):
                    for k0 in (0,256,512):
                        for base in range(0,(tk//32)*8*vectors,32):
                            words=[];high=[]
                            for i in range(base,base+32):
                                col=n0+(i%vectors)*8;kk=k0+i//(vectors*8)*32+(i//vectors)%8
                                words.append(((kk//32)*8+kk%8)*extent+col)
                                if q==13:
                                    pn=(col&~15)|(col&7)|(((kk>>7)&1)<<3)
                                    kg=(kk//256)*16|(((kk>>6)&1)<<3)|(kk&7)
                                    high.append(kg*extent+pn)
                                elif q==14:high.append(((kk//64)*8+kk%8)*extent+col)
                            for is_high,plane in enumerate((words,*([high]if high else[]))):
                                self.assertTrue(all(w%8==0 for w in plane))
                                reuse=(2 if tn==32 else 4)if is_high and q==13 else 2 if is_high and q==14 and tn==16 else 1
                                self.assertEqual(len(set(plane)),32//reuse)
                                for sector in (32,64,128):
                                    coverage=Counter((w*2+b)//sector for w in set(plane)for b in range(16))
                                    # Low N32 spans64B, N16 spans32B per K
                                    # residue. Q5's canonical high plane has
                                    # 16B N8 spans; N-bit3/K32 share words.
                                    # Retain these real limitations explicitly.
                                    expected=(16 if sector==32 else tn)if is_high and q==13 else min(sector,tn*2)
                                    self.assertEqual(set(coverage.values()),{expected})

    def test_inventory_retains_r2_controls_and_holds_io_constant(self):
        from dev.prefill_dense.round3 import pool,stages,composed,CONTROLS,profile_jobs
        results=[]
        for point in range(7):
            arms=pool(point);self.assertEqual(len(arms),6)
            self.assertEqual({a['algorithm']for a in arms},{-1})
            rows=[dict(arm=a,median_us=dict(dequant=20-a['dequant_variant']))for a in arms]
            for storage in (0,1):
                typed=[a for a in arms if a['storage']==storage]
                self.assertEqual(len({a['vector_io']for a in typed}),1)
                self.assertEqual(typed[0]['dequant_variant'],CONTROLS[point])
                self.assertEqual(typed[2]['dequant_variant'],typed[1]['dequant_variant']+1)
            self.assertTrue(all(stages(point,a)==('dequant',)for a in arms))
            final=composed(point,rows);self.assertEqual(len(final),4)
            for control,choice in zip(final[::2],final[1::2]):
                self.assertEqual(control['dequant_variant'],CONTROLS[point])
                self.assertEqual(control['storage'],choice['storage'])
                self.assertEqual(control['vector_io'],choice['vector_io'])
            results.append(dict(point=point,status='PASS',arms=rows))
        jobs=profile_jobs(results);self.assertEqual(len(jobs),24)
        self.assertTrue(all(j['stage']==1 for j in jobs))
        for pair in range(6):
            self.assertEqual([j['role']for j in jobs if j['pair']==pair],['control','candidate','candidate','control'])
        results[5]['status']='FAIL'
        with self.assertRaises(ValueError):profile_jobs(results)

    def test_acu_checks_exact_specialization_including_conversion_mode(self):
        import csv
        from dev.prefill_dense.round3 import check_profile
        from dev.prefill_dense.round2 import cycle_pairs
        with tempfile.TemporaryDirectory()as td:
            path=Path(td)/'raw.csv'
            row={'ID':'0','Kernel Name':'void quactlize::dequant::dense_dequant_pair<(gguf_scale::KType)4, true, 16, 256, true>(...)',
                'launch__grid_dim_x':'320','launch__grid_dim_y':'100','launch__grid_dim_z':'1',
                'launch__block_size':'128','ce__cycles_elapsed.avg':'300000','ce__cycles_elapsed.avg.per_second':'1700000000'}
            def write():
                with path.open('w')as f:
                    w=csv.DictWriter(f,fieldnames=list(row),quoting=csv.QUOTE_ALL);w.writeheader();w.writerow(row)
            job=dict(point=5,stage=1,arm='r3-bf16-dq13');write()
            evidence=check_profile(path,job);self.assertTrue(evidence['fast_fp32'])
            row['Kernel Name']=row['Kernel Name'].replace('256, true>','256, false>');write()
            with self.assertRaises(ValueError):check_profile(path,job)
            row['Kernel Name']=row['Kernel Name'].replace('256, false>','256, true>');row['launch__block_size']='256';write()
            with self.assertRaises(ValueError):check_profile(path,job)
        reports=[dict(pair=0,status='PASS',role=role,evidence=dict(cycles=c,frequency_hz=f))for role,c,f in
                 [('control',100,1e9),('candidate',95,1.7e9),('candidate',95,1.7e9),('control',100,1e9)]]
        self.assertAlmostEqual(cycle_pairs(reports)[0]['cycle_delta_pct'],-5)

    def test_reused_runner_isolated_timing_and_complete_confirmation(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from dev.prefill_dense import round2,round3
        from dev.prefill_dense.spec import workloads
        samples=[]
        class FakeBench:
            copies=2;receipt={'scope':'HOST_TEST_NOT_PERFORMANCE'}
            def __init__(self,*args,**kwargs):pass
            def handle(self,arm):return arm['name']
            def update(self,*args,**kwargs):pass
            def launch(self,*args,**kwargs):pass
            def check(self,*args,**kwargs):return {'relative_l1_error':0}
            def check_weights(self,*args,**kwargs):return {'bad':0}
            def samples(self,h,stage):samples.append((h,stage));return [1.,2.,3.]
            def close(self):pass
        with tempfile.TemporaryDirectory()as td:
            a=SimpleNamespace(arm=None,point=1,output=Path(td),sdk=Path(td),l2_bytes=64<<20,result=Path(td)/'point-1.json')
            with patch('dev.prefill_dense.bench.Bench',FakeBench):
                self.assertEqual(round2.measure(a,None,workloads()[1],None,None,dict(sha256='test'),dict(host_test=True),protocol=round3),0)
            result=json.loads(a.result.read_text())
            self.assertEqual(len(result['arms']),6);self.assertEqual(len(result['complete']),16)
            self.assertEqual(len(result['numerical']),10)
            self.assertEqual(len(samples),40);self.assertEqual({stage for _,stage in samples},{-1,1})
            self.assertTrue(result['round_three']);self.assertFalse(result['round_two'])
            self.assertFalse(result['performance_admitted']);self.assertFalse(result['production_changed'])

if __name__=='__main__':unittest.main()
