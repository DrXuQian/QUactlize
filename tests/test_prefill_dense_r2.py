import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))

class RoundTwoHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();so=Path(cls.temp.name)/'host.so'
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off',
            '-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
            '-I'+str(SDK/'include'),str(ROOT/'tests/prefill_dense_r2_host.cpp'),'-o',str(so)],check=True)
        cls.lib=C.CDLL(str(so))
        cls.lib.dense_r2_host.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
        cls.lib.dense_stripe4.argtypes=[C.c_uint64,C.c_uint,C.c_uint];cls.lib.dense_stripe4.restype=C.c_uint64
        cls.lib.dense_unsigned_index.argtypes=[C.c_uint]*4;cls.lib.dense_unsigned_index.restype=C.c_uint64
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_unsigned_geometry_independent_gguf_bits_and_faults(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference import gguf_kpack as ref
        from tools.kpack_warmup_fixture import prepare_expert
        from dev.bf16_compute.fixture import bf16_bits
        for q in (12,13,14):
            for k in (512,1024):
                s=ref.SPECS[q];n=256;rng=np.random.default_rng(q+k)
                raw=rng.integers(0,256,(n*k//256,s.raw_bytes),dtype='u1')
                for offset in (s.d_offset,s.dmin_offset):
                    if offset>=0:raw[:,offset:offset+2]=rng.uniform(.0001,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                packed=prepare_expert(raw,q,n,k)
                gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
                for bf in (0,1):
                    want=bf16_bits(gold)if bf else gold.astype('<f2').view('<u2')
                    for variant in (4,5,6,7):
                        with self.subTest(q=q,k=k,bf=bf,variant=variant):
                            out=np.empty_like(want)
                            args=(q,bf,variant,*[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                            self.assertEqual(self.lib.dense_r2_host(*args,0),0)
                            self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            self.assertEqual(self.lib.dense_r2_host(*args,1),0)
                            self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            if variant==7:self.assertNotEqual(self.lib.dense_r2_host(*args,2),0)

    def test_striped_ownership_tails_and_per_instruction_transactions(self):
        f=self.lib.dense_stripe4
        for count in (1,3,4,7,31,128,1028,2047,2048,2049,3073,8192):
            cells=[]
            for block in range((count+2047)//2048):
                for tid in range(256):
                    for part in (0,1):
                        i=f(block,tid,part);cells.extend(range(i,min(i+4,count)))
            self.assertEqual(sorted(cells),list(range(count)))
        # Sector count for ONE dynamic vector instruction, not the union of
        # two instructions (which hid the problem in the previous audit).
        for sector in (32,64,128):
            for part in (0,1):
                old={((lane*8+part*4)*4+b)//sector for lane in range(32)for b in range(16)}
                new={(f(0,lane,part)*4+b)//sector for lane in range(32)for b in range(16)}
                self.assertEqual(len(new)*2,len(old))
                narrow={(f(0,lane,part)*2+b)//sector for lane in range(32)for b in range(8)}
                self.assertEqual(len(narrow),256//sector)
        self.assertEqual(f(2**25,255,1),2**25*2048+2044)

    def test_unsigned_planes_keep_64bit_products(self):
        f=self.lib.dense_unsigned_index
        for bits in (1,2,4):
            for extent in (256,5120,2**30):
                for n in (0,7,8,15,255):
                    for k in (0,7,8,31,63,127,128,255,256,2**31-256):
                        if bits==1:
                            col=(n&~15)|(n&7)|(((k>>7)&1)<<3)
                            kg=(k//256)*16|(((k>>6)&1)<<3)|(k&7)
                            want=kg*extent+col
                        else:want=((k//(8*(16//bits)))*8+k%8)*extent+n
                        self.assertEqual(f(bits,n,k,extent),want)

    def test_round_two_screen_is_bounded_and_full_not_sum(self):
        from dev.prefill_dense.round2 import pool,stages,composed,profile_jobs
        for point in range(7):
            arms=pool(point);self.assertEqual(len(arms),14)
            self.assertEqual({a['algorithm']for a in arms},{-1})
            results=[]
            for a in arms:
                row=dict(arm=a,median_us={k:1.+a['dequant_variant']for k in stages(point,a)})
                results.append(row)
            pairs=composed(point,results)
            self.assertEqual(len(pairs),4)
            self.assertEqual({p['storage']for p in pairs},{0,1})
        results=[dict(point=i,status='PASS',arms=[dict(arm=a,median_us={k:float(a['dequant_variant'])for k in stages(i,a)})for a in pool(i)])for i in range(7)]
        jobs=profile_jobs(results)
        self.assertEqual(len(jobs),24)
        self.assertEqual({j['point']for j in jobs},{0,5,6})
        self.assertTrue(all(j['stage']in (0,1,3)for j in jobs))

    def test_acu_uses_cycles_and_validates_selected_geometry(self):
        from dev.prefill_dense.round2 import check_profile,cycle_pairs
        import csv
        with tempfile.TemporaryDirectory()as td:
            p=Path(td)/'raw.csv'
            row={'ID':'0','Kernel Name':'dense_convert_stripe4',
                'launch__grid_dim_x':'25600','launch__grid_dim_y':'1','launch__grid_dim_z':'1',
                'launch__block_size':'256','ce__cycles_elapsed.avg':'235877',
                'ce__cycles_elapsed.avg.per_second':'1700000000'}
            def write():
                with p.open('w')as f:
                    w=csv.DictWriter(f,fieldnames=list(row),quoting=csv.QUOTE_ALL);w.writeheader();w.writerow(row)
            job=dict(point=0,stage=3,arm='r2-bf16-io2');write()
            self.assertEqual(check_profile(p,job)['cycles'],235877)
            row['launch__block_size']='128';write()
            with self.assertRaises(ValueError):check_profile(p,job)
            row['launch__block_size']='256';row['ce__cycles_elapsed.avg']='nan';write()
            with self.assertRaises(ValueError):check_profile(p,job)
        reports=[dict(pair=0,status='PASS',role=role,evidence=dict(cycles=c,frequency_hz=f))for role,c,f in
                 [('control',100,1e9),('candidate',90,1.7e9),('candidate',90,1.7e9),('control',100,1e9)]]
        pairs=cycle_pairs(reports)
        self.assertAlmostEqual(pairs[0]['cycle_delta_pct'],-10)
        self.assertEqual(cycle_pairs(reports[:-1]),[])

    def test_measurement_orchestration_without_device(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from dev.prefill_dense.round2 import measure
        from dev.prefill_dense.spec import workloads
        import json
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
                self.assertEqual(measure(a,None,workloads()[1],None,None,dict(sha256='test'),dict(host_test=True)),0)
            result=json.loads(a.result.read_text())
            self.assertEqual(len(result['arms']),14)
            self.assertEqual(len(result['complete']),16)
            self.assertEqual(len(result['numerical']),18)
            self.assertEqual(len(samples),96)
            self.assertEqual({stage for _,stage in samples},{-1,0,1,3})
            self.assertFalse(result['performance_admitted'])

if __name__=='__main__':unittest.main()
