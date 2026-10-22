import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))

class RoundFourHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();so=Path(cls.temp.name)/'host.so'
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off',
            '-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
            '-I'+str(SDK/'include'),str(ROOT/'tests/prefill_dense_r4_host.cpp'),'-o',str(so)],check=True)
        cls.lib=C.CDLL(str(so));cls.lib.dense_r4_host.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
        cls.lib.dense_r4_layout.argtypes=[C.c_uint]*5;cls.lib.dense_r4_layout.restype=C.c_uint
        cls.lib.dense_full4.argtypes=[C.c_uint64,C.c_uint,C.c_uint];cls.lib.dense_full4.restype=C.c_uint64
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_cooperative_readers_raw_gguf_and_faults(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference import gguf_kpack as ref
        from tools.kpack_warmup_fixture import prepare_expert
        from dev.bf16_compute.fixture import bf16_bits
        for q in (12,13,14):
            for n,k in ((256,512),(256,1024),(512,512)):
                s=ref.SPECS[q];rng=np.random.default_rng(q+n+k+4)
                raw=rng.integers(0,256,(n*k//256,s.raw_bytes),dtype='u1')
                for offset in (s.d_offset,s.dmin_offset):
                    if offset>=0:
                        raw[:,offset:offset+2]=rng.uniform(-.009,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                        raw[:8,offset:offset+2]=np.array([0,0x8000,1,0x8001,0x3555,0xb555,0x0400,0x8400],dtype='<u2').view('u1').reshape(-1,2)
                packed=prepare_expert(raw,q,n,k);gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
                for bf in (0,1):
                    want=bf16_bits(gold)if bf else gold.astype('<f2').view('<u2')
                    for variant in range(14,22):
                        with self.subTest(q=q,n=n,k=k,bf=bf,variant=variant):
                            out=np.full_like(want,0x7fff)
                            args=(q,bf,variant,*[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                            self.assertEqual(self.lib.dense_r4_host(*args,0),0)
                            self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            self.assertEqual(self.lib.dense_r4_host(*args,1),0)
                            self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            if variant==21:self.assertNotEqual(self.lib.dense_r4_host(*args,2),0)
                            if variant>=17:self.assertNotEqual(self.lib.dense_r4_host(*args,4),0)
                            if variant in (17,20):
                                self.assertEqual(self.lib.dense_r4_host(*args,5),0)
                                self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))

    def test_bijection_vector_alignment_and_producer_bank_model(self):
        from collections import Counter
        f=self.lib.dense_r4_layout
        for n,k in ((16,256),(32,128),(32,256),(64,128),(64,256)):
            for plane in (0,1):
                with self.subTest(n=n,k=k,plane=plane):
                    addresses=[f(n,k,plane,p,w)for p in range(n//2)for w in range(k//4)]
                    self.assertEqual(set(addresses),set(range(n//2*(k//4))))
                    for p in range(n//2):
                        for w in range(0,k//4,4):
                            start=f(n,k,plane,p,w);self.assertEqual(start%4,0)
                            self.assertEqual([f(n,k,plane,p,w+i)for i in range(4)],list(range(start,start+4)))
                    coords=[]
                    for i in range((k//32)*8*(n//8)):
                        x=f(n,k,2,i,plane);coords.append((x&255,(x>>8)&255,x>>16))
                    self.assertEqual(set(coords),{(c,r,g)for c in range(0,n,8)for r in range(8)for g in range(k//32)})
                    # 32x4B bank footprint, both 16/32-lane service models.
                    # This is not a substitute for native per-PC ACU evidence.
                    for phase in (16,32):
                        for base in range(0,len(coords),phase):
                            for p in range(4):
                                at=[f(n,k,plane,c//2+p,g*8+r)*(2 if plane else 4)for c,r,g in coords[base:base+phase]]
                                self.assertEqual(len({a//4 for a in at}),phase)
                                banks=Counter((a//4)%32 for a in at)
                                self.assertEqual(max(banks.values()),1)

    def test_full_io_ownership_and_each_instruction_coalescing(self):
        f=self.lib.dense_full4
        for blocks in (1,2,5):
            cells=[f(b,t,p)+j for b in range(blocks)for t in range(256)for p in range(4)for j in range(4)]
            self.assertEqual(sorted(cells),list(range(blocks*4096)))
        for sector in (32,64,128):
            for p in range(4):
                for size in (2,4):
                    addresses=[f(0,lane,p)*size for lane in range(32)]
                    self.assertTrue(all(a%(4*size)==0 for a in addresses))
                    covered={(a+j)//sector for a in addresses for j in range(4*size)}
                    self.assertEqual(len(covered),32*4*size//sector)
        self.assertEqual(f(2**25,255,3),2**25*4096+4092)

    def test_vector_metadata_and_wider_low_plane_footprints(self):
        from collections import Counter
        f=self.lib.dense_r4_layout
        for n,k in ((16,256),(32,128),(32,256),(64,128),(64,256)):
            for sector in (32,64,128):
                # Canonical units copied linearly, every active lane reads16B.
                for unit in (16,36):
                    count=n*unit//16
                    for n0 in (0,n,2*n):
                        start=n0*unit
                        for base in range(0,count,32):
                            active=min(32,count-base)
                            coverage=Counter((start+i*16+j)//sector for i in range(base,base+active)for j in range(16))
                            self.assertEqual(sum(coverage.values()),active*16)
                            self.assertEqual(len(coverage),((start+base*16)%sector+active*16+sector-1)//sector)
                            self.assertTrue(all(0<x<=sector for x in coverage.values()))
                # Each low-code vector instruction, separately. N64 covers
                #128B; narrowerN retains its actual partially-used sectors.
                for base in range(0,k//32*8*(n//8),32):
                    addresses=[]
                    for i in range(base,base+32):
                        coord=f(n,k,2,i,0);col=coord&255;r=(coord>>8)&255;g=coord>>16
                        addresses.append(((g*8+r)*512+col)*2)
                    cover=Counter((a+j)//sector for a in addresses for j in range(16))
                    self.assertEqual(set(cover.values()),{min(sector,n*2)})

    def test_high_plane_duplicate_reads_are_accounted_per_instruction(self):
        f=self.lib.dense_r4_layout
        for n,k in ((16,256),(32,128),(32,256),(64,128),(64,256)):
            for q in (13,14):
                for n0 in (0,n,256):
                    for k0 in (0,256,512):
                        for base in range(0,k//32*8*(n//8),32):
                            addresses=[]
                            for i in range(base,base+32):
                                coord=f(n,k,2,i,1);col=n0+(coord&255);kk=k0+(coord>>16)*32+((coord>>8)&255)
                                if q==13:
                                    pn=(col&~15)|(col&7)|(((kk>>7)&1)<<3)
                                    kg=(kk//256)*16|(((kk>>6)&1)<<3)|(kk&7)
                                    addresses.append((kg*512+pn)*2)
                                else:addresses.append((((kk//64)*8+kk%8)*512+col)*2)
                            unique=(16 if q==13 else 32)if n==64 else(8 if q==13 else 16)
                            self.assertEqual(len(set(addresses)),unique)
                            self.assertTrue(all(a%16==0 for a in addresses))
                            footprints=tuple(len({(a+j)//s for a in addresses for j in range(16)})for s in (32,64,128))
                            self.assertEqual(footprints,{16:(8,8,8),32:(8,4,4),64:(16,8,4)}[n])

    def test_round_inventory_and_isolated_axes(self):
        from dev.prefill_dense import round4 as p
        results=[]
        for point in range(7):
            arms=p.pool(point);self.assertEqual(len(arms),16)
            self.assertEqual(len({a['name']for a in arms}),16)
            self.assertEqual({a['algorithm']for a in arms},{-1})
            rows=[dict(arm=a,median_us={s:30-a['dequant_variant']for s in p.stages(point,a)})for a in arms]
            for storage in (0,1):
                typed=[a for a in arms if a['storage']==storage]
                self.assertEqual({a['dequant_variant']for a in typed},
                    {p.CONTROLS[point],p.BANK[p.CONTROLS[point]],17,18,19,20,21})
                self.assertEqual(len({a['vector_io']for a in typed if a['vector_io']!=3}),1)
            self.assertEqual(sum('input'in p.stages(point,a)for a in arms),4)
            self.assertEqual(sum('dequant'in p.stages(point,a)for a in arms),14)
            complete=p.composed(point,rows);self.assertEqual(len(complete),4)
            self.assertTrue(all(a['dequant_variant']==p.CONTROLS[point]for a in complete[::2]))
            results.append(dict(point=point,status='PASS',arms=rows))
        jobs=p.profile_jobs(results);self.assertEqual(len(jobs),40)
        self.assertEqual({j['stage']for j in jobs},{0,1,3})
        for pair in range(10):
            self.assertEqual([j['role']for j in jobs if j['pair']==pair],['control','candidate','candidate','control'])
        for point in (0,5):
            for r in results[point]['arms']:
                if r['arm']['dequant_variant']==p.CONTROLS[point]and'dequant'in r['median_us']:
                    r['median_us']['dequant']=1.
        self.assertEqual(len(p.profile_jobs(results)),32)
        results[5]['status']='FAIL'
        with self.assertRaises(ValueError):p.profile_jobs(results)

    def test_acu_validates_dtype_geometry_and_metadata_not_only_symbol(self):
        import csv
        from dev.prefill_dense.round4 import check_profile
        with tempfile.TemporaryDirectory()as td:
            path=Path(td)/'raw.csv'
            row={'ID':'0','Kernel Name':'void quactlize::dequant::dense_dequant_cooperative<(gguf_scale::KType)4, true, 16u, 256u, true>(...)',
                'launch__grid_dim_x':'320','launch__grid_dim_y':'100','launch__grid_dim_z':'1',
                'launch__block_size':'128','ce__cycles_elapsed.avg':'300000','ce__cycles_elapsed.avg.per_second':'1700000000'}
            def write():
                with path.open('w')as f:
                    w=csv.DictWriter(f,fieldnames=list(row),quoting=csv.QUOTE_ALL);w.writeheader();w.writerow(row)
            job=dict(point=5,stage=1,arm='r4-bf16-dq19');write()
            self.assertTrue(check_profile(path,job)['cooperative_metadata'])
            row['Kernel Name']=row['Kernel Name'].replace('256u, true>','256u, false>');write()
            with self.assertRaises(ValueError):check_profile(path,job)
            row.update({'Kernel Name':'void quactlize::prefill::dense_convert_full4<true, true>(...)',
                'launch__grid_dim_x':'12800','launch__grid_dim_y':'1','launch__block_size':'256'})
            job=dict(point=5,stage=0,arm='r4-bf16-io3');write()
            self.assertEqual(check_profile(path,job)['vector_io'],3)
            row['Kernel Name']=row['Kernel Name'].replace('<true, true>','<false, true>');write()
            with self.assertRaises(ValueError):check_profile(path,job)
            with self.assertRaises(ValueError):check_profile(path,job|dict(stage=2))

    def test_round4_measurement_orchestration(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from dev.prefill_dense import round2,round4
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
                self.assertEqual(round2.measure(a,None,workloads()[1],None,None,dict(sha256='test'),dict(host_test=True),protocol=round4),0)
            result=json.loads(a.result.read_text())
            self.assertEqual(len(result['arms']),16);self.assertEqual(len(result['complete']),16)
            self.assertEqual(len(result['numerical']),20);self.assertEqual(len(samples),104)
            self.assertEqual({stage for _,stage in samples},{-1,0,1,3})
            self.assertTrue(result['round_four']);self.assertFalse(result['round_three'])
            self.assertFalse(result['performance_admitted']);self.assertFalse(result['production_changed'])

if __name__=='__main__':unittest.main()
