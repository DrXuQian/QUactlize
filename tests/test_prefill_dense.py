import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
if 'PPU_SDK' not in os.environ and not SDK.exists():
    SDK=Path('/root/ppu-sdk/2.1.1')

class DenseHostTest(unittest.TestCase):
    def test_actual_runtime_dtype_and_dynamic_addresses_two_streams(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);lib=root/'CUDA_SDK/targets/x86_64-linux/lib';lib.mkdir(parents=True)
            subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',str(ROOT/'tests/prefill_dense_provider.cpp'),
                '-o',str(lib/'libcublas.so')],check=True)
            exe=root/'host'
            subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),
                '-I'+str(SDK/'include'),str(ROOT/'tests/prefill_dense_runtime_host.cpp'),
                str(ROOT/'quactlize/prefill/dense_runtime.cpp'),'-ldl','-o',str(exe)],check=True)
            subprocess.run([str(exe),str(root)],check=True)

    def test_typed_packed_exchange_matches_independent_gguf(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference import gguf_kpack as ref
        from tools.kpack_warmup_fixture import prepare_expert
        from dev.bf16_compute.fixture import bf16_bits
        with tempfile.TemporaryDirectory()as td:
            so=Path(td)/'host.so'
            subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off',
                '-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
                '-I'+str(SDK/'include'),str(ROOT/'tests/prefill_dense_host.cpp'),'-o',str(so)],check=True)
            lib=C.CDLL(str(so));fn=lib.dense_host;fn.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
            for q in (12,13,14):
                for k in (512,1024):
                    s=ref.SPECS[q];rng=np.random.default_rng(q+k);n=256
                    raw=rng.integers(0,256,(n*k//256,s.raw_bytes),dtype='u1')
                    for offset in (s.d_offset,s.dmin_offset):
                        if offset>=0:raw[:,offset:offset+2]=(rng.uniform(.0001,.009,len(raw))).astype('<f2').view('u1').reshape(-1,2)
                    p=prepare_expert(raw,q,n,k)
                    gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
                    for bf in (0,1):
                        want=bf16_bits(gold)if bf else gold.astype('<f2').view('<u2')
                        for tk in (128,256):
                            with self.subTest(q=q,k=k,bf=bf,tk=tk):
                                out=np.empty_like(want)
                                args=(q,bf,tk,*[p[x].ctypes.data if p[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                                self.assertEqual(fn(*args,0),0)
                                self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                                self.assertEqual(fn(*args,1),0)
                                self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))

    def test_inventory_keeps_controls_and_no_policy_update(self):
        from dev.prefill_dense.spec import workloads,arms
        self.assertEqual(len(workloads()),7)
        pool=arms();self.assertEqual(len(pool),11)
        self.assertEqual({r['dequant_variant']for r in pool},{-1,1,2,3})
        self.assertTrue(all(r['math_mode']==16 for r in pool))
        self.assertEqual(pool[0]['name'],'bf16-old')

    def test_profile_inventory_bounded_and_no_first_kernel_only(self):
        from dev.prefill_dense.spec import profile_jobs,arms
        from tools.run_prefill_dense_opt import check_profile
        results=[dict(point=i,status='PASS',arms=[dict(arm=a,median_us=dict(complete=j+1,dequant=j+1))
                 for j,a in enumerate(arms())])for i in range(7)]
        jobs=profile_jobs(results);self.assertEqual(len(jobs),15)
        self.assertEqual({j['point']for j in jobs},{0,5,6})
        self.assertTrue(all(j['arm']!='bf16-vec'for j in jobs if j['stage']==1))
        self.assertEqual(sum(j['arm'].startswith('bf16-packed')for j in jobs if j['stage']==1),2)
        with tempfile.TemporaryDirectory()as td:
            path=Path(td)/'raw.csv'
            path.write_text('"ID","Kernel Name"\n"0","provider_first"\n"1","provider_reduce"\n')
            self.assertEqual(len(check_profile(path,dict(point=0,arm='bf16-old',stage=2))['launches']),2)
            with self.assertRaises(ValueError):check_profile(path,dict(point=0,arm='bf16-old',stage=3))
            path.write_text('"ID","Kernel Name"\n"0","dense_convert_vec8"\n')
            self.assertEqual(len(check_profile(path,dict(point=0,arm='fp16-packed1',stage=3))['launches']),1)
            path.write_text('"ID","Kernel Name"\n')
            with self.assertRaises(ValueError):check_profile(path,dict(point=0,arm='bf16-old',stage=2))

    def test_packed_affine_layout_model_is_bijective_and_balanced(self):
        # 32 banks x4B, float2 metadata requests: a footprint model, not ACU.
        from collections import Counter
        for groups in (4,8,16):
            address=lambda row,g:row*groups+(g^(row//(32//groups)))
            self.assertEqual({address(r,g)for r in range(32)for g in range(groups)},set(range(32*groups)))
            for g in range(groups):
                banks=Counter((2*address(r,g)+v)%32 for r in range(32)for v in (0,1))
                self.assertEqual(set(banks.values()),{2})

if __name__=='__main__':unittest.main()
