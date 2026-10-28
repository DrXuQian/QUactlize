"""Production contract and numerical regression tests."""
from collections import Counter
import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from ci.support import dense_dequant as run
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
        for n,k in ((512,2048),(2048,512),(4096,2048),(8192,2048),
                    (2048,4096),(1024,5120)):
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
