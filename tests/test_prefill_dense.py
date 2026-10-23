"""Production dense helpers: independent GGUF oracles, maps and ABI negatives."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))

class DenseHelperHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();so=Path(cls.temp.name)/'host.so'
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-ffp-contract=off',
            '-I'+str(ROOT),'-I'+str(ROOT/'quactlize/include'),'-I'+str(ROOT/'third_party/actlize/include'),
            '-I'+str(SDK/'include'),str(ROOT/'tests/prefill_dense_host.cpp'),'-o',str(so)],check=True)
        cls.lib=C.CDLL(str(so));cls.lib.dense_cooperative_host.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
        cls.lib.dense_plane_pair_host.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
        cls.lib.dense_q8_exchange_address.argtypes=[C.c_uint]*3
        cls.lib.dense_q8_exchange_address.restype=C.c_uint
        cls.lib.dense_cooperative_layout.argtypes=[C.c_uint]*5;cls.lib.dense_cooperative_layout.restype=C.c_uint
        cls.lib.dense_full4.argtypes=[C.c_uint64,C.c_uint,C.c_uint];cls.lib.dense_full4.restype=C.c_uint64
        cls.lib.dense_pair_host.argtypes=[C.c_int]*3+[C.c_void_p]*4+[C.c_int]*3
        cls.lib.dense_direct_host.argtypes=[C.c_int]*2+[C.c_void_p]*4+[C.c_int]*2
        cls.lib.dense_q8_host.argtypes=[C.c_int]+[C.c_void_p]*3+[C.c_uint]*2+[C.c_int]
        cls.lib.dense_stripe4.argtypes=[C.c_uint64,C.c_uint,C.c_uint];cls.lib.dense_stripe4.restype=C.c_uint64
        cls.lib.dense_unsigned_index.argtypes=[C.c_uint]*4;cls.lib.dense_unsigned_index.restype=C.c_uint64
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

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

    def test_direct_all_formats_and_precisions(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference import gguf_kpack as ref
        from tools.kpack_warmup_fixture import prepare_expert
        from dev.bf16_compute.fixture import bf16_bits
        for q in range(10,15):
            n,k=256,512
            s=ref.SPECS[q];rng=np.random.default_rng(100+q)
            raw=rng.integers(0,256,(n*k//256,s.raw_bytes),dtype='u1')
            for offset in (s.d_offset,s.dmin_offset):
                if offset>=0:
                    raw[:,offset:offset+2]=rng.uniform(-.009,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                    raw[:8,offset:offset+2]=np.array([0,0x8000,1,0x8001,0x3555,0xb555,0x0400,0x8400],dtype='<u2').view('u1').reshape(-1,2)
            packed=prepare_expert(raw,q,n,k)
            gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
            for bf in (0,1):
                with self.subTest(q=q,bf=bf):
                    want=bf16_bits(gold)if bf else gold.astype('<f2').view('<u2')
                    out=np.full_like(want,0x7fff)
                    args=(q,bf,*[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                    self.assertEqual(self.lib.dense_direct_host(*args),0)
                    self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))

    def test_q8_full_tile_matches_independent_gguf(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from dev.bf16_compute.fixture import planes, bf16_bits
        for n,k in ((256,256),(256,512),(512,1024)):
            rng=np.random.default_rng(n+k)
            raw=rng.integers(0,256,(n,k//32,34),dtype='u1')
            scale=rng.uniform(-.09,.09,(n,k//32)).astype('<f2')
            scale.view('<u2').flat[:8]=[0,0x8000,1,0x8001,0x3555,0xb555,0x0400,0x8400]
            raw[:,:,:2]=scale.view('u1').reshape(n,k//32,2)
            raw.reshape(-1,34)[:8,2:]=np.arange(256,dtype='u1').reshape(8,32)
            packed=planes(raw,8)
            gold=dequantize(raw.reshape(-1),GGMLQuantizationType.Q8_0).reshape(n,k)
            for bf in (0,1):
                want=bf16_bits(gold) if bf else gold.astype('<f2').view('<u2')
                for fault in (0,1):
                    out=np.full_like(want,0x7fff)
                    rc=self.lib.dense_q8_host(bf,packed['low'].ctypes.data,packed['units'].ctypes.data,
                                             out.ctypes.data,n,k,fault)
                    self.assertEqual(rc,0)
                    differs=np.any((out!=want)&(((out|want)&0x7fff)!=0))
                    self.assertEqual(bool(differs),bool(fault))

    def test_q8_coalesced_vector_loads_and_stores(self):
        for base in range(0,128,32):
            loads=[((t//8)*512+(t%8)*8)*2 for t in range(base,base+32)]
            stores=[((t//4)*1024+(t%4)*8)*2 for t in range(base,base+32)]
            for addresses in (loads,stores):
                self.assertTrue(all(a%16==0 for a in addresses))
                self.assertEqual(len({(a+b)//32 for a in addresses for b in range(16)}),16)

    def test_q2_q3_q8_plane_pair_oracle_and_faults(self):
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        from reference import gguf_kpack as ref
        from tools.kpack_warmup_fixture import prepare_expert
        from dev.bf16_compute.fixture import planes, bf16_bits
        for q in (8,10,11):
            for n,k in ((256,512),(768,1024),(256,1536)):
                rng=np.random.default_rng(q+n+k)
                raw=rng.integers(0,256,(n*k//(32 if q==8 else 256),34 if q==8 else ref.SPECS[q].raw_bytes),dtype='u1')
                offset=0 if q==8 else ref.SPECS[q].d_offset
                raw[:,offset:offset+2]=rng.uniform(-.009,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                raw[:10,offset:offset+2]=np.array([0,0x8000,1,0x8001,0x3555,0xb555,0x0400,0x8400,0x7bff,0xfbff],dtype='<u2').view('u1').reshape(-1,2)
                if q==10:
                    off=ref.SPECS[q].dmin_offset
                    raw[:,off:off+2]=rng.uniform(-.009,.009,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                if q==8:raw[:8,2:]=np.arange(256,dtype='u1').reshape(8,32)
                packed=planes(raw.reshape(n,k//32,34),8)if q==8 else prepare_expert(raw,q,n,k)
                gold=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
                for bf in (0,1):
                    with np.errstate(over='ignore'):want=bf16_bits(gold)if bf else gold.astype('<f2').view('<u2')
                    control=np.empty_like(want)
                    ptrs=[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')]
                    if q==8:self.assertEqual(self.lib.dense_q8_host(bf,ptrs[0],ptrs[2],control.ctypes.data,n,k,0),0)
                    else:self.assertEqual(self.lib.dense_direct_host(q,bf,*ptrs,control.ctypes.data,n,k),0)
                    for variant in ((22,23,24,25) if q==8 else (22,23)):
                        with self.subTest(q=q,n=n,k=k,bf=bf,variant=variant):
                            out=np.full_like(want,0x7fff)
                            args=(q,bf,variant,*ptrs,out.ctypes.data,n,k)
                            self.assertEqual(self.lib.dense_plane_pair_host(*args,0),0)
                            # Independent GGUF zero-sign convention may differ;
                            # control/candidate must still match every raw bit.
                            self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            np.testing.assert_array_equal(out,control)
                            for fault in (1,3,*((4,)if q==11 or q==10 and variant==22 else())):
                                self.assertEqual(self.lib.dense_plane_pair_host(*args,fault),0)
                                self.assertTrue(np.any(out!=control))
                            self.assertNotEqual(self.lib.dense_plane_pair_host(*args,2),0)
                            if q==11:self.assertNotEqual(self.lib.dense_plane_pair_host(*args,5),0)
                            if variant in (24,25):
                                self.assertEqual(self.lib.dense_plane_pair_host(*args,6),0)
                                self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))

    def test_q8_exchange_unique_shared_reads_and_unchanged_stores(self):
        from collections import Counter
        fn=self.lib.dense_q8_exchange_address
        for tk in (128,256):
            reads=[]
            for task in range(32*(tk//8)):
                source,other,pair,kk=[fn(tk,task,x) for x in range(4)]
                self.assertEqual(task//32,other//32)
                self.assertEqual((pair,kk),(task//(tk//8),task%(tk//8)*8))
                self.assertEqual(source%4,0)
                reads.extend(range(source,source+4))
                for residue in range(8):
                    owner=task if residue//4==task%2 else other
                    address=fn(tk,owner,0)+residue%4
                    want=(pair*(tk//2)+(kk//16)*8+residue)^(pair&28)
                    self.assertEqual(address,want)
                if pair==4 and task%2==1:
                    # Exact negative for physical-base+4 instead of logical+4.
                    wrong=((pair*(tk//2)+(kk//16)*8)^(pair&28))+4
                    self.assertNotEqual(source,wrong)
            # All physical code words fetched once, rather than twice.
            self.assertEqual(sorted(reads),list(range(32*(tk//2))))
            for start in range(0,32*(tk//8),32):
                for side in (0,1):
                    addresses=[((2*fn(tk,t,2)+side)*5120+fn(tk,t,3))*2 for t in range(start,start+32)]
                    for sector in (32,64,128):
                        cover=Counter((a+b)//sector for a in addresses for b in range(16))
                        self.assertEqual(set(cover.values()),{sector})

    def test_box_runner_fixture_is_accepted_by_actual_readers(self):
        import numpy as np
        from tools.run_dense_dequant import fixture
        for q in (8,10,11,12,13,14):
            packed,golden,receipt=fixture(q,256,512)
            self.assertEqual(set(receipt['planes']),{'low','high','units'})
            ptrs=[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')]
            for bf,want in enumerate(golden):
                for v in ((22,23) if q in (8,10,11) else (8,20,21)):
                    out=np.full_like(want,0x7fff)
                    fn=self.lib.dense_plane_pair_host if q in (8,10,11) else self.lib.dense_pair_host if v==8 else self.lib.dense_cooperative_host
                    self.assertEqual(fn(q,bf,v,*ptrs,out.ctypes.data,256,512,0),0)
                    self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))

    def test_plane_pair_physical_words_and_sector_footprints(self):
        from collections import Counter
        for bits in (1,2,8):
            for tk in (128,256):
                rows=tk*bits//16
                shared=[(p*rows+r)^(p&28&(rows-1))for p in range(32)for r in range(rows)]
                self.assertEqual(sorted(shared),list(range(32*rows)))
                for start in range(0,rows*8,32):
                    addresses=[((i//8)*512+(i%8)*8)*2 for i in range(start,start+32)]
                    for sector in (32,64,128):
                        counts=Counter((a+b)//sector for a in addresses for b in range(16))
                        self.assertEqual(set(counts.values()),{sector})
                # Unique global word ownership, not one load per logical K8.
                reads=[(i//8,2*((i%8)*4+p)+half)for i in range(rows*8)for p in range(4)for half in (0,1)]
                self.assertEqual(len(reads),len(set(reads)))
                self.assertEqual(len(reads),rows*64)
                for start in range(0,32*(tk//8),32):
                    for side in (0,1):
                        stores=[((2*(i//(tk//8))+side)*1024+i%(tk//8)*8)*2 for i in range(start,start+32)]
                        for sector in (32,64,128):
                            counts=Counter((a+b)//sector for a in stores for b in range(16))
                            self.assertEqual(set(counts.values()),{sector})

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
                    for variant in (8,):
                        with self.subTest(q=q,n=n,k=k,bf=bf,variant=variant):
                            out=np.full_like(want,0x7fff)
                            args=(q,bf,variant,*[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                            self.assertEqual(self.lib.dense_pair_host(*args,0),0)
                            self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            for fault in (1,3):
                                self.assertEqual(self.lib.dense_pair_host(*args,fault),0)
                                self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            if variant==8:self.assertNotEqual(self.lib.dense_pair_host(*args,2),0)

    def test_each_output_instruction_is_coalesced(self):
        # A lane owns K8 in each of two N rows. Count each 16-byte store
        # separately: do not hide holes by combining two instructions.
        for tn,tk in ((32,128),):
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
            for tn,tk in ((32,128),):
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
                    for variant in (20,21):
                        with self.subTest(q=q,n=n,k=k,bf=bf,variant=variant):
                            out=np.full_like(want,0x7fff)
                            args=(q,bf,variant,*[packed[x].ctypes.data if packed[x].size else None for x in ('low','high','units')],out.ctypes.data,n,k)
                            self.assertEqual(self.lib.dense_cooperative_host(*args,0),0)
                            self.assertFalse(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            self.assertEqual(self.lib.dense_cooperative_host(*args,1),0)
                            self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))
                            if variant==21:self.assertNotEqual(self.lib.dense_cooperative_host(*args,2),0)
                            if variant>=17:self.assertNotEqual(self.lib.dense_cooperative_host(*args,4),0)
                            if variant in (17,20):
                                self.assertEqual(self.lib.dense_cooperative_host(*args,5),0)
                                self.assertTrue(np.any((out!=want)&(((out|want)&0x7fff)!=0)))

    def test_bijection_vector_alignment_and_producer_bank_model(self):
        from collections import Counter
        f=self.lib.dense_cooperative_layout
        for n,k in ((64,128),(64,256)):
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
        f=self.lib.dense_cooperative_layout
        for n,k in ((64,128),(64,256)):
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
        f=self.lib.dense_cooperative_layout
        for n,k in ((64,128),(64,256)):
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


if __name__=='__main__':unittest.main()
