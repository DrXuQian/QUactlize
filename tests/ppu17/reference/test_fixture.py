"""CPU-only tests for the existing GPTQ importer and lossless native packing."""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare import CONTRACT, Reference, validate
from fixture import decode_checkpoint, synthetic_checkpoint


class FixtureTests(unittest.TestCase):
    def test_original_importer_preserves_codes_scales_and_groups(self):
        qw,qz,sc=synthetic_checkpoint(128,512)
        q,scale=decode_checkpoint(qw,qz,sc)
        expected=np.empty((512,128),np.uint8)
        for k in range(512):
            for n in range(128):
                expected[k,n]=(int(qw[k//8,n])>>(4*(k%8)))&15
        np.testing.assert_array_equal(q,expected)
        np.testing.assert_array_equal(scale.view(np.uint16),sc.view(np.uint16))
        self.assertEqual(set(q.reshape(-1)),set(range(16)))
        self.assertGreater(np.count_nonzero(sc[0]!=sc[1]),120)
        self.assertGreater(len(np.unique(sc)),128)

    def test_incompatible_quantization_is_not_coerced(self):
        qw,qz,sc=synthetic_checkpoint(128,512)
        for kwargs in ({'group_size':64},{'desc_act':True},
                       {'g_idx':np.arange(512)[::-1]//128}):
            with self.assertRaises(ValueError):
                decode_checkpoint(qw,qz,sc,**kwargs)
        with self.assertRaises(ValueError):
            decode_checkpoint(qw,np.full_like(qz,0x77777777),sc)
        with self.assertRaises(ValueError):
            decode_checkpoint(qw,qz,sc.astype(np.float32))
        bad=sc.copy();bad[0,0]=np.inf
        with self.assertRaises(ValueError):
            decode_checkpoint(qw,qz,bad)
        with self.assertRaises(ValueError):
            decode_checkpoint(qw,qz,sc[:1])

    def test_old_affine_experiment_cannot_be_loaded_or_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'manifest.json').write_text(json.dumps({'schema':1,'libraries':{}}))
            with self.assertRaisesRegex(ValueError,'GPTQ'):
                Reference(root,'quactlize')
        with self.assertRaisesRegex(ValueError,'contract'):
            validate({'status':'PASS','rows':[{}],
                      'scope':'COMMON_GROUP64_PHYSICAL_HOPPER_FULL_PREPARED_CALL'})


def native_pack_check(references):
    ref=Reference(references,'quactlize')
    prepacked=json.loads((references/'manifest.json').read_text()).get('quactlize_reader')=='prepacked'
    for n,k in ((128,128),(512,512)):
        q,sc=decode_checkpoint(*synthetic_checkpoint(n,k))
        got=np.zeros(n*k//2,np.uint8)
        ref.check(ref.fn['pack'](0,n,k,q.ctypes.data,got.ctypes.data))
        if prepacked:
            # Independent scalar MMA lane/register formula, not the C++
            # packer's CuTe map or the reference library's pack function.
            nn,kk=np.meshgrid(np.arange(n),np.arange(k))
            local_n,local_k=nn%128,kk%64
            thread=(local_n//64)*128+(local_n%64//16)*32+(local_n%8)*4+(local_k%8)//2
            value=(local_k%2)+(local_n%16//8)*2+(local_k%16//8)*4+(local_k//16)*8
            offset=8192*(nn//128+(kk//64)*(n//128))+thread*32+(value//8)*8+(value%2)*4+(value%8)//2
            decoded=(got[offset//2] >> ((offset%2)*4))&15
            np.testing.assert_array_equal(decoded,q)
            got[0]^=1
            if np.array_equal((got[offset//2] >> ((offset%2)*4))&15,q):
                raise ValueError('corrupt prepacked code was not detected')
            continue
        flat=np.empty(n*k,np.int16)
        flat[0::2]=got&15;flat[1::2]=got>>4
        signed=np.where(flat>=8,flat-16,flat).reshape(n,k).T
        np.testing.assert_array_equal(signed+8,q)
        # Injected bit corruption must break the logical code equality.
        got[0]^=1
        self_bad=((int(got[0])&15)^8)
        if self_bad==int(q[0,0]):
            raise ValueError('corrupt native pack was not detected')
    print('PPU17_GPTQ_HOST '+('MMA-prepacked' if prepacked else 'signed-int4')+' packing cases=2 PASS',flush=True)


def upstream_marlin_check(root,references):
    import torch
    from fixture import marlin_metadata
    paths=root/'vllm/model_executor/layers/quantization/utils'
    env={'torch':torch,'np':np,'GPTQ_MARLIN_TILE':16,'get_pack_factor':lambda bits:32//bits}
    names={'get_weight_perm','marlin_weights','marlin_permute_weights',
           'get_scale_perms','marlin_permute_scales'}
    for name in ('marlin_utils_test.py','marlin_utils.py'):
        tree=ast.parse((paths/name).read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(paths/name),'exec'),env)
    ref=Reference(references,'marlin')
    for n,k in ((128,256),(512,512)):
        q,scale=decode_checkpoint(*synthetic_checkpoint(n,k))
        got=np.zeros(n*k//2,np.uint8)
        ref.check(ref.fn['pack'](0,n,k,q.ctypes.data,got.ctypes.data))
        expected=env['marlin_weights'](torch.from_numpy(q),k,n,4,env['get_weight_perm'](4))
        np.testing.assert_array_equal(expected.numpy().view(np.uint8).reshape(-1),got)
        value=torch.from_numpy(scale)
        if not torch.equal(marlin_metadata(value),env['marlin_permute_scales'](value,k,n,128)):
            raise ValueError('Marlin scale bits/placement differ from upstream')
    print('PPU17_GPTQ_HOST Marlin packing/scales match upstream cases=2 PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--vllm',type=Path)
    p.add_argument('--references',type=Path)
    args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.references:native_pack_check(args.references)
    if args.vllm:upstream_marlin_check(args.vllm,args.references)
