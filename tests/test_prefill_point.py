import copy
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dev.prefill_point.bench import Call, SF, WORKLOAD, call, summarize
from dev.prefill_point.build import build_native, carrier_source, fp32_accumulator_source, once
from tools.run_prefill_point import ARMS, native_stage_names, plan

ROOT = Path(__file__).resolve().parents[1]


class PrefillPointTests(unittest.TestCase):
    def test_quactlize_only_build_does_not_access_llama_or_build_native_adapter(self):
        from tools.run_prefill_point import build
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)
            # No llama_source/build fields: they must not be required here.
            args=SimpleNamespace(output=output,sdk=output/'sdk',no_native=True)
            def command(commands,*_args,**_kwargs):
                (output/'build/libpoint-timer.so').write_bytes(b'HOST TEST TIMER')
            with patch('dev.prefill_point.build.build_native',side_effect=AssertionError('native called'))as native,\
                 patch('dev.strategy.prefill_build.build')as helpers,\
                 patch('tools.build_kpack_dequant.build')as dequant,\
                 patch('dev.native_components.build.run_commands',side_effect=command):
                receipt=build(args)
            native.assert_not_called();helpers.assert_called_once();dequant.assert_called_once()
            self.assertFalse(receipt['native_reference'])
            self.assertNotIn('carrier',receipt);self.assertNotIn('adapter',receipt)
            self.assertFalse((output/'build/native').exists())

    def rows(self):
        return [dict(round=r, rotate=rotate, mode=mode, stage=stage,
                     width=4 if mode == 1 else 1,
                     samples_us=([1., 2., 3., 4.]*3 if mode == 1 else [10.]*3))
                for r in range(3) for rotate in (0, 1) for mode in range(3)
                for stage in range(4 if mode == 0 else 1)]

    def test_fixed_point_not_search(self):
        self.assertEqual((WORKLOAD.qtype, WORKLOAD.m, WORKLOAD.n, WORKLOAD.k), (12, 2048, 1024, 5120))
        self.assertEqual(ARMS, ('native-fp16', 'sf', 'full-bf16'))
        self.assertEqual(dict(SF.parameters)['split'], 1)
        self.assertEqual(dict(SF.parameters)['dequant_config'], 4)
        self.assertEqual(plan()['full_dequant_config'], 12)
        for name in ('production_changed', 'performance_admitted', 'cuda_graph_timing', 'cublas_graph'):
            self.assertFalse(plan()[name])

    def test_descriptor_layout(self):
        self.assertEqual(ctypes.sizeof(Call), 136)
        self.assertEqual(Call.kind.offset, 96)
        self.assertEqual(Call.stride.offset, 128)
        c = call(None, 3, [42, ctypes.c_void_p(19)], [2, 7], stride=5120)
        self.assertEqual(list(c.p)[:2], [42, 19])
        self.assertEqual(list(c.v)[:2], [2, 7])
        self.assertEqual(c.stride, 5120)

    def test_diagnostic_keeps_oracles_not_old_timers(self):
        text = carrier_source()
        self.assertIn('#include "carrier_timing.cuh"', text)
        self.assertLess(text.index('#include "carrier_contract.cuh"'),text.index('proofs.push_back(proof(0));'))
        self.assertLess(text.index('publish(fs::path(argv[2])/(point.id+".inventory.json"),inventory);'),
                        text.index('#include "carrier_contract.cuh"'))
        self.assertIn('last_weight_copy_negative', text)
        self.assertIn('omitted_producer', text)
        self.assertIn('nc.replay(nc.session,copy,-1)', text)
        self.assertNotIn('nc.measure(nc.session,int(i),rotate', text)
        self.assertEqual(text.count('NC_BIND(point_samples)'), 1)
        with self.assertRaises(ValueError): once('two two', 'two', 'one')

    def traits(self):
        return '''
template<ggml_type T> struct batched_mul_mat_traits;
template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};
'''

    def test_explicit_accumulator_keeps_f16_storage_and_native_dispatch(self):
        source=self.traits()+'\n// native-dispatch-with-no-accumulator-env\n'
        adapted,receipt=fp32_accumulator_source(source)
        self.assertEqual(source,self.traits()+'\n// native-dispatch-with-no-accumulator-env\n')
        self.assertIn('compute_type = CUBLAS_COMPUTE_32F;',adapted)
        self.assertIn('using cuda_type = half;',adapted)
        self.assertIn('data_type = CUDA_R_16F;',adapted)
        self.assertIn('static const float val = alpha;',adapted)
        self.assertIn('static const float val = beta;',adapted)
        self.assertIn('// native-dispatch-with-no-accumulator-env',adapted)
        self.assertIn('nc_point_batched_mul_mat_traits',adapted)
        self.assertTrue(receipt['arithmetic_changed'])
        self.assertFalse(receipt['selector_changed']);self.assertFalse(receipt['llama_tree_modified'])
        self.assertEqual(receipt['linked_backend_required'],'PASS_TWO_INPUTS')

    def test_existing_f32_traits_and_unsupported_seams(self):
        source=self.traits().replace('CUBLAS_COMPUTE_16F','CUBLAS_COMPUTE_32F').replace('const half','const float')
        _,receipt=fp32_accumulator_source(source)
        self.assertFalse(receipt['arithmetic_changed'])
        for bad in (source+source,source.replace('CUDA_R_16F','CUDA_R_16BF'),
                    source.replace('static const float val = alpha;','float val = alpha;'),
                    source.replace('CUBLAS_COMPUTE_32F','CUBLAS_COMPUTE_32F_FAST_16F'),
                    source.replace('ggml_get_to_fp16_cuda','ggml_get_to_fp32_cuda')):
            with self.subTest(source=bad),self.assertRaises(ValueError):fp32_accumulator_source(bad)

    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler unavailable')
    def test_real_scalar_bytes_lifetime_and_symbol_isolation(self):
        header='''
#include <cassert>
#include <cstdint>
#include <cstring>
// Only the scalar constants 0/1 are used; no host/GPU arithmetic is simulated.
struct half {
    uint16_t bits;
    constexpr half(double v):bits(v==0 ? 0 : 0x3c00) {}
    bool operator==(half other) const { return bits==other.bits; }
};
enum ggml_type { GGML_TYPE_F16=1 };
enum cudaDataType_t { CUDA_R_16F=2 };
enum cublasComputeType_t { CUBLAS_COMPUTE_16F=64, CUBLAS_COMPUTE_32F=68 };
int ggml_get_to_fp16_cuda(ggml_type) { return 17; }
int ggml_get_to_fp16_nc_cuda(ggml_type) { return 19; }
'''
        adapted,_=fp32_accumulator_source(self.traits())
        checked=adapted.replace('return &val;',
            'static_assert(sizeof(val)==4,"provider scalar must be FP32"); return &val;')
        original_getters='''
extern "C" const void * original_alpha() { return batched_mul_mat_traits<GGML_TYPE_F16>::get_alpha(); }
extern "C" const void * original_beta() { return batched_mul_mat_traits<GGML_TYPE_F16>::get_beta(); }
'''
        body='''
extern "C" const void * original_alpha();
extern "C" const void * original_beta();
int main() {
    using Traits=nc_point_batched_mul_mat_traits<GGML_TYPE_F16>;
    static_assert(Traits::compute_type==CUBLAS_COMPUTE_32F);
    static_assert(Traits::data_type==CUDA_R_16F);
    static_assert(sizeof(Traits::cuda_type)==2);
    const void * a=Traits::get_alpha(), * b=Traits::get_beta();
    // Force the old DSO's identically named original traits to be loaded too.
    assert(*static_cast<const half *>(original_alpha())==half(1));
    assert(*static_cast<const half *>(original_beta())==half(0));
    for(int i=0;i<8;++i) {
        assert(Traits::get_alpha()==a && Traits::get_beta()==b);
        uint32_t ab=0,bb=1;
        std::memcpy(&ab,a,4);std::memcpy(&bb,b,4);
        assert(ab==0x3f800000 && bb==0);
    }
    assert(Traits::convert(GGML_TYPE_F16)==17);
    assert(Traits::convert_nc(GGML_TYPE_F16)==19);
}
'''
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=root/'original.cpp';new=root/'adapter.cpp';binary=root/'test'
            old.write_text(header+self.traits()+original_getters)
            new.write_text(header+checked+body)
            subprocess.run(['g++','-std=c++17','-O0','-fPIC','-shared',str(old),'-o',str(root/'liboriginal.so')],check=True)
            command=['g++','-std=c++17','-O0',str(new),'-L'+str(root),'-Wl,-rpath,'+str(root),
                     '-loriginal','-o',str(binary)]
            subprocess.run(command,check=True);subprocess.run([str(binary)],check=True)
            # Merely changing the compute enum leaves half scalar storage.
            bad=checked.replace('static const float val','static const half val')
            new.write_text(header+bad+body)
            failed=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('provider scalar must be FP32',failed.stderr)

    def test_payload_reuse_builds_native_adapter_in_new_run(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);previous=root/'previous';previous.mkdir()
            old=previous/'native';old.mkdir()
            receipt=old/'adapter.json';receipt.write_text('old receipt remains immutable')
            args=SimpleNamespace(reuse_build=previous,llama_source=root/'llama',
                                 llama_build=root/'build',sdk=root/'sdk')
            target=root/'new'/'native'
            # Stop at the real build seam: never validate or rewrite the old
            # control-script identity merely to reuse unchanged GPU payloads.
            with patch('dev.prefill_point.build.build_adapter',side_effect=RuntimeError('test build seam'))as build:
                with self.assertRaisesRegex(RuntimeError,'test build seam'):
                    build_native(args,target,{})
                build.assert_called_once_with(args.llama_source,args.llama_build,args.sdk,target)
            self.assertEqual(receipt.read_text(),'old receipt remains immutable')

    def test_summary_requires_complete_three_round_inventory(self):
        rows = self.rows()
        out = summarize(rows, ['input', 'expansion', 'main', 'output'])
        self.assertEqual(out['repeated']['sequence_stage_us_sum'], 10)
        self.assertEqual(out['repeated']['isolated_us_sum'], 40)
        self.assertEqual(out['rotating_weights']['complete_us'], 10)
        bads = [rows[:-1], rows+[rows[0]], [r for r in rows if r['round'] != 2]]
        for name, value in [('width', 7), ('samples_us', [float('nan')]*3), ('samples_us', [0.]*3)]:
            bad = copy.deepcopy(rows); bad[0][name] = value; bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError): summarize(bad, ['input', 'expansion', 'main', 'output'])

    def test_native_inventory_matches_actual_ppu_names(self):
        names = ['void convert_cont_ppu_kernel<float, __half>(...)',
                 'void dequantize_block_q4_K(...)', 'cublasGemmEx',
                 'void convert_cont_ppu_kernel<__half, float>(...)']
        rows = [dict(name=n, symbol=n, entry='PROVIDER' if n == 'cublasGemmEx' else 'KERNEL') for n in names]
        self.assertEqual(native_stage_names(rows), ['input', 'expansion', 'main', 'output'])
        with self.assertRaises(ValueError): native_stage_names(rows[::-1])
        with self.assertRaises(ValueError): native_stage_names([dict(name='unknown', symbol='unknown', entry='KERNEL')])

    @unittest.skipUnless(shutil.which('g++'), 'host C++ compiler unavailable')
    def test_native_event_loop_order_and_error_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td)/'point-events'
            subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            str(ROOT/'tests/prefill_point_events_host.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
