from pathlib import Path
import os
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_real_host_compute_contract(tmp_path):
    sdk=Path(os.environ.get("PPU_SDK","/root/ppu-sdk/2.1.1"))
    exe=tmp_path/"bf16-compute"
    result=subprocess.run(["g++","-O2","-std=c++17",f"-I{ROOT}",
        f"-I{ROOT}/quactlize/include",f"-I{ROOT}/third_party/actlize/include",
        f"-I{sdk}/include",f"-I{sdk}/targets/x86_64-linux/include",
        str(ROOT/"tests/bf16_compute_host.cpp"),"-o",str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "BF16_COMPUTE_HOST PASS" in subprocess.check_output([exe],text=True)


def test_compute_is_in_cache_and_source_contract(tmp_path,monkeypatch):
    from test_kpack_jit import fake_compiler
    from quactlize.decode.compiler import DecodeCompiler
    from quactlize.decode.grouped_compiler import GroupedComputeCompiler
    from quactlize.runtime.compiler import source_contract
    from tools.kpack_jit import parent_tuple
    sdk=fake_compiler(tmp_path)
    monkeypatch.setenv("PATH",str(sdk/"bin")+os.pathsep+os.environ["PATH"])
    parent=parent_tuple("q6",[14,0,8,64,256,8,16,2,0,16,-1])
    f16=DecodeCompiler(sdk,tmp_path/"cache",compute_type="f16")
    bf16=DecodeCompiler(sdk,tmp_path/"cache",compute_type="bf16")
    assert source_contract(f16.identity)!=source_contract(bf16.identity)
    assert f16.source(parent,"")!=bf16.source(parent,"")
    assert "#define QKD_USE_BF16_COMPUTE 1" in bf16.source(parent,"")
    assert f16.build(parent)["key"]!=bf16.build(parent)["key"]
    with pytest.raises(ValueError): DecodeCompiler(sdk,tmp_path/"cache",compute_type="auto")
    packed=parent_tuple("q4",[12,0,8,64,256,8,16,2,1,16,-1])
    assert "#define QK_AP 1" in bf16.source(packed,"")
    grouped=GroupedComputeCompiler(sdk,tmp_path/"grouped")
    with pytest.raises(ValueError,match="grouped parent"): grouped.source(parent,"")
    parent=parent|dict(route="fq-grouped",persistent=0)
    assert "#define QK_USE_BF16_COMPUTE 1" in grouped.source(parent,"")


def test_old_half_contract_is_not_silently_promoted():
    dense=(ROOT/"quactlize/decode/dense.cuh").read_text()
    assert "if constexpr(QKD_USE_BF16_COMPUTE) return QK_UNSUPPORTED" in dense
    assert "d->compute_type==quactlize::decode::compute_type" in dense
    grouped=(ROOT/"quactlize/runtime/module.cuh").read_text()
    assert "if constexpr(QK_USE_BF16_COMPUTE) return QK_UNSUPPORTED" in grouped
    assert "d->compute_type==quactlize::runtime::compute_type" in grouped
    simt=(ROOT/"quactlize/execution/simt_activation.cuh").read_text()
    assert "struct Activation<Input, 0> : q4_s1::Activation<Input>" in simt
    assert "__floats2bfloat162_rn(x,y)" in simt
    assert "clamp" not in simt


def test_compute_abi_sizes_match_c(tmp_path):
    import ctypes as C
    from quactlize.decode.native_compute import DenseComputeCall, GroupedComputeCall, ComputeIdentity
    from quactlize.execution.native import SimtCallV2
    source=tmp_path/"sizes.cpp"
    source.write_text('#include "quactlize/decode/api.h"\n#include "quactlize/execution/simt.h"\n'
        '#include <cstdio>\nint main(){std::printf("%zu %zu %zu %zu %zu\\n",sizeof(qkd_dense_call_v2),'
        'sizeof(qk_compute_device_call_v3),sizeof(qkd_compute_identity_v2),sizeof(qk_compute_identity_v3),'
        'sizeof(qkg_simt_call_v2));}\n')
    exe=tmp_path/"sizes"
    subprocess.run(["g++","-std=c++17",f"-I{ROOT}",str(source),"-o",str(exe)],check=True)
    sizes=list(map(int,subprocess.check_output([exe],text=True).split()))
    assert sizes==[C.sizeof(DenseComputeCall),C.sizeof(GroupedComputeCall),C.sizeof(ComputeIdentity),
                   C.sizeof(ComputeIdentity),C.sizeof(SimtCallV2)]


def test_actual_moe_swiglu_and_split_completion_preserve_range(tmp_path):
    chain=(ROOT/"quactlize/runtime/moe_chain.cuh").read_text()
    types=chain[chain.index("struct MixedMoePlan"):chain.index("CUTLASS_HOST_DEVICE bool moe_prepare_m1_supported")]
    body=chain[chain.index("template<class Compute=Half>\nCUTLASS_DEVICE float moe_projection_value"):
               chain.index("__global__ void moe_chain_swiglu(")]
    # Execute the actual scalar boundary bodies, with one host logical lane.
    # This is not a GPU scheduling or MMA numerical admission.
    body=body.replace("__int_as_float(0x7fffffff)","std::numeric_limits<float>::quiet_NaN()")
    directory=(ROOT/"quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp").read_text()
    header=directory[directory.index("struct alignas(16) Header"):directory.index("static_assert(sizeof(Header)")]
    source=tmp_path/"moe-range.cpp"
    source.write_text(r'''
#include <hggc_runtime.h>
#include "cutlass/numeric_types.h"
#include "quactlize/runtime/moe_protocol.h"
#include <cassert>
#include <cmath>
#include <limits>
dim3 threadIdx{0,0,0},blockIdx{0,0,0},blockDim{1,1,1},gridDim{1,1,1};
namespace quactlize::runtime {
using Half=cutlass::half_t;
namespace moe {
''' + header + "}\n" + types + body + r'''
}
int main() {
  using namespace quactlize::runtime;
  using Bf16=cutlass::bfloat16_t;
  moe::Header directory{};
  Bf16 gate(482.842712f),up(504.063690f),down(0.f);
  ComputeMoePlan<Bf16> plan{};
  plan.gate.output=&gate;plan.gate.n=1;plan.gate.splits=1;plan.gate.directory_header=&directory;
  plan.up.output=&up;plan.up.n=1;plan.up.splits=1;
  plan.down.k=1;plan.down.a=&down;
  moe_swiglu_body(plan);
  assert(std::isfinite(float(down)) && float(down)>65504.f);
  assert(down.raw()==Bf16((float(gate)/(1.f+std::exp(-float(gate))))*float(up)).raw());
  Half hgate{float(gate)},hup{float(up)},hdown{0.f};
  qk_moe_plan_v1 legacy=plan;
  legacy.gate.output=&hgate;legacy.up.output=&hup;legacy.down.a=&hdown;
  moe_swiglu_body(legacy);
  assert(std::isinf(float(hdown)));
  float partials[2]={123000.f,120383.484375f};
  auto projection=plan.gate;projection.m=1;projection.splits=2;projection.partials=partials;
  assert(moe_projection_value<Bf16>(projection,0,0)==float(Bf16(243383.484375f)));
  assert(std::isinf(moe_projection_value<Half>(projection,0,0)));
}
''')
    sdk=Path(os.environ.get("PPU_SDK","/root/ppu-sdk/2.1.1"))
    exe=tmp_path/"moe-range"
    result=subprocess.run(["g++","-O2","-std=c++17",f"-I{ROOT}",f"-I{ROOT}/third_party/actlize/include",
        f"-I{sdk}/include",f"-I{sdk}/targets/x86_64-linux/include",str(source),"-o",str(exe)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    subprocess.run([exe],check=True)
