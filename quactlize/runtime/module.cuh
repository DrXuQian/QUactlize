#pragma once
#include "abi.h"
#include "kernel_types.cuh"
#include "grouped_workspace.hpp"
#include "indexed.cuh"
#include "moe_chain.cuh"
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

// A generated translation unit supplies QK_QTYPE/ROUTE/TM/... and the immutable
// parent/build identifiers. It instantiates only that parent, not a sweep.
namespace quactlize::runtime {
constexpr int qtype = QK_QTYPE, route = QK_ROUTE;
constexpr int tm = QK_TM, tn = QK_TN, tk = QK_TK, stages = QK_STAGES;
constexpr bool grouped = route >= QK_GROUPED_FQ;
constexpr bool packed = route == QK_DENSE_FQ || route == QK_GROUPED_FQ;
static_assert(qtype != 8 || (!packed && QK_AP == 0), "Q8_0 requires W8A16 with FP16 scale, no packed-A");
using F = Format<qtype>;
constexpr uint64_t mapping = qtype == 8 ? q8_kpack2::kMappingId : qtype == 12 ? UINT64_C(0x51344b5034540001)
                                        : UINT64_C(0x514b504b54000001);
constexpr bool needs_zero = !packed && qtype != 8;

inline uint64_t align16(uint64_t value) { return (value + 15) & ~UINT64_C(15); }

inline bool compact_device_call(int device_maximum, int experts) {
  return device_maximum > 0 && experts <= 1024;
}

inline int validate(qk_call_v1 const& c, qk_recipe_v1 const& r, int& max_rows, int device_maximum = 0) {
  if (c.version != 1 || c.size != sizeof(c) || r.version != 1 || r.size != sizeof(r) ||
      c.mapping_id != mapping || c.m <= 0 || c.n <= 0 || c.k <= 0 ||
      c.experts <= 0 || c.group_size != F::spec.group_size || c.compute_units <= 0 ||
      c.m > INT32_MAX-256 || c.n > INT32_MAX-256 || c.k > INT32_MAX-256 ||
      c.experts > INT32_MAX-256 ||
      c.n % 256 || c.k % ((qtype == 11 || qtype == 14) ? 512 : 256) || c.k % tk ||
      r.algorithm < QK_ORDINARY || r.algorithm > QK_PERSISTENT ||
      (r.split != 1 && r.split != 2 && r.split != 4 && r.split != 8)) return QK_INVALID;
  if (r.algorithm == QK_PERSISTENT ? (r.grid <= 0 || (!grouped && r.split != 1)) : r.grid != 0)
    return QK_INVALID;
  if (c.k % (tk * r.split) || c.k / (tk * r.split) < stages - 1)
    return QK_UNSUPPORTED;
  max_rows = c.m;
  if constexpr (grouped) {
    // Both grouped schedules use slice-major FP32 partials and an ordered reducer.
    if (uint64_t(c.experts) * r.split > 65535) return QK_UNSUPPORTED;
    if (r.algorithm==QK_PERSISTENT && c.experts>1024) return QK_UNSUPPORTED;
    if (device_maximum > 0) {
      if (c.rows_host || c.rows_device || device_maximum > c.m ||
          device_maximum > INT32_MAX-256 || int64_t(device_maximum)*c.experts < c.m)
        return QK_INVALID;
      max_rows = device_maximum;
      if (r.algorithm == QK_ORDINARY && (c.experts > 65535 || (c.n+tn-1)/tn > 65535))
        return QK_UNSUPPORTED;
    } else {
      if (!c.rows_host) return QK_INVALID;
      int64_t total = 0;
      max_rows = 0;
      for (int i = 0; i < c.experts; ++i) {
        if (c.rows_host[i] < 0) return QK_INVALID;
        total += c.rows_host[i];
        max_rows = std::max(max_rows, c.rows_host[i]);
      }
      if (total != c.m) return QK_INVALID;
    }
    if (int64_t(c.experts)*((int64_t(max_rows)+tm-1)/tm)>INT32_MAX) return QK_INVALID;
    if (r.algorithm==QK_PERSISTENT || compact_device_call(device_maximum,c.experts)) {
      int const bound=quactlize::moe_directory::bounded_entries(c.m,max_rows,c.experts,tm);
      uint64_t const work=uint64_t(bound)*uint64_t((c.n+tn-1)/tn)*r.split;
      if (bound<=0 || work>uint64_t(INT32_MAX)) return QK_UNSUPPORTED;
    }
    if constexpr (route == QK_GROUPED_FQ) {
      if (r.algorithm != QK_PERSISTENT_PARENT) return QK_UNSUPPORTED;
    }
  } else {
    if (c.experts != 1 || c.rows_host || c.rows_device || c.offsets_device) return QK_INVALID;
    if constexpr (QK_AP != 0) if (c.m != 1) return QK_UNSUPPORTED;
    // Preserve the measured dense domain; grouped TM8 has independent admission.
    if constexpr (tm == 8) if (c.m > (packed ? 64 : 7)) return QK_UNSUPPORTED;
    if constexpr (packed) if (r.algorithm != QK_ORDINARY) return QK_UNSUPPORTED;
  }
  return QK_OK;
}

struct Handle {
  virtual int run(hggcStream_t) = 0;
  virtual int bind_indexed(qk_llama_indexed_v1 const&) { return QK_UNSUPPORTED; }
  virtual int moe_projection(qk_moe_projection_v1&) { return QK_UNSUPPORTED; }
  virtual int moe_stage(qk_moe_plan_v1 const&,int,hggcStream_t) { return QK_UNSUPPORTED; }
  virtual ~Handle() = default;
};

template<class T> struct DenseHandle final : Handle {
  typename T::Prepared ordinary;
  typename T::PersistentGemm persistent;
  bool use_persistent = false;

  int prepare(qk_call_v1 const& c, qk_recipe_v1 const& r, int occupancy) {
    auto stream = static_cast<hggcStream_t>(c.stream);
    auto a = static_cast<Half const*>(c.a);
    auto b = static_cast<typename T::Low const*>(c.low);
    auto high = static_cast<typename T::High const*>(c.high);
    auto output = static_cast<Half*>(c.output);
    if (r.algorithm == QK_ORDINARY) {
      return ordinary.initialize(a, b, c.metadata, c.zero, output, c.m, c.n, c.k,
          c.group_size, r.split, static_cast<char*>(c.workspace), c.workspace_bytes,
          stream, high) ? QK_OK : QK_INITIALIZE_ERROR;
    }
    using K = typename T::PersistentKernel;
    using G = typename T::PersistentGemm;
    auto ml = T::Prepared::make_mainloop_arguments(
        a, b, c.metadata, c.zero, c.m, c.n, c.k, c.group_size, high);
    auto sc = cutlass::make_cute_packed_stride(typename K::StrideC{}, cute::make_shape(c.m,c.n,1));
    auto sd = cutlass::make_cute_packed_stride(typename K::StrideD{}, cute::make_shape(c.m,c.n,1));
    typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        {c.m,c.n,c.k,1}, ml, {{1.f,0.f}, static_cast<Half*>(nullptr), sc, output, sd},
        cutlass::KernelHardwareInfo{c.device,c.compute_units}, {}, occupancy, uint32_t(r.grid)};
    if (G::can_implement(args) != cutlass::Status::kSuccess) return QK_UNSUPPORTED;
    if (persistent.initialize(args, nullptr, stream) != cutlass::Status::kSuccess)
      return QK_INITIALIZE_ERROR;
    use_persistent = true;
    return QK_OK;
  }
  int run(hggcStream_t stream) override {
    auto status = use_persistent ? persistent.run(stream) : ordinary.run(stream);
    return status == cutlass::Status::kSuccess ? QK_OK : QK_RUNTIME_ERROR;
  }
};

using Dense = DenseTypes<qtype,QK_TM,QK_TN,QK_TK,QK_WM,QK_WN,QK_STAGES,QK_AP,QK_DN>;
template<bool P, class O = Half, bool Compact = false>
using Group = GroupedTypes<qtype,QK_TM,QK_TN,QK_TK,QK_WM,QK_WN,QK_STAGES,QK_DN,P,O,Compact>;

inline uint64_t scheduler_bytes(int max_rows, int experts, bool directory, int rows) {
  return align16(directory ? quactlize::moe_directory::bounded_workspace_bytes(rows,max_rows,experts,tm)
                           : uint64_t(experts + 1) * sizeof(int));
}

template<class Shape, class Stride>
__global__ void grouped_device_metadata(int const* offsets, Half* output,
    Shape* shapes, Half** outputs, Stride* strides, int* rows, int n, int k, int experts) {
  int e = int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
  if (e >= experts) return;
  int begin=offsets[e], count=offsets[e+1]-begin;
  rows[e]=count; shapes[e]=cute::make_shape(count,n,k);
  outputs[e]=output+int64_t(begin)*n;
  strides[e]=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(count,n,1));
}

template<class Shape, class Stride>
__global__ void grouped_splitk_device_metadata(int const* offsets, float* partials,
    Shape* shapes, float** outputs, Stride* strides, int* rows,
    int m, int n, int k, int experts, int splits) {
  int e = int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
  if (e >= experts) return;
  int begin=offsets[e], count=offsets[e+1]-begin;
  rows[e]=count; shapes[e]=cute::make_shape(count,n,k);
  for (int s=0; s<splits; ++s) {
    int entry=e+s*experts;
    outputs[entry]=partials+(int64_t(s)*m+begin)*n;
    strides[entry]=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(count,n,1));
  }
}

template<bool Persistent, bool Split = false, bool Compact = false> struct GroupedHandle final : Handle {
  static constexpr bool Directory = Persistent || Compact;
  using Output = std::conditional_t<Split,float,Half>;
  using T = Group<Persistent,Output,Compact>;
  using K = typename T::Kernel;
  using G = typename T::Gemm;
  using Shape = moe_grouped_ppu::GroupShape;
  using DStride = moe_grouped_ppu::DStride;
  G gemm;
  using Reduction = cutlass::gemm::device::splitk_parallel::PpuMixedInputSplitKParallelCompactReduction<2>;
  Reduction reduction;
  int splits = 1;
  float* partials = nullptr;
  qk_call_v1 call{};
  int max_rows = 0;
  quactlize::moe_directory::View directory{};
  std::vector<Shape> shapes;
  std::vector<Output*> outputs;
  std::vector<DStride> strides;
  std::vector<int> prefix;
  bool device_only = false;
  Shape* device_shapes = nullptr;
  Output** device_outputs = nullptr;
  DStride* device_strides = nullptr;
  qk_llama_indexed_v1 indexed{};

  int moe_projection(qk_moe_projection_v1& p) override {
    if constexpr (!Directory) return QK_UNSUPPORTED;
    if (!device_only || !indexed.version) return QK_UNSUPPORTED;
    Shape shape{}; DStride stride{};
    auto offset=[](auto const& tuple,auto const& member) {
      return uint32_t(reinterpret_cast<char const*>(&member)-reinterpret_cast<char const*>(&tuple));
    };
    p={}; p.version=1; p.size=sizeof(p);
    p.shape_size=sizeof(Shape); p.stride_size=sizeof(DStride);
    p.shape_offsets[0]=offset(shape,cute::get<0>(shape));
    p.shape_offsets[1]=offset(shape,cute::get<1>(shape));
    p.shape_offsets[2]=offset(shape,cute::get<2>(shape));
    p.stride_offset=offset(stride,cute::get<0>(stride));
    p.m=call.m; p.n=call.n; p.k=call.k; p.experts=call.experts;
    p.tile_m=tm; p.splits=splits; p.device=call.device;
    p.a=const_cast<void*>(call.a); p.output=call.output; p.partials=partials;
    p.shapes=device_shapes; p.outputs=device_outputs; p.strides=device_strides;
    p.offsets=const_cast<int*>(call.offsets_device); p.rows=const_cast<int*>(call.rows_device);
    p.directory_header=directory.header; p.directory_entries=directory.entries;
    p.directory_capacity=directory.capacity; p.io=indexed;
    p.workspace=call.workspace; p.workspace_bytes=call.workspace_bytes;
    return QK_OK;
  }

  int moe_stage(qk_moe_plan_v1 const& plan,int phase,hggcStream_t stream) override {
    if (!indexed.version || plan.version!=1 || plan.size!=sizeof(plan)) return QK_INVALID;
    switch (phase) {
      case QK_MOE_PREPARE:
        if (moe_prepare_m1_supported(plan)) moe_chain_prepare_m1<Shape,DStride><<<1,256,0,stream>>>(plan);
        else moe_chain_prepare<Shape,DStride><<<moe_prepare_blocks(call.experts,call.m),256,0,stream>>>(plan);
        break;
      case QK_MOE_PRODUCER:
        return gemm.run(stream)==cutlass::Status::kSuccess ? QK_OK : QK_RUNTIME_ERROR;
      case QK_MOE_ACTIVATE:
        moe_chain_swiglu<<<dim3(std::min((plan.down.k+255)/256,32),call.m),256,0,stream>>>(plan);
        break;
      case QK_MOE_FINISH:
        return finish_indexed(stream);
      default: return QK_INVALID;
    }
    return hggcGetLastError()==hggcSuccess ? QK_OK : QK_RUNTIME_ERROR;
  }

  int bind_indexed(qk_llama_indexed_v1 const& io) override {
    if constexpr (!Directory) return QK_UNSUPPORTED;
    if (!device_only || call.m>32 || call.experts>1024) return QK_UNSUPPORTED;
    if (indexed.version || io.version!=1 || io.size!=sizeof(io) || io.reserved ||
        io.tokens<=0 || io.topk<=0 || io.channels<=0 || io.topk>call.experts ||
        int64_t(io.tokens)*io.topk!=call.m || io.tokens!=max_rows ||
        io.ids_stride<io.topk || io.a_row_stride<call.k ||
        io.a_token_stride<=0 || io.out_row_stride<call.n ||
        !io.ids || !io.a || !io.output || !io.row_ids ||
        (uintptr_t(io.ids)|uintptr_t(io.a)|uintptr_t(io.output)|uintptr_t(io.row_ids))%4)
      return QK_INVALID;
    constexpr auto limit=INT64_MAX/sizeof(float);
    if (io.a_row_stride>limit/io.channels || io.a_token_stride<int64_t(io.channels)*io.a_row_stride ||
        io.ids_stride>limit/io.tokens || io.a_token_stride>limit/io.tokens ||
        io.out_row_stride>limit/call.m) return QK_INVALID;
    hggcStreamCaptureStatus capture=hggcStreamCaptureStatusNone;
    if (hggcStreamIsCapturing(static_cast<hggcStream_t>(call.stream),&capture)!=hggcSuccess)
      return QK_RUNTIME_ERROR;
    if (capture!=hggcStreamCaptureStatusNone) return QK_UNSUPPORTED;
    indexed=io;
    return QK_OK;
  }

  int prepare(qk_call_v1 const& c, qk_recipe_v1 const& r, int occupancy, int maximum, bool from_device = false) {
    call = c;
    splits = r.split;
    device_only = from_device;
    max_rows = maximum;
    auto stream = static_cast<hggcStream_t>(c.stream);
    if (!device_only) {
      shapes.reserve(c.experts);
    }
    GroupedWorkspace layout;
    uint64_t head = scheduler_bytes(max_rows,c.experts,Directory,c.m);
    if (!grouped_workspace(c.experts,splits,c.m,c.n,head,sizeof(Shape),sizeof(DStride),
        device_only,layout) || layout.total > c.workspace_bytes) return QK_INVALID;
    char* base = static_cast<char*>(c.workspace);
    auto ds = reinterpret_cast<Shape*>(base + layout.shapes);
    auto dp = reinterpret_cast<Output**>(base + layout.outputs);
    auto dd = reinterpret_cast<DStride*>(base + layout.strides);
    Output* destination;
    if constexpr (Split) {
      partials = reinterpret_cast<float*>(base + layout.partials);
      destination = partials;
      typename Reduction::Arguments reduction_args{c.m,c.n,splits,partials,
          layout.partial_bytes,static_cast<Half*>(c.output),c.n};
      if (reduction.initialize(reduction_args)!=cutlass::Status::kSuccess) return QK_INVALID;
    } else destination = static_cast<Half*>(c.output);
    if (!device_only) {
      outputs.resize(size_t(c.experts)*splits); strides.resize(outputs.size());
    }
    prefix.push_back(0);
    int64_t offset = 0;
    bool uniform = true;
    int first_tiles = device_only ? (maximum+tm-1)/tm : (c.rows_host[0] + tm - 1) / tm;
    for (int e = 0; !device_only && e < c.experts; ++e) {
      int rows = c.rows_host[e], tiles = (rows + tm - 1) / tm;
      uniform &= tiles == first_tiles;
      shapes.emplace_back(rows,c.n,c.k);
      for (int s=0; s<splits; ++s) {
        outputs[e+s*c.experts]=destination+(int64_t(s)*c.m+offset)*c.n;
        strides[e+s*c.experts]=cutlass::make_cute_packed_stride(DStride{},cute::make_shape(rows,c.n,1));
      }
      prefix.push_back(prefix.back() + tiles);
      offset += rows;
    }
    auto copy = [&](void* dst, void const* src, size_t bytes) {
      return hggcMemcpyAsync(dst,src,bytes,hggcMemcpyHostToDevice,stream) == hggcSuccess;
    };
    if (!device_only && (!copy(ds,shapes.data(),sizeof(Shape)*c.experts) ||
        !copy(dp,outputs.data(),sizeof(Output*)*outputs.size()) ||
        !copy(dd,strides.data(),sizeof(DStride)*strides.size()))) return QK_RUNTIME_ERROR;
    if (device_only) {
      device_shapes=ds; device_outputs=dp; device_strides=dd;
      call.rows_device=reinterpret_cast<int*>(base + layout.rows);
    }
    auto sa = cutlass::make_cute_packed_stride(typename K::StrideA{},cute::make_shape(max_rows,c.k,c.experts));
    auto sb = cutlass::make_cute_packed_stride(typename K::StrideB{},cute::make_shape(c.n,c.k,c.experts));
    auto ss = cutlass::make_cute_packed_stride(typename T::Mainloop::StrideScale{},
        cute::make_shape(c.n,c.k/c.group_size,c.experts));
    moe_grouped_ppu::GroupProblemShape problem;
    problem.num_groups=c.experts; problem.problem_shapes=ds;
    problem.host_problem_shapes=device_only ? nullptr : shapes.data();
    typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped, problem,
        {static_cast<Half const*>(c.a),sa,static_cast<typename T::Low const*>(c.low),sb,
         static_cast<Half const*>(c.metadata),ss,c.group_size,static_cast<Half const*>(c.zero),c.offsets_device},
        {{},static_cast<Output const**>(nullptr),typename T::Epilogue::StrideC{},dp,dd},
        cutlass::KernelHardwareInfo{c.device,c.compute_units}};
    args.representative_m=max_rows; args.representative_n=c.n; args.representative_k=c.k;
    if constexpr (!std::is_void_v<typename T::High>)
      args.mainloop.ptr_B2=static_cast<typename T::High const*>(c.high);
    if constexpr (Directory) {
      directory=quactlize::moe_directory::make_bounded_view(base,head,c.m,max_rows,c.experts,tm);
      args.directory_header=directory.header; args.directory_entries=directory.entries;
      args.logical_work_upper=uint64_t(directory.capacity)*uint64_t((c.n+tn-1)/tn)*splits;
      args.ctas_per_cu=occupancy; args.grid_ctas_override=r.grid; args.splitk=splits;
    } else {
      args.group_M=call.rows_device; args.mtiles_uniform=uniform?first_tiles:0; args.splitk=splits;
    }
    if (G::can_implement(args) != cutlass::Status::kSuccess) return QK_UNSUPPORTED;
    if (G::get_workspace_size(args)>head) return QK_INVALID;
    if (gemm.initialize(args,base,stream)!=cutlass::Status::kSuccess) return QK_INITIALIZE_ERROR;
    if constexpr (!Directory)
      if (!device_only && !uniform && !copy(base,prefix.data(),prefix.size()*sizeof(int))) return QK_RUNTIME_ERROR;
    return QK_OK;
  }
  int finish_indexed(hggcStream_t stream) {
#define QK_FINISH(S) case S: indexed_finish<S><<<dim3(std::min((call.n+255)/256,32),call.m),256,0,stream>>>( \
        partials,static_cast<Half const*>(call.output),indexed.output,indexed.row_ids, \
        call.m,call.n,indexed.out_row_stride,directory.header); break
    switch (splits) { QK_FINISH(1); QK_FINISH(2); QK_FINISH(4); QK_FINISH(8); }
#undef QK_FINISH
    return hggcGetLastError()==hggcSuccess ? QK_OK : QK_RUNTIME_ERROR;
  }
  int run(hggcStream_t stream) override {
    if (indexed.version) {
      Output* destination;
      if constexpr (Split) destination=partials;
      else destination=static_cast<Half*>(call.output);
      indexed_prepare<tm><<<dim3(std::min((call.k+255)/256,32),call.m),256,0,stream>>>(
          indexed,const_cast<Half*>(static_cast<Half const*>(call.a)),
          const_cast<int*>(call.offsets_device),const_cast<int*>(call.rows_device),
          device_shapes,device_outputs,device_strides,destination,call.m,call.n,call.k,
          call.experts,splits,directory);
      if (hggcGetLastError()!=hggcSuccess) return QK_RUNTIME_ERROR;
      if (gemm.run(stream)!=cutlass::Status::kSuccess) return QK_RUNTIME_ERROR;
      return finish_indexed(stream);
    }
    if (device_only) {
      if constexpr (Split) {
        grouped_splitk_device_metadata<<<(call.experts+127)/128,128,0,stream>>>(call.offsets_device,
            partials,device_shapes,device_outputs,device_strides,const_cast<int*>(call.rows_device),
            call.m,call.n,call.k,call.experts,splits);
      } else {
        grouped_device_metadata<<<(call.experts+127)/128,128,0,stream>>>(call.offsets_device,
            static_cast<Half*>(call.output),device_shapes,device_outputs,device_strides,
            const_cast<int*>(call.rows_device),call.n,call.k,call.experts);
      }
      if (hggcGetLastError()!=hggcSuccess) return QK_RUNTIME_ERROR;
    }
    if constexpr (Directory) {
      if (!quactlize::moe_directory::launch_build<tm>(call.rows_device,call.offsets_device,
          max_rows,call.experts,directory,stream)) return QK_RUNTIME_ERROR;
    }
    if (gemm.run(stream)!=cutlass::Status::kSuccess) return QK_RUNTIME_ERROR;
    if constexpr (Split)
      if (reduction.run(stream)!=cutlass::Status::kSuccess) return QK_RUNTIME_ERROR;
    return QK_OK;
  }
};

template<class G> int resource_query(qk_resources_v1& out) {
  out.shared_bytes=G::GemmKernel::SharedStorageSize;
  if (!ppu_tactics::fits_block_smem(out.shared_bytes)) return QK_UNSUPPORTED;
  out.occupancy=G::maximum_active_blocks();
  if (out.occupancy<=0) return QK_RUNTIME_ERROR;
  return QK_OK;
}

inline int query(qk_call_v1 const& c,qk_recipe_v1 const& r,qk_resources_v1& out, int device_maximum = 0) {
  out={1,sizeof(out),0,0,0,0,0};
  int maximum=0, status=validate(c,r,maximum,device_maximum);
  if (status!=QK_OK) return status;
  int actual_device=-1;
  if (hggcGetDevice(&actual_device)!=hggcSuccess) return QK_RUNTIME_ERROR;
  if (actual_device!=c.device ||
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(actual_device)!=c.compute_units)
    return QK_INVALID;
  if constexpr (grouped) {
    GroupedWorkspace layout;
    bool const compact=compact_device_call(device_maximum,c.experts);
    if (!grouped_workspace(c.experts,r.split,c.m,c.n,
        scheduler_bytes(maximum,c.experts,r.algorithm==QK_PERSISTENT || compact,c.m),
        sizeof(moe_grouped_ppu::GroupShape),sizeof(moe_grouped_ppu::DStride),device_maximum>0,layout))
      return QK_INVALID;
    out.workspace_bytes=layout.total;
    if (r.algorithm==QK_PERSISTENT) {
      status=r.split>1 ? resource_query<typename Group<true,float>::Gemm>(out)
                       : resource_query<typename Group<true>::Gemm>(out);
    } else if (compact) {
      status=r.split>1 ? resource_query<typename Group<false,float,true>::Gemm>(out)
                       : resource_query<typename Group<false,Half,true>::Gemm>(out);
    } else {
      status=r.split>1 ? resource_query<typename Group<false,float>::Gemm>(out)
                       : resource_query<typename Group<false>::Gemm>(out);
    }
  } else {
    if (r.split>1) {
      dense_splitk_parallel_ppu::WorkspacePlan plan;
      if (!dense_splitk_parallel_ppu::query_workspace_plan(c.m,c.n,r.split,plan)) return QK_INVALID;
      out.workspace_bytes=plan.partial_bytes;
      status=resource_query<typename Dense::Prepared::SplitGemm>(out);
    } else if (r.algorithm==QK_PERSISTENT) status=resource_query<typename Dense::PersistentGemm>(out);
    else status=resource_query<typename Dense::Shipping::Gemm>(out);
  }
  if (status!=QK_OK) out.runtime_status=int(hggcPeekAtLastError());
  if (status==QK_OK && r.algorithm==QK_PERSISTENT &&
      uint64_t(r.grid)>uint64_t(c.compute_units)*uint64_t(out.occupancy)) return QK_UNSUPPORTED;
  return status;
}
}  // namespace quactlize::runtime

extern "C" qk_identity_v1 const* quactlize_kpack_identity_v1() {
  using namespace quactlize::runtime;
  static qk_identity_v1 const id{1,sizeof(qk_identity_v1),qtype,route,QK_TM,tn,tk,
      QK_WM,QK_WN,stages,QK_AP,QK_DN,mapping,QK_PARENT,QK_BUILD_KEY};
  return &id;
}
extern "C" int quactlize_kpack_device_v1(char* name,int capacity,int32_t* ordinal,int32_t* cu) {
  if (!name || capacity<1 || !ordinal || !cu) return QK_INVALID;
  hggcDeviceProp properties{};
  if (hggcGetDevice(ordinal)!=hggcSuccess ||
      hggcGetDeviceProperties(&properties,*ordinal)!=hggcSuccess) return QK_RUNTIME_ERROR;
  *cu=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(*ordinal);
  if (*cu<=0 || int(std::strlen(properties.name))>=capacity) return QK_RUNTIME_ERROR;
  std::strcpy(name,properties.name);
  return QK_OK;
}
extern "C" int quactlize_kpack_query_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,qk_resources_v1* out) {
  if (!c || !r || !out) return QK_INVALID;
  try { return quactlize::runtime::query(*c,*r,*out); }
  catch (...) { return QK_RUNTIME_ERROR; }
}
extern "C" int quactlize_kpack_prepare_v1(qk_call_v1 const* c,qk_recipe_v1 const* r,void** handle) {
  using namespace quactlize::runtime;
  if (!handle) return QK_INVALID;
  *handle=nullptr;
  qk_resources_v1 resources{};
  int status=quactlize_kpack_query_v1(c,r,&resources);
  if (status!=QK_OK) return status;
  if (!c->a || !c->low || !c->metadata || !c->output ||
      ((F::spec.high_bits!=0) != (c->high!=nullptr)) ||
      (needs_zero ? c->zero==nullptr : c->zero!=nullptr) ||
      c->workspace_bytes<resources.workspace_bytes ||
      (resources.workspace_bytes && (!c->workspace || (uintptr_t(c->workspace)&15))) ||
      (grouped && (!c->rows_device || !c->offsets_device))) return QK_INVALID;
  std::unique_ptr<Handle> result;
  try {
    if constexpr (grouped) {
      int maximum=0; validate(*c,*r,maximum);
      if (r->algorithm==QK_PERSISTENT && r->split>1) {
        result=std::make_unique<GroupedHandle<true,true>>();
        status=static_cast<GroupedHandle<true,true>*>(result.get())->prepare(*c,*r,resources.occupancy,maximum);
      } else if (r->algorithm==QK_PERSISTENT) {
        result=std::make_unique<GroupedHandle<true>>();
        status=static_cast<GroupedHandle<true>*>(result.get())->prepare(*c,*r,resources.occupancy,maximum);
      } else if (r->split>1) {
        result=std::make_unique<GroupedHandle<false,true>>();
        status=static_cast<GroupedHandle<false,true>*>(result.get())->prepare(*c,*r,resources.occupancy,maximum);
      } else {
        result=std::make_unique<GroupedHandle<false>>();
        status=static_cast<GroupedHandle<false>*>(result.get())->prepare(*c,*r,resources.occupancy,maximum);
      }
    } else {
      result=std::make_unique<DenseHandle<Dense>>();
      status=static_cast<DenseHandle<Dense>*>(result.get())->prepare(*c,*r,resources.occupancy);
    }
    if (status!=QK_OK) {
      // Preserve host backing until any enqueued metadata copy has completed.
      hggcStreamSynchronize(static_cast<hggcStream_t>(c->stream));
      return status;
    }
    *handle=result.release(); return QK_OK;
  } catch (...) {
    hggcStreamSynchronize(static_cast<hggcStream_t>(c->stream));
    return QK_RUNTIME_ERROR;
  }
}
extern "C" int quactlize_kpack_run_v1(void* handle,void* stream) {
  if (!handle) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle*>(handle)->run(static_cast<hggcStream_t>(stream));
}

extern "C" int quactlize_kpack_bind_llama_indexed_v1(void* handle,qk_llama_indexed_v1 const* io) {
  if (!handle || !io) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle*>(handle)->bind_indexed(*io);
}
extern "C" int quactlize_kpack_moe_projection_v1(void* handle,qk_moe_projection_v1* out) {
  if (!handle || !out) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle*>(handle)->moe_projection(*out);
}
extern "C" int quactlize_kpack_moe_stage_v1(void* handle,qk_moe_plan_v1 const* plan,int phase,void* stream) {
  if (!handle || !plan) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle*>(handle)->moe_stage(*plan,phase,static_cast<hggcStream_t>(stream));
}

extern "C" int quactlize_kpack_grouped_query_v2(qk_device_call_v2 const* d,
    qk_recipe_v1 const* r,qk_resources_v1* out) {
  using namespace quactlize::runtime;
  if (!d || !r || !out || d->version!=2 || d->size!=sizeof(*d) ||
      d->reserved || d->max_rows<=0) return QK_INVALID;
  if constexpr (!grouped) return QK_UNSUPPORTED;
  try { return query(d->call,*r,*out,d->max_rows); }
  catch (...) { return QK_RUNTIME_ERROR; }
}

extern "C" int quactlize_kpack_grouped_prepare_v2(qk_device_call_v2 const* d,
    qk_recipe_v1 const* r,void** handle) {
  using namespace quactlize::runtime;
  if (!handle) return QK_INVALID;
  *handle=nullptr;
  qk_resources_v1 resources{};
  int status=quactlize_kpack_grouped_query_v2(d,r,&resources);
  if (status!=QK_OK) return status;
  auto const& c=d->call;
  hggcStreamCaptureStatus capture=hggcStreamCaptureStatusNone;
  if (hggcStreamIsCapturing(static_cast<hggcStream_t>(c.stream),&capture)!=hggcSuccess)
    return QK_RUNTIME_ERROR;
  if (capture!=hggcStreamCaptureStatusNone) return QK_UNSUPPORTED;
  if (!c.a || !c.low || !c.metadata || !c.output || !c.offsets_device ||
      ((F::spec.high_bits!=0)!=(c.high!=nullptr)) || (needs_zero ? c.zero==nullptr : c.zero!=nullptr) ||
      !c.workspace || c.workspace_bytes<resources.workspace_bytes ||
      (uintptr_t(c.workspace)&15) || (uintptr_t(c.offsets_device)&3)) return QK_INVALID;
  if constexpr (grouped) {
    try {
      std::unique_ptr<Handle> result;
      if (r->algorithm==QK_PERSISTENT && r->split>1) {
        auto p=std::make_unique<GroupedHandle<true,true>>();
        status=p->prepare(c,*r,resources.occupancy,d->max_rows,true); result=std::move(p);
      } else if (r->algorithm==QK_PERSISTENT) {
        auto p=std::make_unique<GroupedHandle<true>>();
        status=p->prepare(c,*r,resources.occupancy,d->max_rows,true); result=std::move(p);
      } else if (compact_device_call(d->max_rows,c.experts)) {
        if (r->split>1) {
          auto p=std::make_unique<GroupedHandle<false,true,true>>();
          status=p->prepare(c,*r,resources.occupancy,d->max_rows,true); result=std::move(p);
        } else {
          auto p=std::make_unique<GroupedHandle<false,false,true>>();
          status=p->prepare(c,*r,resources.occupancy,d->max_rows,true); result=std::move(p);
        }
      } else if (r->split>1) {
        auto p=std::make_unique<GroupedHandle<false,true>>();
        status=p->prepare(c,*r,resources.occupancy,d->max_rows,true); result=std::move(p);
      } else {
        auto p=std::make_unique<GroupedHandle<false>>();
        status=p->prepare(c,*r,resources.occupancy,d->max_rows,true); result=std::move(p);
      }
      if (status!=QK_OK) return status;
      *handle=result.release(); return QK_OK;
    } catch (...) { return QK_RUNTIME_ERROR; }
  }
  return QK_UNSUPPORTED;
}
extern "C" void quactlize_kpack_destroy_v1(void* handle) {
  delete static_cast<quactlize::runtime::Handle*>(handle);
}
extern "C" int quactlize_kpack_measure_v1(void* handle,void* stream,int repeats,double* us) {
  if (!handle || !us || repeats<1 || repeats>1024) return QK_INVALID;
  *us=0;
  hggcEvent_t begin{},end{};
  if (hggcEventCreate(&begin)!=hggcSuccess) return QK_RUNTIME_ERROR;
  if (hggcEventCreate(&end)!=hggcSuccess) { hggcEventDestroy(begin); return QK_RUNTIME_ERROR; }
  auto s=static_cast<hggcStream_t>(stream);
  int status=hggcEventRecord(begin,s)==hggcSuccess ? QK_OK : QK_RUNTIME_ERROR;
  for (int i=0; i<repeats && status==QK_OK; ++i) status=quactlize_kpack_run_v1(handle,stream);
  if (status==QK_OK && (hggcEventRecord(end,s)!=hggcSuccess || hggcEventSynchronize(end)!=hggcSuccess))
    status=QK_RUNTIME_ERROR;
  float ms=0;
  if (status==QK_OK && hggcEventElapsedTime(&ms,begin,end)==hggcSuccess) *us=double(ms)*1000/repeats;
  else status=QK_RUNTIME_ERROR;
  hggcEventDestroy(begin); hggcEventDestroy(end);
  return status;
}
