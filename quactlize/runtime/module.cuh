#pragma once
#include "abi.h"
#include "kernel_types.cuh"
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
using F = Format<qtype>;
constexpr uint64_t mapping = qtype == 12 ? UINT64_C(0x51344b5034540001)
                                        : UINT64_C(0x514b504b54000001);

inline uint64_t align16(uint64_t value) { return (value + 15) & ~UINT64_C(15); }

inline int validate(qk_call_v1 const& c, qk_recipe_v1 const& r, int& max_rows, int device_maximum = 0) {
  if (c.version != 1 || c.size != sizeof(c) || r.version != 1 || r.size != sizeof(r) ||
      c.mapping_id != mapping || c.m <= 0 || c.n <= 0 || c.k <= 0 ||
      c.experts <= 0 || c.group_size != F::spec.group_size || c.compute_units <= 0 ||
      c.m > INT32_MAX-256 || c.n > INT32_MAX-256 || c.k > INT32_MAX-256 ||
      c.experts > INT32_MAX-256 ||
      c.n % 256 || c.k % ((qtype == 11 || qtype == 14) ? 512 : 256) || c.k % tk ||
      r.algorithm < QK_ORDINARY || r.algorithm > QK_PERSISTENT ||
      (r.split != 1 && r.split != 2 && r.split != 4 && r.split != 8)) return QK_INVALID;
  if (r.algorithm == QK_PERSISTENT ? (r.grid <= 0 || r.split != 1) : r.grid != 0)
    return QK_INVALID;
  if (c.k % (tk * r.split) || c.k / (tk * r.split) < stages - 1)
    return QK_UNSUPPORTED;
  max_rows = c.m;
  if constexpr (grouped) {
    if (r.split != 1) return QK_INVALID;
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
template<bool P> using Group = GroupedTypes<qtype,QK_TM,QK_TN,QK_TK,QK_WM,QK_WN,QK_STAGES,QK_DN,P>;

inline uint64_t scheduler_bytes(int max_rows, int experts, bool persistent) {
  return align16(persistent ? quactlize::moe_directory::workspace_bytes(max_rows, experts, tm)
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

template<bool Persistent> struct GroupedHandle final : Handle {
  using T = Group<Persistent>;
  using K = typename T::Kernel;
  using G = typename T::Gemm;
  using Shape = moe_grouped_ppu::GroupShape;
  using DStride = moe_grouped_ppu::DStride;
  G gemm;
  qk_call_v1 call{};
  int max_rows = 0;
  quactlize::moe_directory::View directory{};
  std::vector<Shape> shapes;
  std::vector<Half*> outputs;
  std::vector<DStride> strides;
  std::vector<int> prefix;
  bool device_only = false;
  Shape* device_shapes = nullptr;
  Half** device_outputs = nullptr;
  DStride* device_strides = nullptr;

  int prepare(qk_call_v1 const& c, qk_recipe_v1 const& r, int occupancy, int maximum, bool from_device = false) {
    call = c;
    device_only = from_device;
    max_rows = maximum;
    auto stream = static_cast<hggcStream_t>(c.stream);
    if (!device_only) {
      shapes.reserve(c.experts); outputs.reserve(c.experts); strides.reserve(c.experts);
    }
    prefix.push_back(0);
    int64_t offset = 0;
    bool uniform = true;
    int first_tiles = device_only ? (maximum+tm-1)/tm : (c.rows_host[0] + tm - 1) / tm;
    for (int e = 0; !device_only && e < c.experts; ++e) {
      int rows = c.rows_host[e], tiles = (rows + tm - 1) / tm;
      uniform &= tiles == first_tiles;
      shapes.emplace_back(rows,c.n,c.k);
      outputs.push_back(static_cast<Half*>(c.output) + offset * c.n);
      strides.push_back(cutlass::make_cute_packed_stride(DStride{},cute::make_shape(rows,c.n,1)));
      prefix.push_back(prefix.back() + tiles);
      offset += rows;
    }
    uint64_t head = scheduler_bytes(max_rows,c.experts,Persistent);
    char* base = static_cast<char*>(c.workspace);
    auto ds = reinterpret_cast<Shape*>(base + head);
    auto dp = reinterpret_cast<Half**>(base + head + align16(sizeof(Shape)*c.experts));
    auto dd = reinterpret_cast<DStride*>(reinterpret_cast<char*>(dp) + align16(sizeof(Half*)*c.experts));
    auto copy = [&](void* dst, void const* src, size_t bytes) {
      return hggcMemcpyAsync(dst,src,bytes,hggcMemcpyHostToDevice,stream) == hggcSuccess;
    };
    if (!device_only && (!copy(ds,shapes.data(),sizeof(Shape)*c.experts) ||
        !copy(dp,outputs.data(),sizeof(Half*)*c.experts) ||
        !copy(dd,strides.data(),sizeof(DStride)*c.experts))) return QK_RUNTIME_ERROR;
    if (device_only) {
      device_shapes=ds; device_outputs=dp; device_strides=dd;
      call.rows_device=reinterpret_cast<int*>(reinterpret_cast<char*>(dd)+align16(sizeof(DStride)*c.experts));
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
        {{},static_cast<Half const**>(nullptr),typename T::Epilogue::StrideC{},dp,dd},
        cutlass::KernelHardwareInfo{c.device,c.compute_units}};
    args.representative_m=max_rows; args.representative_n=c.n; args.representative_k=c.k;
    if constexpr (!std::is_void_v<typename T::High>)
      args.mainloop.ptr_B2=static_cast<typename T::High const*>(c.high);
    if constexpr (Persistent) {
      directory=quactlize::moe_directory::make_view(base,head,max_rows,c.experts,tm);
      args.directory_header=directory.header; args.directory_entries=directory.entries;
      args.logical_work_upper=uint64_t(directory.capacity)*uint64_t((c.n+tn-1)/tn);
      args.ctas_per_cu=occupancy; args.grid_ctas_override=r.grid; args.splitk=1;
    } else {
      args.group_M=call.rows_device; args.mtiles_uniform=uniform?first_tiles:0; args.splitk=1;
    }
    if (G::can_implement(args) != cutlass::Status::kSuccess) return QK_UNSUPPORTED;
    if (G::get_workspace_size(args)>head) return QK_INVALID;
    if (gemm.initialize(args,base,stream)!=cutlass::Status::kSuccess) return QK_INITIALIZE_ERROR;
    if constexpr (!Persistent)
      if (!device_only && !uniform && !copy(base,prefix.data(),prefix.size()*sizeof(int))) return QK_RUNTIME_ERROR;
    return QK_OK;
  }
  int run(hggcStream_t stream) override {
    if (device_only) {
      grouped_device_metadata<<<(call.experts+127)/128,128,0,stream>>>(call.offsets_device,
          static_cast<Half*>(call.output),device_shapes,device_outputs,device_strides,
          const_cast<int*>(call.rows_device),call.n,call.k,call.experts);
      if (hggcGetLastError()!=hggcSuccess) return QK_RUNTIME_ERROR;
    }
    if constexpr (Persistent) {
      if (!quactlize::moe_directory::launch_build<tm>(call.rows_device,call.offsets_device,
          max_rows,call.experts,directory,stream)) return QK_RUNTIME_ERROR;
    }
    return gemm.run(stream)==cutlass::Status::kSuccess ? QK_OK : QK_RUNTIME_ERROR;
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
    out.workspace_bytes=scheduler_bytes(maximum,c.experts,r.algorithm==QK_PERSISTENT)+
        align16(sizeof(moe_grouped_ppu::GroupShape)*c.experts)+
        align16(sizeof(Half*)*c.experts)+align16(sizeof(moe_grouped_ppu::DStride)*c.experts)+
        (device_maximum > 0 ? align16(sizeof(int)*c.experts) : 0);
    status = r.algorithm==QK_PERSISTENT ? resource_query<typename Group<true>::Gemm>(out)
                                      : resource_query<typename Group<false>::Gemm>(out);
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
      (packed ? c->zero!=nullptr : c->zero==nullptr) ||
      c->workspace_bytes<resources.workspace_bytes ||
      (resources.workspace_bytes && (!c->workspace || (uintptr_t(c->workspace)&15))) ||
      (grouped && (!c->rows_device || !c->offsets_device))) return QK_INVALID;
  std::unique_ptr<Handle> result;
  try {
    if constexpr (grouped) {
      int maximum=0; validate(*c,*r,maximum);
      if (r->algorithm==QK_PERSISTENT) {
        result=std::make_unique<GroupedHandle<true>>();
        status=static_cast<GroupedHandle<true>*>(result.get())->prepare(*c,*r,resources.occupancy,maximum);
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
      ((F::spec.high_bits!=0)!=(c.high!=nullptr)) || (packed ? c.zero!=nullptr : c.zero==nullptr) ||
      !c.workspace || c.workspace_bytes<resources.workspace_bytes ||
      (uintptr_t(c.workspace)&15) || (uintptr_t(c.offsets_device)&3)) return QK_INVALID;
  if constexpr (grouped) {
    try {
      std::unique_ptr<Handle> result;
      if (r->algorithm==QK_PERSISTENT) {
        auto p=std::make_unique<GroupedHandle<true>>();
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
