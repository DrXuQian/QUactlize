// Real CUDA global stores using the production PPU TiledMma's coordinate
// mapping. No PPU MMA instruction is executed on NVIDIA.
#include <cuda_runtime.h>
#include <hggc_runtime.h>
#include "../../tests/kpack_grouped_postops_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Buffer {
  T* ptr = nullptr;
  explicit Buffer(size_t n) { check_cuda(cudaMalloc(&ptr,n*sizeof(T))); }
  ~Buffer() { cudaFree(ptr); }
  void upload(std::vector<T> const& host) {
    check_cuda(cudaMemcpy(ptr,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
  }
};

template<int Q, int TM>
using Types = PostopsLayout<TM>;

template<int Q, int TM, bool WrongEntry>
__global__ void direct_kernel(typename Types<Q,TM>::Epilogue::Params params,
    int const* rows, int const* offsets, int n, int experts, int splits) {
  using namespace cute;
  using T = Types<Q,TM>;
  using Epi = typename T::Epilogue;
  using Mma = typename T::Mainloop::TiledMma;
  int e = blockIdx.z % experts, s = blockIdx.z / experts;
  int m = rows[e], bm = blockIdx.x, bn = blockIdx.y;
  if (bm*TM >= m) return;
  Mma mma;
  auto accum = partition_fragment_C(mma,make_shape(Int<TM>{},_64{}));
  auto identity = make_identity_tensor(make_shape(Int<TM>{},_64{}));
  auto coords = mma.get_thread_slice(threadIdx.x).partition_C(identity);
  for (int i=0; i<size(accum); ++i) {
    int row = offsets[e]+bm*TM+int(get<0>(coords(i)));
    int col = bn*64+int(get<1>(coords(i)));
    accum(i) = float(s*100000+row*1000+col+1);
  }
  __shared__ typename Epi::SharedStorage storage;
  Epi epi(params,storage);
  // Rotating the descriptor index must fail; the stored value retains its
  // true slice/expert tag, so even complete output coverage cannot hide it.
  int entry = e + (WrongEntry ? (s+1)%splits : s)*experts;
  epi(make_shape(m,n,256,experts*splits),typename T::Tile{},
      make_coord(bm,bn,_,entry),accum,mma,make_tuple(m-bm*TM,n-bn*64,0),
      threadIdx.x,reinterpret_cast<char*>(&storage));
}

template<int Q, int TM>
int test_direct() {
  using T = Types<Q,TM>;
  using Epi = typename T::Epilogue;
  using Stride = typename Epi::InternalStrideD;
  int bad_cases=0;
  for (auto rows : {std::vector<int>{1,0,1,0}, std::vector<int>{9,0,17,3}})
  for (int n : {64,66,256}) for (int splits : {2,4,8}) {
    int e=rows.size(),m=0,max_rows=0;
    std::vector<int> offsets{0};
    for (int r:rows) { m+=r; offsets.push_back(m); max_rows=std::max(max_rows,r); }
    int count=splits*m*n;
    Buffer<float> output(count+8);
    Buffer<int> device_rows(e),device_offsets(e+1);
    device_rows.upload(rows); device_offsets.upload(offsets);
    std::vector<float*> pointers(e*splits);
    std::vector<Stride> strides(e*splits);
    for(int s=0;s<splits;++s) for(int expert=0;expert<e;++expert) {
      int entry=expert+s*e;
      pointers[entry]=output.ptr+4+(s*m+offsets[expert])*n;
      strides[entry]=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(rows[expert],n,1));
    }
    Buffer<float*> device_ptrs(pointers.size()); device_ptrs.upload(pointers);
    Buffer<Stride> device_strides(strides.size()); device_strides.upload(strides);
    typename Epi::Params params{{},nullptr,{},device_ptrs.ptr,device_strides.ptr};
    dim3 grid((max_rows+TM-1)/TM,(n+63)/64,e*splits);
    dim3 block(cute::size(typename T::Mainloop::TiledMma{}));
    std::vector<float> initial(count+8,-1234567.f),got(count+8);
    for(bool negative:{false,true}) {
      output.upload(initial);
      if (negative) direct_kernel<Q,TM,true><<<grid,block>>>(params,device_rows.ptr,device_offsets.ptr,n,e,splits);
      else direct_kernel<Q,TM,false><<<grid,block>>>(params,device_rows.ptr,device_offsets.ptr,n,e,splits);
      check_cuda(cudaGetLastError()); check_cuda(cudaDeviceSynchronize());
      check_cuda(cudaMemcpy(got.data(),output.ptr,got.size()*4,cudaMemcpyDeviceToHost));
      int bad=0,guard=0;
      for(int i=0;i<4;++i) guard+=(got[i]!=initial[i])+(got[count+4+i]!=initial[count+4+i]);
      for(int s=0;s<splits;++s) for(int row=0;row<m;++row) for(int col=0;col<n;++col)
        bad+=got[4+(s*m+row)*n+col] != float(s*100000+row*1000+col+1);
      if (guard || (negative ? bad!=count : bad!=0)) ++bad_cases;
    }
  }
  std::printf("CUDA_GROUPED_DIRECT q=%d TM=%d cases=36 positive_and_rotated_slice=1 bad_cases=%d\n",Q,TM,bad_cases);
  return bad_cases;
}

extern "C" int qkg_cuda_direct_test() {
  try {
    return test_direct<10,8>()+test_direct<11,8>()+test_direct<12,8>()+
           test_direct<13,8>()+test_direct<14,8>()+test_direct<12,16>();
  } catch(std::exception const& error) {
    std::fprintf(stderr,"CUDA_GROUPED_DIRECT ERROR %s\n",error.what()); return -1;
  }
}
