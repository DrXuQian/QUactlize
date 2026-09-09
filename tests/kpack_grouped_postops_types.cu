#include "../quactlize/runtime/kernel_types.cuh"
#include "kpack_grouped_postops_layout.hpp"

template<int Q,int TM> struct TypeProof {
  using S1=quactlize::runtime::GroupedTypes<Q,TM,64,256,TM,16,2,64,false>;
  using Split=quactlize::runtime::GroupedTypes<Q,TM,64,256,TM,16,2,64,false,float,true>;
  using Projection=PostopsLayout<TM>;
  using ActualMma=typename Split::Mainloop::TiledMma;
  using TestMma=typename Projection::Mma;
  static_assert(std::is_same_v<typename S1::Epilogue,typename S1::OutputEpilogue>,"S1 must remain exact");
  static_assert(!std::is_same_v<typename Split::Epilogue,typename Split::OutputEpilogue>);
  static_assert(std::is_same_v<typename S1::Mainloop,typename Split::Mainloop>,"mainloop changed");
  static_assert(std::is_same_v<decltype(ActualMma{}.get_layoutC_TV()),decltype(TestMma{}.get_layoutC_TV())>,
      "CUDA projection must be exactly the production C ownership type");
  static_assert(std::is_same_v<typename Split::Epilogue::StrideD,typename Projection::Epilogue::StrideD>);
  static_assert(sizeof(typename Split::Epilogue::Params)==sizeof(typename Projection::Epilogue::Params));
  static_assert(sizeof(typename Split::Epilogue::SharedStorage)==sizeof(typename Split::OutputEpilogue::SharedStorage));
};

static_assert(sizeof(TypeProof<10,8>)+sizeof(TypeProof<11,8>)+sizeof(TypeProof<12,8>)+
    sizeof(TypeProof<13,8>)+sizeof(TypeProof<14,8>)+sizeof(TypeProof<12,16>)==6);
