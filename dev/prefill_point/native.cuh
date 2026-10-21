// Opt-in single-point diagnosis; never changes normal component campaigns.
#define POINT_RT(name) cuda##name
#include "events.hpp"
#undef POINT_RT

extern "C" int nc_point_samples(void * ptr,int rotate,int mode,int index,int samples,double * values) {
    return nc::guard([&] {
        auto & s=*static_cast<nc::Session*>(ptr);
        const int stages=int(s.records.at(0)->stages.size());
        const int copies=rotate?int(s.records.size()):1;
        std::vector<bool> resident(stages);
        for(int j=0;j<stages;++j) {
            const auto name=s.records.at(0)->stages.at(j).description["symbol"].get<std::string>();
            resident[j]=name.find("convert")!=std::string::npos;
        }
        prefill_point::sample(s.ctx.stream(),stages,copies,mode,index,samples,[&](int j,int i) {
            // Isolated casts use resident scratch, as in the original
            // component campaign. Weight expansion/provider still rotate.
            const int copy=mode==0 && resident[j] ? 0 : i;
            auto & stage=s.records.at(copy)->stages.at(j);
            if(stage.provider)return int(stage.provider());
            auto & k=stage.kernel;
            if(k.driver_entry) {
                auto & p=k.driver;
                return int(cuLaunchKernel(p.func,p.gridDimX,p.gridDimY,p.gridDimZ,
                    p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes,
                    reinterpret_cast<CUstream>(s.ctx.stream()),p.kernelParams,p.extra));
            }
            auto & p=k.runtime;
            return int(cudaLaunchKernel(p.func,p.gridDim,p.blockDim,p.kernelParams,p.sharedMemBytes,s.ctx.stream()));
        },values);
    });
}
