// Appended to a generated copy of the exact native host dispatch functions.
// Kernels/heuristics are not rewritten. The linked backend owns their code.
namespace nc {
inline void dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * out) {
    ggml_cuda_set_device(ctx.device);
    if (out->op==GGML_OP_MUL_MAT) ggml_cuda_mul_mat(ctx,out->src[0],out->src[1],out);
    else if (out->op==GGML_OP_MUL_MAT_ID) ggml_cuda_mul_mat_id(ctx,out);
    else throw std::runtime_error("not a matrix operator");
    check(cudaGetLastError(),"native dispatch launch");
}
struct Session {
    ggml_backend_cuda_context ctx;
    std::vector<std::unique_ptr<ggml_cuda_pool>> pools;
    std::vector<std::unique_ptr<Recorder>> records;
    std::vector<void *> allocations;
    std::string text;
    explicit Session(int device) : ctx(device) { ctx.stream(); }
    ~Session() { cudaStreamSynchronize(ctx.stream()); records.clear(); pools.clear();
        for(auto p:allocations) cudaFree(p); }
    void * allocate(size_t bytes) {
        void * p=nullptr;check(cudaMalloc(&p,bytes),"helper allocation");allocations.push_back(p);return p;
    }
    void record(ggml_tensor * out) {
        // A separate native pool per cold weight copy also gives dequantized
        // weights distinct storage. Reusing one expanded-B buffer would turn
        // a nominal cold GGUF ring into a warm provider benchmark.
        if (ctx.pools[ctx.device][0]) throw std::runtime_error("recording pool was reused");
        for (int i=0;i<3;++i) dispatch(ctx,out);
        check(cudaStreamSynchronize(ctx.stream()),"native warmup");
        auto r=std::make_unique<Recorder>(ctx.stream());
        try {
            r->begin(); active=r.get(); dispatch(ctx,out); active=nullptr; r->end();
        } catch (...) { active=nullptr; throw; }
        if (r->stages.empty()) throw std::runtime_error("empty native component inventory");
        if (!records.empty()) {
            if (records[0]->stages.size()!=r->stages.size()) throw std::runtime_error("native stage count changed across ring");
            for (size_t j=0;j<r->stages.size();++j)
                if (records[0]->stages[j].description!=r->stages[j].description)
                    throw std::runtime_error("native launch/config changed across ring");
        }
        pools.push_back(std::move(ctx.pools[ctx.device][0]));
        records.push_back(std::move(r));
    }
};
struct KernelGraph {
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t instance=nullptr;
    explicit KernelGraph(const std::vector<Stage *> & stages) {
        check(cudaGraphCreate(&graph,0),"single-stage graph create");
        try {
            cudaGraphNode_t previous=nullptr;
            for (auto stage:stages) {
                if (stage->provider) throw std::runtime_error("provider in graph forbidden");
                cudaGraphNode_t node=nullptr;
                stage->kernel.add(&node,graph,previous); previous=node;
            }
            check(cudaGraphInstantiateWithFlags(&instance,graph,0),"single-stage instantiate");
        } catch (...) { if (graph) cudaGraphDestroy(graph); throw; }
    }
    void run(cudaStream_t stream) { check(cudaGraphLaunch(instance,stream),"single-stage replay"); }
    ~KernelGraph() { if(instance) cudaGraphExecDestroy(instance); if(graph) cudaGraphDestroy(graph); }
};
inline thread_local std::string error;
template<class F> int guard(F fn) {
    try {fn();error.clear();return 0;}
    catch (const std::exception & e) {error=e.what();return 1;}
}
} // namespace nc
extern "C" const char * nc_error() { return nc::error.c_str(); }
extern "C" void * nc_create(int device) {
    void * result=nullptr; nc::guard([&] {result=new nc::Session(device);}); return result;
}
extern "C" void nc_destroy(void * p) { delete static_cast<nc::Session *>(p); }
extern "C" int nc_record(void * p, ggml_tensor * out) {
    return nc::guard([&] {static_cast<nc::Session *>(p)->record(out);});
}
extern "C" const char * nc_inventory(void * p) {
    auto & s=*static_cast<nc::Session *>(p);nc::Json list=nc::Json::array();
    for (auto & stage:s.records.at(0)->stages) list.push_back(stage.description);
    s.text=list.dump();return s.text.c_str();
}
extern "C" int nc_replay(void * p, int copy, int omitted) {
    return nc::guard([&] {
        auto & s=*static_cast<nc::Session *>(p);
        auto & stages=s.records.at(copy)->stages;
        for (size_t i=0;i<stages.size();++i) if (int(i)!=omitted) {
            if(stages[i].provider) {
                if(stages[i].provider()!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("ordinary native provider failed");
            } else { nc::KernelGraph graph({&stages[i]});graph.run(s.ctx.stream());
                nc::check(cudaStreamSynchronize(s.ctx.stream()),"validation stage complete"); }
        }
        nc::check(cudaStreamSynchronize(s.ctx.stream()),"native replay complete");
    });
}
extern "C" int nc_seed(void * p, int limit) {
    return nc::guard([&] {
        auto & s=*static_cast<nc::Session *>(p);
        for(auto & record:s.records) for(int i=0;i<limit;++i) {
            auto & stage=record->stages.at(i);
            if(stage.provider) {
                if(stage.provider()!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("native prefix provider failed");
            } else { nc::KernelGraph graph({&stage});graph.run(s.ctx.stream());
                nc::check(cudaStreamSynchronize(s.ctx.stream()),"native prefix kernel complete"); }
        }
        nc::check(cudaStreamSynchronize(s.ctx.stream()),"native prefix complete");
    });
}
extern "C" int nc_zero_weight(void * p, void * weight, uint64_t bytes) {
    return nc::guard([&] {
        auto & s=*static_cast<nc::Session *>(p);
        if(!weight || !bytes)throw std::runtime_error("empty weight negative control");
        nc::check(cudaMemsetAsync(weight,0,bytes,s.ctx.stream()),"last-weight-copy negative");
        nc::check(cudaStreamSynchronize(s.ctx.stream()),"weight negative complete");
    });
}
// The shared llama GPU routing helper used by the PoC Quactlize caller. It is
// independent of N/K/qtype; never charge top-k selection (IDs are input here).
extern "C" int nc_route(void * p, int tokens, int experts, int topk, int channels, const int * host_ids) {
    return nc::guard([&] {
        auto & s=*static_cast<nc::Session *>(p);auto stream=s.ctx.stream();
        if(tokens<=0 || experts<=0 || topk<=0 || topk>experts || (channels!=1&&channels!=topk) || !host_ids || !s.records.empty())
            throw std::runtime_error("invalid standalone routing request");
        const int rows=tokens*topk;
        auto ids=static_cast<int *>(s.allocate(size_t(rows)*4));
        auto src=static_cast<int *>(s.allocate(size_t(rows)*4));
        auto dst=static_cast<int *>(s.allocate(size_t(rows)*4));
        auto bounds=static_cast<int *>(s.allocate(size_t(experts+1)*4));
        auto launch=[&] {ggml_cuda_launch_mm_ids_helper(ids,src,dst,bounds,experts,tokens,topk,channels,
            topk,channels,false,stream);nc::check(cudaGetLastError(),"routing helper launch");};
        for(int repeat:{0,1,0}) {
            std::vector<int> input(host_ids,host_ids+rows),a(rows,-1),b(rows,-1),c(experts+1,-1),count(experts,0);
            for(auto & e:input) {if(e<0||e>=experts)throw std::runtime_error("expert outside routing domain");
                e=(e+7*repeat)%experts;++count[e];}
            nc::check(cudaMemcpyAsync(ids,input.data(),size_t(rows)*4,cudaMemcpyHostToDevice,stream),"routing ids upload");
            nc::check(cudaMemsetAsync(src,0xff,size_t(rows)*4,stream),"routing source poison");
            nc::check(cudaMemsetAsync(dst,0xff,size_t(rows)*4,stream),"routing destination poison");
            nc::check(cudaMemsetAsync(bounds,0xff,size_t(experts+1)*4,stream),"routing bounds poison");
            launch();nc::check(cudaStreamSynchronize(stream),"routing oracle sync");
            nc::check(cudaMemcpy(a.data(),src,size_t(rows)*4,cudaMemcpyDeviceToHost),"routing source read");
            nc::check(cudaMemcpy(b.data(),dst,size_t(rows)*4,cudaMemcpyDeviceToHost),"routing destination read");
            nc::check(cudaMemcpy(c.data(),bounds,size_t(experts+1)*4,cudaMemcpyDeviceToHost),"routing bounds read");
            std::vector<bool> seen(rows,false);int prefix=0;
            for(int e=0;e<experts;++e) {
                if(c[e]!=prefix || c[e+1]!=prefix+count[e])throw std::runtime_error("routing prefix/count oracle differs");
                for(int r=c[e];r<c[e+1];++r) {
                    int slot=b[r];
                    if(slot<0||slot>=rows||seen[slot]||input[slot]!=e || a[r]!=slot/topk*channels+slot%topk%channels)
                        throw std::runtime_error("routing permutation/source oracle differs");
                    seen[slot]=true;
                }
                prefix+=count[e];
            }
            if(prefix!=rows)throw std::runtime_error("routing output incomplete");
        }
        auto record=std::make_unique<nc::Recorder>(stream);record->begin();launch();record->end();
        if(record->stages.size()!=1)throw std::runtime_error("shared routing changed from one kernel; review inventory");
        s.records.push_back(std::move(record));
    });
}
// Required by qkp_stage_prepare on every FULL call. This is a memory-API
// component, not a GEMM kernel or an invented launch-latency correction.
extern "C" int nc_reset_samples(void * p, int count, double * out) {
    return nc::guard([&] {
        if(count<9 || !out)throw std::runtime_error("invalid reset sample count");
        auto & s=*static_cast<nc::Session *>(p);auto stream=s.ctx.stream();auto ptr=s.allocate(256);
        nc::check(cudaMemsetAsync(ptr,0x7f,4,stream),"error status poison");
        for(int i=0;i<11;++i)nc::check(cudaMemsetAsync(ptr,0,4,stream),"error reset warmup");
        nc::check(cudaStreamSynchronize(stream),"error reset warmup complete");
        std::vector<cudaEvent_t> ev(2*count,nullptr);
        try {
            for(auto & e:ev)nc::check(cudaEventCreate(&e),"reset event create");
            for(int i=0;i<count;++i) {
                nc::check(cudaEventRecord(ev[2*i],stream),"reset start");
                nc::check(cudaMemsetAsync(ptr,0,4,stream),"four-byte error reset");
                nc::check(cudaEventRecord(ev[2*i+1],stream),"reset stop");
            }
            nc::check(cudaEventSynchronize(ev.back()),"reset timer completion");
            for(int i=0;i<count;++i) {float ms=0;nc::check(cudaEventElapsedTime(&ms,ev[2*i],ev[2*i+1]),"reset interval");
                if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("invalid reset time");out[i]=ms*1000;}
            int value=-1;nc::check(cudaMemcpy(&value,ptr,4,cudaMemcpyDeviceToHost),"error reset oracle");
            if(value)throw std::runtime_error("error reset did not clear status");
        } catch(...) {cudaStreamSynchronize(stream);for(auto e:ev)if(e)cudaEventDestroy(e);throw;}
        for(auto e:ev)nc::check(cudaEventDestroy(e),"reset event destroy");
    });
}
extern "C" int nc_measure(void * p, int index, int rotate, int rounds, int samples, double * us) {
    return nc::guard([&] {
        if(rounds<3 || samples<3 || !us) throw std::runtime_error("invalid native sample request");
        auto & s=*static_cast<nc::Session *>(p);auto stream=s.ctx.stream();
        std::vector<nc::Stage *> stages;
        size_t copies=rotate?s.records.size():1;
        if(rotate && copies<2) throw std::runtime_error("cold stage requires multiple copies");
        for(size_t i=0;i<copies;++i) stages.push_back(&s.records.at(i)->stages.at(index));
        bool provider=bool(stages[0]->provider);
        std::unique_ptr<nc::KernelGraph> graph;
        int divisor=1;
        if(!provider) {
            std::vector<nc::Stage *> repeated;
            do {repeated.insert(repeated.end(),stages.begin(),stages.end());} while(repeated.size()<16);
            divisor=int(repeated.size());graph=std::make_unique<nc::KernelGraph>(repeated);
        }
        auto run=[&](size_t i) {
            if(graph) graph->run(stream);
            else if(stages[i%copies]->provider()!=CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("ordinary provider measurement failed");
        };
        int warmups=provider?int(copies)*((std::max(11,int(copies*2))+int(copies)-1)/int(copies)):2;
        // Precreate all events. No Python or readback between samples. For
        // providers walk the whole ring each round, not only its first slots.
        int per_sample=provider?int(copies):1;
        int count=rounds*samples*per_sample;
        std::vector<cudaEvent_t> events(2*count,nullptr);
        try {
            for(auto & e:events) nc::check(cudaEventCreate(&e),"native timed event create");
            for(int i=0;i<warmups;++i) run(i);
            // Continuous same-stream submissions. Do not put event creation,
            // readback or a CPU/Python pause between warmup and measurement.
            for(int i=0;i<count;++i) {
                nc::check(cudaEventRecord(events[2*i],stream),"native stage start");
                run(i);
                nc::check(cudaEventRecord(events[2*i+1],stream),"native stage stop");
            }
            nc::check(cudaEventSynchronize(events.back()),"native timing complete");
            for(int i=0;i<count;++i) {
                float ms=0;
                nc::check(cudaEventElapsedTime(&ms,events[2*i],events[2*i+1]),"native stage interval");
                if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("invalid native duration");
                us[i]=ms*1000/divisor; // retain EVERY ordinary provider call
            }
        } catch(...) {cudaStreamSynchronize(stream);for(auto e:events)if(e)cudaEventDestroy(e);throw;}
        for(auto e:events) nc::check(cudaEventDestroy(e),"native timed event destroy");
    });
}
