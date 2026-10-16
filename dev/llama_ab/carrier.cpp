// Test-only carrier. Both arms execute the linked llama backend dispatcher.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static void checked(int rc, const char * step) {
    if (rc) throw std::runtime_error(std::string(step) + " rc=" + std::to_string(rc));
}
template<class T> static T symbol(void * lib, const char * name) {
    auto p = reinterpret_cast<T>(dlsym(lib, name));
    if (!p) throw std::runtime_error(std::string("missing runtime symbol: ") + name);
    return p;
}
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static void publish(const fs::path & path, const json & value) {
    require(!fs::exists(path), "refusing to overwrite an existing result");
    fs::create_directories(path.parent_path());
    auto tmp = path; tmp += ".tmp";
    require(!fs::exists(tmp), "result staging already exists");
    { std::ofstream out(tmp); out << value.dump(2) << '\n'; require(out.good(), "result write failed"); }
    fs::rename(tmp, path);
}
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16);
}
static float value(uint64_t i, uint32_t seed) {
    return (int(mix(uint32_t(i) ^ mix(uint32_t(i >> 32)) ^ seed) & 65535) - 32768) / 65536.f;
}

struct Runtime {
    int (*create)(void **, unsigned);
    int (*destroy)(void *);
    int (*elapsed)(float *, void *, void *);
    int (*memory)(size_t *, size_t *);
    Runtime(const std::string & path) {
        void * lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(dlerror());
        create = symbol<decltype(create)>(lib, "hggcEventCreateWithFlags");
        destroy = symbol<decltype(destroy)>(lib, "hggcEventDestroy");
        elapsed = symbol<decltype(elapsed)>(lib, "hggcEventElapsedTime");
        memory = symbol<decltype(memory)>(lib, "hggcMemGetInfo");
    }
};
struct Events {
    Runtime & rt;
    ggml_backend_event start{}, stop{};
    Events(Runtime & r, ggml_backend_dev_t dev) : rt(r) {
        start.device = stop.device = dev;
        checked(rt.create(&start.context, 0), "create timed start");
        checked(rt.create(&stop.context, 0), "create timed stop");
    }
    ~Events() { if (start.context) rt.destroy(start.context); if (stop.context) rt.destroy(stop.context); }
};
struct Context {
    ggml_context * p = nullptr;
    explicit Context(size_t bytes) { p = ggml_init({bytes, nullptr, true}); require(p, "ggml context failed"); }
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
    ~Context() { ggml_free(p); }
};
struct Buffer {
    ggml_backend_buffer_t p = nullptr;
    ~Buffer() { ggml_backend_buffer_free(p); }
};
struct Backend {
    ggml_backend_t p = nullptr;
    Backend() { p = ggml_backend_cuda_init(0); require(p, "CUDA backend missing"); }
    ~Backend() { if (p) { ggml_backend_synchronize(p); ggml_backend_free(p); } }
};

struct Weights {
    int q, n, k, experts, copies;
    size_t bytes, row_bytes;
    uint64_t raw_fingerprint = UINT64_C(14695981039346656037);
    Context contexts[2];
    Buffer buffers[2];
    std::vector<ggml_tensor *> tensors[2];
    std::vector<uint8_t> raw;
    Weights(const json & w, int count, Runtime & rt) :
        q(w["qtype"]), n(w["n"]), k(w["k"]), experts(w["experts"]), copies(count),
        contexts{Context((count+8)*ggml_tensor_overhead()), Context((count+8)*ggml_tensor_overhead())} {
        Backend probe;
        auto dev = ggml_backend_get_device(probe.p);
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto native = ggml_backend_cuda_buffer_type(0);
        ggml_backend_buffer_type_t packed = nullptr;
        for (auto name : {"ggml_backend_dev_get_override_bufts", "ggml_backend_dev_get_extra_bufts"}) {
            auto get = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(reg, name));
            if (get) for (auto types = get(dev); types && *types; ++types) {
                if (strstr(ggml_backend_buft_name(*types), "KPACK")) {
                    require(!packed || packed == *types, "ambiguous K-pack buffer"); packed = *types;
                }
            }
        }
        require(packed && packed != native, "K-pack buffer is unavailable in this llama build");
        row_bytes = ggml_row_size(ggml_type(q), k);
        bytes = row_bytes * size_t(n) * experts;
        raw.resize(bytes);
        std::vector<float> values(size_t(k)*32), importance(k, 1.f);
        for (int e = 0; e < experts; ++e) for (int first = 0; first < n; first += 32) {
            int rows = std::min(32, n-first);
            for (size_t i = 0; i < size_t(rows)*k; ++i)
                values[i] = value((uint64_t(e)*n+first)*k+i, 710+q*1009);
            auto size = ggml_quantize_chunk(ggml_type(q), values.data(),
                raw.data()+(size_t(e)*n+first)*row_bytes, 0, rows, k, importance.data());
            require(size == size_t(rows)*row_bytes, "quantizer byte count differs");
        }
        for (uint8_t v : raw) { raw_fingerprint ^= v; raw_fingerprint *= UINT64_C(1099511628211); }
        size_t free_bytes=0, total=0;
        checked(rt.memory(&free_bytes, &total), "memory query");
        std::printf("LLAMA_AB_MEMORY %s\n", json({{"free_bytes",free_bytes},{"total_bytes",total},
                    {"weight_pair_bytes",uint64_t(bytes)*copies*2},{"copies",copies}}).dump().c_str());
        std::fflush(stdout);
        require(double(bytes)*copies*2 < free_bytes*.78,
            "cold pair exceeds free VRAM budget; do not shrink the cold ring or omit experts");
        for (int arm=0; arm<2; ++arm) {
            auto type = arm ? packed : native;
            for (int i=0; i<copies; ++i)
                tensors[arm].push_back(ggml_new_tensor_3d(contexts[arm].p, ggml_type(q), k, n, experts));
            buffers[arm].p = ggml_backend_alloc_ctx_tensors_from_buft(contexts[arm].p, type);
            require(buffers[arm].p, "weight allocation failed");
            ggml_backend_buffer_set_usage(buffers[arm].p, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto t : tensors[arm]) {
                ggml_set_name(t, "ab_weight_loading");
                ggml_backend_tensor_set(t, raw.data(), 0, bytes);
            }
            std::printf("LLAMA_AB_WEIGHTS arm=%d copies=%d bytes=%zu buft=%s\n", arm, copies, bytes,
                        ggml_backend_buft_name(type)); std::fflush(stdout);
        }
        ggml_backend_synchronize(probe.p);
    }
};

struct Point {
    Weights & weight;
    int m, topk, channels, copies;
    bool indexed;
    std::string id;
    std::unique_ptr<Context> ctx[2];
    Buffer buffers[2];
    ggml_tensor * inputs[2]{}, * ids[2]{};
    std::vector<ggml_tensor *> outputs[2];
    std::vector<ggml_cgraph *> graphs[2];
    // Backends are destroyed before graph buffers and weight handles.
    Backend backends[2];
    std::vector<float> input;
    std::vector<int32_t> owners;
    Point(Weights & weights, const json & row) : weight(weights) {
        const auto & w = row["workload"];
        m=w["m"]; topk=w["topk"]; channels=w["channels"]; copies=row["copies"];
        indexed=w["operation"]=="indexed"; id=row["id"];
        require(copies>=2 && copies<=weight.copies, "invalid ring count");
        owners=row["owners"].get<std::vector<int32_t>>();
        require(owners.size()==size_t(m)*topk, "routing extent differs");
        for (auto e : owners) require(e>=0 && e<weight.experts, "expert out of bounds");
        input.resize(size_t(m)*channels*weight.k);
        const size_t ctx_bytes=(copies*8+32)*ggml_tensor_overhead()+copies*ggml_graph_overhead_custom(8, false);
        for (int arm=0; arm<2; ++arm) {
            ctx[arm]=std::make_unique<Context>(ctx_bytes);
            auto c=ctx[arm]->p;
            inputs[arm]=ggml_new_tensor_3d(c, GGML_TYPE_F32, weight.k, indexed?channels:m, indexed?m:1);
            ggml_set_input(inputs[arm]);
            if (indexed) { ids[arm]=ggml_new_tensor_2d(c, GGML_TYPE_I32, topk, m); ggml_set_input(ids[arm]); }
            for (int i=0; i<copies; ++i) {
                auto wt=weight.tensors[arm][i];
                ggml_set_name(wt, ("ab_"+id.substr(0,20)+"_"+std::to_string(i)).c_str());
                auto out=indexed?ggml_mul_mat_id(c,wt,inputs[arm],ids[arm]):ggml_mul_mat(c,wt,inputs[arm]);
                require(ggml_backend_supports_op(backends[arm].p,out),
                    arm?"caller does not support candidate operation (dense-only MR cannot cover MoE)":"native operation unsupported");
                ggml_set_output(out); outputs[arm].push_back(out);
                auto graph=ggml_new_graph_custom(c,8,false); ggml_build_forward_expand(graph,out);
                graphs[arm].push_back(graph);
            }
            buffers[arm].p=ggml_backend_alloc_ctx_tensors(c,backends[arm].p);
            require(buffers[arm].p,"point allocation failed");
            if (indexed) ggml_backend_tensor_set(ids[arm],owners.data(),0,owners.size()*sizeof(int32_t));
        }
    }
    void launch(int arm, int index) {
        require(ggml_backend_graph_compute_async(backends[arm].p,graphs[arm][index])==GGML_STATUS_SUCCESS,
                "backend graph execution failed");
    }
    void set_input(int repeat) {
        for (size_t i=0;i<input.size();++i) input[i]=value(i,931+repeat*541);
        for (int arm=0;arm<2;++arm) ggml_backend_tensor_set(inputs[arm],input.data(),0,input.size()*4);
    }
    json check(int index) {
        std::vector<float> got[2];
        const size_t count=size_t(m)*topk*weight.n;
        for (int arm=0;arm<2;++arm) {
            got[arm].resize(count);
            ggml_backend_synchronize(backends[arm].p);
            ggml_backend_tensor_get(outputs[arm][index],got[arm].data(),0,count*4);
            for (size_t i=0;i<count;++i) if (!std::isfinite(got[arm][i])) {
                throw std::runtime_error("nonfinite/unwritten output arm="+std::to_string(arm)+
                    " copy="+std::to_string(index)+" first="+std::to_string(i));
            }
        }
        double diff=0,norm=0;
        for (size_t i=0;i<count;++i) { diff+=std::pow(double(got[0][i])-got[1][i],2); norm+=double(got[0][i])*got[0][i]; }
        double relative=std::sqrt(diff/std::max(norm,1e-30));
        if (!(relative<.02)) throw std::runtime_error("candidate/native relative L2 error="+
            std::to_string(relative)+" exceeds .02, copy="+std::to_string(index));
        const auto decode=ggml_get_type_traits(ggml_type(weight.q))->to_float;
        require(decode,"raw GGUF independent decoder unavailable");
        std::vector<float> b(weight.k);
        double maximum[2]{};
        for (int s=0;s<64;++s) {
            size_t pos=(uint64_t(mix(s+471))*count)>>32;
            size_t row=pos/weight.n, col=pos%weight.n;
            int expert=owners[row];
            size_t ar=(row/topk)*channels+(row%topk)%channels;
            decode(weight.raw.data()+(size_t(expert)*weight.n+col)*weight.row_bytes,b.data(),weight.k);
            double dot=0,denom=0;
            for (int k=0;k<weight.k;++k) { double x=double(input[ar*weight.k+k])*b[k];dot+=x;denom+=std::abs(x); }
            for (int arm=0;arm<2;++arm) maximum[arm]=std::max(maximum[arm],std::abs(got[arm][pos]-dot)/std::max(denom,1e-30));
        }
        if (!(maximum[0]<.005 && maximum[1]<.005)) throw std::runtime_error(
            "independent GGUF spot dot error native="+std::to_string(maximum[0])+
            " candidate="+std::to_string(maximum[1])+" exceeds .005 abs-dot normalization");
        return {{"pair_relative_l2",relative},{"raw_spot_normalized_error",{maximum[0],maximum[1]}},
                {"spot_cells",64},{"finite_cells_per_arm",count},{"copy",index}};
    }
    json sample(int arm, Runtime & rt, int traversals) {
        Events events(rt,ggml_backend_get_device(backends[arm].p));
        auto start=Clock::now();
        ggml_backend_event_record(&events.start,backends[arm].p);
        for (int t=0;t<traversals;++t) for (int i=0;i<copies;++i) launch(arm,i);
        ggml_backend_event_record(&events.stop,backends[arm].p);
        ggml_backend_event_synchronize(&events.stop);
        double wall=seconds(start)*1e6/(copies*traversals);
        float ms=0;checked(rt.elapsed(&ms,events.start.context,events.stop.context),"event interval");
        require(std::isfinite(ms) && ms>0,"invalid event interval");
        return {{"event_us",double(ms)*1000/(copies*traversals)},{"wall_us",wall}};
    }
};

int main(int argc, char ** argv) {
    try {
        require(argc==3,"usage: llama-ab-carrier REQUEST.json OUTPUT_DIR");
        require(getenv("GGML_CUDA_DISABLE_GRAPHS"),"CUDA Graph must be disabled for this eager comparison");
        json request;std::ifstream file(argv[1]);file>>request;
        Runtime rt(request["runtime_library"]);
        auto rows=request["points"];require(!rows.empty(),"empty family");
        int copies=0;for (const auto & p:rows) copies=std::max(copies,int(p["copies"]));
        auto started=Clock::now();Weights weights(rows[0]["workload"],copies,rt);
        const double setup=seconds(started);
        for (const auto & row:rows) {
            auto begin=Clock::now();
            std::printf("LLAMA_AB_START id=%s\n",row["id"].get<std::string>().c_str());std::fflush(stdout);
            Point point(weights,row);
            json proof=json::array();
            for (int repeat=0;repeat<2;++repeat) {
                point.set_input(repeat);
                for (int arm=0;arm<2;++arm) for (int copy=0;copy<point.copies;++copy) {
                    std::vector<float> poison(ggml_nelements(point.outputs[arm][copy]),NAN);
                    ggml_backend_tensor_set(point.outputs[arm][copy],poison.data(),0,poison.size()*4);
                    point.launch(arm,copy);
                }
                // All copies have distinct packed allocations; validate every copy.
                for (int copy=0;copy<point.copies;++copy) proof.push_back(point.check(copy));
            }
            for (int arm=0;arm<2;++arm) for (int warm=0;warm<3;++warm) {
                for (int copy=0;copy<point.copies;++copy) point.launch(arm,copy);
                ggml_backend_synchronize(point.backends[arm].p);
            }
            const double setup_point=seconds(begin);
            json rounds=json::array();
            for (int r=0;r<request["rounds"].get<int>();++r) {
                json blocks=json::array();
                for (int arm:{0,1,1,0}) {
                    json ev=json::array(),wall=json::array();
                    for (int s=0;s<request["samples"].get<int>();++s) {
                        auto v=point.sample(arm,rt,1);ev.push_back(v["event_us"]);wall.push_back(v["wall_us"]);
                    }
                    blocks.push_back({{"arm",arm},{"event_us",ev},{"wall_us",wall}});
                }
                rounds.push_back(blocks);
            }
            proof.push_back(point.check(point.copies-1));
            json result={{"id",row["id"]},{"workload",row["workload"]},{"correctness","PASS"},
                {"scope","LLAMA_EAGER_COMPLETE_OPERATOR_EVENTS_NOT_KERNEL_ONLY_OR_MODEL_TPOT"},
                {"proof",proof},{"rounds",rounds},{"ring",row["ring"]},{"setup_family_seconds",setup},
                {"setup_point_seconds",setup_point},{"point_seconds",seconds(begin)},
                {"cuda_graph",false},{"jit_and_warmup_excluded",true},{"native_dispatcher",true},
                {"authority",request["authority"]}};
            result["fixture"]={{"raw_bytes",weights.bytes},{"raw_fnv1a64",weights.raw_fingerprint},
                {"weight_seed",710+weights.q*1009},{"activation_seeds",{931,1472}},
                {"kind","SEEDED_GGML_QUANTIZER_NOT_MODEL_WEIGHTS"}};
            std::ifstream maps("/proc/self/maps");
            std::string line;
            std::vector<std::string> images;
            while (std::getline(maps,line)) {
                auto pos=line.find('/');
                if (pos!=std::string::npos && line.find(".so",pos)!=std::string::npos)
                    images.push_back(line.substr(pos));
            }
            std::sort(images.begin(),images.end());images.erase(std::unique(images.begin(),images.end()),images.end());
            result["loaded_images"]=images;
            publish(fs::path(argv[2])/(row["id"].get<std::string>()+".json"),result);
            std::printf("LLAMA_AB_MEASURED id=%s seconds=%.3f\n",row["id"].get<std::string>().c_str(),seconds(begin));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr,"LLAMA_AB_FAIL: %s\n",e.what());return 1;
    }
}
