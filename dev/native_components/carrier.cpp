// Native-only fixture, linked against the caller's existing backend. Sharing
// fixture code does not require any K-pack buffer/provider or caller branch.
#define main historical_ab_main
#include "../llama_ab/carrier.cpp"
#undef main
#include <cxxabi.h>

struct NativeComponents {
    const char * (*error)();
    void * (*create)(int);
    void (*destroy)(void *);
    int (*record)(void *, ggml_tensor *);
    int (*replay)(void *, int, int);
    int (*seed)(void *, int);
    const char * (*inventory)(void *);
    int (*measure)(void *, int, int, int, int, double *);
    int (*route)(void *, int, int, int, int, const int *);
    int (*reset_samples)(void *, int, double *);
    int (*zero_weight)(void *, void *, uint64_t);
    void * session = nullptr;
    explicit NativeComponents(const std::string & path) {
        void * lib=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!lib) throw std::runtime_error(dlerror());
#define NC_BIND(name) name=symbol<decltype(name)>(lib,"nc_" #name)
        NC_BIND(error);NC_BIND(create);NC_BIND(destroy);NC_BIND(record);
        NC_BIND(replay);NC_BIND(seed);NC_BIND(inventory);NC_BIND(measure);
        NC_BIND(route);NC_BIND(reset_samples);
        NC_BIND(zero_weight);
#undef NC_BIND
        session=create(0);require(session,error());
    }
    void check(int rc) { if(rc) throw std::runtime_error(error()); }
    ~NativeComponents() { if(session) destroy(session); }
};

static std::string demangle(const std::string & symbol) {
    int status=0;char * value=abi::__cxa_demangle(symbol.c_str(),nullptr,nullptr,&status);
    std::string result=status?symbol:value;std::free(value);return result;
}
static std::string native_role(const json & stage) {
    if(stage["entry"]=="PROVIDER") return "provider";
    std::string name=demangle(stage["symbol"]);
    const std::string anon="(anonymous namespace)";
    for(size_t pos=name.find(anon);pos!=std::string::npos;pos=name.find(anon)) name.replace(pos,anon.size(),"__anonymous");
    // Classify the outer entry, not types nested in its arguments/templates.
    auto pos=name.find('<');auto args=name.find('(');
    if(args!=std::string::npos && (pos==std::string::npos||args<pos)) pos=args;
    std::string head=name.substr(0,pos);
    if(head.find("fixup")!=std::string::npos || head.find("reduce")!=std::string::npos) return "reducer";
    if(head.find("mul_mat")!=std::string::npos || head.find("gemm")!=std::string::npos ||
       head.find("gemv")!=std::string::npos) return "producer";
    if(head.find("dequantize")!=std::string::npos) return "expansion";
    if(head.find("scatter")!=std::string::npos) return "postprocess";
    if(head.find("gather")!=std::string::npos || head.find("quantize")!=std::string::npos ||
       head.find("convert")!=std::string::npos || head.find("get_rows")!=std::string::npos) return "adapter";
    if(head.find("mm_ids")!=std::string::npos || head.find("bounds_to")!=std::string::npos ||
       head.find("computeBlockInfo")!=std::string::npos || head.find("k_compute_batched_ptrs")!=std::string::npos)
        return "prepare";
    return "unclassified";
}
static void poison(Point & point, int copy) {
    std::vector<float> values(ggml_nelements(point.outputs[0][copy]),NAN);
    ggml_backend_tensor_set(point.outputs[0][copy],values.data(),0,values.size()*4);
}
static std::vector<float> download(Point & point, int copy) {
    std::vector<float> result(ggml_nelements(point.outputs[0][copy]));
    ggml_backend_tensor_get(point.outputs[0][copy],result.data(),0,result.size()*4);return result;
}
static void compare_native(const std::vector<float> & a, const std::vector<float> & b) {
    double diff=0,norm=0;
    require(a.size()==b.size(),"native output extent differs");
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"native adapter produced nonfinite output");
        diff+=std::pow(double(a[i])-b[i],2);norm+=double(a[i])*a[i];
    }
    require(std::sqrt(diff/std::max(norm,1e-30))<1e-5,"extracted native call differs from linked backend");
}
static int helpers(const json & request,const fs::path & output) {
    Backend backend; // initialize the native device/launch policy once
    const int rounds=request.value("rounds",3),samples=request.value("samples",3);
    for(const auto & row:request["helper_points"]) {
        NativeComponents nc(request["adapter_library"]);
        std::vector<double> values(rounds*samples);json inventory;
        const bool routing=row["kind"]=="routing";
        if(routing) {
            auto ids=row["owners"].get<std::vector<int>>();const auto & w=row["workload"];
            require(ids.size()==size_t(int(w["m"]))*int(w["topk"]),"routing extent differs");
            nc.check(nc.route(nc.session,w["m"],w["experts"],w["topk"],w["channels"],ids.data()));
            inventory=json::parse(nc.inventory(nc.session));
            nc.check(nc.measure(nc.session,0,0,rounds,samples,values.data()));
        } else {
            require(row["kind"]=="reset","unknown helper");
            nc.check(nc.reset_samples(nc.session,int(values.size()),values.data()));
        }
        json times=json::array();
        for(int r=0;r<rounds;++r) {
            json t=json::array();for(int s=0;s<samples;++s)t.push_back(values[r*samples+s]);times.push_back(t);
        }
        json component={{"role",routing?"routing":"reset"},{"samples_us",times},{"correctness","PASS"},
            {"first_use_excluded",true},{"host_launch_cost_added",false},
            {"timed_kernel_types",routing?json(1):json(nullptr)},
            {"timing",routing?"ISOLATED_KERNEL_GRAPH_EVENTS":"ORDINARY_MEMORY_EVENTS"},
            {"cuda_graph",routing},{"internal_reduction_included",false},
            {"cache","REPEATED_RESIDENT_SCRATCH"},{"identity",request["authority"]}};
        publish(output/(row["id"].get<std::string>()+".json"),json{
            {"id",row["id"]},{"component",component},{"inventory",inventory},{"contract",row},
            {"accuracy",routing?"FULL_PERMUTATION_COUNTS_SOURCES_TWO_ROUTINGS":"POISONED_STATUS_CLEARED"},
            {"authority",request["authority"]},{"production_changed",false}});
        std::printf("COMPONENT_HELPER_DONE kind=%s id=%s\n",row["kind"].get<std::string>().c_str(),row["id"].get<std::string>().c_str());
        std::fflush(stdout);
    }
    return 0;
}
int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: native-components-carrier REQUEST.json OUTPUT_DIR");
        require(getenv("GGML_CUDA_DISABLE_GRAPHS"),"disable caller graph capture, not per-kernel isolated timers");
        json request;std::ifstream file(argv[1]);file>>request;
        if(request.contains("helper_points"))return helpers(request,argv[2]);
        Runtime rt(request["runtime_library"]);
        const auto & rows=request["points"];require(!rows.empty(),"empty native family");
        int copies=0;for(const auto & row:rows) copies=std::max(copies,int(row["copies"]));
        Weights weights(rows[0]["workload"],copies,rt,1,request.at("raw_fixture").get<std::string>());
        for(const auto & row:rows) {
            auto start=Clock::now();
            std::printf("NATIVE_COMPONENT_START point=%s\n",row["id"].get<std::string>().c_str());std::fflush(stdout);
            Point point(weights,row);
            json proofs=json::array();
            point.set_input(0);point.launch(0,0);proofs.push_back(point.check(0));
            auto original=download(point,0);
            NativeComponents nc(request["adapter_library"]);
            for(int copy=0;copy<point.copies;++copy) nc.check(nc.record(nc.session,point.outputs[0][copy]));
            json inventory=json::parse(nc.inventory(nc.session));
            int first_main=-1;
            for(size_t i=0;i<inventory.size();++i) {
                auto & stage=inventory[i];stage["name"]=demangle(stage["symbol"]);
                require(stage["name"].get<std::string>().find("quactlize")==std::string::npos,
                        "native arm executed a Quactlize kernel");
                stage["role"]=native_role(stage);
                if(stage["role"]=="producer"||stage["role"]=="provider") if(first_main<0) first_main=int(i);
            }
            publish(fs::path(argv[2])/(point.id+".inventory.json"),inventory);
            require(first_main>=0,"native producer not identified; inventory retained");
            for(auto & stage:inventory) require(stage["role"]!="unclassified","unclassified native component; no timing admitted");
            for(int copy=0;copy<point.copies;++copy) {
                poison(point,copy);nc.check(nc.replay(nc.session,copy,-1));
                compare_native(original,download(point,copy));proofs.push_back(point.check(copy));
            }
            const int last=point.copies-1;
            nc.check(nc.zero_weight(nc.session,weights.tensors[0][last]->data,weights.bytes));
            nc.check(nc.replay(nc.session,last,-1));
            bool weight_failed=false;
            try {point.check(last);compare_native(original,download(point,last));}
            catch(const std::exception &) {weight_failed=true;}
            require(weight_failed,"last weight copy was not consumed");
            nc.check(nc.replay(nc.session,0,-1));proofs.push_back(point.check(0));
            ggml_backend_tensor_set(weights.tensors[0][last],weights.raw.data(),0,weights.bytes);
            nc.check(nc.replay(nc.session,last,-1));proofs.push_back(point.check(last));
            point.set_input(1);point.launch(0,0);proofs.push_back(point.check(0));
            auto expected_changed=download(point,0);poison(point,0);
            nc.check(nc.replay(nc.session,0,first_main));
            bool failed=false;
            try { point.check(0);compare_native(expected_changed,download(point,0)); }
            catch(const std::exception &) {failed=true;}
            require(failed,"omitting the main stage was not detected");
            for(int copy=0;copy<point.copies;++copy) {
                poison(point,copy);nc.check(nc.replay(nc.session,copy,-1));proofs.push_back(point.check(copy));
            }
            // Original library vs extracted call with changed activation as well.
            auto changed=download(point,0);point.launch(0,0);point.check(0);
            compare_native(changed,download(point,0));
            point.set_input(0);
            const int rounds=request.value("rounds",3),samples=request.value("samples",3);
            json components=json::array();
            for(size_t i=0;i<inventory.size();++i) {
                auto stage=inventory[i];std::string role=stage["role"];
                bool rotate=role=="producer"||role=="provider"||role=="expansion";
                nc.check(nc.seed(nc.session,int(i)));
                int width=samples*(role=="provider"?point.copies:1);
                std::vector<double> values(rounds*width);
                nc.check(nc.measure(nc.session,int(i),rotate,rounds,samples,values.data()));
                json times=json::array();
                for(int r=0;r<rounds;++r) {
                    json t=json::array();for(int s=0;s<width;++s)t.push_back(values[r*width+s]);times.push_back(t);
                }
                stage.update({{"samples_us",times},{"correctness","PASS"},{"first_use_excluded",true},
                    {"host_launch_cost_added",false},{"timed_kernel_types",role=="provider"?json(nullptr):json(1)},
                    {"timing",role=="provider"?"ORDINARY_PROVIDER_EVENTS":"ISOLATED_KERNEL_GRAPH_EVENTS"},
                    {"cuda_graph",role!="provider"},{"internal_reduction_included",role=="provider"},
                    {"cache",rotate?"ROTATING_ACTIVE_READ_SET_AT_LEAST_2P25_L2":"REPEATED_RESIDENT_SCRATCH"},
                    {"identity",request["authority"]},{"stage",i}});
                components.push_back(stage);
            }
            for(int copy=0;copy<point.copies;++copy) {
                poison(point,copy);nc.check(nc.replay(nc.session,copy,-1));proofs.push_back(point.check(copy));
            }
            json result={{"id",point.id},{"workload",row["workload"]},{"components",components},
                {"correctness","PASS"},{"proof",proofs},{"linked_backend_comparison","PASS_TWO_INPUTS"},
                {"omitted_producer","DETECTED"},{"ring",row["ring"]},{"authority",request["authority"]},
                {"last_weight_copy_negative","DETECTED_COPY0_UNCHANGED"},
                {"wall_seconds",seconds(start)},{"production_changed",false},
                {"scope","NATIVE_INDEPENDENT_COMPONENTS_NOT_MODEL_TPOT"}};
            result["fixture"]=request["raw_fixture_receipt"];
            std::ifstream maps("/proc/self/maps");std::string line;std::vector<std::string> images;
            while(std::getline(maps,line)) {auto pos=line.find('/');
                if(pos!=std::string::npos && line.find(".so",pos)!=std::string::npos)images.push_back(line.substr(pos));}
            std::sort(images.begin(),images.end());images.erase(std::unique(images.begin(),images.end()),images.end());
            result["loaded_images"]=images;
            publish(fs::path(argv[2])/(point.id+".json"),result);
            std::printf("NATIVE_COMPONENT_DONE point=%s stages=%zu seconds=%.3f\n",point.id.c_str(),components.size(),seconds(start));
            std::fflush(stdout);
        }
        return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"NATIVE_COMPONENT_FAIL: %s\n",e.what());return 1;}
}
