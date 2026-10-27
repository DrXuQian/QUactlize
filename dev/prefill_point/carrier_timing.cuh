// Included inside a GENERATED copy of the original carrier's main. Existing
// fixture/oracle/negative controls stay identical; only its timers are replaced.
require(request.value("point_probe",false),"not a point probe");
const bool random_dense_probe=request.value("random_dense_probe",false);
if(random_dense_probe) {
    require((weights.q==8 || (weights.q>=10 && weights.q<=14)) && point.m>=9 && point.m<=8192,
            "random dense diagnostic requires Q2..Q6/Q8 M9..8192");
} else if(request.value("cohort_prefill_probe",false)) {
    require((weights.q==8 || (weights.q>=10 && weights.q<=14)) && point.m>=128 && point.m<=8192,
            "cohort prefill diagnostic requires Q2..Q6/Q8 M128..8192");
} else if(request.value("model_prefill_probe",false)) {
    require((weights.q==12 || weights.q==14) && point.m==2048,
            "model prefill diagnostic requires Q4/Q6 M2048");
} else {
    require(weights.q==12 && weights.n==1024 && weights.k==5120 && point.m==2048,
            "single-point probe workload changed");
}
int providers=0;
for(const auto & stage:inventory) if(stage["entry"]=="PROVIDER") {
    ++providers;
    const auto & a=stage["arguments"];
    require(stage["symbol"]=="cublasGemmEx" && a.size()==19 &&
        a[8]==2 && a[11]==2 && a[15]==2 && a[17]==68,
        "native provider is not Full-FP16 with FP32 accumulation");
}
// A random transition M may use native MMQ/GEMV without a cuBLAS provider.
// Its exact kernel/launch inventory is checked against Asys in the parser.
// Provider calls keep the original F16/F32 four-stage contract.
require((providers==1 && inventory.size()==4) ||
        (random_dense_probe && providers==0 && !inventory.empty()),"native path contract changed");
json point_timings=json::array();
if(request.value("trace_only",false)) {
    nc.check(nc.point_profile(nc.session));
    proofs.push_back(proof(point.copies-1));compare_native(original,download(point,point.copies-1));
}
for(int round=0;!request.value("trace_only",false) && round<rounds;++round) for(int pass=0;pass<2;++pass) {
    const int rotate=(pass+round)%2;
    for(int mode=0;mode<3;++mode) {
        for(int stage=0;stage<(mode==0?int(inventory.size()):1);++stage) {
            const int copies=rotate?point.copies:1;
            const int width=mode==1?int(inventory.size()):1;
            std::vector<double> us(std::max(samples,copies)*width);
            nc.check(nc.point_samples(nc.session,rotate,mode,stage,samples,us.data()));
            point_timings.push_back({{"round",round},{"rotate",rotate},{"mode",mode},
                {"stage",stage},{"width",width},{"samples_us",us}});
            if(mode==2) {
                const int last=(std::max(samples,copies)-1)%copies;
                proofs.push_back(proof(last));compare_native(original,download(point,last));
            }
        }
    }
}
