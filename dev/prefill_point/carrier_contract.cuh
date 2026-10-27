// Included after saving the actual native inventory, BEFORE correctness or
// timing. Precision mismatch is an infrastructure failure, not kernel error.
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
        "native provider is not Full-FP16 with FP32 accumulation; check point adapter and linked reference build");
}
// A random transition M may use native MMQ/GEMV without a cuBLAS provider.
// Preserve its actual body/arithmetic and check the inventory in the parser.
require((providers==1 && inventory.size()==4) ||
        (random_dense_probe && providers==0 && !inventory.empty()),"native path contract changed");
std::printf("NATIVE_POINT_PRECISION storage=%s accumulator=%s linked_backend=PENDING\n",
            providers ? "F16" : "LINKED_NATIVE_BODY", providers ? "F32" : "LINKED_NATIVE_BODY");
std::fflush(stdout);
