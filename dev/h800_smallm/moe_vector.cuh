namespace quactlize::runtime {
// Pair stores reduce global-store issue count. The half-aligned guard case
// keeps scalar head/tail elements; no store crosses the caller's row extent.
CUTLASS_DEVICE void moe_m1_gather_vector(qk_moe_plan_v1 const& plan,int lane,int stride) {
  auto const& p=plan.gate;
  auto gate=static_cast<Half*>(p.a),up=static_cast<Half*>(plan.up.a);
  if ((p.k&1) || (!plan.merged && ((uintptr_t(gate)^uintptr_t(up))&3))) {
    moe_m1_gather(plan,lane,stride); return;
  }
  int const head=(uintptr_t(gate)&3)?1:0;
  int const pairs=(p.k-head)/2,tail=head+2*pairs;
  if (lane==0) {
    for (int r=0;r<8;++r) {
      if (head) {
        Half v=Half(p.io.a[0]);gate[int64_t(r)*p.k]=v;
        if (!plan.merged) up[int64_t(r)*p.k]=v;
      }
      if (tail<p.k) {
        Half v=Half(p.io.a[tail]);gate[int64_t(r)*p.k+tail]=v;
        if (!plan.merged) up[int64_t(r)*p.k+tail]=v;
      }
    }
  }
  for (int pair=lane;pair<pairs;pair+=stride) {
    int col=head+2*pair;
    auto value=__floats2half2_rn(p.io.a[col],p.io.a[col+1]);
    #pragma unroll
    for (int r=0;r<8;++r) {
      *reinterpret_cast<__half2*>(gate+int64_t(r)*p.k+col)=value;
      if (!plan.merged) *reinterpret_cast<__half2*>(up+int64_t(r)*p.k+col)=value;
    }
  }
}
} // namespace quactlize::runtime
