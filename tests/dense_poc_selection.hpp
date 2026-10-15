#pragma once
// Frozen AUTO/dense control flow from llama.cpp quactlize abaccee355cc,
// ggml/src/ggml-cuda/quactlize-execution.cu. Uses the unchanged old APIs;
// not the new provider's select(). A MISS is resolved against the old dense
// shipping registry, not against the new general coverage parent.
struct PocDense {
    bool legacy=false,full=false,direct=false,q4=false,native_f16=false;
    int route=QK_DENSE_FQ,sf_config=-1;
    qks_smallm_choice_v1 small{};
    qkg_q4_decode_config_v1 q4_config{};
    qks_choice_v1 tc{};
    qks_prefill_choice_v1 prefill{};
};
inline PocDense poc_dense(void* runtime,quactlize_ppu_dense_params_v1 const& p) {
    assert(p.m>0);
    PocDense out;
    qkg_call_v1 c{1,sizeof(c)};c.qtype=p.qtype;c.n=p.n;c.k=p.k;c.experts=1;c.rows=p.m;
    c.mode=QKG_DENSE;c.input_type=QKG_F32;c.channels=c.topk=1;c.a_row_stride=p.k;c.out_row_stride=p.n;
    bool matched=false,table_tc=false,q4_tc=false;
    if (p.m<=8) {
        qkg_simt_call_v2 typed{2,sizeof(typed),c,QK_COMPUTE_F16};
        qks_smallm_choice_v2 selection{};
        int rc=quactlize_kpack_dispatch_query_smallm_v3(runtime,&typed,&p.arrangement,&selection);
        assert(rc==QKS_OK || rc==QKS_MISS);
        if (rc==QKS_OK) {
            matched=true;out.small=selection.base;
            out.direct=out.small.kind!=QKS_SMALLM_TC;out.q4=out.small.kind==QKS_SMALLM_Q4;
            out.q4_config=selection.q4;table_tc=!out.direct;
            if (table_tc) out.tc=out.small.tc;
        }
    }
    if (!matched) {
        if (p.qtype==12) {
            qkg_sizes_v1 sizes{};
            int rc=quactlize_kpack_q4_decode_select_v1(&c,&p.arrangement,&out.q4_config,&sizes);
            assert(rc==QKG_OK || rc==QKG_SHAPE);
            out.direct=out.q4=rc==QKG_OK;
        } else {
            int rc=quactlize_kpack_dispatch_query_smallm_v1(runtime,&c,&p.arrangement,&out.small);
            assert(rc==QKS_OK || rc==QKS_MISS);
            if (rc==QKS_OK) {
                out.direct=out.small.kind==QKS_SMALLM_SIMT;table_tc=!out.direct;
                if (table_tc) out.tc=out.small.tc;
            }
        }
    }
    if (out.direct) return out;
    qks_request_v1 r{1,sizeof(r),p.qtype,QK_DENSE_FQ,p.m,p.n,p.k,1,p.m,p.arrangement.mapping_id};
    auto query=[&](bool decode=false) {
        if (p.m<=8) return quactlize_kpack_dispatch_query_dense_io_v1(runtime,&r,QKD_F32,decode,&out.tc);
        return decode ? quactlize_kpack_dispatch_query_decode_v1(runtime,&r,&out.tc) :
            quactlize_kpack_dispatch_query_v1(runtime,&r,&out.tc);
    };
    int status=table_tc?QKS_OK:QKS_MISS;
    if (!table_tc && p.qtype==12) {
        status=query(true);assert(status==QKS_OK || status==QKS_MISS);q4_tc=status==QKS_OK;
    }
    int prefill=-1;
    if (!table_tc && !q4_tc && p.m>1 && p.qtype!=8) {
        int rc=quactlize_kpack_dispatch_prefill_v1(&r,7,&out.prefill);
        assert(rc==QKS_OK || rc==QKS_MISS);
        if (rc==QKS_OK) prefill=out.prefill.route;
    }
    if (prefill==2) {out.full=true;return out;}
    if (prefill==1) out.sf_config=out.prefill.dequant_config;
    bool sf=false;
    if (p.qtype==8) {
        r.route=QK_DENSE_SF;if (!table_tc) status=query();assert(status==QKS_OK);sf=true;
    } else if (prefill==1 || (table_tc && !std::strncmp(out.tc.parent,"sf",2))) {
        r.route=QK_DENSE_SF;if (!table_tc) status=query();sf=status==QKS_OK;
    }
    if (!sf && !q4_tc && !table_tc) {r.route=QK_DENSE_FQ;status=query();}
    assert(status==QKS_OK || status==QKS_MISS);
    out.legacy=status==QKS_MISS;out.route=sf?QK_DENSE_SF:QK_DENSE_FQ;out.native_f16=p.m>8;
    return out;
}
inline void compare_poc(PocDense const& old,quactlize::dense::Plan const& now) {
    using K=quactlize::dense::Kind;
    if (old.legacy) {
        auto const& p=now.params;
        assert(now.kind==K::Tc && now.request.route==QK_DENSE_FQ && now.info.compute_type==0);
        assert(now.tc.policy==QKS_LEGACY_DENSE && now.native_f16 && now.legacy_adapters);
        assert(now.copy_input && now.copy_output && !now.plane_bytes);
        int tm,tn,tk,wm,wn,st,split=1;
        if (p.qtype==12) {
            auto const& row=ppu_q4_kpack4_shipping::row(ppu_q4_kpack4_shipping::default_config(p.m,p.n,p.k));
            tm=row.tile_m;tn=row.tile_n;tk=row.tile_k;wm=row.warp_m;wn=row.warp_n;st=row.stages;split=row.split;
        } else {
            ppu_dense_shipping::ConfigId id{};
            assert(ppu_kquant_measured_policy::find_dense_config(nullptr,p.qtype,p.m,p.n,p.k,id));
            auto const& row=ppu_dense_shipping::kConfigs[static_cast<int>(id)];
            tm=row.tile_m;tn=row.tile_n;tk=ppu_formats::for_qtype(p.qtype).fully_quantized_tile_k;
            wm=row.warp_m;wn=row.warp_n;st=row.stages;
        }
        char parent[192];
        std::snprintf(parent,sizeof(parent),"fqk_tc_q%d_l%d_a0_tm%d_tn%d_tk%d_wm%d_wn%d_s%d_bc0_ap0_dn0",
            p.qtype,p.qtype==12?1:2,tm,tn,tk,wm,wn,st);
        assert(!std::strcmp(parent,now.tc.parent));
        assert(now.tc.algorithm==QK_ORDINARY && now.tc.grid==0 && now.tc.split==split);
        assert(now.metadata_type==QK_METADATA_F16);
    } else if (old.direct) {
        assert(now.kind==(old.q4?K::Q4:K::Simt));
        if (old.q4) assert(!std::memcmp(&old.q4_config,&now.small.q4,sizeof(old.q4_config)));
        else assert(!std::memcmp(&old.small.simt,&now.small.base.simt,sizeof(old.small.simt)));
        assert(!now.copy_input && !now.copy_output && now.info.compute_type==0);
    } else if (old.full) {
        assert(now.kind==K::Full && now.prefill.dequant_config==old.prefill.dequant_config);
        assert(now.info.route==QUACTLIZE_DENSE_FULL_BF16 && now.info.compute_type==1);
    } else {
        assert(now.kind==K::Tc && now.request.route==old.route && now.info.compute_type==0);
        assert(std::string(now.tc.parent)==old.tc.parent && std::string(now.tc.build_key)==old.tc.build_key);
        assert(now.tc.algorithm==old.tc.algorithm && now.tc.split==old.tc.split && now.tc.grid==old.tc.grid);
        assert(now.tc.policy==old.tc.policy && now.tc.workspace_bytes==old.tc.workspace_bytes);
        assert(now.native_f16==old.native_f16 && now.metadata_type==QK_METADATA_F16);
        auto selected=quactlize::dispatch::select_smallm(quactlize::dense::simt_call(now),now.params.arrangement);
        if(selected.dequant_config>=0) {
            // New independently confirmed SF helper, not a change to the
            // frozen PoC's default prepass. Every other contract stays equal.
            assert(old.sf_config==-1 && now.prefill.version && now.prefill.route==1);
            assert(now.prefill.dequant_config==selected.dequant_config);
        } else assert((now.prefill.version && now.prefill.route==1 ? now.prefill.dequant_config : -1)==old.sf_config);
    }
}
