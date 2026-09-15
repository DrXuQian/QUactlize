#include "simt_validation.hpp"

#define QKG_SIMT_DECLARE(Q) \
    extern "C" int qkg_simt_launch_##Q(qkg_call_v1 const*, qkg_simt_config_v1 const*); \
    extern "C" bool qkg_simt_supported_##Q(qkg_simt_config_v1 const*);
QKG_SIMT_DECLARE(8)
QKG_SIMT_DECLARE(10)
QKG_SIMT_DECLARE(11)
QKG_SIMT_DECLARE(12)
QKG_SIMT_DECLARE(13)
QKG_SIMT_DECLARE(14)
#undef QKG_SIMT_DECLARE

extern "C" int quactlize_kpack_simt_query_v1(qkg_call_v1 const* c,
    qkg_simt_config_v1 const* f, quactlize_ppu_placed_arrangement_v2 const* a,
    qkg_sizes_v1* out) {
    if (!c || !f || !a || !out) return QKG_INVALID;
    *out={};
    int rc=quactlize::execution::simt::query(*c,*f,a,*out);
    if (rc) return rc;
#define QKG_SIMT_SUPPORTED(Q) case Q: return qkg_simt_supported_##Q(f) ? QKG_OK : QKG_INVALID;
    switch (c->qtype) {
        QKG_SIMT_SUPPORTED(8)
        QKG_SIMT_SUPPORTED(10)
        QKG_SIMT_SUPPORTED(11)
        QKG_SIMT_SUPPORTED(12)
        QKG_SIMT_SUPPORTED(13)
        QKG_SIMT_SUPPORTED(14)
        default: return QKG_FORMAT;
    }
#undef QKG_SIMT_SUPPORTED
}

extern "C" int quactlize_kpack_simt_run_v1(qkg_call_v1 const* c,
    qkg_simt_config_v1 const* f, quactlize_ppu_placed_arrangement_v2 const* a) {
    qkg_sizes_v1 sizes{};
    int rc=quactlize_kpack_simt_query_v1(c,f,a,&sizes);
    if (rc) return rc;
    rc=quactlize::execution::simt::buffers(*c,sizes);
    if (rc) return rc;
#define QKG_SIMT_CASE(Q) case Q: return qkg_simt_launch_##Q(c,f);
    switch (c->qtype) {
        QKG_SIMT_CASE(8)
        QKG_SIMT_CASE(10)
        QKG_SIMT_CASE(11)
        QKG_SIMT_CASE(12)
        QKG_SIMT_CASE(13)
        QKG_SIMT_CASE(14)
        default: return QKG_FORMAT;
    }
#undef QKG_SIMT_CASE
}
