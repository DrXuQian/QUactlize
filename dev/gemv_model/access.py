"""Exact source-load footprints; distinguish them from emitted/DRAM traffic."""

from dev.gemv_simt.access import pattern
from dev.gemv_ppu.access_pattern import footprint
from quactlize.execution.simt_codegen import Config


def access(point, c, bases):
    cfg = Config(c.variant, c.columns, c.warps, c.values, c.split)
    result = pattern(point.q, cfg, point.physical_n, point.k, bases=bases)
    if c.direct_meta:
        streams = [s for s in result["streams"] if not s["name"].startswith("metadata-")]
        groups, cols = result["lane_groups"], result["lane_columns"]
        lanes = [l for l, g in enumerate(groups) if g < point.k // 16]
        for p in range(c.values):
            ptr = [bases["units"] + ((groups[l] // 32) * point.n + cols[l] + p) * 36
                   + ((groups[l] // 16) & 1) * 18 for l in lanes]
            for name, offsets, width in (("d", ptr, 2),
                    ("sc", [x + 2 + (groups[l] & 15) for x, l in zip(ptr, lanes)], 1)):
                streams.append(dict(name=f"metadata-direct-{name}-p{p}", lanes=lanes, addresses=offsets,
                                    width_bytes=width, footprint={str(s): footprint(offsets, width, s) for s in (32, 64, 128)}))
        result["streams"] = streams
    result.update(B_contiguous_bytes_per_k_worker_group=2*c.tile_n,
                  fast_code_dequant="LOP3/mantissa half2 extraction to exact integer codes; FP32 FMA",
                  metadata_reader="direct Q6 d+int8" if c.direct_meta else "H32" if point.q in (12,13) else "existing",
                  fixed_shape=c.fixed, paired_N4=point.paired,
                  hoisted_B_words_per_thread=8*c.values if point.q == 8 and c.hoist else None)
    result["aggregate_source_footprint"] = {str(s): {
        k: sum(stream["footprint"][str(s)][k] for stream in result["streams"])
        for k in ("lane_bytes", "unique_bytes", "sector_bytes")}
        for s in (32, 64, 128)}
    return result
