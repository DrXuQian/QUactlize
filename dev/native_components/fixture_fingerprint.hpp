// Host-only fixture checksum, same IEEE CRC32 as Python zlib.crc32.
#pragma once
#include <cstddef>
#include <cstdint>

inline uint32_t nc_fixture_crc32(const void * data, size_t size) {
    uint32_t table[256];
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t v = i;
        for (int bit = 0; bit < 8; ++bit) v = (v >> 1) ^ (0xedb88320u & (0u - (v & 1u)));
        table[i] = v;
    }
    auto bytes = static_cast<const unsigned char *>(data);
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ bytes[i]) & 255] ^ (crc >> 8);
    return crc ^ 0xffffffffu;
}
