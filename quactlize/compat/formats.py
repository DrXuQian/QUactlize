"""Native16 GGUF IDs and raw block extents; no GGUF/PyTorch/device dependency.

These declarations do not register types in a separately installed gguf reader.
Q2_0 follows DrXuQian/llama.cpp 356a231c12910c545bfff0f781ed3be779b1ba5a.
"""
GGML_TYPE_Q2_0 = 42
RAW_BLOCK_SIZES = {
    7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110),
    12: (256, 144), 13: (256, 176), 14: (256, 210), GGML_TYPE_Q2_0: (64, 18),
}
