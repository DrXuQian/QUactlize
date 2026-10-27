// CPU-only carrier for the caller's unchanged TP split function.
#include "llama-hparams.h"
#include "llama-arch.h"
#include "ggml-backend.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <stdexcept>

static std::map<int, int64_t> block_sizes;

extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    auto it = block_sizes.find(int(type));
    if (it == block_sizes.end() || it->second <= 0) {
        throw std::runtime_error("unknown storage block size for qtype " + std::to_string(type));
    }
    return it->second;
}

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    char message[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + message);
}

struct header_model {
    llama_hparams hparams{};
    llm_arch arch;
    std::map<std::string, ggml_tensor> weights;
    std::vector<float> splits;
    const float * tensor_split() const { return splits.data(); }
    const ggml_tensor * get_tensor(const char * name) const {
        auto it = weights.find(name);
        return it == weights.end() ? nullptr : &it->second;
    }
    const ggml_tensor * get_tensor_meta(const char * name) const { return get_tensor(name); }
};

struct llama_meta_device_get_split_state_userdata {
    const header_model * model;
    size_t n_devices;
    // Newer callers may consult loader metadata before model placement. This
    // carrier registers ALL header tensors above; no lazy loader is needed.
    const header_model * loader = nullptr;
};

// Generated from LLAMA_CI_DIR/src/llama-model.cpp, never a copied TP rule set.
#include "tp-selector.inc"

int main() {
    std::string active;
    try {
        std::string marker, arch;
        size_t world;
        header_model model;
        auto & h = model.hparams;
        std::cin >> marker >> arch >> world;
        GGML_ASSERT(marker == "QZ_TP_HEADER_V1" && world > 0 && world <= GGML_BACKEND_META_MAX_DEVICES);
        if (arch == "qwen35moe") model.arch = LLM_ARCH_QWEN35MOE;
        else if (arch == "deepseek4") model.arch = LLM_ARCH_DEEPSEEK4;
        else throw std::runtime_error("unsupported architecture");
        model.splits.assign(world, 1.0f);
        std::cin >> h.n_layer_all >> h.n_layer_nextn >> h.n_embd >> h.n_ff_exp
                 >> h.ssm_d_conv >> h.ssm_d_inner >> h.ssm_d_state >> h.ssm_dt_rank >> h.ssm_n_group
                 >> h.dsv4_o_group_count >> h.dsv4_o_lora_rank >> h.dsv4_hc_mult
                 >> h.n_embd_head_k_full >> h.n_embd_head_v_full >> h.n_embd_head_k_swa >> h.n_embd_head_v_swa;
        GGML_ASSERT(h.n_layer_all > 0 && h.n_layer_all <= LLAMA_MAX_LAYERS && h.n_layer_nextn < h.n_layer_all);
        for (size_t i = 0; i < h.n_layer_all; ++i) {
            int recurrent, swa;
            std::cin >> h.n_head_arr[i] >> h.n_head_kv_arr[i] >> h.n_ff_arr[i] >> recurrent >> swa;
            h.is_recr_impl[i] = recurrent;
            h.is_swa_impl[i] = swa;
        }
        size_t count;
        std::cin >> count;
        std::vector<std::string> targets;
        for (size_t i = 0; i < count; ++i) {
            std::string name;
            int type, target;
            int64_t block;
            ggml_tensor t{};
            std::cin >> std::quoted(name) >> type >> block >> target;
            for (auto & d : t.ne) std::cin >> d;
            GGML_ASSERT(name.size() < sizeof(t.name));
            std::memcpy(t.name, name.c_str(), name.size()+1);
            t.type = static_cast<ggml_type>(type);
            GGML_ASSERT(model.weights.emplace(name, t).second);
            auto inserted = block_sizes.emplace(type, block);
            GGML_ASSERT(inserted.second || inserted.first->second == block);
            if (target) targets.push_back(name);
        }
        GGML_ASSERT(std::cin.good());
        std::string extra;
        GGML_ASSERT(!(std::cin >> extra));
        llama_meta_device_get_split_state_userdata ud{&model, world};
        for (const auto & name : targets) {
            active = name;
            const auto * t = model.get_tensor(name.c_str());
            const auto s = llama_meta_device_get_split_state(t, &ud);
            GGML_ASSERT(s.n_segments > 0 && s.n_segments <= 16);
            GGML_ASSERT(s.axis == GGML_BACKEND_SPLIT_AXIS_0 || s.axis == GGML_BACKEND_SPLIT_AXIS_1 ||
                        s.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            std::cout << "{\"name\":" << std::quoted(name) << ",\"qtype\":" << int(t->type)
                      << ",\"axis\":" << int(s.axis) << ",\"segments\":[";
            for (size_t segment = 0; segment < s.n_segments; ++segment) {
                if (segment) std::cout << ',';
                std::cout << "{\"repeat\":" << s.nr[segment] << ",\"extents\":[";
                for (size_t rank = 0; rank < world; ++rank) {
                    if (rank) std::cout << ',';
                    std::cout << s.ne[segment*world+rank];
                }
                std::cout << "]}";
            }
            std::cout << "]}\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "TP_HEADER_FAIL tensor=" << active << " error=" << error.what() << '\n';
        return 1;
    }
}
