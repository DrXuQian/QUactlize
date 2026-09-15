// Exercise the real caller's dlsym path, without a CUDA runtime or a model.
#include "quactlize-execution-lib.h"
#include "ggml-impl.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern "C" void ggml_abort(char const*, int, char const* format, ...) {
    va_list args;
    va_start(args,format);
    std::vfprintf(stderr,format,args);
    va_end(args);
    std::fputc('\n',stderr);
    std::exit(86);
}

extern "C" void ggml_log_internal(ggml_log_level, char const* format, ...) {
    va_list args;
    va_start(args,format);
    std::vfprintf(stderr,format,args);
    va_end(args);
}

int main() {
    auto api=ggml_quactlize_execution_library();
    return api && api->query_smallm && api->full_image ? 0 : 1;
}
