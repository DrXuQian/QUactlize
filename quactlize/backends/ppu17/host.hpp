#pragma once
#include "api.h"

namespace quactlize::ppu17 {
int error(int code, char const *message);
int validate_problem(q17_problem_v1 const *);
int validate_call(q17_call_v1 const *, bool expansion);
int validate_launch_options(q17_launch_options_v1 const *, bool compiled);
}
