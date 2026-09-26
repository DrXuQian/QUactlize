#pragma once
#include <functional>
#include "api.h"

struct q17_plan_v1 {
  int device;
  std::function<int()> enqueue;
};
