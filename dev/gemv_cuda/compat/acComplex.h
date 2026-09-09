#pragma once
#include <cuComplex.h>
using acFloatComplex = cuFloatComplex;
using acDoubleComplex = cuDoubleComplex;
#define make_acFloatComplex make_cuFloatComplex
#define make_acDoubleComplex make_cuDoubleComplex
#define acCrealf cuCrealf
#define acCimagf cuCimagf
#define acCreal cuCreal
#define acCimag cuCimag
