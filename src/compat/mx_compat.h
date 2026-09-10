// Tensor-backend selector: lets the rest of src/ use `mx::` unchanged on either backend.
#pragma once

#if defined(MJMLX_BACKEND_MLX)
#include <mlx/linalg.h>
#include <mlx/mlx.h>
namespace mx = mlx::core;
#elif defined(MJMLX_BACKEND_MKX)
#include "compat/mx_compat_mkx.h"
#else
#error "No MJMLX_TENSOR_BACKEND selected (define MJMLX_BACKEND_MLX or MJMLX_BACKEND_MKX)"
#endif
