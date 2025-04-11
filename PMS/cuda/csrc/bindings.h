#ifndef PMS_CUDA_BINDINGS_H
#define PMS_CUDA_BINDINGS_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>
#include <random>

#define N_THREADS 256

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

namespace pms {

std::tuple<
    torch::Tensor, 
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
propagation_tensor(
    const int patch_size,
    const int num_iters,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const bool is_integer_disp,
    const bool is_fource_fpw,
    torch::Tensor img_left,
    torch::Tensor grad_left,
    torch::Tensor img_right,
    torch::Tensor grad_right
);

}

#endif // PMS_CUDA_BINDINGS_H