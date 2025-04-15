#include "bindings.h"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace pms {

namespace cg = cooperative_groups;

__global__ void initiate_cost(
    const int patch_size,
    const int width,
    const int height,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const float3* img_left,
    const float3* img_right,
    const float2* grad_left,
    const float2* grad_right,
    float3* plane_left,
    float3* plane_right,
    float* cost_left,
    float* cost_right
){
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const float3 plane_l = plane_left[y * width + x];
    cost_left[y * width + x] = compute_cost(patch_size,
                                            width,
                                            height,
                                            x,
                                            y,
                                            gamma,
                                            alpha,
                                            tau_col,
                                            tau_grad,
                                            min_disp,
                                            max_disp,
                                            plane_l,
                                            img_right,
                                            grad_right,
                                            img_left,
                                            grad_left);
    const float3 plane_r = plane_right[y * width + x];
    cost_right[y * width + x] = compute_cost(patch_size,
                                            width,
                                            height,
                                            x,
                                            y,
                                            gamma,
                                            alpha,
                                            tau_col,
                                            tau_grad,
                                            -max_disp,
                                            -min_disp,
                                            plane_r,
                                            img_left,
                                            grad_left,
                                            img_right,
                                            grad_right);
}

std::tuple<
    torch::Tensor, 
    torch::Tensor>
initiate_cost_tensor(
    const int patch_size,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    torch::Tensor img_left,
    torch::Tensor img_right,
    torch::Tensor grad_left,
    torch::Tensor grad_right,
    torch::Tensor plane_left,
    torch::Tensor plane_right
){
    const int width = img_left.size(2);
    const int height = img_left.size(1);

    torch::Tensor cost_left = torch::zeros({height, width}, img_left.options());
    torch::Tensor cost_right = torch::zeros({height, width}, img_left.options());

    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    initiate_cost<<<grid_size, block_size>>>(
        patch_size,
        width,
        height,
        gamma,
        alpha,
        tau_col,
        tau_grad,
        min_disp,
        max_disp,
        (float3*)img_left.contiguous().data_ptr<float>(),
        (float3*)img_right.contiguous().data_ptr<float>(),
        (float2*)grad_left.contiguous().data_ptr<float>(),
        (float2*)grad_right.contiguous().data_ptr<float>(),
        (float3*)plane_left.contiguous().data_ptr<float>(),
        (float3*)plane_right.contiguous().data_ptr<float>(),
        cost_left.contiguous().data_ptr<float>(),
        cost_right.contiguous().data_ptr<float>()
    );

    return std::make_tuple(cost_left, cost_right);
}

__global__ void propagation_kernel(
    const int dir,
    const int patch_size,
    const int width,
    const int height,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const bool is_integer_disp,
    const bool is_fource_fpw,
    const float3* img_left,
    const float3* img_right,
    const float2* grad_left,
    const float2* grad_right,
    float3* plane_left,
    float3* plane_right,
    float* cost_left,
    float* cost_right,
    float* rand_disp_,
    float3* rand_norm_
) { 

	int y = (dir == 1) ? 0 : height - 1;
	for (int i = 0; i < height; i++) {
		int x = (dir == 1) ? 0 : width - 1;
		for (int j = 0; j < width; j++) {
            spatial_propagation(dir,
                                patch_size,
                                width,
                                height,
                                x,
                                y,
                                gamma,
                                alpha,
                                tau_col,
                                tau_grad,
                                min_disp,
                                max_disp,
                                img_right,
                                grad_right,
                                img_left,
                                grad_left,
                                plane_left,
                                cost_left);

			if (!is_fource_fpw) {
				plane_refine(
                    patch_size,
                    width,
                    height,
                    x,
                    y,
                    gamma,
                    alpha,
                    tau_col,
                    tau_grad,
                    min_disp,
                    max_disp,
                    is_integer_disp,
                    is_fource_fpw,
                    img_left,
                    grad_left,
                    img_right,
                    grad_right,
                    plane_right,
                    cost_right,
                    plane_left,
                    cost_left,
                    rand_disp_,
                    rand_norm_);
			}

			view_propagation(
                            patch_size,
                            width,
                            height,
                            x,
                            y,
                            gamma,
                            alpha,
                            tau_col,
                            tau_grad,
                            min_disp,
                            max_disp,
                            img_left,
                            grad_left,
                            img_right,
                            grad_right,
                            plane_left,
                            plane_right,
                            cost_right);

			x += dir;
		}
		y += dir;
	}
}

std::tuple<
    torch::Tensor, 
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
propagation_tensor(
    const int num_iter,
    const int patch_size,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const bool is_integer_disp,
    const bool is_fource_fpw,
    torch::Tensor img_left,
    torch::Tensor img_right,
    torch::Tensor grad_left,
    torch::Tensor grad_right,
    torch::Tensor plane_left,
    torch::Tensor plane_right,
    torch::Tensor cost_left,
    torch::Tensor cost_right,
    torch::Tensor rand_disp,
    torch::Tensor rand_norm
){
    const int dir = (num_iter%2==0) ? 1 : -1;
    
    const int width = img_left.size(2);
    const int height = img_left.size(1);

    // Launch kernel
    dim3 block_size(1, 1);
    dim3 grid_size(1, 1);

    propagation_kernel<<<grid_size, block_size>>>(
        dir,
        patch_size,
        width,
        height,
        gamma,
        alpha,
        tau_col,
        tau_grad,
        min_disp,
        max_disp,
        is_integer_disp,
        is_fource_fpw,
        (float3*)img_left.contiguous().data_ptr<float>(),
        (float3*)img_right.contiguous().data_ptr<float>(),
        (float2*)grad_left.contiguous().data_ptr<float>(),
        (float2*)grad_right.contiguous().data_ptr<float>(),
        (float3*)plane_left.contiguous().data_ptr<float>(),
        (float3*)plane_right.contiguous().data_ptr<float>(),
        cost_left.contiguous().data_ptr<float>(),
        cost_right.contiguous().data_ptr<float>(),
        (float*)rand_disp.contiguous().data_ptr<float>(),
        (float3*)rand_norm.contiguous().data_ptr<float>()
    );

    return std::make_tuple(plane_left, plane_right, cost_left, cost_right);
}

} // namespace pms