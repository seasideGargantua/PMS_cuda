#include "bindings.h"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace pms {

namespace cg = cooperative_groups;

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
    const float2* grad_left,
    const float3* img_right,
    const float2* grad_right,
    float3* plane_left,
    float3* plane_right,
    float* cost_left,
    float* cost_right,
    float* rand_disp_,
    float* rand_norm_
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
    const int patch_size,
    const int num_iter,
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
){
    const int dir = (num_iter%2==0) ? 1 : -1;
    
    const int width = img_left.size(2);
    const int height = img_left.size(1);

    torch::Tensor plane_left = torch::zeros({height, width, 3}, img_left.options());
    torch::Tensor plane_right = torch::zeros({height, width, 3}, img_left.options());
    torch::Tensor cost_left = torch::zeros({height, width}, img_left.options());
    torch::Tensor cost_right = torch::zeros({height, width}, img_left.options());

    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    // 在主机代码中生成随机数
    std::vector<float> random_disparities(width * height);
    std::vector<float> random_normals(width * height);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_disp(min_disp, max_disp);
    std::uniform_real_distribution<float> rand_norm(-1.0f, 1.0f);

    for (int i = 0; i < width * height; i++) {
        random_disparities[i] = rand_disp(gen);
        random_normals[i] = rand_norm(gen);
    }

    // 将随机数传递到设备
    float* d_random_disparities;
    float* d_random_normals;
    cudaMalloc(&d_random_disparities, random_disparities.size() * sizeof(float));
    cudaMalloc(&d_random_normals, random_normals.size() * sizeof(float));
    cudaMemcpy(d_random_disparities, random_disparities.data(), random_disparities.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_random_normals, random_normals.data(), random_normals.size() * sizeof(float), cudaMemcpyHostToDevice);

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
        (float2*)grad_left.contiguous().data_ptr<float>(),
        (float3*)img_right.contiguous().data_ptr<float>(),
        (float2*)grad_right.contiguous().data_ptr<float>(),
        (float3*)plane_left.contiguous().data_ptr<float>(),
        (float3*)plane_right.contiguous().data_ptr<float>(),
        cost_left.contiguous().data_ptr<float>(),
        cost_right.contiguous().data_ptr<float>(),
        d_random_disparities,
        d_random_normals
    );

    return std::make_tuple(plane_left, plane_right, cost_left, cost_right);
}

} // namespace pms