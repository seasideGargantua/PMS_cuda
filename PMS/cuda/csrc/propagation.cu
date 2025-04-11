#include "bindings.h"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "curand_kernel.h"

namespace pms {

namespace cg = cooperative_groups;

std::tuple<torch::Tensor, 
          torch::Tensor,
          torch::Tensor,
          torch::tensor>
propagation_tensor(
    const int patch_size,
    const int num_iters,
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
    const int dir = (num_iter_%2==0) ? 1 : -1
    
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

    curandState* devStates;
	cudaMalloc(&devStates, block_size * grid_size * sizeof(curandState));
	srand(time(0));

    setup_kernel<<<grid_size, block_size>>>(devStates, rand());

    propagation_kernel<<<grid_size, block_size>>>(
        dir,
        patch_size,
        width,
        height,
        alpha,
        tau_col,
        tau_grad,
        min_disp,
        max_disp,
        is_integer_disp,
        is_fource_fpw,
        devStates,
        (float3*)img_left.contiguous().data_ptr<float>(),
        (float2*)grad_left.contiguous().data_ptr<float>(),
        (float3*)img_right.contiguous().data_ptr<float>(),
        (float2*)grad_right.contiguous().data_ptr<float>(),
        (float3*)plane_left.contiguous().data_ptr<float>(),
        (float3*)plane_right.contiguous().data_ptr<float>(),
        (float3*)cost_left.contiguous().data_ptr<float>(),
        (float3*)cost_right.contiguous().data_ptr<float>()
    );

    return std::make_tuple(plane_left, plane_right, cost_left, cost_right);
}

__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //获取线程号0~blocks*THREAD_NUM-1  grid划分成1维，block划分为1维
	curand_init(seed, tid, 0, &state[tid]);// initialize the state
}

__global__ void propagation_kernel(
    const int dir,
    const int patch_size,
    const int width,
    const int height,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const bool is_integer_disp,
    const bool is_fource_fpw,
    curandState* globalState,
    const float3* img_left,
    const float2* grad_left,
    const float3* img_right,
    const float2* grad_right,
    float3* plane_left,
    float3* plane_right,
    float* cost_left,
    float* cost_right
) { 

	int tid = blockIdx.x *blockDim.x + threadIdx.x; //获取线程号0~blocks*THREAD_NUM-1  grid划分成1维，block划分为1维
	curandState localState = globalState[tid];

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
                    alpha,
                    tau_col,
                    tau_grad,
                    min_disp,
                    max_disp,
                    is_integer_disp,
                    is_fource_fpw,
                    localState,
                    img_left,
                    grad_left,
                    img_right,
                    grad_right,
                    plane_right,
                    cost_right,
                    plane_left,
                    cost_left);
			}

			view_propagation(
                            patch_size,
                            width,
                            height,
                            x,
                            y,
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

inline __device__ void spatial_propagation(
    const int dir,
    const int patch_size,
    const int width,
    const int height,
    const int x,
    const int y,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const float3* img_right,
    const float3* grad_right,
    const float3* img_left,
    const float2* grad_left,
    float3* plan_left,
    float* cost_left
) {
    
    float3& plane_p = plane_left[y * width + x];
    float& cost_p = cost_left[y * width + x];

	const int xd = x - dir;
	if (xd >= 0 && xd < width) {
		float3 plane_pd = plane_left[y * width + xd];
		if (plane_pd != plane_p) {
			const float cost_pd = compute_cost(patch_size,
                                                width,
                                                height,
                                                x,
                                                y,
                                                alpha,
                                                tau_col,
                                                tau_grad,
                                                min_disp,
                                                max_disp,
                                                p,
                                                img_right,
                                                grad_right,
                                                img_left,
                                                grad_left);
			if (cost_pd < cost_p) {
				plane_p = plane_pd;
				cost_p = cost_pd;
			}
		}
	}

	const int yd = y - dir;
	if (yd >= 0 && yd < height) {
		float3 plane_pd = plane_left[yd * width + x];
		if (plane_pd != plane_p) {
			const float cost_pd = compute_cost(patch_size,
                                                width,
                                                height,
                                                x,
                                                y,
                                                alpha,
                                                tau_col,
                                                tau_grad,
                                                min_disp,
                                                max_disp,
                                                p,
                                                img_right,
                                                grad_right,
                                                img_left,
                                                grad_left);
			if (cost_pd < cost_p) {
				plane_p = plane_pd;
				cost_p = cost_pd;
			}
		}
	}
}

inline __device__ void view_propagation(
    const int patch_size,
    const int width,
    const int height,
    const int x,
    const int y,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const float3* img_left,
    const float2* grad_left,
    const float3* img_right,
    const float2* grad_right,
    const float3* plane_left,
    float3* plane_right,
    float* cost_right
) {
    
    const int p = y * width + x;
	const float3 plane_p = plane_left[p];

	const float d_p = to_disparity(plane_p, x, y);

	const int xr = lround(x - d_p);
	if (xr < 0 || xr >= width) {
		return;
	}

	const int q = y * width + xr;
	float3& plane_q = plane_right[q];
	float& cost_q = cost_right[q];

	const float3 plane_p2q = to_another_view(plane_p);
	const float d_q = to_disparity(plane_p2q, xr, y);
	const auto cost = compute_cost(patch_size,
                                    width,
                                    height,
                                    x,
                                    y,
                                    alpha,
                                    tau_col,
                                    tau_grad,
                                    min_disp,
                                    max_disp,
                                    plane_p2q,
                                    img_right,
                                    grad_right,
                                    img_left,
                                    grad_left);
	if (cost < cost_q) {
		plane_q = plane_p2q;
		cost_q = cost;
	}
}

inline __device__ void plane_refine(
    const int patch_size,
    const int width,
    const int height,
    const int x,
    const int y,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const bool is_integer_disp,
    const bool is_fource_fpw,
    const curandState localState,
    const float3* img_left,
    const float2* grad_left,
    const float3* img_right,
    const float2* grad_right,
    const float3* plane_right,
    const float* cost_right,
    float3* plane_left,
    float* cost_left
) {

	float3& plane_p = plane_left[y * width + x];
	float& cost_p = cost_left[y * width + x];

	float d_p = to_disparity(plane_p, x, y);
	float3 norm_p = to_normal(plane_p);

	float disp_update = (max_disp - min_disp) / 2.0f;
	float norm_update = 1.0f;
	const float stop_thres = 0.1f;

	while (disp_update > stop_thres) {

		float disp_rd = curand_normal(localState) * disp_update;
		if (is_integer_disp) {
			disp_rd = static_cast<float>(round(disp_rd));
		}

		const float d_p_new = d_p + disp_rd;
		if (d_p_new < min_disp || d_p_new > max_disp) {
			disp_update /= 2;
			norm_update /= 2;
			continue;
		}

		float3 norm_rd;
		if (!is_fource_fpw) {
			norm_rd.x = curand_normal(localState) * norm_update;
			norm_rd.y = curand_normal(localState) * norm_update;
			float32 z = curand_normal(localState) * norm_update;
			while (z == 0.0f) {
				z = curand_normal(localState) * norm_update;
			}
			norm_rd.z = z;
		}
		else {
			norm_rd.x = 0.0f; norm_rd.y = 0.0f;	norm_rd.z = 0.0f;
		}

		float3 norm_p_new = norm_p + norm_rd;
		normalize(norm_p_new);

		float3 plane_new = init_from_norm(norm_p_new, d_p_new, x, y);

		if (plane_new != plane_p) {
			const float cost = compute_cost(patch_size,
                                            width,
                                            height,
                                            x,
                                            y,
                                            alpha,
                                            tau_col,
                                            tau_grad,
                                            min_disp,
                                            max_disp,
                                            plane_p2q,
                                            img_right,
                                            grad_right,
                                            img_left,
                                            grad_left);

			if (cost < cost_p) {
				plane_p = plane_new;
				cost_p = cost;
				d_p = d_p_new;
				norm_p = norm_p_new;
			}
		}

		disp_update /= 2.0f;
		norm_update /= 2.0f;
    }
}

} // namespace pms