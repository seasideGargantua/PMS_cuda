#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <random>

#include <ATen/Dispatch.h>
#define COST_PUNISH 120.0f

namespace pms{

inline __device__ float to_disparity(float3 p, int x, int y){
    return p.x * float(x) + p.y * float(y) + p.z;
}

inline __device__ float3 to_another_view(float3 p){
    float denom = 1 / (p.x - 1.f);
	return { p.x * denom, p.y * denom, p.z * denom };
}

inline __device__ float3 normalize(float3& p){
    float norm = sqrt(p.x * p.x + p.y * p.y + p.z*p.z);
    if (norm > 0.0f) {
        p.x /= norm;
        p.y /= norm;
        p.z /= norm;
    }
}

inline __device__ float3 to_normal(float3 p){
    float3 n = {p.x, p.y, -1.0f};
    normalize(n);
    return n;
}

inline __device__ float3 init_from_norm(float3 n, float d, int x, int y){
    float3 p = {-n.x / n.z, -n.y / n.z, (n.x * x + n.y * y + n.z * d) / n.z};
    return p;
}

inline __device__ bool equal_plane(float3 p1, float3 p2){
    return (abs(p1.x - p2.x) < 0.01f && abs(p1.y - p2.y) < 0.01f && abs(p1.z - p2.z) < 0.01f);
}

/*****************************************************/
/*                    Compute cost                   */
/*****************************************************/

inline __device__ float compute(
    const int width,
    const int height,
    const int x, 
    const int y, 
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float3 col_p, 
    const float2 grad_p, 
    const float d,
    const float3* img_right,
    const float2* grad_right
) {
    const float xr = x - d;
    if (xr < 0.0f || xr >= static_cast<float>(width)) {
        return (1 - alpha) * tau_col + alpha * tau_grad;
    }

    const float3 col_q = img_right[y * width + static_cast<int>(xr)];
    const float dc = min(abs(col_p.x - col_q.x) + abs(col_p.y - col_q.y) + abs(col_p.z - col_q.z), tau_col);


    const float2 grad_q = grad_right[y * width + static_cast<int>(xr)];
    const float dg = min(abs(grad_p.x - grad_q.x) + abs(grad_p.y - grad_q.y), tau_grad);

    return (1 - alpha) * dc + alpha * dg;
}

inline __device__ float compute_cost(
    const int patch_size,
    const int width,
    const int height,
    const int x,
    const int y,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const float3 p,
    const float3* img_right,
    const float2* grad_right,
    const float3* img_left,
    const float2* grad_left
) {
    
    const int pat = patch_size / 2;
    const float3 col_p = img_left[y*width + x];
    float cost = 0.0f;
    for (int r = -pat; r <= pat; r++) {
        const int yr = y + r;
        for (int c = -pat; c <= pat; c++) {
            const int xc = x + c;
            if (yr < 0 || yr > height - 1 || xc < 0 || xc > width - 1) {
                continue;
            }
            const float d = to_disparity(p, xc, yr);
            if (d < min_disp || d > max_disp) {
                cost += COST_PUNISH;
                continue;
            }

            const float3& col_q = img_left[yr*width + xc];
            const float dc = abs(col_p.x - col_q.x) + abs(col_p.y - col_q.y) + abs(col_p.z - col_q.z);
            const float w = exp(-dc / gamma);

            const float2 grad_q = grad_left[yr*width + xc];
            cost += w * compute(width,
                                height,
                                x, 
                                y, 
                                alpha,
                                tau_col,
                                tau_grad,
                                col_q, 
                                grad_q, 
                                d,
                                img_right,
                                grad_right);
        }
    }
    return cost;
}

/*******************************************/
/*                Propagation              */
/*******************************************/

inline __device__ void spatial_propagation(
    const int dir,
    const int patch_size,
    const int width,
    const int height,
    const int x,
    const int y,
    const float gamma,
    const float alpha,
    const float tau_col,
    const float tau_grad,
    const float min_disp,
    const float max_disp,
    const float3* img_right,
    const float2* grad_right,
    const float3* img_left,
    const float2* grad_left,
    float3* plane_left,
    float* cost_left
) {
    
    float3& plane_p = plane_left[y * width + x];
    float& cost_p = cost_left[y * width + x];

	const int xd = x - dir;
	if (xd >= 0 && xd < width) {
		float3 plane_pd = plane_left[y * width + xd];
		if (!equal_plane(plane_pd, plane_p)) {
			const float cost_pd = compute_cost(patch_size,
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
                                                plane_pd,
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
		if (!equal_plane(plane_pd, plane_p)) {
			const float cost_pd = compute_cost(patch_size,
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
                                                plane_pd,
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
    const float gamma,
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
                                    gamma,
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
    const float3* plane_right,
    const float* cost_right,
    float3* plane_left,
    float* cost_left,
    float* rand_disp_,
    float* rand_norm_
) {

    float rand_d = rand_disp_[y*width+x];
	float rand_n = rand_norm_[y*width+x];

	float3& plane_p = plane_left[y * width + x];
	float& cost_p = cost_left[y * width + x];

	float d_p = to_disparity(plane_p, x, y);
	float3 norm_p = to_normal(plane_p);

	float disp_update = (max_disp - min_disp) / 2.0f;
	float norm_update = 1.0f;
	const float stop_thres = 0.1f;

	while (disp_update > stop_thres) {

		float disp_rd = rand_d * disp_update;
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
			norm_rd.x = rand_n * norm_update;
			norm_rd.y = rand_n * norm_update;
			float z = rand_n * norm_update;
			while (z == 0.0f) {
				z = rand_n * norm_update;
			}
			norm_rd.z = z;
		}
		else {
			norm_rd.x = 0.0f; norm_rd.y = 0.0f;	norm_rd.z = 0.0f;
		}

		float3 norm_p_new = {norm_p.x + norm_rd.x, 
                             norm_p.y + norm_rd.y, 
                             norm_p.z + norm_rd.z};
		normalize(norm_p_new);

		float3 plane_new = init_from_norm(norm_p_new, d_p_new, x, y);

		if (!equal_plane(plane_new, plane_p)) {
			const float cost = compute_cost(patch_size,
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
                                            plane_new,
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

