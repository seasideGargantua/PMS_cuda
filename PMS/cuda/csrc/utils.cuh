#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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
    float3 n(p.x, p.y, -1.0f);
    normalize(n);
    return n;
}

inline __device__ float3 init_from_norm(float3 n, float d, int x, int y){
    p.z = (n.x * x + n.y * y + n.z * d) / n.z;
    float3 p(-n.x / n.z, -n.y / n.z, (n.x * x + n.y * y + n.z * d) / n.z);
    return p;
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
    const float dc = std::min(abs(col_p.x - col_q.x) + abs(col_p.y - col_q.y) + abs(col_p.z - col_q.z), tau_col);


    const float3 grad_q = grad_right[y * width + static_cast<int>(xr)];
    const float dg = std::min(abs(grad_p.x - grad_q.x) + abs(grad_p.y - grad_q.y), tau_grad_);

    return (1 - alpha) * dc + alpha * dg;
}

inline __device__ float compute_cost(
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
            const float w = exp(-dc / gamma_);

            const float grad_q = grad_left[yr*width + xc];
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

} // namespace pms

