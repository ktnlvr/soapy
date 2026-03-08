#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

__host__ __device__ int W_idx(int l_max, int n_max, int n, int l, int b) {
    return b*(l_max+1)*n_max + l*n_max + n;
}

__host__ __device__ int E_idx(int n_max, int l, int b) {
    return l*n_max + b;
}

__host__ __device__ int xi_idx(int l_max, int l, int m, int k) {
    return l*(l_max+1)*(l_max+1) + m*(l_max+1) + k;
}

// TODO: pass in the thread count
__host__ __device__ int c_idx(int tid, int n_max, int l_max, int n, int l, int m) {
    return n*(l_max+1)*(l_max+1)*256 + l*(l_max+1)*256 + m*256 + tid;
}

__global__ void compute_c_nlm_kernel(
    int N_p, int n_max, int l_max,
    const float *W_nlb, const float *E_lb, const float *xi_lmk,
    const float *x_p, const float *y_p, const float *z_p,
    float *c_partial_real, float *c_partial_imag)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int p = tid; p < N_p; p += stride) {
        float xp = x_p[p];
        float yp = y_p[p];
        float zp = z_p[p];

        float R2 = xp*xp + yp*yp + zp*zp;
        float R = sqrtf(R2);

        float2 xy = make_float2(xp, yp);

        for (int n = 0; n < n_max; n++) {
            for (int l = 0; l <= l_max; l++) {
                for (int m = 0; m <= l; m++) {
                    float2 temp_sum = make_float2(0.0f, 0.0f);

                    float2 xy_m = make_float2(1.0f, 0.0f);
                    for (int i = 0; i < m; i++) {
                        float2 t = xy_m;
                        xy_m.x = t.x*xy.x - t.y*xy.y;
                        xy_m.y = t.x*xy.y + t.y*xy.x;
                    }

                    for (int b = 0; b < n_max; b++) {
                        float w = W_nlb[W_idx(l_max,n_max,n,l,b)];
                        float e = E_lb[E_idx(n_max, l,b)];
                        float exp_factor = expf(e * R2);

                        float2 sum_k = make_float2(0.0f, 0.0f);

                        for (int k = m; k <= l; k++) {
                            float xi = xi_lmk[xi_idx(l_max, l,m,k)];
                            float z_term = 1;
                            for (int i = 0; i < k - m; i++)
                                z_term *= zp;

                            float R_term = 1;
                            for (int i = 0; i < l - k; i++)
                                R_term *= R;

                            float factor = xi * z_term * R_term;
                            sum_k.x += factor;
                            sum_k.y += 0.0f;
                        }

                        float2 t;
                        t.x = xy_m.x*sum_k.x - xy_m.y*sum_k.y;
                        t.y = xy_m.x*sum_k.y + xy_m.y*sum_k.x;

                        temp_sum.x += w * exp_factor * t.x;
                        temp_sum.y += w * exp_factor * t.y;
                    }

                    int idx = c_idx(tid,n_max,l_max,n,l,m);
                    c_partial_real[idx] += temp_sum.x;
                    c_partial_imag[idx] += temp_sum.y;
                }
            }
        }
    }
}
