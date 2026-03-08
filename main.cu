#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <math.h>
#include <stdlib.h>

#include "include/matrix.h"
#include "include/xyz.h"
#include "include/kernels.h"

static inline int index_xi_lmk(int l, int m, int k, int l_max) {
  return l * (l_max + 1) * (l_max + 1) + m * (l_max + 1) + k;
}

static inline int index_alpha_bl(int l, int b, int n_max) {
  return l * n_max + b;
}

static inline int index_beta_lnb(int l, int n, int b, int n_max) {
  return l * n_max * n_max + n * n_max + b;
}

float compute_xi_lmk(int l, int m, int k) {
  float num = tgamma((l + k - 1) / 2 + 1);
  float den =
      tgamma(k - m + 1) * tgamma(l - k + 1) * tgamma((l - k + 1) / 2 + 1);
  return num / den;
}

void precompute_xi_lmk(float *xi_lmk, int l_max) {
  for (int l = 0; l <= l_max; l++)
    for (int m = 0; m <= l; m++)
      for (int k = 0; k <= l; k++)
        xi_lmk[index_xi_lmk(l, m, k, l_max)] = compute_xi_lmk(l, m, k);
}

void get_basis_gto(float r_cut, int n_max, int l_max, float **alphas_full_out,
                   float **betas_full_out) {
  float threshold = 1e-3;

  int L = l_max + 1;

  float *alphas_full = (float *)malloc(L * n_max * sizeof(float));
  float *betas_full = (float *)malloc(L * n_max * n_max * sizeof(float));

  for (int l = 0; l <= l_max; l++) {
    float *a = (float *)malloc(n_max * sizeof(float));
    float *alphas = (float *)malloc(n_max * sizeof(float));

    for (int i = 0; i < n_max; i++)
      a[i] = 1.0 + (r_cut - 1.0) * i / (n_max - 1);

    for (int i = 0; i < n_max; i++)
      alphas[i] = -log(threshold / pow(a[i], l)) / (a[i] * a[i]);

    gsl_matrix *m = gsl_matrix_alloc(n_max, n_max);

    for (int i = 0; i < n_max; i++)
      for (int j = 0; j < n_max; j++)
        gsl_matrix_set(m, i, j, alphas[i] + alphas[j]);

    gsl_matrix *S = gsl_matrix_alloc(n_max, n_max);

    float pref = 0.5 * gsl_sf_gamma(l + 1.5);

    for (int i = 0; i < n_max; i++)
      for (int j = 0; j < n_max; j++) {
        float val = gsl_matrix_get(m, i, j);
        gsl_matrix_set(S, i, j, pref * pow(val, -(l + 1.5)));
      }

    gsl_matrix *S_inv = gsl_matrix_alloc(n_max, n_max);
    matrix_inverse(S, S_inv);

    gsl_matrix *betas = gsl_matrix_alloc(n_max, n_max);
    matrix_sqrt(S_inv, betas);

    for (int i = 0; i < n_max; i++)
      alphas_full[l * n_max + i] = alphas[i];

    for (int i = 0; i < n_max; i++)
      for (int j = 0; j < n_max; j++)
        betas_full[l * n_max * n_max + i * n_max + j] =
            gsl_matrix_get(betas, i, j);

    gsl_matrix_free(m);
    gsl_matrix_free(S);
    gsl_matrix_free(S_inv);
    gsl_matrix_free(betas);

    free(a);
    free(alphas);
  }

  *alphas_full_out = alphas_full;
  *betas_full_out = betas_full;
}

float *precompute_K_nlm(const float *alpha_bl, const float *beta_lnb,
                         int n_max, int l_max, float sigma) {
  int L = l_max + 1;

  float *K_nlm = (float *)malloc(n_max * L * L * sizeof(float));

  for (int l = 0; l <= l_max; l++) {
    float pow_2_l = pow(2.0, l);

    for (int m = 0; m <= l; m++) {

      float numerator = (2 * l + 1) * tgamma(l - m + 1);
      float denominator = 4 * M_PI * tgamma(l + m + 1);

      float lambda_lm = pow_2_l * sqrt(numerator / denominator);

      for (int n = 0; n < n_max; n++) {

        float sum_b = 0.0;

        for (int b = 0; b < n_max; b++) {

          float ab = alpha_bl[l * n_max + b];

          float bb = beta_lnb[l * n_max * n_max + n * n_max + b];

          float denom = pow(1.0 + 2.0 * ab * sigma * sigma, l + 1.5);

          sum_b += bb / denom;
        }

        K_nlm[n * L * L + l * L + m] =
            lambda_lm * pow(2 * M_PI * sigma * sigma, 1.5) * sum_b;
      }
    }
  }

  return K_nlm;
}

float *precompute_W_nlb(const float *alpha_bl, const float *beta_lnb,
                         int l_max, int n_max, float sigma) {
  int L = l_max + 1;
  float *W_nlb = (float *)malloc(n_max * L * n_max * sizeof(float));
  if (!W_nlb)
    return NULL;

  for (int l = 0; l <= l_max; l++) {
    for (int n = 0; n < n_max; n++) {
      for (int b = 0; b < n_max; b++) {

        float ab = alpha_bl[index_alpha_bl(l, b, n_max)];
        float denom = pow(1.0 + 2.0 * ab * sigma * sigma, 1.5);

        float bb = beta_lnb[index_beta_lnb(l, n, b, n_max)];

        W_nlb[n * L * n_max + l * n_max + b] = bb / denom;
      }
    }
  }

  return W_nlb;
}

float *precompute_E_lb(const float *alpha_bl, int l_max, int n_max,
                        float sigma) {
  int L = l_max + 1;
  float *E_lb = (float *)malloc(L * n_max * sizeof(float));
  if (!E_lb)
    return NULL;

  for (int l = 0; l <= l_max; l++) {
    for (int b = 0; b < n_max; b++) {
      float ab = alpha_bl[index_alpha_bl(l, b, n_max)];
      float denom = 1.0 + 2.0 * ab * sigma * sigma;
      E_lb[index_alpha_bl(l, b, n_max)] = -ab / denom;
    }
  }

  return E_lb;
}
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    // Initialise lazy CUDA context
    cudaFree(0);

    // ----------------------
    // Create two streams
    // ----------------------
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);

    int n_max = 2;
    int l_max = 3;
    int sigma = 1;

    int size = (l_max + 1) * (l_max + 1) * (l_max + 1);
    float *xi_lmk_cpu = (float *)malloc(size * sizeof(float));
    precompute_xi_lmk(xi_lmk_cpu, l_max);

    float *alpha_bl;
    float *beta_lnb;
    get_basis_gto(5.0, 6, 4, &alpha_bl, &beta_lnb);

    int N_p;
    float *x_out, *y_out, *z_out;
    int err = read_xyz_coords("random_hydrogens.xyz", &x_out, &y_out, &z_out, &N_p);
    if (err) { fprintf(stderr,"Error reading XYZ file: %d\n", err); return 1; }

    float *K = precompute_K_nlm(alpha_bl, beta_lnb, n_max, l_max, sigma);
    float *E = precompute_E_lb(alpha_bl, l_max, n_max, sigma);
    float *W = precompute_W_nlb(alpha_bl, beta_lnb, l_max, n_max, sigma);

    int L = l_max + 1;
    size_t size_alpha = L * n_max;        
    size_t size_beta  = L * n_max * n_max; 
    size_t size_K     = n_max * L * L;    
    size_t size_E     = L * n_max;        
    size_t size_W     = n_max * L * n_max;
    int size_xi       = L * L * L;
    size_t total_size = (size_alpha + size_beta + size_K + size_E + size_W + size_xi) * sizeof(float);

    float *d_mem;
    cudaMalloc(&d_mem, total_size);
    float *d_alpha = d_mem;
    float *d_beta  = d_alpha + size_alpha;
    float *d_K     = d_beta + size_beta;
    float *d_E     = d_K + size_K;
    float *d_W     = d_E + size_E;
    float *d_xi    = d_W + size_W;

    float *xyz_dev;
    cudaMalloc(&xyz_dev, 3 * N_p * sizeof(float));
    float *d_x_out = xyz_dev;
    float *d_y_out = xyz_dev + N_p;
    float *d_z_out = xyz_dev + 2 * N_p;

    cudaMemcpyAsync(d_alpha, alpha_bl, size_alpha*sizeof(float), cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_beta,  beta_lnb, size_beta*sizeof(float),  cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_K,     K,        size_K*sizeof(float),     cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_E,     E,        size_E*sizeof(float),     cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_W,     W,        size_W*sizeof(float),     cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_xi,    xi_lmk_cpu, size_xi*sizeof(float),  cudaMemcpyHostToDevice, transfer_stream);

    cudaMemcpyAsync(d_x_out, x_out, N_p * sizeof(float), cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_y_out, y_out, N_p * sizeof(float), cudaMemcpyHostToDevice, transfer_stream);
    cudaMemcpyAsync(d_z_out, z_out, N_p * sizeof(float), cudaMemcpyHostToDevice, transfer_stream);

    size_t size_c = N_p * n_max * L * L * sizeof(float);
    float *d_c_real, *d_c_imag;
    cudaMalloc(&d_c_real, size_c);
    cudaMalloc(&d_c_imag, size_c);

    float *c_real_host = (float *)malloc(size_c);
    float *c_imag_host = (float *)malloc(size_c);

    cudaStreamSynchronize(transfer_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, compute_stream);

    int threads = 256;
    int blocks = (N_p + threads - 1)/threads;

    compute_c_nlm_kernel<<<blocks, threads, 0, compute_stream>>>(
        N_p, n_max, l_max,
        d_W, d_E, d_xi,
        d_x_out, d_y_out, d_z_out,
        d_c_real, d_c_imag
    );

    cudaEventRecord(stop, compute_stream);

    cudaStreamSynchronize(compute_stream);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float elapsed_us = milliseconds * 1000.0;
    float elapsed_s  = milliseconds / 1000.0;

    printf("Elapsed time: %.6f μs (%.6f s)\n", elapsed_us, elapsed_s);

    cudaStreamSynchronize(compute_stream);

    cudaFree(d_mem);
    cudaFree(xyz_dev);
    cudaFree(d_c_real);
    cudaFree(d_c_imag);

    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(xi_lmk_cpu);
    free(alpha_bl);
    free(beta_lnb);
    free(x_out);
    free(y_out);
    free(z_out);
    free(K);
    free(E);
    free(W);

    return 0;
}