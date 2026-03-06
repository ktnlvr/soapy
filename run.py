from numba import jit
import numba as nb
import math
import numpy as np
from numba import cuda

from ase.io import read

from scipy.special import gamma
from scipy.linalg import sqrtm, inv

from caching import cache_load_or_compute

import time


start_time = 0.0

import time

start_time = 0.0

def tic():
    global start_time
    start_time = time.time()

def toc():
    elapsed_us = (time.time() - start_time) * 1_000_000
    elapsed_s  = elapsed_us / 1_000_000
    print(f"Elapsed time: {elapsed_us:.6f} μs ({elapsed_s:.6f} s)")


def get_basis_gto(r_cut, n_max, l_max):
    a = np.linspace(1, r_cut, n_max)
    threshold = 1e-3
    alphas_full = np.zeros((l_max + 1, n_max))
    betas_full = np.zeros((l_max + 1, n_max, n_max))

    for l in range(l_max + 1):
        alphas = -np.log(threshold / np.power(a, l)) / a**2
        m = np.zeros((alphas.shape[0], alphas.shape[0])) + alphas
        m = m + m.T
        S = 0.5 * gamma(l + 3.0 / 2.0) * m ** (-l - 3.0 / 2.0)
        betas = sqrtm(inv(S))
        if betas.dtype == np.complex128:
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lower n_max or increase r_cut."
            )
        alphas_full[l, :] = alphas
        betas_full[l, :, :] = betas
    return alphas_full, betas_full


def precompute_xi_lmk(l_max):
    xi = np.zeros((l_max + 1, l_max + 1, l_max + 1))

    for l in range(l_max + 1):
        for m in range(l + 1):
            for k in range(m, l + 1):
                if (k - l) % 2 != 0:
                    xi[l, m, k] = 0.0
                else:
                    num = math.gamma((l + k - 1)/2 + 1)
                    den = (
                        math.gamma(k - m + 1) *
                        math.gamma(l - k + 1) *
                        math.gamma((l + k - 1)/2 - l + 1)
                    )
                    xi[l, m, k] = num / den
    return xi

def precompute_W_nlb(alpha_bl, beta_lnb, sigma):
    denom = (1.0 + 2.0 * alpha_bl * sigma * sigma) ** 1.5
    return (beta_lnb / denom[:, None, :]).transpose(1, 0, 2)

def precompute_K_nlm(alpha_bl, beta_lnb, sigma):
    n_max = beta_lnb.shape[1]
    l_max = alpha_bl.shape[0] - 1

    K_nlm = np.zeros((n_max, l_max+1, l_max+1))

    for l in range(l_max + 1):
        for m in range(l + 1):
            #            ----------------------|
            #         l  | (2l + 1) * (l - m)! |
            # 𝜆_lm = 2   | ~~~~~~~~~~~~~~~~~~~ |
            #            v    4𝜋(l + m!)       | 
            #

            numerator = (2*l + 1) * math.factorial(l - m)
            denominator = 4*math.pi * math.factorial(l + m)
            lambda_lm = (2**l) * math.sqrt(numerator / denominator)

            for n in range(n_max):
                #                            B_lbn
                # \Sigma_N_b = ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #               \sqrt { 1 + 2 a_lb 𝜎^2 }^(2l+3)
                sum_b = 0.0
                for b in range(n_max):
                    ab = alpha_bl[l, b]
                    bb = beta_lnb[l, n, b]
                    denom = (1 + 2*ab*sigma*sigma)**(l + 3/2)
                    sum_b += bb / denom

                # \sqrt{ 2 𝜋 𝜎^2 }^3
                K_nlm[n, l, m] = lambda_lm * (2*math.pi*sigma*sigma)**(3/2) * sum_b

    return K_nlm


def precompute_E_lb(alpha_bl, sigma):
    denom = 1.0 + 2.0 * alpha_bl * sigma * sigma
    E_lb = -alpha_bl / denom
    return E_lb

from numba import cuda, float32, int32

kernel_sig = (
    int32, int32, int32,           # N_p, n_max, l_max
    float32[:, :, :],              # W_nlb (n_max, l_max+1, n_max_b)
    float32[:, :],                 # E_lb (l_max+1, n_max)
    float32[:, :, :],              # xi_lmk (l_max+1, l_max+1, l_max+1)
    float32[:],                     # x_p
    float32[:],                     # y_p
    float32[:],                     # z_p
    float32[:, :, :, :],           # c_partial_real
    float32[:, :, :, :]            # c_partial_imag
)

# C_nlm = K_lm 
#   \Sigma_b W_nlb 
#   \Sigma_p exp( E_lb * R^2 )
#      (x_p + iy_p)^m \Sigma_{k=m}^l 
#         Xi_lmk z^{k-m}_p R^{l-k}_p
@cuda.jit(kernel_sig)
def compute_c_nlm_kernel(N_p, n_max, l_max,
                         W_nlb, E_lb, xi_lmk,
                         x_p, y_p, z_p,
                         c_partial_real, c_partial_imag):

    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    xy_m = np.complex64(0.0)
    temp_sum = np.complex64(0.0)

    for p in range(tid, N_p, stride):
        xp = x_p[p]
        yp = y_p[p]
        zp = z_p[p]

        R2 = xp*xp + yp*yp + zp*zp
        xy = np.complex64(xp + 1j*yp)

        for n in range(n_max):
            for l in range(l_max + 1):
                for m in range(l + 1):

                    temp_sum = np.complex64(0.0)

                    xy_m = np.complex64(1.0)
                    for i in range(m):
                        xy_m *= xy

                    for b in range(W_nlb.shape[2]):
                        w = W_nlb[n, l, b]
                        e = E_lb[l, b]
                        exp_factor = math.exp(e * R2)

                        sum_k = np.complex64(0.0)

                        for k in range(m, l + 1):

                            xi = xi_lmk[l, m, k]

                            z_term = zp ** (k - m)
                            R_term = math.sqrt(R2) ** (l - k)

                            sum_k += np.complex64(xi * z_term * R_term)

                        temp_sum += w * exp_factor * xy_m * sum_k

                    c_partial_real[tid, n, l, m] += temp_sum.real
                    c_partial_imag[tid, n, l, m] += temp_sum.imag

def main():
    r_cut = 50
    n_max = 2
    l_max = 3
    sigma = 1

    compute_c_nlm_kernel.compile(kernel_sig)

    xi_lmk_table = cache_load_or_compute(
        "xi_lmk", precompute_xi_lmk, l_max
    )
    print(f"Xi_lmk = {xi_lmk_table.shape}")

    alpha_bl, beta_lnb = cache_load_or_compute(
        "alpha_bl-beta_lnb", get_basis_gto, r_cut, n_max, l_max
    )

    print(f"A_bl = {alpha_bl.shape}")
    print(f"B_lnb = {beta_lnb.shape}")

    atoms = read('random_hydrogens.xyz')
    positions = atoms.positions
    N_p = len(positions)
    x_p, y_p, z_p = positions[:,0], positions[:,1], positions[:,2]

    print(f"N_p = {len(positions)}")

    k_nlm = cache_load_or_compute(
        "k_nlm", precompute_K_nlm, alpha_bl, beta_lnb, sigma
    )

    print(f"K_nlm = {k_nlm.shape}")

    w_nlb = cache_load_or_compute(
        "W_nlb", precompute_W_nlb, alpha_bl, beta_lnb, sigma
    )
    print(f"W_nlb = {w_nlb.shape}")
    
    e_lb = cache_load_or_compute(
        "E_lb", precompute_E_lb, alpha_bl, sigma
    )
    print(f"E_lb = {e_lb.shape}")

    def pinned_to_device(array, stream):
        pinned = cuda.pinned_array(array.shape, dtype=array.dtype)
        pinned[:] = array
        return cuda.to_device(pinned, stream=stream)

    stream = cuda.stream()

    tic()

    xi_lmk_dev = pinned_to_device(xi_lmk_table, stream)
    W_nlb_dev  = pinned_to_device(w_nlb, stream)
    E_lb_dev   = pinned_to_device(e_lb, stream)

    x_p_dev = pinned_to_device(x_p, stream)
    y_p_dev = pinned_to_device(y_p, stream)
    z_p_dev = pinned_to_device(z_p, stream)

    device = cuda.get_current_device()

    threads_per_block = 256
    blocks_per_grid = 4 * device.MULTIPROCESSOR_COUNT

    num_threads = threads_per_block * blocks_per_grid

    c_partial_real_dev = cuda.pinned_array(
        (num_threads, n_max, l_max+1, l_max+1), dtype=np.float32
    )

    c_partial_imag_dev = cuda.pinned_array(
        (num_threads, n_max, l_max+1, l_max+1), dtype=np.float32
    )

    cuda.profile_start()

    compute_c_nlm_kernel[blocks_per_grid, threads_per_block](
        N_p, n_max, l_max, W_nlb_dev, E_lb_dev, xi_lmk_dev,
        x_p_dev, y_p_dev, z_p_dev,
        c_partial_real_dev, c_partial_imag_dev
    )
    cuda.synchronize()

    c_arr = k_nlm * (np.sum(c_partial_real_dev, axis=0) + 1j * np.sum(c_partial_imag_dev, axis=0))
    toc()

    cuda.profile_stop()

    print(c_arr.shape)

main()
