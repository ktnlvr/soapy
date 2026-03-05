from numba import jit
import numba as nb
import math
import numpy as np

from ase.io import read

from scipy.special import gamma
from scipy.linalg import sqrtm, inv

import time


def tic():
    global start_time
    start_time = time.time()

def toc():
    global start_time, elapsed_time
    elapsed_time += (time.time() - start_time) * 1_000_000
    print(elapsed_time)


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

# C_nlm = K_lm 
#   \Sigma_b W_nlb 
#   \Sigma_p exp( E_lb * R^2 )
#      (x_p + iy_p)^m \Sigma_{k=m}^l 
#         Xi_lmk z^{k-m}_p R^{l-k}_p

def compute_c_nlm(n, 
    l, m,
    K_nlm,
    E_nlm,
    R2,
    xy_pow,
    R_pow,
    z_pow,
    xi_lmk
):
    ...

def mathcalK(p, x, y, z, n, l, m, E, R_pow, xi_lmk):
    ...

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

def main():
    r_cut = 50
    n_max = 2
    l_max = 3
    sigma = 1

    xi_lmk_table = precompute_xi_lmk(l_max)
    assert not np.any(np.isnan(xi_lmk_table))
    print(f"Xi_lmk = {xi_lmk_table.shape}")

    alpha_bl, beta_lnb = get_basis_gto(r_cut, n_max, l_max)
    print(f"A_bl = {alpha_bl.shape}")
    print(f"B_lnb = {beta_lnb.shape}")

    atoms = read('random_hydrogens.xyz')
    positions = atoms.positions
    x_p, y_p, z_p = positions[:,0], positions[:,1], positions[:,2]

    print(f"N_p = {len(positions)}")

    k_nlm = precompute_K_nlm(alpha_bl, beta_lnb, sigma)
    print(f"K_nlm = {k_nlm.shape}")

    w_nlb = precompute_W_nlb(alpha_bl, beta_lnb, sigma)
    print(f"W_nlb = {w_nlb.shape}")

    e_lb = precompute_E_lb(alpha_bl, sigma)
    print(f"E_lb = {e_lb.shape}")

    c_arr = np.zeros((n_max, l_max+1, l_max+1), dtype=complex)
    c = []

    tic()
    for nn in range(n_max):
        c.append([])
        for ln in range(l_max+1):
            c[nn].append([])
            for mn in range(ln + 1):
                c_val = compute_c_nlm(nn, ln, mn, k_nlm, x_p, y_p, z_p, sigma)
                c_arr[nn, ln, mn] = c_val
                c[nn][ln].append(c_val)
    toc()

main()
