import math
import numpy as np
from ase.io import read
from scipy.special import gamma
from scipy.linalg import sqrtm, inv

import time

itter = 0

start_time = 0
elapsed_time = 0

def tic():
    global start_time
    start_time = 0
    start_time = time.time()

def toc():
    global start_time, elapsed_time
    elapsed_time += time.time() - start_time

def get_basis_gto(r_cut, n_max, l_max):
    a = np.linspace(1, r_cut, n_max)
    threshold = 1e-3

    alphas_full = np.zeros((l_max + 1, n_max))
    betas_full = np.zeros((l_max + 1, n_max, n_max))

    for l in range(0, l_max + 1):
        alphas = -np.log(threshold / np.power(a, l)) / a**2
        m = np.zeros((alphas.shape[0], alphas.shape[0]))
        m[:, :] = alphas
        m = m + m.transpose()
        S = 0.5 * gamma(l + 3.0 / 2.0) * m ** (-l - 3.0 / 2.0)
        betas = sqrtm(inv(S))
        if betas.dtype == np.complex128:
            raise ValueError(
                "Could not calculate normalization factors for the radial "
                "basis in the domain of real numbers. Lowering the number of "
                "radial basis functions (n_max) or increasing the radial "
                "cutoff (r_cut) is advised."
            )
        alphas_full[l, :] = alphas
        betas_full[l, :, :] = betas
    return alphas_full, betas_full

# the offset of xi_lmk into a flat xi_lmk array
def xi_lmk_offset(l, m, k):
    offset_l = (l * (l + 1) * (l + 2)) // 6
    offset_m = (m * (2*l - m + 3)) // 2
    offset_k = k - m
    return offset_l + offset_m + offset_k

def xi_lmk(l, m, k):
    if (k - l ) % 2 != 0:
        value = 0.0
    else:
        num = math.gamma((l + k - 1)/2 + 1)
        den = (
                math.gamma(k - m + 1)
                * math.gamma(l - k + 1)
                * math.gamma((l + k - 1)/2 - l + 1)
                )
        value = num / den
    xi_lmk_table[xi_lmk_offset(l, m, k)] = value
    return value

def compute_c_nlm(n,l,m,alpha_bl,beta_nbl,x_p,y_p,z_p,sigma):
    global itter
    numerator   = (2*l + 1) * math.factorial(l - m)
    denominator = 4*math.pi * math.factorial(l + m)
    lambda_lm = (2**l) * math.sqrt(numerator / denominator)
    
    prefactor = lambda_lm
    
    c_nlm = 0.0 + 0.0j
    N_b = len(alpha_bl[0,:]) 
    N_p = len(x_p) 
    for b in range(N_b):
        ab = alpha_bl[l,b]
        bb = beta_nbl[l,n,b]
        denom = (math.sqrt(1 + 2*ab*sigma*sigma))**(2*l + 3)
        factor_b = bb / denom
        
        sum_p = 0.0 + 0.0j
        for p in range(N_p):
            rx, ry, rz = x_p[p], y_p[p], z_p[p]
            rp = math.sqrt(rx**2 + ry**2 + rz**2)
            exp_factor = math.exp(- ab * rp*rp / (1 + 2 * ab * sigma*sigma))
            xy_complex = complex(rx, ry) ** (m)
            itter += 1
            
            rp_l_minus_m = rp**(l - m) if rp != 0 else 0.0
            
            sum_k = 0.0
            for k in range(m, l + 1):
                xi_val = xi_lmk(l, m, k)
                term_k = rz**(k - m) * rp**(m - k)
                sum_k += xi_val * term_k
            sum_p += exp_factor * xy_complex * rp_l_minus_m * sum_k 
        
        c_nlm += factor_b * sum_p*(-1)**m
    c_nlm *= prefactor*math.sqrt(2*sigma*sigma*math.pi)**3
    
    return c_nlm


if __name__ == "__main__":
    r_cut = 50
    n_max = 2
    l_max = 3

    global xi_lmk_table
    xi_lmk_size = (l_max + 1) * (l_max + 2) * (l_max + 3) // 6
    xi_lmk_table = np.zeros(xi_lmk_size)
    xi_lmk_table[:] = np.nan

    tic()
    alpha_bl, beta_nbl = get_basis_gto(r_cut, n_max, l_max)

    np.save("alpha_bl.npy", alpha_bl)
    np.save("beta_nbl.npy", beta_nbl)

    toc()
    print(alpha_bl[:,:])
    print(beta_nbl[:,:,:])
    print(alpha_bl.shape)
    print(beta_nbl.shape)
    c = []
    P = []

    atoms = read('random_hydrogens.xyz')

    positions = atoms.positions
    x_p = positions[:, 0]
    y_p = positions[:, 1]
    z_p = positions[:, 2]

    np.save("positions_py.npy", np.hstack((positions, atoms.numbers[:, np.newaxis])))
    print(np.hstack((positions, atoms.numbers[:, np.newaxis])))

    sigma = 1

    # from here
    c_arr = np.zeros((n_max, l_max+1, l_max+1), dtype=complex)

    for nn in range(n_max):
        c.append([])
        for ln in range(l_max+1):
            c[nn].append([])
            for mn in range(ln + 1):
                c_val = compute_c_nlm(nn, ln, mn, alpha_bl, beta_nbl, x_p, y_p, z_p, sigma)
                c_arr[nn, ln, mn] = c_val
                c[nn][ln].append(c_val)

    np.save('c_nlm_py.npy', np.stack((c_arr.real, c_arr.imag), axis=-1))

    accum = 0
    for ln in range(l_max + 1):
        for nn in range(n_max):
            for nnn in range(nn,n_max):
                for mn in range(ln + 1):
                    if mn == 0:
                        accum += math.pi*math.sqrt(8/(2*ln + 1))*(c[nn][ln][mn]*(c[nnn][ln][mn].conjugate())).real
                    else:
                        accum += 2*math.pi*math.sqrt(8/(2*ln + 1))*(c[nn][ln][mn]*(c[nnn][ln][mn].conjugate())).real
                P.append(accum)
                accum = 0

    print(xi_lmk_table.shape)
    assert not np.any(np.isnan(xi_lmk_table))
    np.save("xi_lmk_py.npy", xi_lmk_table)

print("c; ", c)
print(len(P))
print(P)
elapsed_time *= 1000000
print(f"Elapsed time: {elapsed_time:.6f} milli seconds")
print(P[0])
