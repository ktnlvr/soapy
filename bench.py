import os
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

def nrmse(a, b):
    """Compute the Normalized Root Mean Square Error (NRMSE) of a relative to b, ignoring NaNs and infs."""
    mask = ~np.isnan(a) & ~np.isnan(b) & ~np.isinf(a) & ~np.isinf(b) & ~np.isneginf(a) & ~np.isneginf(b)
    if not np.any(mask):
        return float('inf')

    a_valid = a[mask]
    b_valid = b[mask]

    mse = np.mean((a_valid - b_valid) ** 2)
    rmse = np.sqrt(mse)
    norm = np.nanmax(b_valid) - np.nanmin(b_valid)
    return rmse / norm if norm != 0 else float('inf')

def color_for_diff(diff):
    """Return color based on the magnitude of the difference."""
    abs_diff = abs(diff)
    if abs_diff < 1e-3:
        return Fore.GREEN  # small difference
    elif abs_diff < 1e-1:
        return Fore.YELLOW  # moderate difference
    else:
        return Fore.RED  # large difference

def print_matrix(label, matrix):
    """Nicely print a labeled matrix (handles up to N-D arrays)."""
    print(f"\n{label} (shape {matrix.shape}):")

    if np.isscalar(matrix):
        print(f"{matrix:8.4f}")
        return

    # Flatten if more than 2D
    if matrix.ndim > 2:
        flat_matrix = matrix.reshape(-1, matrix.shape[-1])
        print(f"Note: Flattened higher dimensions for display.")
        matrix = flat_matrix

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    for row in matrix:
        for val in row:
            print(f"{val:8.4f}", end=" ")
        print()
    print()


def print_difference_matrix(a, b):
    """Print a colorized difference matrix (a - b)."""
    diff = a - b
    print("Difference matrix (C++ - Python):")

    if np.isscalar(diff):
        color = color_for_diff(diff)
        print(f"{color}{diff:8.4f}{Style.RESET_ALL}\n")
        return

    if diff.ndim > 2:
        flat_diff = diff.reshape(-1, diff.shape[-1])
        print(f"Note: Flattened higher dimensions for display.")
        diff = flat_diff

    if diff.ndim == 1:
        diff = diff.reshape(1, -1)

    for row in diff:
        for val in row:
            color = color_for_diff(val)
            print(f"{color}{val:8.4f}{Style.RESET_ALL}", end=" ")
        print()
    print()


def main():
    files = os.listdir('.')

    cpp_files = {f[:-8]: f for f in files if f.endswith('_cpp.npy')}
    py_files = {f[:-7]: f for f in files if f.endswith('_py.npy')}

    common_keys = sorted(set(cpp_files.keys()) & set(py_files.keys()))

    if not common_keys:
        print("No matching _cpp.npy and _py.npy file pairs found.")
        return

    for key in common_keys:
        cpp_data = np.load(cpp_files[key])
        py_data = np.load(py_files[key])

        if cpp_data.shape != py_data.shape:
            print(f"\nSkipping {key}: shape mismatch {cpp_data.shape} vs {py_data.shape}")
            continue

        error = nrmse(cpp_data, py_data)

        print(f"\n{'=' * 80}")
        print(f"Comparing matrices for: {key}")

        print_matrix("Python Matrix", py_data)
        print_difference_matrix(cpp_data, py_data)
        print_matrix("C++ Matrix", cpp_data)

        print(f"NRMSE = {100 * error:.6f}%")
        print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
