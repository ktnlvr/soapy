import os
import hashlib
import numpy as np
import pickle

from ase.io import read

TMP_DIR = "/tmp/soap_cache"
os.makedirs(TMP_DIR, exist_ok=True)

def get_hash(*args):
    m = hashlib.sha256()
    for arg in args:
        if isinstance(arg, np.ndarray):
            m.update(arg.tobytes())
        else:
            m.update(np.array(arg, dtype=np.float64).tobytes())
    return m.hexdigest()

def cache_load_or_compute(name_prefix, compute_fn, *args):
    hash_str = get_hash(*args)
    file_path = os.path.join(TMP_DIR, f"{name_prefix}_{hash_str}.pkl")

    if os.path.exists(file_path):
        print(f"Loading cached {name_prefix} from {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Computing {name_prefix}...")
        arr = compute_fn(*args)
        with open(file_path, "wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        return arr


def cached_read_xyz(filename):
    stat = os.stat(filename)
    cache_key = f"{os.path.basename(filename)}_{stat.st_mtime}_{stat.st_size}.pkl"
    cache_path = os.path.join(TMP_DIR, cache_key)
    
    if os.path.exists(cache_path):
        print(f"Using cached version of {filename}")
        with open(cache_path, "rb") as f:
            atoms = pickle.load(f)
        return atoms
    
    print(f"Reading {filename} from disk")
    atoms = read(filename)
    
    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(atoms, f)
    
    return atoms