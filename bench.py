import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import time

WARMUP_RUNS = 2
BENCH_RUNS = 5
MAX_POWER = 7

# regex to extract elapsed time in μs
elapsed_regex = re.compile(r"Elapsed time: ([0-9.]+) μs")

def run_script(script):
    """Run a script and return elapsed time (from output) in seconds"""
    result = subprocess.run(["python3", script], capture_output=True, text=True)
    out = result.stdout
    match = elapsed_regex.search(out)
    elapsed = float(match.group(1)) * 1e-6 if match else None  # μs → s
    return elapsed, out

def benchmark(script_name, powers):
    total_times = []
    elapsed_times = []

    for p in powers:
        n = 10 ** p
        print(f"\n=== Benchmarking {script_name} for 10^{p} ({n}) ===")
        
        # Warmup runs (generate once before warmups)
        print(f"Generating input for 10^{p}...")
        subprocess.run(["python3", "generate.py", str(n)], capture_output=True)
        
        print(f"Warmup runs ({WARMUP_RUNS})...")
        for i in range(WARMUP_RUNS):
            elapsed, _ = run_script(script_name)
            print(f"  Warmup run {i+1} done, elapsed = {elapsed*1e6:.2f} μs" if elapsed else f"  Warmup run {i+1} done")

        # Benchmark runs
        total_list = []
        elapsed_list = []
        print(f"Benchmark runs ({BENCH_RUNS})...")
        for i in range(BENCH_RUNS):
            # regenerate input before each run
            subprocess.run(["python3", "generate.py", str(n)], capture_output=True)
            
            start = time.time()
            elapsed, _ = run_script(script_name)
            end = time.time()
            
            total_list.append(end - start)
            if elapsed is not None:
                elapsed_list.append(elapsed)
            print(f"  Run {i+1}: total_time = {end-start:.6f}s, elapsed_time = {elapsed*1e6:.2f} μs" if elapsed else f"  Run {i+1}: total_time = {end-start:.6f}s")

        total_times.append((np.mean(total_list), np.std(total_list)))
        elapsed_times.append((np.mean(elapsed_list), np.std(elapsed_list)) if elapsed_list else (None, None))

    return total_times, elapsed_times

powers = list(range(0, MAX_POWER+1))

# Benchmark run.py
run_total, run_elapsed = benchmark("run.py", powers)

# Benchmark target_implementation.py
target_total, target_elapsed = benchmark("target_implementation.py", powers)

# Convert to arrays for plotting
def unpack(tuples):
    mean, std = zip(*tuples)
    return mean, std

run_total_mean, run_total_std = unpack(run_total)
run_elapsed_mean, run_elapsed_std = unpack(run_elapsed)
target_total_mean, target_total_std = unpack(target_total)
target_elapsed_mean, target_elapsed_std = unpack(target_elapsed)

# Plotting
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.errorbar(powers, run_total_mean, yerr=run_total_std, label="run.py", fmt='-o')
plt.errorbar(powers, target_total_mean, yerr=target_total_std, label="target_implementation.py", fmt='-o')
plt.xlabel("Input size (10^n)")
plt.ylabel("Total time (s)")
plt.xticks(powers, [f"10^{p}" for p in powers])
plt.yscale("log")
plt.title("Total Time Benchmark")
plt.legend()
plt.grid(True, which="both", ls="--")

# Convert elapsed to microseconds for readability
run_elapsed_us = [t*1e6 if t is not None else 0 for t in run_elapsed_mean]
run_elapsed_std_us = [s*1e6 if s is not None else 0 for s in run_elapsed_std]
target_elapsed_us = [t*1e6 if t is not None else 0 for t in target_elapsed_mean]
target_elapsed_std_us = [s*1e6 if s is not None else 0 for s in target_elapsed_std]

plt.subplot(1,2,2)
plt.errorbar(powers, run_elapsed_us, yerr=run_elapsed_std_us, label="run.py", fmt='-o')
plt.errorbar(powers, target_elapsed_us, yerr=target_elapsed_std_us, label="target_implementation.py", fmt='-o')
plt.xlabel("Input size (10^n)")
plt.ylabel("Elapsed time (μs)")
plt.xticks(powers, [f"10^{p}" for p in powers])
plt.yscale("log")
plt.title("Elapsed Time Benchmark")
plt.legend()
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("benchmark_results.png")
