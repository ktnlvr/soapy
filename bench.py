import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import ttest_ind
import csv

WARMUP_RUNS = 2
BENCH_RUNS = 10
MAX_POWER = 7

# regex to extract elapsed time in μs
elapsed_regex = re.compile(r"Elapsed time: ([0-9.]+) μs")

def run_script(*cmd):
    """Run a script and return elapsed time (from output) in seconds"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout
    match = elapsed_regex.search(out)
    elapsed = float(match.group(1)) * 1e-6 if match else None  # μs → s
    return elapsed, out

def benchmark(script_name, powers):
    total_times = []
    elapsed_times = []
    raw_elapsed = []  # store all individual elapsed times for stats

    for p in powers:
        n = 10 ** p
        print(f"\n=== Benchmarking {script_name} for 10^{p} ({n}) ===")
        
        # Warmup runs
        print(f"Generating input for 10^{p}...")
        subprocess.run(["python3", "generate.py", str(n)], capture_output=True)
        print(f"Warmup runs ({WARMUP_RUNS})...")
        for i in range(WARMUP_RUNS):
            elapsed, _ = run_script(*script_name)
            print(f"  Warmup run {i+1} done, elapsed = {elapsed*1e6:.2f} μs" if elapsed else f"  Warmup run {i+1} done")

        # Benchmark runs
        total_list = []
        elapsed_list = []
        print(f"Benchmark runs ({BENCH_RUNS})...")
        for i in range(BENCH_RUNS):
            subprocess.run(["python3", "generate.py", str(n)], capture_output=True)
            start = time.time()
            elapsed, _ = run_script(*script_name)
            end = time.time()
            total_list.append(end - start)
            if elapsed is not None:
                elapsed_list.append(elapsed)
            print(f"  Run {i+1}: total_time = {end-start:.6f}s, elapsed_time = {elapsed*1e6:.2f} μs" if elapsed else f"  Run {i+1}: total_time = {end-start:.6f}s")

        total_times.append((np.mean(total_list), np.std(total_list)))
        elapsed_times.append((np.mean(elapsed_list), np.std(elapsed_list)) if elapsed_list else (None, None))
        raw_elapsed.append(elapsed_list)

    return total_times, elapsed_times, raw_elapsed

powers = list(range(0, MAX_POWER+1))

# Benchmark run.py
run_total, run_elapsed, run_raw = benchmark(["./build/main"], powers)

# Benchmark target_implementation.py
target_total, target_elapsed, target_raw = benchmark(["python3", "target_implementation.py"], powers)

# Convert to arrays for plotting
def unpack(tuples):
    mean, std = zip(*tuples)
    return mean, std

run_total_mean, run_total_std = unpack(run_total)
run_elapsed_mean, run_elapsed_std = unpack(run_elapsed)
target_total_mean, target_total_std = unpack(target_total)
target_elapsed_mean, target_elapsed_std = unpack(target_elapsed)

# Convert elapsed to μs
run_elapsed_us = [t*1e6 if t is not None else 0 for t in run_elapsed_mean]
run_elapsed_std_us = [s*1e6 if s is not None else 0 for s in run_elapsed_std]
target_elapsed_us = [t*1e6 if t is not None else 0 for t in target_elapsed_mean]
target_elapsed_std_us = [s*1e6 if s is not None else 0 for s in target_elapsed_std]

# Compute p-values and faster implementation
p_values = []
faster_impl = []
for i, (r, t) in enumerate(zip(run_raw, target_raw)):
    if r and t:
        stat, p = ttest_ind(r, t, equal_var=False)
        p_values.append(p)
        faster = "run.py" if np.mean(r) < np.mean(t) else "target_implementation.py"
        faster_impl.append(faster)
    else:
        p_values.append(None)
        faster_impl.append(None)

# Save results to CSV
with open("benchmark_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Input size", 
        "run_total_mean(s)", "run_total_std(s)", 
        "target_total_mean(s)", "target_total_std(s)",
        "run_elapsed_mean(μs)", "run_elapsed_std(μs)",
        "target_elapsed_mean(μs)", "target_elapsed_std(μs)",
        "p_value", "faster_implementation"
    ])
    for i, p in enumerate(powers):
        writer.writerow([
            f"10^{p}",
            run_total_mean[i], run_total_std[i],
            target_total_mean[i], target_total_std[i],
            run_elapsed_us[i], run_elapsed_std_us[i],
            target_elapsed_us[i], target_elapsed_std_us[i],
            f"{p_values[i]:.6f}" if p_values[i] is not None else "",
            faster_impl[i] if faster_impl[i] else ""
        ])

# Plotting
plt.figure(figsize=(14,5))

# Total time
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

# Elapsed time with p-values
plt.subplot(1,2,2)
plt.errorbar(powers, run_elapsed_us, yerr=run_elapsed_std_us, label="run.py", fmt='-o')
plt.errorbar(powers, target_elapsed_us, yerr=target_elapsed_std_us, label="target_implementation.py", fmt='-o')
plt.xlabel("Input size (10^n)")
plt.ylabel("Elapsed time (μs)")
plt.xticks(powers, [f"10^{p}" for p in powers])
plt.yscale("log")
plt.title("Elapsed Time Benchmark with Significance")
plt.legend()
plt.grid(True, which="both", ls="--")

# Annotate p-values
for i, p in enumerate(p_values):
    if p is not None and p < 0.05:
        y = max(run_elapsed_us[i]+run_elapsed_std_us[i], target_elapsed_us[i]+target_elapsed_std_us[i]) * 1.1
        plt.text(powers[i], y, f"*p={p:.3f}*", ha='center', color='red', fontsize=9)

plt.tight_layout()
plt.savefig("benchmark_results.png")
plt.show()
