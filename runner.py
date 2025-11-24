import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        stderr_output = result.stderr.strip()
        value = int(stderr_output.split()[0])
        return value
    except Exception as e:
        print(f"Error running {command}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark different implementations")
    parser.add_argument("max_power", type=int, help="Maximum power of 10 for generate.py")
    parser.add_argument("--trials", type=int, default=9, help="Number of trials (default 9)")
    parser.add_argument("-r", "--skip-reference", action="store_true",
                        help="Skip running the reference implementation")
    args = parser.parse_args()

    max_power = args.max_power
    trials = args.trials

    implementations = {
        "reference": "python3 reference_implementation.py",
        "soap": "python3 soap.py",
        "sycl": "./build/sycl"
    }

    if args.skip_reference:
        implementations.pop("reference")

    impl_names = list(implementations.keys())

    # Store mean times for plotting and final comparison matrix
    all_results = {name: [] for name in impl_names}

    for p in range(1, max_power + 1):
        n_points = 10 ** p
        print(f"\n=== Generating {n_points} points (10^{p}) ===")
        subprocess.run(f"python generate.py {n_points}", shell=True, check=True)

        power_results = {name: [] for name in impl_names}

        for name, cmd in implementations.items():
            print(f"Running {name} for {trials} trials...")
            for _ in range(trials):
                elapsed = run_command(cmd)
                if elapsed is not None:
                    power_results[name].append(elapsed)

        # Store mean times
        mean_times = {name: np.mean(power_results[name]) for name in impl_names}
        for name in impl_names:
            all_results[name].append(mean_times[name])

        # Print summary
        print(f"\nSummary for 10^{p} points:")
        for name in impl_names:
            mean = np.mean(power_results[name])
            std = np.std(power_results[name])
            print(f"{name}: mean = {mean:.2f} μs, std = {std:.2f} μs")

        # ---- SPEEDUP MATRIX FOR THIS POWER ----
        print(f"\nSpeedup matrix for 10^{p} points (values > 1 mean row is faster):")
        print("               " + "  ".join(f"{b:>10}" for b in impl_names))
        for a in impl_names:
            row = []
            for b in impl_names:
                speedup = mean_times[b] / mean_times[a]
                row.append(f"{speedup:10.3f}")
            print(f"{a:>12}  " + " ".join(row))

    # ---- FINAL AVERAGE SPEEDUP MATRIX OVER ALL POWERS ----
    print("\n=== Final Average Speedup Matrix (averaged over all powers) ===")
    final_matrix = np.zeros((len(impl_names), len(impl_names)))

    for i, a in enumerate(impl_names):
        for j, b in enumerate(impl_names):
            # average speedup across all problem sizes
            ratios = np.array(all_results[b]) / np.array(all_results[a])
            final_matrix[i, j] = np.mean(ratios)

    print("               " + "  ".join(f"{b:>10}" for b in impl_names))
    for i, a in enumerate(impl_names):
        row = " ".join(f"{final_matrix[i,j]:10.3f}" for j in range(len(impl_names)))
        print(f"{a:>12}  {row}")

    # ---- PLOT ----
    plt.figure(figsize=(10,6))
    x = np.arange(1, max_power + 1)
    for name, times in all_results.items():
        plt.plot(x, times, marker='o', label=name)

    plt.xticks(x, [f"10^{i}" for i in x])
    plt.xlabel("Number of points")
    plt.ylabel("Mean execution time (μs, log scale)")
    plt.yscale("log")
    plt.title("Benchmark of Implementations")
    plt.grid(True, linestyle='--', alpha=0.5, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
