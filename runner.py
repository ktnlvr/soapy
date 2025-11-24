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

    all_results = {name: [] for name in implementations}

    for p in range(1, max_power + 1):
        n_points = 10 ** p
        print(f"\n=== Generating {n_points} points (10^{p}) ===")
        subprocess.run(f"python generate.py {n_points}", shell=True, check=True)

        power_results = {name: [] for name in implementations}

        for name, cmd in implementations.items():
            print(f"Running {name} for {trials} trials...")
            for _ in range(trials):
                elapsed = run_command(cmd)
                if elapsed is not None:
                    power_results[name].append(elapsed)

        # Store mean for plotting
        for name in implementations:
            mean_time = np.mean(power_results[name])
            all_results[name].append(mean_time)

        # Print nice summary for this power
        print(f"\nSummary for 10^{p} points:")
        for name in implementations:
            mean = np.mean(power_results[name])
            std = np.std(power_results[name])
            print(f"{name}: mean = {mean:.2f} μs, std = {std:.2f} μs")

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
