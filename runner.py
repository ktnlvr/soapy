import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt

def run_command(command):
    """Run a command and return the integer value from stderr."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Assume the integer we want is the first number in stderr
        stderr_output = result.stderr.strip()
        value = int(stderr_output.split()[0])
        return value
    except Exception as e:
        print(f"Error running {command}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark different implementations")
    parser.add_argument("power", type=int, help="Power of 10 for generate.py")
    parser.add_argument("--trials", type=int, default=9, help="Number of trials (default 12)")
    parser.add_argument("-r", "--skip-reference", action="store_true",
                    help="Skip running the reference implementation")
    args = parser.parse_args()

    power = args.power
    trials = args.trials
    n_points = 10**power

    print(f"Generating {n_points} points...")
    subprocess.run(f"python3 generate.py {n_points}", shell=True, check=True)

    implementations = {
        "reference_implementation.py": "python3 reference_implementation.py",
        "soap.py": "python3 soap.py",
        "sycl": "./build/sycl"
    }

    if args.skip_reference:
        implementations.pop("reference_implementation.py")

    results = {name: [] for name in implementations}

    # Run each implementation 'trials' times
    for name, cmd in implementations.items():
        print(f"Running {name} for {trials} trials...")
        for _ in range(trials):
            elapsed = run_command(cmd)
            if elapsed is not None:
                results[name].append(elapsed)

    # Compute statistics
    means = {name: np.mean(times) for name, times in results.items()}
    stds = {name: np.std(times) for name, times in results.items()}

    print("\nResults (in microseconds):")
    for name in implementations:
        print(f"{name}: mean = {means[name]:.2f}, std = {stds[name]:.2f}")

    # Plot results
    labels = list(implementations.keys())
    means_list = [means[label] for label in labels]
    stds_list = [stds[label] for label in labels]

    plt.figure(figsize=(8,6))
    plt.bar(labels, means_list, yerr=stds_list, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylabel("Time (μs)")
    plt.title(f"Benchmark results for 10^{power} points")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
