import random
import sys

NUM_POINTS = 1_000_000
LOW, HIGH = -10, 10

if len(sys.argv) > 1:
    try:
        NUM_POINTS = int(sys.argv[1])
    except ValueError:
        print("Invalid number of points. Using default:", NUM_POINTS)

with open("random_hydrogens.xyz", "w") as f:
    f.write(f"{NUM_POINTS}\n\n")
    
    for _ in range(NUM_POINTS):
        x = random.uniform(LOW, HIGH)
        y = random.uniform(LOW, HIGH)
        z = random.uniform(LOW, HIGH)
        f.write(f"H {x} {y} {z}\n")
