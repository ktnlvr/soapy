import random

NUM_POINTS = 1_000_000
LOW, HIGH = -10, 10

with open("random_hydrogens.xyz", "w") as f:
    f.write(f"{NUM_POINTS}\n\n")

    for _ in range(NUM_POINTS):
        x = random.uniform(LOW, HIGH)
        y = random.uniform(LOW, HIGH)
        z = random.uniform(LOW, HIGH)
        f.write(f"H {x} {y} {z}\n")
