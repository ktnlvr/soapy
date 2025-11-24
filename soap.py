from ase.io import read
from dscribe.descriptors import SOAP
import time
from sys import stderr

r_cut = 50
n_max = 2
l_max = 3
sigma = 1

start_time = 0
elapsed_time = 0


def tic():
    global start_time
    start_time = time.time()


def toc():
    global start_time, elapsed_time
    elapsed_time += (time.time() - start_time) * 1_000_000


atoms = read("random_hydrogens.xyz")

soap = SOAP(
    species=atoms.get_chemical_symbols(),
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    sigma=sigma,
    periodic=False
)

tic()
soap_vector = soap.create(atoms, centers=[[0, 0, 0]])
toc()

print("Time (μs):", elapsed_time)
print("Size of the SOAP vector:", soap_vector.size)
print("SOAP vector:", soap_vector)

stderr.write(f"{int(elapsed_time)}\n")
