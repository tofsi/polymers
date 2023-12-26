import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

plt.style.use("seaborn-v0_8")
np.random.uniform()


# Adapted from "Generating uniformly distributed numbers on a sphere"
def gen_sphere_uniform_variates(l, N):
    result = np.zeros(shape=[N, 3], dtype=float)

    # Generate random variates
    cos_theta = np.random.uniform(low=-1, high=1, size=N)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(low=0, high=2 * np.pi, size=N)

    # Map to cartesian coordinates
    result[:, 0] = l * sin_theta * np.cos(phi)
    result[:, 1] = l * sin_theta * np.sin(phi)
    result[:, 2] = l * cos_theta

    if N == 1:
        return result[0, :]  # 1d array
    else:
        return result  # 2d array


def gen_kuhn_chain(l, N):
    monomers = gen_sphere_uniform_variates(l, N)
    return monomers.cumsum(axis=0)


def distance_squared(p1, p2):
    diff = p1 - p2
    return np.dot(diff, diff)


def self_avoiding(polymer, min_distance_squared):
    N = polymer.shape[0]
    for i, first_monomer in enumerate(polymer):
        for j in range(max([0, i - 2])):
            if distance_squared(first_monomer, polymer[j]) < min_distance_squared:
                return False
    return True


def gen_self_avoiding_chain(l, monomer_radius, N, max_iterations=np.inf):
    if monomer_radius > l / 2:
        raise ValueError(
            "impossible chain, monomer radius cannot be larger than bond length"
        )
    min_distance_squared = (monomer_radius * 2) ** 2
    polymer = gen_kuhn_chain(l, N)
    i = 1
    while not self_avoiding(polymer, min_distance_squared) and i < max_iterations:
        polymer = gen_kuhn_chain(l, N)
        i += 1
    return polymer


def main():
    pass


if __name__ == "__main__":
    main()
