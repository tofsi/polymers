import numpy as np
import itertools


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


def distance_squared(p1, p2):
    diff = p1 - p2
    return np.dot(diff, diff)


def self_avoiding(chain, min_distance_squared):
    N = chain.shape[0]
    for i, first_monomer in enumerate(chain):
        for j in range(max([0, i - 2])):
            if distance_squared(first_monomer, chain[j]) < min_distance_squared:
                return False
    return True


class Polymer:
    """Basic polymer class"""

    def __init__(self, l, N):
        self.l = l
        self.N = N
        self.chain = self.generate()

    def generate(self):
        return np.zeros([self.N, 3], dtype=float)


class KuhnPolymer(Polymer):
    """Basic freely jointed chain"""

    def __init__(self, l, N):
        super(KuhnPolymer, self).__init__(l, N)

    def generate(self):
        monomers = gen_sphere_uniform_variates(self.l, self.N)
        return monomers.cumsum(axis=0)


class SelfAvoidingKuhnPolymer(KuhnPolymer):
    """Self avoiding freely jointed chain"""

    def __init__(self, l, N, monomer_radius, max_iterations=np.inf):
        self.monomer_radius = monomer_radius
        self.max_iterations = max_iterations
        super(KuhnPolymer, self).__init__(l, N)

    def generate(self):
        if self.monomer_radius > self.l / 2:
            raise ValueError(
                "impossible chain, monomer radius cannot be larger than bond length"
            )
        min_distance_squared = (self.monomer_radius * 2) ** 2
        chain = super(SelfAvoidingKuhnPolymer, self).generate()
        i = 1
        while (
            not self_avoiding(chain, min_distance_squared) and i < self.max_iterations
        ):
            chain = super(SelfAvoidingKuhnPolymer, self).generate()
            i += 1
        return chain


def main():
    pass


if __name__ == "__main__":
    main()
