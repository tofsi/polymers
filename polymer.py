import numpy as np
from random import choice
import itertools


class MaxIterationsExceededError(Exception):
    pass


# Adapted from "Generating uniformly distributed numbers on a sphere"
def gen_sphere_uniform_variate(l):
    result = np.zeros(shape=[3], dtype=float)

    # Generate random variates
    cos_theta = np.random.uniform(low=-1, high=1)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(low=0, high=2 * np.pi)

    # Map to cartesian coordinates
    result[0] = l * sin_theta * np.cos(phi)
    result[1] = l * sin_theta * np.sin(phi)
    result[2] = l * cos_theta

    return result


# Adapted from "Generating uniformly distributed numbers on a sphere"
def gen_sphere_uniform_variates(l, n):
    result = np.zeros(shape=[n, 3], dtype=float)

    # Generate random variates
    cos_theta = np.random.uniform(low=-1, high=1, size=n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.uniform(low=0, high=2 * np.pi, size=n)

    # Map to cartesian coordinates
    result[:, 0] = l * sin_theta * np.cos(phi)
    result[:, 1] = l * sin_theta * np.sin(phi)
    result[:, 2] = l * cos_theta

    return result


def distance_squared(p1, p2):
    diff = p1 - p2
    return np.dot(diff, diff)


def is_self_avoiding(chain, min_distance_squared):
    for i, first_monomer in enumerate(chain):
        for j in range(max([0, i - 3])):
            if distance_squared(first_monomer, chain[j]) < min_distance_squared:
                return False
    return True


def is_self_avoiding_lattice(chain):
    for i, first_monomer in enumerate(chain):
        for j in range(max([0, i - 3])):
            if tuple(first_monomer) == tuple(chain[j]):
                return False
    return True


class Polymer:
    """Basic polymer class"""

    def __init__(self, l, n):
        self.l = l
        if n < 1:
            raise ValueError("Length of chain must be at least 1")
        self.n = n

        self.chain = self.generate()

    def generate(self):
        return np.zeros([self.n + 1, 3], dtype=float)


class KuhnPolymer(Polymer):
    """Basic freely jointed chain"""

    def __init__(self, l, n):
        super(KuhnPolymer, self).__init__(l, n)

    def generate(self):
        monomers = gen_sphere_uniform_variates(self.l, self.n)
        return np.concatenate((np.zeros([1, 3], dtype=float), monomers.cumsum(axis=0)))


class SelfAvoidingKuhnPolymer(KuhnPolymer):
    """Self avoiding freely jointed chain"""

    def __init__(self, l, n, monomer_radius, max_iterations=np.inf):
        self.monomer_radius = monomer_radius
        self.max_iterations = max_iterations
        super(KuhnPolymer, self).__init__(l, n)

    def generate(self):
        """if self.monomer_radius > self.l / 2:
        raise ValueError(
            "impossible chain, monomer radius cannot be larger than bond length"
        )"""
        min_distance_squared = (self.monomer_radius * 2) ** 2
        monomers = np.zeros([self.n, 3], dtype=float)
        i = 1
        while True:
            monomers[0, :] = gen_sphere_uniform_variate(self.l)
            for i, previous_monomer in enumerate(monomers[:-1, :]):
                while True:
                    monomers[i + 1, :] = gen_sphere_uniform_variate(self.l)
                    if (
                        distance_squared(0, previous_monomer + monomers[i + 1, :])
                        > min_distance_squared
                    ):
                        break
            chain = np.concatenate(
                (np.zeros([1, 3], dtype=float), monomers.cumsum(axis=0))
            )
            i += 1
            if is_self_avoiding(chain, min_distance_squared):
                return chain
            elif i >= self.max_iterations:
                raise MaxIterationsExceededError(
                    "Maximum iterations exceeded when generating polymer"
                )
        return None


class LatticePolymer(Polymer):
    def __init__(self, n):
        super(LatticePolymer, self).__init__(1, n)

    def generate(self):
        monomers = np.zeros([self.n, 3], dtype=int)
        directions = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ]
        for i in range(self.n):
            monomers[i, :] = choice(directions)
        return np.concatenate((np.zeros([1, 3], dtype=int), monomers.cumsum(axis=0)))


class SelfAvoidingLatticePolymer(LatticePolymer):
    def __init__(self, n, max_iterations=np.inf):
        self.max_iterations = max_iterations
        super(LatticePolymer, self).__init__(1, n)

    def generate(self):
        monomers = np.zeros([self.n, 3], dtype=int)
        directions = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ]

        # chain = super(SelfAvoidingLatticePolymer, self).generate()
        i = 0
        monomers = np.zeros([self.n, 3], dtype=float)
        monomers[0] = choice(directions)
        while True:
            for i, previous_monomer in enumerate(monomers[:-1, :]):
                monomers[i + 1, :] = choice(
                    list(
                        filter(lambda d: d != tuple(-1 * previous_monomer), directions)
                    )
                )
            chain = np.concatenate(
                (np.zeros([1, 3], dtype=int), monomers.cumsum(axis=0))
            )
            i += 1
            if is_self_avoiding_lattice(chain):
                return chain
            elif i >= self.max_iterations:
                raise MaxIterationsExceededError(
                    "Maximum iterations exceeded when generating polymer"
                )
        return None


def main():
    pass


if __name__ == "__main__":
    main()
