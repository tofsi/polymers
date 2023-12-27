from polymer import *
from scipy import stats


class PolymerStats:
    def __init__(self, n_draws, N, polymer_gen_func, polymer_args):
        self.r_squared = np.zeros(n_draws, dtype=float)
        self.polymers = np.zeros([n_draws, N, 3], dtype=float)
        for i in range(n_draws):
            self.polymers[i, :, :] = polymer_gen_func(*polymer_args)
            self.r_squared[i] = distance_squared(
                self.polymers[i, 0, :], self.polymers[i, -1, :]
            )
        self.mean_r_squared = self.r_squared.mean()


def main():
    l = 1.0
    monomer_radius = 0.45
    N = 40
    n_draws = 10

    polymer_info = PolymerStats(
        n_draws, N, gen_self_avoiding_chain, (l, monomer_radius, N)
    )
    print(polymer_info.mean_r_squared)


if __name__ == "__main__":
    main()
