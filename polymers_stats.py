from polymer import *
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-dark")


class PolymerSample:
    def __init__(self, n_draws, polymer):
        self.r_squared = np.zeros(n_draws, dtype=float)
        self.chains = np.zeros([n_draws, polymer.N + 1, 3], dtype=float)

        self.polymer = polymer
        for i in range(n_draws):
            self.chains[i, :, :] = self.polymer.chain
            self.r_squared[i] = distance_squared(
                self.polymer.chain[0, :], self.polymer.chain[-1, :]
            )
            self.polymer.chain = self.polymer.generate()

        self.polymer.chain = (
            None  # self.polymer is only needed to store the polymer parameters
        )

        self.mean_r_squared = self.r_squared.mean()


def main():
    # l = 1.0
    # monomer_radius = 0.45
    # N = 40
    # n_draws = 10

    # polymer_info = PolymerSample(n_draws, N, KuhnPolymer, (l, N))
    r_squared = np.zeros(1000, dtype=float)
    N_values = [i for i in range(1, 1001)]
    for i, N in enumerate(N_values):
        r_squared[i] = PolymerSample(100, N, KuhnPolymer(1.0, N)).mean_r_squared
    plt.plot(N_values, r_squared, marker="o")
    plt.show()


if __name__ == "__main__":
    main()
