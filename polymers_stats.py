from polymer import *
from scipy import stats

# import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-dark")


class PolymerSample:
    def __init__(self, n_draws, polymer, print_progress=False):
        self.r_squared = np.zeros(n_draws, dtype=float)
        self.C_n = np.zeros(n_draws, dtype=float)
        self.chains = np.zeros([n_draws, polymer.n + 1, 3], dtype=float)
        self.end_to_end = np.zeros([n_draws, 3], dtype=float)
        self.polymer = polymer
        if print_progress:
            print("Generating {} sample".format(str(type(self.polymer))))
        for i in range(n_draws):
            self.chains[i, :, :] = self.polymer.chain
            self.end_to_end[i, :] = self.polymer.chain[-1, :] - self.polymer.chain[0, :]
            self.r_squared[i] = distance_squared(
                self.polymer.chain[0, :], self.polymer.chain[-1, :]
            )
            self.C_n[i] = self.r_squared[i] / (self.polymer.n * self.polymer.l**2)
            self.polymer.chain = self.polymer.generate()
            if print_progress and ((i * 100) / n_draws) % 10 == 0:
                print((i * 100) // n_draws, R"% done")

        self.polymer.chain = (
            None  # self.polymer is only needed to store the polymer parameters
        )

        self.mean_r_squared = self.r_squared.mean()


def main():
    # l = 1.0
    # monomer_radius = 0.45
    # n = 40
    # n_draws = 10

    # polymer_info = PolymerSample(n_draws, n, KuhnPolymer, (l, n))
    r_squared = np.zeros(1000, dtype=float)
    n_values = [i for i in range(1, 1001)]
    for i, n in enumerate(n_values):
        r_squared[i] = PolymerSample(100, KuhnPolymer(1.0, n)).mean_r_squared
    plt.plot(n_values, r_squared, marker="o")
    plt.show()


if __name__ == "__main__":
    main()
