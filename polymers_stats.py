from polymer import *
from scipy import stats
import pandas as pd
import ast
import re

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

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


def save_C_n_data():
    n_max = 65
    n_points = 20
    n_r_values = 20
    n_values = pd.Series(
        np.exp(np.linspace(1, np.log(n_max), n_points)).astype(int)
    ).unique()
    n_draws = 1000
    r_values = np.linspace(0.05, 0.5, n_r_values)

    kuhn_C_n_data = pd.DataFrame(columns=["n", "r", "C_n"])
    lattice_C_n_data = pd.DataFrame(columns=["n", "C_n"])
    i = 0
    print("Generating C_n Data")
    for n in n_values:
        for C_n in PolymerSample(n_draws * 10, SelfAvoidingLatticePolymer(n)).C_n:
            lattice_C_n_data = pd.concat(
                [lattice_C_n_data, pd.DataFrame([{"n": n, "C_n": C_n}])],
                ignore_index=True,
            )
        for r in r_values:
            for C_n in PolymerSample(n_draws, SelfAvoidingKuhnPolymer(1.0, n, r)).C_n:
                kuhn_C_n_data = pd.concat(
                    [kuhn_C_n_data, pd.DataFrame([{"n": n, "r": r, "C_n": C_n}])],
                    ignore_index=True,
                )

            i += 1
            if ((i * 100) / (n_points * n_r_values)) % 10 == 0:
                print((i * 100) // (n_points * n_r_values), R"% done")

    kuhn_C_n_data.to_csv("data/kuhn_C_n.csv")
    lattice_C_n_data.to_csv("data/lattice_C_n.csv")


def fit_C_n_data():
    kuhn_C_n_data = pd.read_csv("data/kuhn_C_n.csv")
    lattice_C_n_data = pd.read_csv("data/lattice_C_n.csv")
    kuhn_C_n_data = kuhn_C_n_data.map(lambda v: np.log(v))
    print(kuhn_C_n_data)

    # kuhn_C_n_data = kuhn_C_n_data.map(lambda v: np.log(v))


def main():
    save_C_n_data()
    fit_C_n_data()


if __name__ == "__main__":
    main()
