from polymer import *
from polymers_stats import PolymerSample
import matplotlib.pyplot as plt


from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats

plt.style.use("seaborn-v0_8")
# sns.set_style("whitegrid")


def plot_sphere_variates():
    sphere_variates = gen_sphere_uniform_variates(1.0, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(sphere_variates[:, 0], sphere_variates[:, 1], sphere_variates[:, 2])
    ax.set_aspect("equal")
    plt.show()


# method adapted from
# https://stackoverflow.com/questions/45324258/draw-many-spheres-efficiently
def plot_spheres(centers, radius, resolution, ax):
    cos_theta = np.linspace(-1, 1, resolution)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.linspace(0, 2 * np.pi, resolution)
    x = np.outer(np.cos(phi), sin_theta)
    y = np.outer(np.sin(phi), sin_theta)
    z = np.outer(np.ones(resolution), cos_theta)
    for k in range(centers.shape[0]):
        ax.plot_surface(
            radius * x + centers[k, 0],
            radius * y + centers[k, 1],
            radius * z + centers[k, 2],
            color="cyan",
            alpha=1.0,
            linewidth=0,
        )


def plot_p(p):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(
        p.chain[:, 0],
        p.chain[:, 1],
        p.chain[:, 2],
        linestyle="-",
        linewidth=5.0,
        c="orange",
    )
    if hasattr(p, "monomer_radius"):
        plot_spheres(p.chain, p.monomer_radius, 40, ax)

    ax.set_aspect("equal")
    plt.show()


def plot_Cn(polymer_sample_dict, ax):
    for name, polymer_samples in polymer_sample_dict.items():
        n_values = list(map(lambda s: s.polymer.n, polymer_samples))
        C_n_values = list(map(lambda s: np.mean(s.C_n), polymer_samples))
        ax.plot(n_values, C_n_values, linestyle="--", marker="o", label=name)


def end_distributions():
    n_samples = 1000

    kuhn_sample = PolymerSample(n_samples, KuhnPolymer(1.0, 70))
    self_avoiding_kuhn_sample = PolymerSample(
        n_samples, SelfAvoidingKuhnPolymer(1.0, 70, 1.0 / np.sqrt(5))
    )
    lattice_sample = PolymerSample(n_samples, LatticePolymer(70))
    self_avoiding_lattice_sample = PolymerSample(
        n_samples, SelfAvoidingLatticePolymer(70)
    )

    samples = {
        "Freely Jointed": kuhn_sample,
        "Freely Jointed (Self Avoiding)": self_avoiding_kuhn_sample,
        "Integer Lattice": lattice_sample,
        "Integer Lattice (Self Avoiding)": self_avoiding_lattice_sample,
    }

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for i, name_and_sample in enumerate(samples.items()):
        name, sample = name_and_sample
        components = sample.end_to_end.reshape([3 * n_samples])

        print(
            "{}: mean {}, Standard Deviation {}".format(
                name, components.mean(), components.std()
            )
        )

        qqplot(components, ax=ax[i % 2, i // 2], line="r")
        ax[i % 2, i // 2].set_title(name)

    plt.savefig("images/qqplots.eps")


def C_n_plots():
    # plot_p(1.0, 59, 0.45)
    fig, ax = plt.subplots()
    n_max = 70
    n_points = 10
    n_values = np.exp(np.linspace(0, np.log(n_max), n_points)).astype(int)
    n_draws = 1000
    lattice_samples = list(
        map(lambda n: PolymerSample(n_draws, LatticePolymer(n)), n_values)
    )
    saw_lattice_samples = list(
        map(lambda n: PolymerSample(n_draws, SelfAvoidingLatticePolymer(n)), n_values)
    )
    kuhn_samples = list(
        map(lambda n: PolymerSample(n_draws, KuhnPolymer(1, n)), n_values)
    )
    saw_kuhn_samples = list(
        map(
            lambda n: PolymerSample(
                n_draws, SelfAvoidingKuhnPolymer(1, n, 1 / np.sqrt(5))
            ),
            n_values,
        )
    )

    samples = {
        "Freely Jointed": kuhn_samples,
        "Freely Jointed (Self Avoiding)": saw_kuhn_samples,
        "Integer Lattice": lattice_samples,
        "Integer Lattice (Self Avoiding)": saw_lattice_samples,
    }
    plot_Cn(samples, ax)

    n_range = np.array(range(1, n_max))
    ax.plot(
        n_range,
        (7 * n_range - 2) / (5 * n_range),
        label="Lower Bound (Self Avoiding)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("mean square end-to-end distance")
    ax.legend()
    plt.savefig("log_C_n_plot.eps")


def main2():
    while True:
        plot_p(SelfAvoidingKuhnPolymer(1.0, 110, 0.5))


if __name__ == "__main__":
    C_n_plots()
