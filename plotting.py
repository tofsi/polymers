from polymer import *
from polymers_stats import PolymerSample
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")


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


def plot_p(l, N, monomer_radius=0.0):
    p = SelfAvoidingKuhnPolymer(l, N, monomer_radius)
    # p = gen_self_avoiding_chain(l, monomer_radius, N)s
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
    plot_spheres(p.chain, monomer_radius, 40, ax)
    ax.set_aspect("equal")
    plt.show()


def plot_mean_r_squared(polymer_sample_dict, ax, theoretical_func):
    for polymer_samples in polymer_sample_dict.values():
        N_values = list(map(lambda s: s.polymer.N, polymer_samples))
        mean_r_squared_values = list(map(lambda s: s.mean_r_squared, polymer_samples))
        ax.plot(
            N_values,
            mean_r_squared_values,
            linestyle="--",
            marker="o",
        )
    if theoretical_func:
        ax.plot(N_values, list(map(theoretical_func, N_values)))
        ax.legend([*polymer_sample_dict.keys(), "Theoretical"])
    else:
        ax.legend(polymer_sample_dict.keys())


def main():
    # plot_p(1.0, 59, 0.45)
    fig, ax = plt.subplots()
    N_values = list(range(2, 50, 10))

    kuhn_samples = list(
        map(lambda N: PolymerSample(10000, KuhnPolymer(1.0, N)), N_values)
    )
    saw_samples = list(
        map(lambda N: PolymerSample(10, SelfAvoidingKuhnPolymer(1.0, N, 0.5)), N_values)
    )
    plot_mean_r_squared(
        {"Kuhn Chain": kuhn_samples, "Self Avoiding Chain": saw_samples},
        ax,
        lambda N: N,
    )
    plt.show()


if __name__ == "__main__":
    main()
