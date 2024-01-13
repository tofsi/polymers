from polymer import *
from polymers_stats import PolymerSample
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-dark")


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


def plot_mean_r_squared(n_draws_list, N_list, l, ax, theoretical_func):
    if len(n_draws_list) != len(N_list):
        raise ValueError("n_draws must be of same length as N")
    mean_r_squared = []
    for n_draws, N in zip(n_draws_list, N_list):
        mean_r_squared.append(
            PolymerSample(n_draws, N, KuhnPolymer, (l, N)).mean_r_squared
        )
    ax.plot(
        mean_r_squared,
        linestyle="-",
        marker="o",
        c="purple",
        mec="cyan",
        mfc="cyan",
    )
    if theoretical_func:
        ax.plot(N_list, [theoretical_func(N) for N in N_list])
        ax.legend(["Sample", "Theoretical"])


def main():
    # plot_p(1.0, 70, 0.45)
    fig, ax = plt.subplots()
    plot_mean_r_squared([2000] * 1000, list(range(2, 1002)), 1.0, ax, lambda N: N)
    plt.show()


if __name__ == "__main__":
    main()
