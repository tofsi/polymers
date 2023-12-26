from main import *
from polymers_stats import PolymerStats

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


def plot_kuhn_chain(l, N, monomer_radius=0.0):
    kuhn_chain = gen_self_avoiding_chain(l, monomer_radius, N)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.plot(
        kuhn_chain[:, 0],
        kuhn_chain[:, 1],
        kuhn_chain[:, 2],
        linestyle="-",
        linewidth=5.0,
        c="orange",
    )
    plot_spheres(kuhn_chain, monomer_radius, 40, ax)
    ax.set_aspect("equal")
    plt.show()


def plot_mean_r_squared(n_draws_list, N_list, l, monomer_radius, ax):
    polymer_info = []
    if len(n_draws_list) != len(N_list):
        raise ValueError("n_draws must be of same length as N")
    for n_draws, N in zip(n_draws_list, N_list):
        polymer_info.append(
            PolymerStats(n_draws, N, gen_self_avoiding_chain, (l, monomer_radius, N))
        )
    ax.plot(
        [p.mean_r_squared for p in polymer_info],
        linestyle="--",
        marker="o",
        c="purple",
        mec="cyan",
        mfc="cyan",
    )


def main():
    # plot_kuhn_chain(1.0, 70, 0.45)
    pass


if __name__ == "__main__":
    main()
