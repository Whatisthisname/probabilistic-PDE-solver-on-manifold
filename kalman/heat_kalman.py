import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import kalman.jax_kalman as jaxk
from kalman import IWPprior


def PIVP_heat_solve(
    *,
    laplace_matrix: jnp.array,
    initial_value: jnp.array,
    derivatives: int = 2,
    timesteps: int = 100,
    delta_time: float = 0.1,
):
    r"""
    Args:
        jax.numpy.array laplace_matrix:
            The domain-specific laplace matrix

        jax.numpy.array initial_value:
            The initial value of the solution, full state space, so also including derivatives.

        int derivatives
            Integer specifying the smoothness of our solution, like the 'k' in C^k. Lower is faster.

        int timesteps
            The amount of steps to compute. Lower is faster.

        int delta_time
            The amount with which to advance time after each step.
    """

    grid = len(laplace_matrix)
    print(grid)

    q = derivatives
    """Amount of derivatives we model"""

    curvature_matrix = jnp.zeros((grid, derivatives * grid + grid))
    curvature_matrix = curvature_matrix.at[:grid, :grid].set(-laplace_matrix)

    time_1derivative_matrix = (
        jnp.zeros((grid, derivatives * grid + grid))
        .at[:, grid : 2 * grid]
        .set(jnp.eye(grid))
    )

    time_2derivative_matrix = (
        jnp.zeros((grid, derivatives * grid + grid))
        .at[:, 2 * grid : 3 * grid]
        .set(jnp.eye(grid))
    )

    SDE_coef, SDE_noise = IWPprior.get_IWP_Prior_SDE_coefficients(
        size=grid, derivatives=derivatives
    )

    print("Built IWP prior")

    F, Q = jaxk.get_discrete_system_coeffs(SDE_coef, SDE_noise, delta_time)

    print("Discretized IWP prior")
    # wave_pde_error_matrix = (
    #     curvature_matrix - time_2derivative_matrix - time_1derivative_matrix * 0.5
    # )
    # wave_pde_error_matrix = wave_pde_error_matrix.at[0, :].set(0)
    # wave_pde_error_matrix = wave_pde_error_matrix.at[grid - 1, :].set(0)
    # wave_pde_error_matrix = wave_pde_error_matrix.at[0, 0].set(1)
    # wave_pde_error_matrix = wave_pde_error_matrix.at[grid - 1, grid - 1].set(1)

    heat_pde_error_matrix = 1 * curvature_matrix - time_1derivative_matrix
    # heat_pde_error_matrix = heat_pde_error_matrix.at[0, :].set(0)
    # heat_pde_error_matrix = heat_pde_error_matrix.at[grid - 1, :].set(0)
    # heat_pde_error_matrix = heat_pde_error_matrix.at[0, 0].set(1)
    # heat_pde_error_matrix = heat_pde_error_matrix.at[grid - 1, grid - 1].set(1)

    R = jnp.zeros((grid, grid))

    initial_cov = jnp.zeros((grid * (1 + q), grid * (1 + q)))

    observations = jnp.zeros((timesteps, grid))

    filter_means, filter_covs, pred_means, pred_covs = jaxk.batch_filter(
        F, Q, heat_pde_error_matrix, R, initial_value, initial_cov, observations
    )
    print("Filtered on PDE observations")
    smooth_means, smooth_covs = jaxk.batch_smooth(
        F, filter_means, filter_covs, pred_means, pred_covs
    )
    print("Smoothed PDE observations, returning")

    return smooth_means, smooth_covs


if __name__ == "__main__":
    grid = 20
    domain = jnp.linspace(0, 1, grid)
    h = domain[1] - domain[0]

    # Laplace operator, ∆
    laplace = (
        jnp.diag(2 * jnp.ones(grid))
        - jnp.diag(jnp.ones(grid - 1), -1)
        - jnp.diag(jnp.ones(grid - 1), 1)
    )

    laplace = laplace.at[0, -1].set(-1)  # make it loop
    laplace = laplace.at[-1, 0].set(-1)
    laplace /= h**2

    derivatives = 2
    initial_value = jnp.zeros(grid * (1 + derivatives))
    initial_value = initial_value.at[:grid].set(jnp.sin(jnp.pi * domain))

    print(initial_value.round(2))

    means, covs = PIVP_heat_solve(
        laplace_matrix=laplace,
        initial_value=initial_value,
        derivatives=derivatives,
        timesteps=100,
        delta_time=0.002,
    )

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for z, i in enumerate(range(len(means))):
        d = 1 + 1 - (z + 1) / len(means)
        dist_scale = 1 / d
        domain = jnp.linspace((1 - dist_scale) / 2, 1 - (1 - dist_scale) / 2, grid)
        mean = means[i] * dist_scale
        std = jnp.sqrt(jnp.diagonal(covs[i])) * dist_scale
        offset = -1 * i / len(means)
        ax[0].plot(domain, offset + mean[:grid], color=cmap(i / len(means)))
        ax[0].fill_between(
            domain,
            offset + mean[:grid] - 3 * std[:grid],
            offset + mean[:grid] + 3 * std[:grid],
            alpha=0.5 + 0.5 * (i / len(means)),
            color=cmap(i / len(means)),
        )

    plt.show()
