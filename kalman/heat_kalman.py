from typing import Literal
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax_kalman as jaxk
import IWPprior


def PIVP_heat_solve(
    *,
    laplace_matrix: jnp.array,
    initial_mean: jnp.array,
    derivatives: int = 2,
    timesteps: int = 100,
    delta_time: float = 0.1,
    ornstein_uhlenbeck_prior: bool = False,
    noise_scale: float = 1,
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

    print("Starting PIVP_heat_solve")

    grid = len(laplace_matrix)
    print(grid)

    q = derivatives
    """Amount of derivatives we model"""

    curvature_matrix = jnp.zeros((grid, q * grid + grid))
    curvature_matrix = curvature_matrix.at[:grid, :grid].set(-laplace_matrix)

    time_1derivative_matrix = (
        jnp.zeros((grid, q * grid + grid)).at[:, grid : 2 * grid].set(jnp.eye(grid))
    )

    time_2derivative_matrix = (
        jnp.zeros((grid, q * grid + grid)).at[:, 2 * grid : 3 * grid].set(jnp.eye(grid))
    )

    SDE_coef, SDE_noise = IWPprior.get_IWP_Prior_SDE_coefficients(
        size=grid, derivatives=q
    )

    if ornstein_uhlenbeck_prior:
        SDE_coef = SDE_coef.at[-grid:, -grid:].set(-laplace_matrix)

    # SDE_noise *= noise_scale

    print("Built IWP prior")

    A, Q = jaxk.fast_get_discrete_system_coeffs(SDE_coef, SDE_noise, delta_time)

    print("Discretized IWP prior")

    heat_pde_error_matrix = curvature_matrix - time_2derivative_matrix

    R = jnp.zeros((grid, grid))

    initial_cov = jnp.zeros((grid * (1 + q), grid * (1 + q)))

    observations = jnp.zeros((timesteps, grid))

    filter_means, filter_covs, pred_means, pred_covs = jaxk.batch_filter(
        A, Q, heat_pde_error_matrix, R, initial_mean, initial_cov, observations
    )
    print("Filtered on PDE observations")
    smooth_means, smooth_covs = jaxk.batch_smooth(
        A, filter_means, filter_covs, pred_means, pred_covs
    )
    print("Smoothed PDE observations, returning")

    return smooth_means, smooth_covs


def PIVP_heat_solve_dense(
    *,
    laplace_matrix: jnp.array,
    initial_mean: jnp.array,
    derivatives: int,
    delta_time: float = 0.1,
    observation_indicator: jnp.array,
    noise_scale: float = 1,
    use_heat_prior: bool = False,
    PDE: Literal["heat", "wave"] = "heat",
):
    r"""
    Args:
        jax.numpy.array laplace_matrix:
            The domain-specific laplace matrix

        jax.numpy.array initial_value:
            The initial value of the solution, full state space, so also including derivatives.

        int derivatives
            Integer k > 0 specifying the smoothness of our time input, like the 'k' in C^k.
            Lower is faster. We need the time derivative to be at least C^1 to define the derivative.

        int delta_time
            The amount with which to advance time after each step.
    """

    print("Starting PIVP_heat_solve_dense")

    # assert (derivatives > 0) and (
    #     timesteps > 0
    # ), "Derivatives and timesteps must be > 0"

    grid = len(laplace_matrix)
    print(grid)

    q = derivatives
    """Amount of derivatives we model"""

    LOO = jnp.zeros((grid, q * grid + grid))
    LOO = LOO.at[:grid, :grid].set(-laplace_matrix)

    OIO = jnp.zeros((grid, q * grid + grid)).at[:, grid : 2 * grid].set(jnp.eye(grid))

    OOI = (
        jnp.zeros((grid, q * grid + grid)).at[:, 2 * grid : 3 * grid].set(jnp.eye(grid))
    )

    SDE_coef, SDE_noise = IWPprior.get_IWP_Prior_SDE_coefficients(
        size=grid, derivatives=q
    )
    SDE_noise *= noise_scale

    # insert -L into lower right corner
    if use_heat_prior:
        SDE_coef = SDE_coef.at[-grid:, -grid:].set(-laplace_matrix)

    A, Q = jaxk.fast_get_discrete_system_coeffs(SDE_coef, SDE_noise, delta_time)

    if PDE == "heat":
        pde_error_matrix = LOO - OIO
    elif PDE == "wave":
        pde_error_matrix = LOO - OOI

    observation_noise = jnp.zeros((grid, grid))

    initial_cov = jnp.diag(jnp.array([1e-5] * grid + [1] * (q * grid)))
    #   [0, 0, 0]
    # = [0, I, 0]
    #   [0, 0, I]

    filter_means, filter_covs, pred_means, pred_covs = (
        jaxk.batch_filter_optional_zero_observation(
            A,
            Q,
            pde_error_matrix,
            observation_noise,
            initial_mean,
            initial_cov,
            observation_indicator,
        )
    )

    print("Filtered on PDE observations")
    smooth_means, smooth_covs = jaxk.batch_smooth(
        A, filter_means, filter_covs, pred_means, pred_covs
    )
    print("Smoothed PDE observations, returning")

    return smooth_means, smooth_covs


def PIVP_solve_dense_non_linear(
    *,
    laplace_matrix: jnp.array,
    initial_value: jnp.array,
    derivatives: int,
    timesteps: int,
    delta_time: float = 0.1,
    observation_indicator: jnp.array,
    noise_scale: float = 1,
    nonlinear_observation_function,
    ornstein_uhlenbeck_prior: bool = False,
    PDE: Literal["heat", "wave"] = "heat",
):
    r"""
    Args:
        jax.numpy.array laplace_matrix:
            The domain-specific laplace matrix

        jax.numpy.array initial_value:
            The initial value of the solution, full state space, so also including derivatives.

        int derivatives
            Integer k > 0 specifying the smoothness of our time input, like the 'k' in C^k.
            Lower is faster. We need the time derivative to be at least C^1 to define the derivative.

        int timesteps
            The amount of steps to compute. Lower is faster.

        int delta_time
            The amount with which to advance time after each step.
    """

    print("Starting PIVP_heat_solve_dense_non_linear")

    assert (derivatives > 0) and (
        timesteps > 0
    ), "Derivatives and timesteps must be > 0"

    grid = len(laplace_matrix)
    print(grid)

    q = derivatives
    """Amount of derivatives we model"""

    SDE_coef, SDE_noise = IWPprior.get_IWP_Prior_SDE_coefficients(
        size=grid, derivatives=q
    )

    # SDE_noise *= noise_scale

    if ornstein_uhlenbeck_prior:
        SDE_coef = SDE_coef.at[-grid:, -grid:].set(-laplace_matrix)

    print("Built IWP prior")

    A, Q = jaxk.fast_get_discrete_system_coeffs(SDE_coef, SDE_noise, delta_time)

    print("Discretized IWP prior")

    R = jnp.zeros((grid, grid))

    initial_cov = jnp.zeros((grid * (1 + q), grid * (1 + q)))
    initial_cov = initial_cov.at[grid:, grid:].set(
        jnp.eye(derivatives * grid) * 1
    )  # TODO choose

    observations = jnp.zeros((len(observation_indicator) * (q + 1), grid))

    filter_means, filter_covs, pred_means, pred_covs = (
        jaxk.batch_filter_optional_observation_nonlinear_observation(
            A,
            Q,
            nonlinear_observation_function,
            R,
            initial_value,
            initial_cov,
            observations,
            observation_indicator,
        )
    )

    print("Filtered on PDE observations")
    smooth_means, smooth_covs = jaxk.batch_smooth(
        A, filter_means, filter_covs, pred_means, pred_covs
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
        initial_mean=initial_value,
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
