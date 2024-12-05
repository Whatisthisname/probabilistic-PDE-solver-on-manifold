from typing import Literal
import matplotlib.pyplot as plt

import jax.numpy as jnp
import app.probabilistic_numerics._covariance_kalman_impl as jaxk
import app.probabilistic_numerics._cholesky_kalman_impl as cholk
import probabilistic_numerics.IWPprior as IWPprior
import jax

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)


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


def PIVP_heat_solve_cholesky(
    *,
    laplace_matrix: jnp.array,
    initial_mean: jnp.array,
    derivatives: int = 2,
    timesteps: int = 100,
    delta_time: float = 0.1,
    ornstein_uhlenbeck_prior: bool = False,
    length_scale: float = 1,
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

    SDE_noise *= length_scale

    print("Built IWP prior")

    A, Q = IWPprior.perform_matrix_fraction_decomposition(
        SDE_coef, SDE_noise, delta_time
    )

    print("Discretized IWP prior")

    heat_pde_error_matrix = curvature_matrix - time_2derivative_matrix

    R = jnp.eye((grid)) * 0.001
    Ch_R = jnp.linalg.cholesky(R)

    initial_cov = jnp.zeros((grid * (1 + q), grid * (1 + q)))

    observations = jnp.zeros((timesteps, grid))

    filtered, reverse_parameters = cholk.jax_batch_extended_filter(
        prior_mean=initial_mean,
        prior_cholvariance=initial_cov,
        A_cond_obs=heat_pde_error_matrix,
        b_cond_obs=jnp.zeros(grid),
        Ch_cond_obs=Ch_R,
        A_cond_state=A,
        b_cond_state=jnp.zeros(grid * (1 + q)),
        Ch_cond_state=Q,
        observations=observations,
    )

    print("Filtered on PDE observations")
    last_filtered = cholk.CholGauss(filtered.mean[-1], filtered.chol_cov[-1])
    smooth_means, smooth_chol_covs, _samples = cholk.jax_batch_smooth_and_sample(
        last_filtered,
        reverse_parameters,
        n_samples=0,
    )
    print("Smoothed PDE observations, returning")

    return smooth_means, smooth_chol_covs


def _taylor_coef_diag(h: float, state: int, derivatives: int) -> jnp.ndarray:
    elements: jnp.ndarray = jnp.arange(derivatives, -1, -1)
    factorial = jnp.cumprod(jnp.arange(1, derivatives + 1))[::-1]
    factorial = jnp.concatenate([factorial, jnp.array([1])])
    elements = jnp.power(h, elements)
    elements = elements / factorial
    return jnp.repeat(elements, state) * jnp.sqrt(h)


def get_taylor_add_h(h: float, state: int, derivatives: int) -> jnp.ndarray:
    diag = _taylor_coef_diag(h, state, derivatives)
    # return jnp.eye(state * (1 + derivatives))
    return jnp.diag(diag)


def get_taylor_remove_h(h: float, state: int, derivatives: int) -> jnp.ndarray:
    diag = _taylor_coef_diag(h, state, derivatives)
    # return jnp.eye(state * (1 + derivatives))
    return jnp.diag(1 / diag)


def solve_nonlinear_IVP(
    *,
    prior_dynamics: jnp.array,
    initial_mean: jnp.array,
    derivatives: int = 2,
    timesteps: int = 100,
    delta_time: float = 0.1,
    prior: Literal["iwp", "heat", "wave"],
    observation_function,
    observation_uncertainty,
    update_indicator,
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

    q = derivatives
    n_state = len(initial_mean) // (q + 1)

    SDE_coef, SDE_noise = IWPprior.get_IWP_Prior_SDE_coefficients(
        size=n_state, derivatives=q
    )

    if prior == "heat":  # place prior dynamics in bottom right
        SDE_coef = SDE_coef.at[-n_state:, -n_state:].set(prior_dynamics)
    elif prior == "wave":  # place prior dynamics in left of bottom right
        SDE_coef = SDE_coef.at[-n_state:, -2 * n_state : -n_state].set(prior_dynamics)

    taylor_add_h = get_taylor_add_h(delta_time, n_state, q)
    taylor_remove_h = get_taylor_remove_h(delta_time, n_state, q)

    A = taylor_remove_h @ SDE_coef @ taylor_add_h
    Q = taylor_remove_h @ SDE_noise

    A, Q = IWPprior.perform_matrix_fraction_decomposition(
        SDE_coef, SDE_noise, delta_time
    )

    print("hi")
    print(Q)

    if rank := jnp.linalg.matrix_rank(Q) < n_state * (1 + q):
        print(
            "WARNING: Rank of Q is",
            rank,
            ", which is not full rank. Applying 1e-8 diagonal jitter",
        )
        Q = Q + jnp.eye(n_state * (1 + q)) * 1e-8

    if (
        jnp.linalg.matrix_rank(observation_uncertainty)
        < observation_uncertainty.shape[0]
    ):
        observation_uncertainty_chol = jnp.zeros_like(observation_uncertainty)
    else:
        observation_uncertainty_chol = jnp.linalg.cholesky(observation_uncertainty)

    filtered, reverse_parameters = cholk.jax_batch_extended_filter(
        prior_mean=taylor_remove_h @ initial_mean,
        prior_cholvariance=jnp.zeros((n_state * (q + 1), n_state * (q + 1))),
        observation_function=lambda state, time, step: observation_function(
            taylor_add_h @ state, time, step
        ),
        Ch_cond_obs=observation_uncertainty_chol,
        A_cond_state=A,
        b_cond_state=jnp.zeros(n_state * (1 + q)),
        Ch_cond_state=jnp.linalg.cholesky(Q),
        update_indicator=update_indicator,
        timesteps=timesteps,
        delta_time=delta_time,
    )

    last_filtered = cholk.CholGauss(filtered.mean[-1], filtered.chol_cov[-1])
    smooth_means, smooth_chol_covs, _samples = cholk.jax_batch_smooth_and_sample(
        last_filtered,
        reverse_parameters,
        n_samples=0,
    )

    stds = jnp.sqrt(
        jnp.diagonal(
            jnp.einsum("ijk,ilk->ijl", smooth_chol_covs, smooth_chol_covs),
            axis1=1,
            axis2=2,
        )
    )

    # for each time step, we have a state vector of size grid * (1 + q).
    # The taylor transform is (grid * (1 + q)) x (grid * (1 + q)).
    # we want to apply the taylor transform to each time step:
    transformed_mean = jnp.einsum("ij,kj->ki", taylor_add_h, smooth_means)
    transformed_std = jnp.einsum("ij,kj->ki", taylor_add_h, stds)
    return transformed_mean, transformed_std


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
