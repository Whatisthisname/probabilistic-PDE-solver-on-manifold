from typing import Literal
import jax
import jax.numpy as jnp
import probabilistic_numerics._cholesky_kalman_impl as cholk
import probabilistic_numerics.IWPprior as IWPprior

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)


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
    n_samples: int = 0,
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
    smooth_means, smooth_chol_covs, samples = cholk.jax_batch_smooth_and_sample(
        last_filtered,
        reverse_parameters,
        n_samples=n_samples,
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
