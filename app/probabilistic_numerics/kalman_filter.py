from typing import Literal
import matplotlib.pyplot as plt

import jax.numpy as jnp
import probabilistic_numerics._covariance_kalman_impl as jaxk
import probabilistic_numerics._cholesky_kalman_impl as cholk
import probabilistic_numerics.IWPprior as IWPprior
import jax

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
    # return jnp.eye(state * (1 + derivatives))  # TODO
    return jnp.diag(diag)


def get_taylor_remove_h(h: float, state: int, derivatives: int) -> jnp.ndarray:
    diag = _taylor_coef_diag(h, state, derivatives)
    # return jnp.eye(state * (1 + derivatives))
    return jnp.diag(1 / diag)


def solve_nonlinear_IVP(
    *,
    prior_matrix: jnp.array,
    initial_mean: jnp.array,
    initial_cov_diag: jnp.array,
    derivatives: int,
    n_solution_points: int,
    delta_time: float,
    prior_type: Literal["iwp", "heat", "wave", "jerk", "all", "two"],
    observation_function,
    observation_uncertainty: jnp.array,
    update_indicator,
    n_samples: int,
):
    q = derivatives
    state_size = len(initial_mean) // (q + 1)

    SDE_coef, SDE_noise = IWPprior.get_IWP_Prior_SDE_coefficients(
        size=state_size, derivatives=q
    )

    if prior_type == "heat":  # place prior matrix in bottom right
        SDE_coef = SDE_coef.at[-state_size:, -1 * state_size :].set(prior_matrix)
    elif prior_type == "wave":  # place prior matrix in left of bottom right
        SDE_coef = SDE_coef.at[-state_size:, -2 * state_size : -1 * state_size].set(
            prior_matrix
        )
    elif prior_type == "jerk":  # place prior matrix in left of left of bottom right
        SDE_coef = SDE_coef.at[-state_size:, -3 * state_size : -2 * state_size].set(
            prior_matrix
        )
    elif (
        prior_type == "all"
    ):  # place prior matrix in left of bottom right and bottom right
        SDE_coef = SDE_coef.at[-state_size:, -state_size:].set(prior_matrix)
        SDE_coef = SDE_coef.at[-state_size:, -2 * state_size : -state_size].set(
            prior_matrix
        )
    elif (
        prior_type == "two"
    ):  # place prior matrix in left of bottom right and bottom right
        SDE_coef = SDE_coef.at[-state_size:, -2 * state_size :].set(prior_matrix)

    unprecondition_matrix = get_taylor_add_h(delta_time, state_size, q)
    precondition_matrix = get_taylor_remove_h(delta_time, state_size, q)

    SDE_coef = precondition_matrix @ SDE_coef @ unprecondition_matrix
    SDE_noise = precondition_matrix @ SDE_noise

    A, Q = IWPprior.perform_matrix_fraction_decomposition(
        SDE_coef, SDE_noise, delta_time
    )

    if (rank := jnp.linalg.matrix_rank(Q)) < state_size * (1 + q):
        print(
            "WARNING: Rank of Q is",
            rank,
            " and should be ",
            state_size * (1 + q),
            ", which is not full rank. Applying 1e-8 diagonal jitter",
        )
        Q = Q + jnp.eye(state_size * (1 + q)) * 1e-8

    if (
        jnp.linalg.matrix_rank(observation_uncertainty)
        < observation_uncertainty.shape[0]
    ):
        observation_uncertainty_chol = jnp.zeros_like(observation_uncertainty)
    else:
        observation_uncertainty_chol = jnp.linalg.cholesky(observation_uncertainty)

    filtered, reverse_parameters = cholk.jax_batch_extended_filter(
        prior_mean=precondition_matrix @ initial_mean,
        prior_cholcov=jnp.diag(jnp.sqrt(initial_cov_diag)),
        observation_function=lambda state, time, step: observation_function(
            unprecondition_matrix @ state, time, step
        ),
        Ch_cond_obs=observation_uncertainty_chol,
        A_cond_state=A,
        b_cond_state=jnp.zeros(state_size * (1 + q)),
        Ch_cond_state=jnp.linalg.cholesky(Q),
        update_indicator=update_indicator,
        n_solution_points=n_solution_points,
        delta_time=delta_time,
    )

    last_filtered = cholk.CholGauss(filtered.mean[-1], filtered.chol_cov[-1])
    smooth_means, smooth_chol_covs, samples = cholk.jax_batch_smooth_and_sample(
        last_filtered,
        reverse_parameters,
        n_samples=n_samples,
    )

    # print(smooth_chol_covs.shape)s
    # print(smooth_means.shape, last_filtered.mean.shape)
    # smooth_means = jnp.concatenate(
    #     [smooth_means[1:], last_filtered.mean.reshape(1, -1)], axis=0
    # )
    # smooth_chol_covs = jnp.concatenate(
    #     [
    #         smooth_chol_covs[1:],
    #         last_filtered.chol_cov.reshape(
    #             1, state_size * (1 + q), state_size * (1 + q)
    #         ),
    #     ],
    #     axis=0,
    # )

    # print(smooth_means)

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
    transformed_samples = jnp.einsum("tiq,kq->tik", samples, unprecondition_matrix)
    transformed_mean = jnp.einsum("ij,kj->ki", unprecondition_matrix, smooth_means)
    transformed_std = jnp.einsum("ij,kj->ki", unprecondition_matrix, stds)
    return transformed_samples, transformed_mean, transformed_std
