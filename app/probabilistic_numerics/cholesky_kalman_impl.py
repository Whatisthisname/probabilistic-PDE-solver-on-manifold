from collections import namedtuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp

CholGauss = namedtuple("CholGauss", ["mean", "chol_cov"])

ReverseParams = namedtuple(
    "reverse_params", ["A_rev_list", "b_rev_list", "Ch_rev_list"]
)

LikelihoodMarginalParams = namedtuple(
    "LikelihoodMarginalParams", ["A_rev", "b_rev", "Ch_rev", "m_out", "Ch_out"]
)
"""
Holds parameters for the likelihood and marginal distribution of the state.
Specifically, from P(X) and P(Y|X), we computed P(X|Y) and P(Y) where

P(X|Y) = N(`A_rev`*Y + `b_rev`, `Ch_rev`)
P(Y) = N(`m_out`, `Ch_out`)
"""


def chol_marginal_and_reverse(
    m_in, Ch_in, A_cond, b_cond, Ch_cond
) -> LikelihoodMarginalParams:
    """
    P(X) = N(m_in, Ch_in)
    P(Y|X) = N(A_cond @ X + b_cond, Ch_cond)

    Returns P(X|Y) = N(A_rev*Y + b_rev, Ch_rev)
    and P(Y) = N(m_out, Ch_out)
    """
    d_in = Ch_in.shape[0]
    d_out = Ch_cond.shape[0]

    zeros = jnp.zeros((d_out, d_in))
    qr_block = jnp.block(
        [
            [Ch_cond.T, zeros],
            [Ch_in.T @ A_cond.T, Ch_in.T],
        ]
    )
    R = jnp.linalg.qr(mode="r", a=qr_block)
    R_1 = R[:d_out, :d_out]
    R_2 = R[:d_out, -d_in:]
    R_3 = R[-d_in:, -d_in:]

    Ch_rev = R_3.T
    Ch_out = R_1.T
    m_out = A_cond @ m_in + b_cond

    # A_rev = jnp.linalg.solve(R_1, R_2).T  # maybe error
    A_rev = jsp.linalg.solve_triangular(R_1, R_2).T  # maybe error
    b_rev = m_in - A_rev @ m_out

    return LikelihoodMarginalParams(A_rev, b_rev, Ch_rev, m_out, Ch_out)


def chol_marginal_and_reverse_deterministic_conditional(
    m_in, Ch_in, A_cond, b_cond
) -> LikelihoodMarginalParams:
    R = jnp.linalg.qr((A_cond @ Ch_in.T).T, mode="r")
    Ch_out = R.T

    A_rev = jnp.linalg.solve(
        R, jnp.linalg.solve(R.T, (Ch_in @ Ch_in.T @ A_cond.T).T).T
    ).T
    # R_inv = jnp.linalg.inv(R)
    # A_rev = (Ch_in @ Ch_in.T @ A_cond.T) @ R_inv @ R_inv.T

    Ch_rev = Ch_in - A_rev @ A_cond @ Ch_in
    m_out = A_cond @ m_in + b_cond
    b_rev = m_in - A_rev @ m_out

    return LikelihoodMarginalParams(A_rev, b_rev, Ch_rev, m_out, Ch_out)


def get_posterior_and_marginal(
    prior_mean,
    prior_cholvariance,
    A_cond,
    b_cond,
    Ch_cond,
    observation,
) -> CholGauss:
    output = chol_marginal_and_reverse(
        prior_mean,
        prior_cholvariance,
        A_cond,
        b_cond,
        Ch_cond,
    )
    # form X | Y and compute its mean and covariance
    return CholGauss(
        output.A_rev @ observation + output.b_rev,
        output.Ch_rev,
        # output.b_rev,
        # output.Ch_rev,
    ), CholGauss(output.m_out, output.Ch_out)


@jax.jit
def jax_batch_filter(
    prior_mean,
    prior_cholvariance,
    A_cond_obs,
    b_cond_obs,
    Ch_cond_obs,
    A_cond_state,
    b_cond_state,
    Ch_cond_state,
    observations,
) -> tuple[list[CholGauss], ReverseParams]:
    Carry = namedtuple("Carry", ["prior", "output_noise_scale_acc"])

    def loop(carry: Carry, observation):
        filtered_state, observation_marginal = get_posterior_and_marginal(
            carry.prior.mean,
            carry.prior.chol_cov,
            A_cond_obs,
            b_cond_obs,
            Ch_cond_obs,
            observation,
        )
        # computing the probability of the observation given previous observations
        # taking the squared mahalanobis norm
        log_observation_prob = jnp.sum(
            jsp.linalg.solve_triangular(
                observation_marginal.chol_cov.T, observation - observation_marginal.mean
            )
            ** 2
        )

        state_state_dist = chol_marginal_and_reverse(
            filtered_state.mean,
            filtered_state.chol_cov,
            A_cond_state,
            b_cond_state,
            Ch_cond_state,
        )

        predicted_next_state = CholGauss(
            mean=state_state_dist.m_out, chol_cov=state_state_dist.Ch_out
        )

        carry = Carry(
            prior=predicted_next_state,
            output_noise_scale_acc=log_observation_prob + carry.output_noise_scale_acc,
        )
        state = (
            filtered_state,
            ReverseParams(
                A_rev_list=state_state_dist.A_rev,
                b_rev_list=state_state_dist.b_rev,
                Ch_rev_list=state_state_dist.Ch_rev,
            ),
        )
        return carry, state

    (carry, (filtered_states, reverse_parameters)) = jax.lax.scan(
        loop,
        init=Carry(
            prior=CholGauss(prior_mean, prior_cholvariance),
            output_noise_scale_acc=0.0,
        ),
        xs=observations,
    )

    gamma = jnp.sqrt(carry.output_noise_scale_acc / (len(observations) + 2))

    scaled_filtered_states = CholGauss(
        mean=filtered_states.mean, chol_cov=filtered_states.chol_cov * gamma
    )
    scaled_reverse_parameters = ReverseParams(
        A_rev_list=reverse_parameters.A_rev_list,
        b_rev_list=reverse_parameters.b_rev_list,
        Ch_rev_list=reverse_parameters.Ch_rev_list * gamma,
    )

    return (
        scaled_filtered_states,
        scaled_reverse_parameters,
    )


@partial(jax.jit, static_argnames=("observation_function", "timesteps"))
def jax_batch_extended_filter(
    prior_mean,
    prior_cholvariance,
    observation_function,
    Ch_cond_obs,
    A_cond_state,
    b_cond_state,
    Ch_cond_state,
    update_indicator,
    timesteps,
    delta_time,
) -> tuple[list[CholGauss], ReverseParams]:
    Carry = namedtuple("Carry", ["prior", "output_noise_scale_acc"])

    def integrate_observation(prior: CholGauss, time: float, step: int):
        obs_jac = jax.jacobian(observation_function, argnums=0)(prior.mean, time, step)
        filtered_state, observation_marginal = get_posterior_and_marginal(
            prior.mean,
            prior.chol_cov,
            obs_jac,
            observation_function(prior.mean, time, step) - obs_jac @ prior.mean,
            Ch_cond_obs,
            observation=jnp.zeros(obs_jac.shape[0]),
        )

        # computing the probability of the observation given previous observations
        # taking the squared mahalanobis norm
        log_observation_prob = jnp.sum(
            jsp.linalg.solve_triangular(
                observation_marginal.chol_cov.T,
                observation_marginal.mean,
            )
            ** 2
        )
        return filtered_state, log_observation_prob

    def loop(carry: Carry, input):
        time, step, update = input

        filtered_state, gamma = jax.lax.cond(
            update,
            integrate_observation,  # <- if true
            lambda prior, time, step: (carry.prior, 0.0),  # <- if false
            *(carry.prior, time, step),
        )

        state_state_dist = chol_marginal_and_reverse(
            filtered_state.mean,
            filtered_state.chol_cov,
            A_cond_state,
            b_cond_state,
            Ch_cond_state,
        )

        predicted_next_state = CholGauss(
            mean=state_state_dist.m_out, chol_cov=state_state_dist.Ch_out
        )

        carry = Carry(
            prior=predicted_next_state,
            output_noise_scale_acc=carry.output_noise_scale_acc + gamma,
        )
        state = (
            filtered_state,
            ReverseParams(
                A_rev_list=state_state_dist.A_rev,
                b_rev_list=state_state_dist.b_rev,
                Ch_rev_list=state_state_dist.Ch_rev,
            ),
        )
        return carry, state

    (carry, (filtered_states, reverse_parameters)) = jax.lax.scan(
        loop,
        init=Carry(
            prior=CholGauss(prior_mean, prior_cholvariance),
            output_noise_scale_acc=0.0,
        ),
        xs=(
            jnp.arange(timesteps) * delta_time,  # time
            jnp.arange(timesteps),  # step
            update_indicator,  # update
        ),
    )

    gamma = jnp.sqrt(carry.output_noise_scale_acc / (jnp.sum(update_indicator) + 2))

    filtered_states: CholGauss
    # filtered_states.chol_cov.at[:].multiply(gamma)
    # reverse_parameters.Ch_rev_list.at[:].multiply(gamma)

    scaled_filtered_states = CholGauss(
        mean=filtered_states.mean, chol_cov=filtered_states.chol_cov * gamma
    )
    scaled_reverse_parameters = ReverseParams(
        A_rev_list=reverse_parameters.A_rev_list,
        b_rev_list=reverse_parameters.b_rev_list,
        Ch_rev_list=reverse_parameters.Ch_rev_list * gamma,
    )

    return (
        scaled_filtered_states,
        scaled_reverse_parameters,
    )


@partial(jax.jit, static_argnames=("n_samples"))
def jax_batch_smooth_and_sample(
    last_filtered_state: CholGauss,
    reverse_parameters: ReverseParams,
    n_samples: int,
) -> tuple[list[CholGauss]]:
    Carry = namedtuple("Carry", ["next_state_smoothed", "last_sample"])

    noise = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(  # (T, n_samples, state_dim)
            reverse_parameters.A_rev_list.shape[0],
            n_samples,
            last_filtered_state.mean.shape[0],
        ),
    )

    def loop(carry: Carry, rev_params):
        rev_params, noise = rev_params
        smooth_state = chol_marginal_and_reverse(
            carry.next_state_smoothed.mean,
            carry.next_state_smoothed.chol_cov,
            rev_params.A_rev_list,
            rev_params.b_rev_list,
            rev_params.Ch_rev_list,
        )

        t_sample = (
            jnp.einsum("ij, kj -> ki", rev_params.A_rev_list, carry.last_sample)
            + rev_params.b_rev_list
            + jnp.einsum("ij, kj -> ki", rev_params.Ch_rev_list, noise)
        )

        next_state_smoothed = CholGauss(
            mean=smooth_state.m_out, chol_cov=smooth_state.Ch_out
        )

        carry = Carry(next_state_smoothed=next_state_smoothed, last_sample=t_sample)
        output = next_state_smoothed.mean, next_state_smoothed.chol_cov, t_sample

        return carry, output

    T_sample = last_filtered_state.mean + jnp.einsum(
        "ij, kj -> ki", last_filtered_state.chol_cov, noise[0]
    )
    # ^^^multiply each sample by the cholesky factor

    (_carry, (means, chol_covs, samples)) = jax.lax.scan(
        loop,
        init=Carry(next_state_smoothed=last_filtered_state, last_sample=T_sample),
        xs=(reverse_parameters, noise),
        reverse=True,
    )

    return means, chol_covs, samples
