import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


@jax.jit
def get_discrete_system_coeffs(SDE_coef, SDE_noise, time_step):
    A = jax.scipy.linalg.expm(SDE_coef * time_step)

    outer = SDE_noise @ SDE_noise.T

    def integrand(t):
        expm = jax.scipy.linalg.expm(SDE_coef * t)
        return expm @ outer @ expm.T

    points = jnp.linspace(0, time_step, 10)
    evaluations = jax.vmap(integrand)(points)

    Q = jax.scipy.integrate.trapezoid(y=evaluations, x=points, axis=0)

    return A, Q


# random number generator
rng = jax.random.PRNGKey(1)


def filter(
    prev_mean,
    prev_cov,
    A,
    Q,
    H,
    R,
    new_observation,
):
    # predictive mean:
    pred_mean = A @ prev_mean
    # predictive covariance:
    pred_cov = A @ prev_cov @ A.T + Q
    # observation residual:
    resid = new_observation - H @ pred_mean
    # observation covariance / innovation covariance:
    innov_cov = H @ pred_cov @ H.T + R
    # Kálmán gain:
    gain = pred_cov @ H.T @ jnp.linalg.inv(innov_cov)

    filter_mean = pred_mean + gain @ resid
    filter_cov = (jnp.eye(H.shape[1]) - gain @ H) @ pred_cov

    return (filter_mean, filter_cov), (pred_mean, pred_cov)


@jax.jit
def batch_smooth(A, filter_means, filter_covs, pred_means, pred_covs):
    def rts_smoother(
        filter_mean,
        filter_cov,
        A,
        next_pred_mean,
        next_pred_cov,
        next_smooth_mean,
        next_smooth_cov,
    ):
        G = filter_cov @ A.T @ jnp.linalg.inv(next_pred_cov)
        m = filter_mean + G @ (next_smooth_mean - next_pred_mean)
        P = filter_cov + G @ (next_smooth_cov - next_pred_cov) @ G.T
        return m, P

    def rts_body(
        smoothed_mean_and_cov: tuple[None, None],
        filter_and_preds: tuple[tuple[None, None], tuple[None, None]],
    ):
        next_smooth_mean, next_smooth_cov = smoothed_mean_and_cov
        (filter_mean, filter_cov), (pred_mean, pred_cov) = filter_and_preds
        smooth_mean, smooth_cov = rts_smoother(
            filter_mean,
            filter_cov,
            A,
            pred_mean,
            pred_cov,
            next_smooth_mean,
            next_smooth_cov,
        )
        return (smooth_mean, smooth_cov), (smooth_mean, smooth_cov)

    pair = ((filter_means[:-1], filter_covs[:-1]), (pred_means[1:], pred_covs[1:]))

    _, smoothed = jax.lax.scan(
        f=rts_body,
        init=(filter_means[-1], filter_covs[-1]),
        xs=pair,
        reverse=True,
    )
    smooth_means, smooth_covs = smoothed
    smooth_means = jnp.concatenate(
        [smooth_means, jnp.expand_dims(filter_means[-1], axis=0)], axis=0
    )
    smooth_covs = jnp.concatenate(
        [smooth_covs, jnp.expand_dims(filter_covs[-1], axis=0)], axis=0
    )
    return (smooth_means, smooth_covs)


def step(state, noise):
    return A @ state + noise, A @ state + noise


@jax.jit
def batch_filter(A, Q, H, R, mean, cov, observations):
    """Batch filter a sequence of observations

    Args:
        A : jnp.array
            State evolution matrix, shape (state_dim, state_dim)
        Q : jnp.array
            State noise covariance, shape (state_dim, state_dim)
        H : jnp.array
            Observation model matrix, shape (obs_dim, state_dim)
        R : jnp.array
            Observation noise covariance, shape (obs_dim, obs_dim)
        mean : jnp.array
            Initial state mean, shape (state_dim,)
        cov : jnp.array
            Initial state covariance, shape (state_dim, state_dim)
        observations : jnp.array
            Observations, shape (n_timesteps, obs_dim)

    Returns:
        filter_means : jnp.array
            Filtered state means, shape (n_timesteps, state_dim)

        filter_covs : jnp.array
            Filtered state covariances, shape (n_timesteps, state_dim, state_dim)

        pred_means : jnp.array
            Predicted state means, shape (n_timesteps, state_dim)

        pred_covs : jnp.array
            Predicted state covariances, shape (n_timesteps, state_dim, state_dim)
    """

    def body_fun(mean_and_cov, obs):
        curr_mean, curr_cov = mean_and_cov
        (new_mean, new_cov), (pred_mean, pred_cov) = filter(
            curr_mean, curr_cov, A, Q, H, R, obs
        )
        return (new_mean, new_cov), ((new_mean, new_cov), (pred_mean, pred_cov))

    _, ((filter_means, filter_covs), (pred_means, pred_covs)) = jax.lax.scan(
        f=body_fun, init=(mean, cov), xs=observations
    )

    return filter_means, filter_covs, pred_means, pred_covs


@jax.jit
def batch_filter_variable_observation_matrix(
    A, Q, Hlist, Rlist, Hindicator, Rindicator, mean, cov, observations
):
    """Batch filter a sequence of observations

    Args:
        A : jnp.array
            State evolution matrix, shape (state_dim, state_dim)
        Q : jnp.array
            State noise covariance, shape (state_dim, state_dim)
        H : jnp.array
            Observation model matrix, shape (obs_dim, state_dim)
        R : jnp.array
            Observation noise covariance, shape (obs_dim, obs_dim)
        mean : jnp.array
            Initial state mean, shape (state_dim,)
        cov : jnp.array
            Initial state covariance, shape (state_dim, state_dim)
        observations : jnp.array
            Observations, shape (n_timesteps, obs_dim)

    Returns:
        filter_means : jnp.array
            Filtered state means, shape (n_timesteps, state_dim)

        filter_covs : jnp.array
            Filtered state covariances, shape (n_timesteps, state_dim, state_dim)

        pred_means : jnp.array
            Predicted state means, shape (n_timesteps, state_dim)

        pred_covs : jnp.array
            Predicted state covariances, shape (n_timesteps, state_dim, state_dim)
    """

    def body_fun(mean_and_cov, h_and_r_and_obs):
        curr_mean, curr_cov = mean_and_cov
        h_ind, r_ind, obs = h_and_r_and_obs
        (new_mean, new_cov), (pred_mean, pred_cov) = filter(
            curr_mean, curr_cov, A, Q, Hlist[h_ind], Rlist[r_ind], obs
        )
        return (new_mean, new_cov), ((new_mean, new_cov), (pred_mean, pred_cov))

    _, ((filter_means, filter_covs), (pred_means, pred_covs)) = jax.lax.scan(
        f=body_fun, init=(mean, cov), xs=(Hindicator, Rindicator, observations)
    )

    return filter_means, filter_covs, pred_means, pred_covs


if __name__ == "__main__":
    A = jnp.array([[0, 1, 0], [-0.3, -0.4, 1], [0, 0, -0.1]])
    A = jnp.array([[0, 1, 0], [-1, -0.1, 1], [0, 0, -0.1]])
    """State evolution"""

    Q = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1.0]])
    """State noise covariance"""

    A, Q = get_discrete_system_coeffs(A, Q, 1.0)

    H = jnp.array([[1.0, 0, 0.0]])
    # H = jnp.eye(3)
    """Observation model"""
    R = jnp.array([[0.1]])
    """Observation model noise covariance"""

    timesteps = jnp.arange(0, 10)
    Qnoise = (
        jax.random.multivariate_normal(
            rng, mean=jnp.zeros(A.shape[0]), cov=Q, shape=(timesteps.shape[0],)
        )
        * 1.0
    )

    mean = jnp.array([2, 0, 0.0])
    cov = jnp.array([[0, 0, 0.0], [0, 0, 0], [0, 0, 0]])

    _, states = jax.lax.scan(f=step, init=mean, xs=Qnoise)

    observations = jax.lax.map(lambda x: H @ x, states)

    observations.at[30].set(2.0)

    filter_means, filter_covs, pred_means, pred_covs = batch_filter(
        A, Q, H, R, mean, cov, observations
    )

    (smooth_means, smooth_covs) = batch_smooth(
        A, filter_means, filter_covs, pred_means, pred_covs
    )

    filter_stds = jnp.sqrt(jnp.diagonal(filter_covs, axis1=1, axis2=2))
    smooth_stds = jnp.sqrt(jnp.diagonal(smooth_covs, axis1=1, axis2=2))

    fig, ax = plt.subplots((4), figsize=(10, 7))

    names = ["position", "velocity", "acceleration"]
    for i in range(3):
        filter_mean = filter_means[:, i]
        smooth_mean = smooth_means[:, i]

        filter_std = filter_stds[:, i] * 3
        smooth_std = smooth_stds[:, i] * 3

        ax[i + 1].set_title(f"Predicted {names[i]}")
        ax[i + 1].plot(timesteps, filter_mean, label="predicted", color="red")
        ax[i + 1].plot(timesteps, smooth_mean, label="predicted", color="blue")
        ax[i + 1].plot(timesteps, states[:, i], label="state", color="black")

        # ax[i + 1].fill_between(
        #     timesteps, filter_mean - filter_std, filter_mean + filter_std, label="stds", alpha=0.5, color="red"
        # )
        # ax[i + 1].fill_between(
        #     timesteps, smooth_mean - smooth_std, smooth_mean + smooth_std, label="stds", alpha=0.5, color="blue"
        # )
        # ax[i + 1].legend()
        ax[i + 1].grid()
        ax[i + 1].set_ylim(-3, 3)

    plt.show()
