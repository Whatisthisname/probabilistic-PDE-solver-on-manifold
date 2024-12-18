import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def get_IWP_Prior_SDE_coefficients(
    *, size: int, derivatives: int
) -> tuple[jax.numpy.array, jax.numpy.array]:
    q = derivatives
    transition = jnp.diag(jnp.ones(q * size), k=size)
    noise = jnp.diag(jnp.concatenate((jnp.zeros(size * q), jnp.ones(size))))
    return transition, noise


@jax.jit
def perform_matrix_fraction_decomposition(SDE_coef, SDE_noise, delta_time):
    state = SDE_coef.shape[0]

    blocked = jnp.block(
        [[SDE_coef, SDE_noise], [jnp.zeros((state, state)), -SDE_coef.T]]
    )

    expd = jax.scipy.linalg.expm(blocked * delta_time)

    A = expd[:state, :state]

    # solution = expd @ jnp.block([[jnp.zeros((state, state))], [jnp.eye(state)]])
    # Q = solution[:state, :state] @ A.T
    Q = expd[:state, state:] @ A.T

    return A, Q  # 0.5 * (Q + Q.T)


if __name__ == "__main__":
    print(get_IWP_Prior_SDE_coefficients(size=3, derivatives=2))
