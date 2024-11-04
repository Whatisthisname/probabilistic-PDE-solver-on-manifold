import jax
import jax.numpy as jnp


def get_IWP_Prior_SDE_coefficients(
    *, size: int, derivatives: int
) -> tuple[jax.numpy.array, jax.numpy.array]:
    q = derivatives
    transition = jnp.diag(jnp.ones(q * size), k=size)
    noise = jnp.diag(jnp.concatenate((jnp.zeros(size * q), jnp.ones(size))))
    return transition, noise


if __name__ == "__main__":
    print(get_IWP_Prior_SDE_coefficients(size=3, derivatives=2))
