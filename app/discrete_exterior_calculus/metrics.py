from functools import partial
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy.integrate as inte


def euclidean(point: jnp.ndarray):
    return jnp.eye(2)


def sauron_metric(point: jnp.ndarray):
    ydiff = 0.5 + 0.5 * (point[0] ** 2)
    xdiff = 1
    e1 = jnp.array([1, 0])
    e2 = jnp.array([0, 1])
    return jnp.column_stack((xdiff * e1, ydiff * e2))


def sheet_metric(point: jnp.ndarray):
    ydiff = 0.2
    xdiff = 1
    e1 = jnp.array([1, 0])
    e2 = jnp.array([0, 1])
    return jnp.column_stack((xdiff * e1, ydiff * e2))


def hyperbolic_metric(point: jnp.ndarray):
    norm_squared = jnp.sum(point**2)  # * 2
    # norm_squared = jnp.clip(norm_squared, 0.0, 0.9)
    magnitude = 1 / (1 - norm_squared) ** 2
    # magnitude = 0.1 + (norm_squared)
    return jnp.eye(2) * magnitude


def semi_hyperbolic_metric(point: jnp.ndarray):
    norm_squared = jnp.sum(point**2) * 2
    # norm_squared = jnp.clip(norm_squared, 0.0, 0.9)
    # magnitude = 1 / (1 - norm_squared) ** 2
    magnitude = 0.1 + (norm_squared)
    return jnp.eye(2) * magnitude


def spacecollapse(point: jnp.ndarray):
    # scale = jnp.array([[0.1, 0], [0, 0.4]])
    scale = jnp.array([[0.05, 0], [0, 1.3]]) * 0.5
    return (
        100
        * jnp.eye(2)
        * jnp.clip(
            (
                1
                - jnp.exp(
                    -((jnp.linalg.norm(point @ jnp.linalg.inv(scale) @ point)) ** 2)
                )
            ),
            0.5,
            1,
        )
    )


def hardcollapse(point: jnp.ndarray):
    scale = jnp.array([[0.2, 0], [0, 0.7]]) * 0.3

    return jnp.eye(2) * jax.lax.cond(
        jnp.linalg.norm(point @ jnp.linalg.inv(scale) @ point) < 1,
        lambda: 0.05,
        lambda: 1.0,
    )


def spaceexpand(point: jnp.ndarray):
    scale = jnp.array([[0.03, 0], [0, 0.8]]) * 0.5
    return jnp.eye(2) * (
        1
        + 10
        * (jnp.exp(-((jnp.linalg.norm(point @ jnp.linalg.inv(scale) @ point)) ** 2)))
    )


def vertical_stretch(point: jnp.ndarray):
    return jnp.array([[0.5**2, 0], [0, 1**2]])


def embed_to_slope(point):
    return jnp.array([point[0], point[1], jnp.sqrt(2) * point[1]])


# Example Usage
# @jax.jit
def projected_slope_metric(point):
    jacobian = jax.jacobian(embed_to_slope)

    return jacobian(point).T @ jacobian(point)

    # return jax.lax.cond(
    #     jnp.linalg.norm(point) == 0,
    #     lambda: jnp.eye(2),
    #     lambda: ,
    # )


def embed_to_bell(point):
    scale = jnp.array([[0.1, 0.1], [0.1, 0.8]]) * 0.5
    z = jnp.exp(-((jnp.linalg.norm(point @ jnp.linalg.inv(scale) @ point)) ** 2))
    return jnp.array([point[0], point[1], z])


# Example Usage
# @jax.jit
def projected_bell_metric(point):
    jacobian = jax.jacobian(embed_to_bell)

    return jax.lax.cond(
        jnp.linalg.norm(point) == 0,
        lambda: jnp.eye(2),
        lambda: jacobian(point).T @ jacobian(point),
    )


@jax.jit
def _length_element(t, g1, g2, p1, p2):
    diff = p1 - p2
    return jnp.sqrt(diff.T @ g1 @ diff * (1 - t) + diff @ g2 @ diff * t)


# @jax.jit
def _real_length_element(t, p1, p2, metric):
    diff = p1 - p2
    g = metric((1 - t) * p1 + t * p2)
    return jnp.sqrt(diff.T @ g @ diff)


@partial(jax.jit, static_argnums=(2,))
def measure_distance(p1, p2, metric):
    g1 = metric(p1)
    g2 = metric(p2)
    n = 20
    # length_elements = jax.vmap(lambda t: _length_element(t, g1, g2, p1, p2))(
    #     jnp.linspace(0, 1, n)
    # )
    length_elements = jax.vmap(lambda t: _real_length_element(t, p1, p2, metric))(
        jnp.linspace(0, 1, n)
    )
    return inte.trapezoid(y=length_elements, dx=1 / (n - 1))
