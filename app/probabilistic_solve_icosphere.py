from typing import Literal
import jax
import jax.numpy as jnp
from discrete_exterior_calculus import DEC
from icosphere import icosphere
from jax import config
import numpy as np
from probabilistic_numerics import heat_kalman

config.update("jax_enable_x64", True)

end_time = 10
return_times = jnp.linspace(0, end_time, 100)

first_order_problems = ["heat", "heat and tanh", "heat small tanh"]
second_order_problems = ["wave", "wave and tanh"]


def solve(
    isosphere_nu: int,
    timesteps: int,
    derivatives: int,
    prior: Literal["heat", "wave", "iwp"],
    problem: str,
):
    # nu:       1   2   3   4    5    6    7    8    9    10
    # vertices: 12, 42, 92, 162, 252, 362, 492, 642, 812, 1002
    vertices, faces = icosphere(nu=isosphere_nu)
    n = len(vertices)

    mesh = DEC.Mesh(vertices, faces)
    ymost_point = jnp.argmax(vertices[:, 1])
    yleast_point = jnp.argmin(vertices[:, 1])

    q = derivatives

    O = jnp.zeros((n, n))
    I = jnp.eye(n)

    E_0 = jnp.block([I] + [O] * q)
    E_1 = jnp.block([O] * 1 + [I] + [O] * (q - 1))
    E_2 = jnp.block([O] * 2 + [I] + [O] * (q - 2))
    E_3 = jnp.block([O] * 3 + [I] + [O] * (q - 3))
    E_4 = jnp.block([O] * 4 + [I] + [O] * (q - 4))

    if problem == "heat and tanh":

        def f(u):
            return jnp.tanh(mesh.laplace_matrix @ u) + mesh.laplace_matrix @ u

    if problem == "heat small tanh":

        def f(u):
            return 0.1 * jnp.tanh(mesh.laplace_matrix @ u) + mesh.laplace_matrix @ u

    if problem == "heat":

        def f(u):
            return mesh.laplace_matrix @ u

    if problem == "wave":

        def f(u):
            return mesh.laplace_matrix @ u

    if problem == "wave and tanh":

        def f(u):
            return mesh.laplace_matrix @ u + jnp.tanh(mesh.laplace_matrix @ u)

    if problem in second_order_problems:

        def non_linear_observation_function(state, time, step):
            return (f(E_0 @ state)) - E_2 @ state

    if problem in first_order_problems:

        def non_linear_observation_function(state, time, step):
            return (f(E_0 @ state)) - E_1 @ state

    initial_value = np.zeros(n * (q + 1))
    initial_value[ymost_point] = 2.0
    initial_value[yleast_point] = -2.0

    if True:
        if problem in first_order_problems:
            if q >= 1:
                initial_value[n : 2 * n] = f(initial_value[:n])
            if q >= 2:
                J_f = jax.jacfwd(f)(initial_value[:n])
                initial_value[2 * n : 3 * n] = J_f @ f(initial_value[:n])
            if q >= 3:
                H_f = jax.jacfwd(jax.jacfwd(f))(initial_value[:n])
                first_term = jnp.tensordot(H_f, f(initial_value[:n]), axes=1) @ f(
                    initial_value[:n]
                )
                second_term = J_f @ (J_f @ f(initial_value[:n]))
                initial_value[3 * n : 4 * n] = first_term + second_term
            if q >= 4:
                raise AssertionError("q >= 4 not supported, numerically unstable")
        if problem in second_order_problems:
            if q >= 2:
                initial_value[2 * n : 3 * n] = f(initial_value[:n])
            if q >= 3:
                J_f = jax.jacfwd(f)(initial_value[:n])
                initial_value[3 * n : 4 * n] = J_f @ f(initial_value[:n])
            if q >= 4:
                raise AssertionError("q >= 4 not supported, numerically unstable")

    delta_time = end_time / timesteps
    solution_times = jnp.linspace(0, end_time, timesteps, endpoint=False)

    update_indicator = jnp.ones(timesteps, dtype=bool)
    update_indicator = update_indicator.at[0].set(False)
    # update_indicator = update_indicator.at[timesteps // 2 :].set(False)

    _samples, kalman_sol, u_std = heat_kalman.solve_nonlinear_IVP(
        prior_matrix=mesh.laplace_matrix,
        initial_mean=initial_value,
        derivatives=derivatives,
        timesteps=timesteps,
        delta_time=delta_time,
        prior=prior,
        observation_function=non_linear_observation_function,
        update_indicator=update_indicator,
        observation_uncertainty=jnp.zeros((n, n)),
        n_samples=0,
    )

    interpolated_means = jax.vmap(
        lambda v: jnp.interp(return_times, solution_times, v)
    )(kalman_sol.T).T
    interpolated_stds = jax.vmap(lambda v: jnp.interp(return_times, solution_times, v))(
        u_std.T
    ).T
    return interpolated_means, interpolated_stds
