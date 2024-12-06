from typing import Literal
import jax
import jax.numpy as jnp
from discrete_exterior_calculus import DEC
from icosphere import icosphere
from jax import config
import numpy as np
from probabilistic_numerics import kalman_filter

config.update("jax_enable_x64", True)

end_time = 10
return_times = jnp.linspace(0, end_time, 100)

first_order_problems = ["heat", "heat and tanh", "heat small tanh", "heat and tanh u"]
second_order_problems = ["wave", "wave and tanh"]


def solve(
    isosphere_nu: int,
    n_solution_points: int,
    derivatives: int,
    prior_type: Literal["heat", "wave", "iwp"],
    prior_scale: float,
    vector_field: callable,
    order: int,
):
    assert derivatives >= order
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
    E_1 = jnp.block([O] + [I] + [O] * (q - 1))
    E_2 = jnp.block([O, O] + [I] + [O] * (q - 2))

    initial_value = jnp.zeros(n * (q + 1))
    initial_value = initial_value.at[ymost_point].set(1.0)
    initial_value = initial_value.at[yleast_point].set(-1.0)

    from probdiffeq.taylor import odejet_padded_scan

    if order == 1:

        def non_linear_observation_function(state, time, step):
            return vector_field(E_0 @ state, mesh.laplace_matrix) - E_1 @ state

        def taylor_vector_field(y):
            return vector_field(y, mesh.laplace_matrix)

        tcoeffs = odejet_padded_scan(taylor_vector_field, (E_0 @ initial_value,), num=q)
        del taylor_vector_field

        initial_value = jnp.array(tcoeffs).flatten()
        initial_cov_diag = jnp.zeros_like(initial_value)

    if order == 2:

        def non_linear_observation_function(state, time, step):
            return (
                vector_field(E_0 @ state, E_1 @ state, mesh.laplace_matrix)
                - E_2 @ state
            )

        def taylor_vector_field(y, dy):
            return vector_field(y, dy, mesh.laplace_matrix)

        tcoeffs = odejet_padded_scan(
            taylor_vector_field,
            (E_0 @ initial_value, E_1 @ initial_value),
            num=q - 1,
        )
        del taylor_vector_field

        initial_value = jnp.array(tcoeffs).flatten()
        initial_cov_diag = jnp.zeros_like(initial_value)

    delta_time = end_time / (n_solution_points - 1)  # -1 because we start at 0
    solution_times = jnp.linspace(0, end_time, n_solution_points, endpoint=True)

    # print("now")
    # print(non_linear_observation_function(initial_value, 0, 0))
    # print(initial_value)
    # print(initial_cov_diag)

    # initial_value = jnp.zeros_like(initial_value)
    # initial_cov_diag = jnp.zeros_like(initial_cov_diag)

    # def non_linear_observation_function(state, time, step):
    #     return E_0 @ state - E_1 @ state

    # print(non_linear_observation_function(initial_value, 0, 0))

    _samples, kalman_sol, u_std = kalman_filter.solve_nonlinear_IVP(
        prior_matrix=-mesh.laplace_matrix * prior_scale,
        initial_mean=initial_value,
        initial_cov_diag=initial_cov_diag,
        derivatives=q,
        n_solution_points=n_solution_points,
        delta_time=delta_time,
        prior_type=prior_type,
        observation_function=non_linear_observation_function,
        update_indicator=jnp.ones(n_solution_points, dtype=bool),
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
