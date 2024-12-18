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
return_times = jnp.linspace(0, end_time, 1500, endpoint=True)


def solve(
    mesh: DEC.Mesh,
    n_solution_points: int,
    derivatives: int,
    prior_type: Literal["heat", "wave", "iwp"],
    vector_field: callable,
    order: int,
):
    assert derivatives >= order
    n = len(mesh.vertices)
    ymost_point = jnp.argmax(mesh.vertices[:, 1])
    yleast_point = jnp.argmin(mesh.vertices[:, 1])

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

    delta_time = end_time / (n_solution_points - 1)  # -1 because we start at 0
    solution_times = jnp.linspace(0, end_time, n_solution_points, endpoint=True)
    initial_cov_diag = jnp.zeros_like(initial_value)

    _samples, means, stds = kalman_filter.solve_nonlinear_IVP(
        prior_matrix=-mesh.laplace_matrix,
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

    mesh.dump_to_JSON(
        f"experiment_mesh_steps={n_solution_points}_prior={prior_type}_q={q}_order={order}.json",
        {
            "means": {"data": means[:, :n], "start": 0, "end": len(means)},
            "stds": {"data": stds[:, :n], "start": 0, "end": len(stds)},
            "initial": initial_value[:n],
        },
    )

    return means, stds
