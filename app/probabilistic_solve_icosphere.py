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


def solve(
    isosphere_nu: int,
    timesteps: int,
    derivatives: int,
    prior: Literal["heat", "wave", "iwp"],
):
    # nu:       1   2   3   4    5    6    7    8    9    10
    # vertices: 12, 42, 92, 162, 252, 362, 492, 642, 812, 1002
    vertices, faces = icosphere(nu=isosphere_nu)
    n = len(vertices)

    mesh = DEC.Mesh(vertices, faces)
    ymost_point = jnp.argmax(vertices[:, 1])

    q = derivatives

    O = jnp.zeros((n, n))
    I = jnp.eye(n)

    E_0 = jnp.block([I] + [O] * q)
    E_1 = jnp.block([O] * 1 + [I] + [O] * (q - 1))
    E_2 = jnp.block([O] * 2 + [I] + [O] * (q - 2))
    E_3 = jnp.block([O] * 3 + [I] + [O] * (q - 3))
    E_4 = jnp.block([O] * 4 + [I] + [O] * (q - 4))

    def non_linear_observation_function(state, time):
        return jnp.tanh((mesh.laplace_matrix @ E_0 @ state)) - E_2 @ state

    initial_value = np.zeros(n * (q + 1))
    initial_value[ymost_point] = 1
    initial_value[2 * n : 3 * n] = jnp.tanh((mesh.laplace_matrix @ initial_value[:n]))

    if False:
        if q > 2:
            init = (
                jax.jacobian(lambda x: jnp.tanh((mesh.laplace_matrix @ x)))(
                    initial_value[:n]
                )
                @ initial_value[:n]
            )
            initial_value[3 * n : 4 * n] = init
        if q > 3:
            # Compute Jacobian of f(u)
            J = jax.jacobian(lambda x: jnp.tanh((mesh.laplace_matrix @ x)))(
                initial_value[:n]
            )
            # Compute Hessian of f(u), which is the Jacobian of the Jacobian
            H = jax.jacobian(
                jax.jacobian(lambda x: jnp.tanh((mesh.laplace_matrix @ x)))
            )(initial_value[:n])
            # Compute first derivative
            u_t = initial_value[2 * n : 3 * n]
            # Compute second derivative
            u_tt = initial_value[3 * n : 4 * n]
            # Tensor contraction for the first term: (H @ u_t) @ u_t
            first_term = jnp.einsum("ijk,j,k->i", H, u_t, u_t)
            # Second term: J @ u_tt
            second_term = jnp.dot(J, u_tt)
            # Third derivative

            initial_value[4 * n : 5 * n] = first_term + second_term

    delta_time = end_time / timesteps
    solution_times = jnp.linspace(0, end_time, timesteps)

    update_indicator = jnp.ones(timesteps, dtype=bool)
    update_indicator = update_indicator.at[0].set(False)

    kalman_sol, u_std = heat_kalman.solve_nonlinear_IVP(
        laplace_matrix=-mesh.laplace_matrix,
        initial_mean=initial_value,
        derivatives=derivatives,
        timesteps=timesteps,
        delta_time=delta_time,
        prior=prior,
        observation_function=non_linear_observation_function,
        update_indicator=update_indicator,
    )

    interpolated_means = jax.vmap(
        lambda v: jnp.interp(return_times, solution_times, v)
    )(kalman_sol.T).T
    interpolated_stds = jax.vmap(lambda v: jnp.interp(return_times, solution_times, v))(
        u_std.T
    ).T
    return interpolated_means, interpolated_stds
