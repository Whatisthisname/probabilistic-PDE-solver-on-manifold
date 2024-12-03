import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.integrate as inte
import jax
import jax.numpy as jnp
import app.probabilistic_numerics._covariance_kalman_impl as jaxk

rng = jax.random.PRNGKey(0)
grid = 50

space_collocation = jnp.linspace(0, 1, grid)
h = space_collocation[1] - space_collocation[0]

q = 2
"""Amount of derivatives we model"""

# Laplace operator, -∆
neg_laplace = (
    jnp.diag(-2 * jnp.ones(grid))
    + jnp.diag(jnp.ones(grid - 1), 1)
    + jnp.diag(jnp.ones(grid - 1), -1)
) / h**2

forward_diff = (jnp.diag(-1 * jnp.ones(grid)) + jnp.diag(jnp.ones(grid - 1), 1)) / h


O = jnp.zeros((grid, grid))
I = jnp.eye(grid)
L = neg_laplace
F = forward_diff

# for boundary conditions: u(∂) = 0
OIO = jnp.eye(grid)
OIO = OIO.at[0, 0].set(0)
OIO = OIO.at[-1, -1].set(0)

value_matrix = jnp.block([I, O, O])
curvature_matrix = jnp.block([L, O, O])
grad_matrix = jnp.block([F, O, O])
time_derivative_matrix = jnp.block([O, I, O])
time_2derivative_matrix = jnp.block([O, O, I])

SDE_coef = jnp.block(
    [
        [O, I, O],
        [O, O, I],
        [O, O, O],
    ]
)

SDE_noise = jnp.block(
    [
        [O, O, O],
        [O, O, O],
        [O, O, I],
    ]
)


F, Q = jaxk.get_discrete_system_coeffs(SDE_coef, SDE_noise, 0.005)

wave_pde_error_matrix = (
    curvature_matrix - time_2derivative_matrix - time_derivative_matrix * 0.5
)
wave_pde_error_matrix = wave_pde_error_matrix.at[0, :].set(0)
wave_pde_error_matrix = wave_pde_error_matrix.at[grid - 1, :].set(0)
wave_pde_error_matrix = wave_pde_error_matrix.at[0, 0].set(1)
wave_pde_error_matrix = wave_pde_error_matrix.at[grid - 1, grid - 1].set(1)

heat_pde_error_matrix = 0.03 * curvature_matrix - time_derivative_matrix
heat_pde_error_matrix = heat_pde_error_matrix.at[0, :].set(0)
heat_pde_error_matrix = heat_pde_error_matrix.at[grid - 1, :].set(0)
heat_pde_error_matrix = heat_pde_error_matrix.at[0, 0].set(1)
heat_pde_error_matrix = heat_pde_error_matrix.at[grid - 1, grid - 1].set(1)

moving_wave_pde_error_matrix = grad_matrix - time_2derivative_matrix * 0.5
moving_wave_pde_error_matrix = moving_wave_pde_error_matrix.at[0, :].set(0)
moving_wave_pde_error_matrix = moving_wave_pde_error_matrix.at[grid - 1, :].set(0)
moving_wave_pde_error_matrix = moving_wave_pde_error_matrix.at[0, 0].set(1)
moving_wave_pde_error_matrix = moving_wave_pde_error_matrix.at[grid - 1, grid - 1].set(
    1
)

poisson_equation_error_matrix = curvature_matrix
poisson_equation_error_matrix = poisson_equation_error_matrix.at[0, :].set(0)
poisson_equation_error_matrix = poisson_equation_error_matrix.at[grid - 1, :].set(0)
poisson_equation_error_matrix = poisson_equation_error_matrix.at[grid // 2, :].set(0)
poisson_equation_error_matrix = poisson_equation_error_matrix.at[0, 0].set(1)
poisson_equation_error_matrix = poisson_equation_error_matrix.at[
    grid - 1, grid - 1
].set(1)
poisson_equation_error_matrix = poisson_equation_error_matrix.at[
    grid // 2, grid // 2
].set(1)


R = jnp.zeros(
    (wave_pde_error_matrix.shape[0], wave_pde_error_matrix.shape[0])
)  # + 1e-4 * jnp.eye(pde_error_matrix.shape[0])


initial = jnp.zeros(grid * (1 + q))
initial = initial.at[grid // 2].set(1)
initial = initial.at[:grid].set(jnp.sin(3 * jnp.pi * space_collocation))

cov = jnp.zeros((grid * (1 + q), grid * (1 + q)))

H_indices = jnp.zeros(100, dtype=jnp.int32)
R_indices = jnp.zeros(100, dtype=jnp.int32)
H_indices = H_indices.at[0].set(1)
H_indices = H_indices.at[-1].set(1)

observations = jnp.ones((2, grid))

# H_list = np.concatenate([pde_error_matrix, position_observation_matrix], axis=0)

# filter_means, filter_covs, pred_means, pred_covs = jaxk.batch_filter_variable_observation_matrix(F, Q, H_indices, R_indices, H_list, R_list, initial, cov, observations)
filter_means, filter_covs, pred_means, pred_covs = jaxk.batch_filter(
    F, Q, poisson_equation_error_matrix, R, initial, cov, observations
)
smooth_means, smooth_covs = jaxk.batch_smooth(
    F, filter_means, filter_covs, pred_means, pred_covs
)


cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for z, i in enumerate(range(len(observations))):
    d = 1 + 1 - (z + 1) / len(observations)
    dist_scale = 1 / d
    domain = jnp.linspace((1 - dist_scale) / 2, 1 - (1 - dist_scale) / 2, grid)
    mean = smooth_means[i] * dist_scale
    std = jnp.sqrt(jnp.diagonal(smooth_covs[i])) * dist_scale
    offset = -1 * i / len(observations)
    ax[0].plot(domain, offset + mean[:grid], color=cmap(i / len(observations)))
    ax[0].fill_between(
        domain,
        offset + mean[:grid] - 3 * std[:grid],
        offset + mean[:grid] + 3 * std[:grid],
        alpha=0.5 + 0.5 * (i / len(observations)),
        color=cmap(i / len(observations)),
    )

plt.show()
