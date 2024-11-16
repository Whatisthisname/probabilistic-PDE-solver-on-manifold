import numpy as np
from discrete_exterior_calculus.DECMesh import Mesh
import kalman.heat_kalman as hk
from solver import heat_solver as hs
import scipy.sparse as sps
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

mesh_name = "sphere_small"
mesh = Mesh.from_obj(f"meshes/{mesh_name}.obj")

boundary_nodes = mesh.boundary_mask.copy()

np.random.seed(0)

n = len(mesh.vertices)

boundary_values = np.zeros(n)
boundary_values[0] = 0

derivatives = 2

initial_value = np.zeros(n * (derivatives + 1))
above_mask = (mesh.vertices[:, 1] > 0).astype(float)
distance_to_center = np.linalg.norm(mesh.vertices[:, [0, 2]] * [1.5, 3], axis=1)
rbf = np.exp(-3 * (distance_to_center**2)) * 1
initial_value[:n] = above_mask * rbf


timesteps = 175
timesteps = 185
timesteps = 190
timesteps = 191
# timesteps = 192 # breaks, unstable numerically
# timesteps = 193 # also breaks


obs = np.zeros(timesteps).astype(bool)
obs[:] = True


delta_time = 0.05
pde_type = "wave"
OU = False

# pde: u_tt = -u_xx * u


O = jnp.zeros((n, n))
I = jnp.eye(n)
curvature_matrix = jnp.block([-mesh.laplace_matrix, O, O])
value = jnp.block([I, O, O])
first_time_derivative = jnp.block([O, I, O])
second_time_derivative = jnp.block([O, O, I])


# H @ u
# H(u)


# def non_linear_observation_function(state):
#    return curvature_matrix @ state - first_time_derivative @ state

# u_t = -u∆u


def non_linear_observation_function(state):
    return (E_0 @ state) * (curvature_matrix @ state) - first_time_derivative @ state


p_means, p_covs = hk.PIVP_solve_dense_non_linear(
    laplace_matrix=-mesh.laplace_matrix,
    initial_value=initial_value,
    derivatives=derivatives,
    timesteps=sum(obs),
    delta_time=delta_time,
    observation_indicator=obs,
    ornstein_uhlenbeck_prior=OU,
    noise_scale=1,
    PDE=pde_type,
    nonlinear_observation_function=non_linear_observation_function,
)


n_means = hs.numeric_solve_heat_equation(
    initial_condition=initial_value[:n],
    L=sps.csr_matrix(mesh.laplace_matrix),
    delta_time=delta_time,
    boundary_nodes=boundary_nodes,
    boundary_values=boundary_values,
    f=np.zeros(n),
    timesteps=timesteps,
)


mesh.dump_to_JSON(
    f"{pde_type}{'_OU' if OU else ''}_{mesh_name}.json",
    {
        "Initial Value": initial_value[:n],
        "Probabilistic Heat Equation": {
            "data": p_means[:, :n],
            "start": 0,
            "end": 1,
        },
        "Probabilistic Standard Dev.": {
            # "data": np.sqrt(np.diagonal(p_covs[:, :n], axis1=1, axis2=2)),
            "data": np.sqrt(
                np.maximum(np.diagonal(p_covs[:, :n], axis1=1, axis2=2), 0)
            ),
            "start": 0,
            "end": 1,
        },
        "Numerical Heat Equation": {
            "data": n_means,
            "start": 0,
            "end": 1,
        },
        "Absolute Difference": {
            "data": np.abs(p_means[:, :n] - n_means),
            "start": 0,
            "end": 1,
        },
        "areas": np.diagonal(mesh.star0.toarray()),
    },
)


# take 4 random points on the surface:
idxs = np.random.choice(np.arange(n), 5)

rows = 1
cols = len(idxs)
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

for column in range(cols):
    i = idxs[column]
    ax = axs[column]
    ax.plot(p_means[:, i], label="Process Mean", color="blue")
    ax.scatter(np.where(obs)[0], np.zeros(obs.sum()), label="PDE constraint")
    ax.fill_between(
        np.arange(len(p_means)),
        p_means[:, i] - 3 * np.sqrt(np.maximum(p_covs[:, i, i], 0)),
        p_means[:, i] + 3 * np.sqrt(np.maximum(p_covs[:, i, i], 0)),
        alpha=0.5,
        color="blue",
        label=r"$\pm3\sigma$",
    )
    ax.plot(n_means[:, i], label="numerical solution", linestyle="--", color="black")
    ax.legend()
    ax.set_ylim(-2, 2)
plt.show()
