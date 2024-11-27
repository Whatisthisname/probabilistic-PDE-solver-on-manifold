import numpy as np
from app.discrete_exterior_calculus.DEC import Mesh
from probabilistic_numerics import heat_kalman
import scipy.sparse as sps
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax


from icosphere import icosphere

# nu:       1   2   3   4    5    6    7    8    9    10
# vertices: 12, 42, 92, 162, 252, 362, 492, 642, 812, 1002
vertices, faces = icosphere(nu=3)

mesh_name = f"icosphere_v={len(vertices)}"
# mesh = Mesh.from_obj(f"meshes/{mesh_name}.obj")
mesh = mesh = Mesh(vertices, faces)

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

timesteps = 200

delta_time = 0.05
heat_prior = False

O = jnp.zeros((n, n))
I = jnp.eye(n)

curvature_matrix = jnp.block([-mesh.laplace_matrix, O, O])
value = jnp.block([I, O, O])
solution = jnp.block([I, O, O])
first_time_derivative = jnp.block([O, I, O])
second_time_derivative = jnp.block([O, O, I])

# get the eigenvalues and vectors of the laplace matrix
# and sort them in descending order
eigenvalues, eigenvectors = jnp.linalg.eigh(mesh.laplace_matrix)
idx = jnp.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx].T.real


initial_value[:n] = eigenvectors[15] * 5


def non_linear_observation_function(state, time):
    return -jnp.tanh((curvature_matrix @ state)) - second_time_derivative @ state


import time

start = time.time()

update_indicator = jnp.tile(jnp.array([True] + [True] * 9), timesteps // 10)
timesteps = len(update_indicator)

p_means, p_stds = heat_kalman.solve_nonlinear_IVP(
    laplace_matrix=-mesh.laplace_matrix,
    initial_mean=initial_value,
    derivatives=derivatives,
    timesteps=timesteps,
    delta_time=delta_time,
    heat_prior=heat_prior,
    observation_function=non_linear_observation_function,
    update_indicator=update_indicator,
    length_scale=0.001,
)


print("Time taken:", jnp.round(time.time() - start, 2))

newname = f"nonlinear_{'heat_prior' if heat_prior else ''}_{mesh_name}.json"
print("Saving to", newname)
mesh.dump_to_JSON(
    newname,
    {
        "Initial Value": initial_value[:n],
        "Probabilistic Heat Equation": {
            "data": p_means[:, :n],
            "start": 0,
            "end": 1,
        },
        "Probabilistic Standard Dev.": {
            "data": (p_stds[:, :n]),
            "start": 0,
            "end": 1,
        },
        "eigenvectors": {
            "data": eigenvectors,
            "start": 0,
            "end": 1,
        },
        "areas": np.diagonal(mesh.star0.toarray()),
    },
)


# take some random points on the surface:
idxs = np.random.choice(np.arange(n), 5)

rows = 1
cols = len(idxs)
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

for column in range(cols):
    i = idxs[column]
    ax = axs[column]
    ax.plot(p_means[:, i], label="Process Mean", color="blue")
    ax.scatter(
        np.where(update_indicator)[0],
        np.zeros(update_indicator.sum()),
        label="PDE constraint",
    )
    ax.fill_between(
        np.arange(len(p_means)),
        p_means[:, i] - 3 * p_stds[:, i],
        p_means[:, i] + 3 * p_stds[:, i],
        alpha=0.5,
        color="blue",
        label=r"$\pm3\sigma$",
    )
    # ax.plot(n_means[:, i], label="numerical solution", linestyle="--", color="black")
    ax.legend()
    ax.set_ylim(-2, 2)
plt.show()
