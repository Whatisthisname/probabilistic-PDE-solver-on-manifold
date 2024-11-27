import jax
import jax.numpy as jnp
import numpy as np
from app.discrete_exterior_calculus.DEC import Mesh
from probabilistic_numerics import heat_kalman as hk
from traditional_numerics import heat_solver as hs
import scipy.sparse as sps
import matplotlib.pyplot as plt


mesh_name = "sphere_small"
mesh = Mesh.from_obj(f"meshes/{mesh_name}.obj")

boundary_nodes = mesh.boundary_mask.copy()

np.random.seed(0)

n = len(mesh.vertices)

boundary_values = np.zeros(n)

derivatives = 2

initial_value = np.zeros(n * (derivatives + 1))
above_mask = (mesh.vertices[:, 1] > 0).astype(float)
distance_to_center = np.linalg.norm(mesh.vertices[:, [0, 2]] * [1.5, 3], axis=1)
rbf = np.exp(-3 * (distance_to_center**2)) * 1
initial_value[:n] = above_mask * rbf
initial_value = jnp.array(initial_value)


timesteps = 175
timesteps = 185
timesteps = 190
timesteps = 191
# timesteps = 200
# timesteps = 192 # breaks, numerically unstable
# timesteps = 193 # also breaks

# timesteps = 50

obs = jnp.ones(timesteps).astype(bool)
delta_time = 0.05
pde_type = "wave"
OU = False

compiled = jax.jit(
    hk.PIVP_heat_solve_dense,
    static_argnames=(
        "derivatives",
        "use_heat_prior",
        "PDE",
    ),
)

p_means, p_covs = compiled(
    laplace_matrix=-mesh.laplace_matrix,
    initial_mean=initial_value,
    derivatives=derivatives,
    delta_time=delta_time,
    observation_indicator=obs,
    use_heat_prior=OU,
    noise_scale=1,
    PDE=pde_type,
)


@jax.jit
def get_sd_from_cov(full_state_covs):
    return jnp.sqrt(
        jnp.maximum(jnp.diagonal(full_state_covs[:, :n], axis1=1, axis2=2), 0)
    )


eigenval, eigenvec = jnp.linalg.eig(mesh.laplace_matrix)
idx = np.argsort(eigenval.real)[::-1]
eigenvec = eigenvec[:, idx].T[:30].real


print("done, starting to write to .json")

newname = f"{pde_type}{'_OU' if OU else ''}_{mesh_name}_t=0..{timesteps}.json"
mesh.dump_to_JSON(
    newname,
    {
        "Initial Value": initial_value[:n],
        f"Probabilistic {pde_type.capitalize()} Equation": {
            "data": p_means[:, :n],
            "start": 0,
            "end": 1,
        },
        "Probabilistic Standard Dev.": {
            # "data": np.sqrt(np.diagonal(p_covs[:, :n], axis1=1, axis2=2)),
            "data": get_sd_from_cov(p_covs),
            "start": 0,
            "end": 1,
        },
        "eigvectors": {
            "start": 0,
            "end": len(eigenvec) - 1,
            "data": 5 * eigenvec,
        },
        "areas": np.diagonal(mesh.star0.toarray()),
    },
)
print("saved to", newname)

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
    # ax.plot(n_means[:, i], label="numerical solution", linestyle="--", color="black")
    ax.legend()
    ax.set_ylim(-2, 2)
plt.show()
