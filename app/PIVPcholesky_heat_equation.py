import numpy as np
from discrete_exterior_calculus import DECMesh
from probabilistic_numerics import heat_kalman
from traditional_numerics import heat_solver as hs
import scipy.sparse as sps

name = "sphere_small"
mesh = DECMesh.Mesh.from_obj(f"meshes/{name}.obj")

boundary_nodes = mesh.boundary_mask.copy()

n = len(mesh.vertices)

boundary_values = np.zeros(n)
boundary_values[0] = 0

derivatives = 2

initial_value = np.zeros(n * (derivatives + 1))
above_mask = (mesh.vertices[:, 1] > 0).astype(float)
distance_to_center = np.linalg.norm(mesh.vertices[:, [0, 2]] * [1.5, 3], axis=1)
rbf = np.exp(-3 * (distance_to_center**2)) * 1
initial_value[:n] = above_mask * rbf

timesteps = 1000
delta_time = 0.005

p_means, p_chol_covs = heat_kalman.PIVP_heat_solve_cholesky(
    laplace_matrix=-mesh.laplace_matrix,
    initial_mean=initial_value,
    derivatives=derivatives,
    timesteps=timesteps,
    delta_time=delta_time,
    length_scale=1,
)


n_means = hs.numeric_solve_heat_equation(
    initial_condition=initial_value[:n],
    L=sps.csr_matrix(mesh.laplace_matrix),
    delta_time=delta_time,
    boundary_nodes=boundary_nodes,
    boundary_values=boundary_values,
    f=np.zeros(n),
    timesteps=timesteps - 1,
)


newname = f"chol_{name}, dt={delta_time}, t={timesteps}, heat.json"
mesh.dump_to_JSON(
    newname,
    {
        "Initial Value": initial_value[:n],
        "Probabilistic Heat Equation": {
            "data": p_means[:, :n],
            "start": 0,
            "end": 1,
        },
        # "Probabilistic Variance": {
        #     "data": np.diagonal(p_chol_covs[:, :n], axis1=1, axis2=2),
        #     "start": 0,
        #     "end": 1,
        # },
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
print("saved as", newname)
