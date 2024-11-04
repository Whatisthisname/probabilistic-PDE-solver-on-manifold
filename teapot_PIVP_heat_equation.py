import numpy as np

# sys.path.append("..")
from DECMesh import Mesh
import kalman.heat_kalman as hk

mesh = Mesh.from_obj("meshes/teapot.obj")

boundary_nodes = mesh.boundary_mask.copy()

L_sps = mesh.laplace_matrix

boundary_values = np.zeros(len(mesh.vertices))
boundary_values[0] = 0

source_term = np.zeros(len(mesh.vertices))
source_term = (mesh.vertices[:, 1] < 0.2).astype(float)


means, covs = hk.PIVP_heat_solve(
    laplace_matrix=L_sps,
    initial_value=source_term,
    derivatives=2,
    timesteps=100,
    delta_time=0.1,
)


boundary_nodes[0] = True

mesh.dump_to_JSON(
    "teapot_PIVP.json",
    {
        "Source Term": source_term,
        "Heat Equation": {"data": means, "start": 0, "end": (len(means) - 1) * 0.002},
    },
)
