import numpy as np
from discrete_exterior_calculus.DECMesh import Mesh
import kalman.heat_kalman as hk
from solver import heat_solver as hs
import scipy.sparse as sps
import matplotlib.pyplot as plt

name = "sphere_small"
mesh = Mesh.from_obj(f"meshes/{name}.obj")

boundary_nodes = mesh.boundary_mask.copy()

np.random.seed(0)

n = len(mesh.vertices)

boundary_values = np.zeros(n)
boundary_values[0] = 0

derivatives = 1

initial_value = np.zeros(n * (derivatives + 1))
initial_value[:n] = (mesh.vertices[:, 1] < 0.3).astype(float)
initial_value[:n] = np.sign(np.random.rand(n) - 0.5)

timesteps = 40

stepsizes = np.logspace(-4, -2, 10, endpoint=True)
obs = np.ones(timesteps + 1).astype(bool)

T = 0.4
num_steps = 10000
num_dt = T / num_steps

n_means = hs.numeric_solve_heat_equation(
    initial_condition=initial_value[:n],
    L=sps.csr_matrix(mesh.laplace_matrix),
    delta_time=num_dt,
    boundary_nodes=boundary_nodes,
    boundary_values=boundary_values,
    f=np.zeros(n),
    timesteps=num_steps,
)

n_domain = np.arange(0, num_steps + 1) * num_dt

data = []

for j, delta_time in enumerate(stepsizes):
    print(f"Step {j + 1}/{len(stepsizes)}")

    probnum_dt = delta_time
    probnum_steps = int(T / probnum_dt)
    print(f"probnum steps: {probnum_steps}")

    obs = np.ones(probnum_steps + 1).astype(bool)

    p_means, _p_covs = hk.PIVP_heat_solve_dense(
        laplace_matrix=-mesh.laplace_matrix,
        initial_mean=initial_value,
        derivatives=derivatives,
        timesteps=sum(obs),
        delta_time=probnum_dt,
        observation_indicator=obs,
        use_heat_prior=False,
        noise_scale=1,
    )

    ou_p_means, _ou_p_covs = hk.PIVP_heat_solve_dense(
        laplace_matrix=-mesh.laplace_matrix,
        initial_mean=initial_value,
        derivatives=derivatives,
        timesteps=sum(obs),
        delta_time=probnum_dt,
        observation_indicator=obs,
        use_heat_prior=True,
        noise_scale=1,
    )

    p_domain = np.arange(0, probnum_steps + 1) * probnum_dt

    print(len(n_domain))
    print(len(n_means[:, 0]))

    ref_point_trajec = np.interp(p_domain, n_domain, n_means[:, 0])

    l_inf_error = np.max(np.abs(ref_point_trajec - p_means[:, 0]))
    ou_l_inf_error = np.max(np.abs(ref_point_trajec - ou_p_means[:, 0]))

    data.append((delta_time, l_inf_error, ou_l_inf_error))


data = np.array(data)

# plotting the error on a log_log scale
plt.figure()
plt.loglog(
    data[:, 0], data[:, 1], label="Probabilistic Solver, spatially independent prior"
)
plt.loglog(
    data[:, 0],
    data[:, 2],
    label="Probabilistic Solver, Ornstein-Uhlenbeck prior",
)
plt.xlabel("Timestep size")
plt.ylabel("L_inf error")
plt.legend()
plt.show()
