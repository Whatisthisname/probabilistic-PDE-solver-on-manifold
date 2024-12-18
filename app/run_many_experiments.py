import jax
import jax.numpy as jnp
from icosphere import icosphere
import probabilistic_solve_on_mesh
import probabilistic_solve_icosphere
from discrete_exterior_calculus import DEC
import shelve
import itertools
from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController, Kvaerno5


from persistent_storage import (
    # get_value,
    # set_value,
    # remove_value,
    # wipe_db,
    experiment_setup,
    build_experiment_name,
)


jax.config.update("jax_enable_x64", True)

# problem_names = experiment_setup.keys()
problem_names = ["heat and tiny square", "heat"]

for problem_name in problem_names:
    try:
        print("Running experiments for problem:", problem_name)
        data = experiment_setup[problem_name]
        priors, derivatives, timesteps, problem_title, vf, order, domain = (
            data["priors"],
            data["derivatives"],
            data["timesteps"],
            data["problem_title"],
            data["vector_field"],
            data["order"],
            data["domain"],
        )

        # wipe the database
        with shelve.open(f"./dbs/{problem_name}", "c", writeback=True) as f:
            f.clear()

        if jnp.min(jnp.asarray(derivatives)) < order:
            raise ValueError(
                f"Derivatives {derivatives} must be at least {order} for order {order} method"
            )

        if domain == "icosphere":
            # nu:       1   2   3   4    5    6    7    8    9    10
            # vertices: 12, 42, 92, 162, 252, 362, 492, 642, 812, 1002
            nu = 1
            vertices, faces = icosphere(nu=nu)
            n = len(vertices)
            mesh = DEC.Mesh(vertices, faces)
            laplacian = mesh.laplace_matrix

        else:
            import potpourri3d as pp3d

            mesh = DEC.Mesh.from_obj(f"./meshes/{domain}.obj")
            V, F = pp3d.read_mesh(
                f"/Users/theoruterwurtzen/Desktop/MSc Thesis/code/meshes/{domain}.obj"
            )
            # initialize the glue map and edge lengths arrays from the input data
            import intrinsic_triang as inttri

            G = inttri.build_gluing_map(F)
            l = inttri.build_edge_lengths(V, F)

            inttri.flip_to_delaunay(F, G, l)

            laplacian = jnp.asarray(inttri.build_cotan_laplacian(F, l).toarray())

            n = len(mesh.vertices)
            vertices = mesh.vertices

        zmost_point = jnp.argmax(vertices[:, 2])
        zleast_point = jnp.argmin(vertices[:, 2])
        xmost_point = jnp.argmax(vertices[:, 0])
        xleast_point = jnp.argmin(vertices[:, 0])
        ymost_point = jnp.argmax(vertices[:, 1])
        yleast_point = jnp.argmin(vertices[:, 1])

        def calc_diffrax_sol(steps: int):
            if order == 1:

                def vector_field(t, y, args):
                    return vf(y, laplacian)

            if order == 2:

                def vector_field(t, y, args):
                    return jnp.concatenate(
                        (y[n : 2 * n], vf(y[:n], y[n : 2 * n], laplacian))
                    )

            u0 = jnp.zeros(n)
            u0 = u0.at[ymost_point].set(1.0)
            u0 = u0.at[yleast_point].set(-1.0)

            if order == 1:
                y0 = u0
            if order == 2:
                y0 = jnp.concatenate([u0, jnp.zeros(n)])

            # Solve the system
            sol = diffeqsolve(
                ODETerm(vector_field),
                Kvaerno5(),
                t0=0,
                t1=10,
                dt0=0.01,
                y0=y0,
                saveat=SaveAt(ts=jnp.linspace(0, 10, steps, endpoint=True)),
                stepsize_controller=PIDController(rtol=1e-8, atol=1e-8),
                max_steps=50000,
            )

            diffrax_sol = sol.ys[:, zleast_point]  # Displacement solutions over time
            return diffrax_sol

        import time
        import numpy as np
        import tqdm

        product = list(itertools.product(priors, derivatives, timesteps))
        iter = tqdm.tqdm(product)

        for prior, q, timestep in iter:
            experiment_name = build_experiment_name(prior, q, timestep)
            iter.set_description(f"Running experiment: {experiment_name}")

            means, stds, runtime, rmse, diff = [None] * 5
            with shelve.open("./dbs/" + problem_name, "r") as f:
                if experiment_name in f:
                    means, stds, runtime, rmse, diff = f[experiment_name]

            if means is None:
                fastest_time = 1e9
                for _ in range(1):
                    start_time = time.time()
                    if domain == "icosphere":
                        means, stds = probabilistic_solve_icosphere.solve(
                            isosphere_nu=nu,
                            n_solution_points=timestep,
                            derivatives=q,
                            prior_type=prior,
                            vector_field=vf,
                            order=order,
                        )
                    else:
                        means, stds = probabilistic_solve_on_mesh.solve(
                            mesh=mesh,
                            n_solution_points=timestep,
                            derivatives=q,
                            prior_type=prior,
                            vector_field=vf,
                            order=order,
                        )
                    try:
                        means = means[:, zleast_point]
                        stds = stds[:, zleast_point]
                    except Exception as err:
                        print(f"Experiment {experiment_name} failed because of {err}")
                    end_time = time.time()
                    if end_time - start_time < fastest_time:
                        fastest_time = end_time - start_time

                diffrax_sol = calc_diffrax_sol(timestep)

                diff = means - diffrax_sol
                rmse = jnp.sqrt(jnp.mean(diff[:-1] ** 2))
                means = means.astype(np.float32)
                stds = stds.astype(np.float32)

                with shelve.open("dbs/" + problem_name, "w") as f:
                    f[experiment_name] = (
                        means.astype(np.float32),
                        stds.astype(np.float32),
                        fastest_time,
                        rmse,
                        diff,
                    )

            else:
                continue
    except Exception as err:
        print(f"Experiment {experiment_name} failed because of {err}")
        continue
