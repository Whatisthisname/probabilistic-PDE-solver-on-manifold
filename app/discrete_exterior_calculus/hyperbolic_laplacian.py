from functools import partial
from discrete_exterior_calculus import DEC
import jax.numpy as jnp
import jax
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

jax.config.update("jax_enable_x64", True)


# @jax.jit
def hyperbolic_distance(a, b):
    if jnp.linalg.norm(a) > 1e-5:
        anorm = jnp.linalg.norm(a)
        a = (a / anorm) * (1 - (anorm - 1) ** 2) ** 0.1
    if jnp.linalg.norm(b) > 1e-5:
        bnorm = jnp.linalg.norm(b)
        b = (b / bnorm) * (1 - (bnorm - 1) ** 2) ** 0.1

    dist = jnp.arccosh(
        1
        + 2
        * jnp.sum((a - b) ** 2, axis=1)
        / ((1 - jnp.sum(a**2, axis=1)) * (1 - jnp.sum(b**2, axis=1)))
    )
    return dist


def different_distance(a, b):
    dist = (
        jnp.exp(
            0.5
            * jnp.arccosh(
                1
                + 2
                * jnp.sum((a - b) ** 2, axis=1)
                / ((1 - jnp.sum(a**2, axis=1)) * (1 - jnp.sum(b**2, axis=1)))
            )
        )
        - 1
    ) * jnp.exp(jnp.linalg.norm(a - b, axis=1))
    return dist


@jax.jit
def distance(a, b):
    return jnp.linalg.norm(a - b, axis=1)


def get_triangle_area_and_interior_angles_from_lengths(tris):
    # Compute the area of each triangle
    a = tris[:, 2 + 1]
    b = tris[:, 2 + 2]
    c = tris[:, 2 + 3]

    s = (a + b + c) / 2
    befsqrt = s * (s - a) * (s - b) * (s - c)
    areas = jnp.sqrt(befsqrt)

    # Compute the interior cotangent of each triangle
    cot_a = (b**2 + c**2 - a**2) / (4 * areas)
    cot_b = (a**2 + c**2 - b**2) / (4 * areas)
    cot_c = (a**2 + b**2 - c**2) / (4 * areas)

    return areas, jnp.stack([cot_a, cot_b, cot_c], axis=1)


# @jax.jit
def get_triangle_area_and_interior_angles_from_distance_function_and_positions(
    tris, vertices, distance_func
):
    length_triples = distance_func(
        vertices[tris[:, [1, 0, 0]]].reshape(len(tris) * 3, 2),
        vertices[tris[:, [2, 2, 1]]].reshape(len(tris) * 3, 2),
    )
    length_triples = length_triples.reshape(len(tris), 3)

    # Compute the area of each triangle
    a = length_triples[:, 0]
    b = length_triples[:, 1]
    c = length_triples[:, 2]
    s = (a + b + c) / 2

    areas = jnp.sqrt(s * (s - a) * (s - b) * (s - c))

    # Compute the interior cotangent of each triangle
    cot_a = (b**2 + c**2 - a**2) / (4 * areas)
    cot_b = (a**2 + c**2 - b**2) / (4 * areas)
    cot_c = (a**2 + b**2 - c**2) / (4 * areas)

    return areas, jnp.stack([cot_a, cot_b, cot_c], axis=1)


def compute_star0(areas, tri_map):
    """
    Computes the Hodge star operator on 0-forms, holds the dual areas of each vertex
    """
    star0_diagonal = np.array([np.sum(areas[np.array(tris)]) / 3.0 for tris in tri_map])
    return sps.diags(star0_diagonal)


def compute_star1_circ(faces, edges, cotangents):
    edge_to_idx_map = {}
    for i, (a, b) in enumerate(edges):
        edge_to_idx_map[(a, b)] = i

    A = np.zeros((len(edges)), dtype=float)
    for (a, b, c), (cot_a, cot_b, cot_c) in zip(faces, cotangents):
        A[edge_to_idx_map[DEC.edge(a, b)]] += cot_c * 0.5
        A[edge_to_idx_map[DEC.edge(b, c)]] += cot_a * 0.5
        A[edge_to_idx_map[DEC.edge(c, a)]] += cot_b * 0.5

    return np.diag(A)


def compute_neg_laplacian(mesh, faces_and_lengths):
    areas, cotangents = get_triangle_area_and_interior_angles_from_lengths(
        np.array(faces_and_lengths).reshape(-1, 6)
    )

    star0 = compute_star0(areas, mesh.tri_map)

    star1_circ = compute_star1_circ(mesh.faces, mesh.edges, cotangents)

    d0 = mesh.compute_d0()
    laplacian = -spsla.spsolve(star0, d0.T @ star1_circ @ d0)
    return laplacian


if __name__ == "__main__":
    name = "small-disc"
    mesh = DEC.Mesh.from_obj(f"../meshes/{name}.obj", lazy=False)

    vertices = jnp.array(mesh.vertices[:, [0, 2]])

    areas, cotangents = (
        get_triangle_area_and_interior_angles_from_distance_function_and_positions(
            mesh.faces, vertices, hyperbolic_distance
        )
    )

    star0 = compute_star0(areas, mesh.tri_map)
    star1_circ = compute_star1_circ(mesh.faces, mesh.edges, cotangents)
    print(jnp.isnan(areas).any())
    # exit()

    import scipy.sparse.linalg as spsla

    print(np.max(np.abs((star0 - mesh.star0.todense()))))
    print(np.max(np.abs((star1_circ - mesh.star1_circ))))

    laplacian = -spsla.spsolve(star0, mesh.d0.T @ star1_circ @ mesh.d0)

    print(np.max(np.abs((laplacian - mesh.laplace_matrix))))

    import numpy as np
    from app.discrete_exterior_calculus import DEC
    from app.probabilistic_numerics import kalman_filter
    from traditional_numerics import heat_solver as hs
    import scipy.sparse as sps

    print("jnp.max(laplacian)")
    print(jnp.max(laplacian))
    boundary_nodes = mesh.boundary_mask.copy()

    n = len(mesh.vertices)

    boundary_values = np.sin(3 * np.arctan2(mesh.vertices[:, 0], mesh.vertices[:, 2]))

    derivatives = 2

    initial_value = np.zeros(n * (derivatives + 1))
    above_mask = (mesh.vertices[:, 1] > 0).astype(float) * 0 + 1
    distance_to_center = np.linalg.norm(mesh.vertices[:, [0, 2]] * [1.5, 3], axis=1)
    rbf = np.exp(-3 * (distance_to_center**2)) * 1

    initial_value[:n] = above_mask * rbf
    initial_value[:n][mesh.boundary_mask] = boundary_values[mesh.boundary_mask]
    # initial_value[n : 2 * n] = -laplacian @ initial_value[:n]

    timesteps = 10
    delta_time = 0.5

    p_means, p_chol_covs = kalman_filter.PIVP_heat_solve_cholesky(
        laplace_matrix=-laplacian,
        initial_mean=initial_value,
        derivatives=derivatives,
        timesteps=timesteps,
        delta_time=delta_time,
        length_scale=1,
    )

    l1_n_means = hs.numeric_solve_heat_equation(
        initial_condition=initial_value[:n],
        L=sps.csr_matrix(laplacian),
        delta_time=delta_time,
        boundary_nodes=boundary_nodes,
        boundary_values=boundary_values,
        f=np.zeros(n),
        timesteps=timesteps - 1,
    )

    l2_orig_n_means = hs.numeric_solve_heat_equation(
        initial_condition=initial_value[:n],
        L=sps.csr_matrix(-mesh.laplace_matrix),
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
            "Numerical Heat Equation, hyperbolic laplace": {
                "data": l2_orig_n_means,
                "start": 0,
                "end": 1,
            },
            "Numerical Heat Equation, euclidean laplace": {
                "data": l1_n_means,
                "start": 0,
                "end": 1,
            },
            "Absolute Difference": {
                "data": np.abs(l2_orig_n_means - l1_n_means),
                "start": 0,
                "end": 1,
            },
            "areas": np.diagonal(mesh.star0.toarray()),
        },
    )
    print("saved as", newname)
