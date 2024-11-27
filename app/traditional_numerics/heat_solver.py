import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import numpy as np


def modify_poisson_system(L, f, boundary_nodes, boundary_values):
    L = L.tocsr()  # Convert L to CSR format for efficient row operations
    f_modified = f.copy()

    for i in boundary_nodes:
        # Zero out the i-th row in L
        L.data[L.indptr[i] : L.indptr[i + 1]] = 0
        # Set the diagonal element to 1
        L[i, i] = 1
        # Set the corresponding entry in f to the boundary value

    return L, f_modified


def solve_poisson_equation(L, source, boundary_nodes, boundary_values):
    source = source.copy()
    source[boundary_nodes] = boundary_values[boundary_nodes]
    L_modified = matrix_to_identity_at_boundary(L, boundary_nodes)
    # print("rank of L:", np.linalg.matrix_rank(L_modified.toarray()))
    # print("shape of L:", L_modified.shape)
    # L and f have been modified so the solution will ensure the boundary conditions are satisfied
    u = spla.spsolve(L_modified, source)

    return u


### HEAT EQUATION


def numeric_solve_heat_equation(
    initial_condition, L, delta_time, boundary_nodes, boundary_values, f, timesteps
):
    n = len(initial_condition)
    u_n = initial_condition.copy()
    u_n[boundary_nodes] = boundary_values[boundary_nodes]

    # Modify the system matrix once (since boundary conditions are time-independent)
    I = sparse.identity(n, format="csr")  # Identity matrix
    A = I - delta_time * L  # System matrix for Backward Euler
    # Convert A to CSR format for efficient row operations
    A = A.tocsr()

    # Precompute h * f if f is time-independent
    hf = delta_time * f

    steps = [u_n.copy()]
    for _ in range(timesteps):
        # Construct right-hand side
        b = u_n.copy()

        # For interior nodes, add source term
        b[~boundary_nodes] += hf[~boundary_nodes]

        # For boundary nodes, enforce boundary conditions
        b[boundary_nodes] = boundary_values[boundary_nodes]

        # Solve for the next time step
        u_next = spla.spsolve(A, b)

        steps.append(u_next.copy())
        # Update the solution
        u_n = u_next

    return np.array(steps)


def matrix_to_identity_at_boundary(M, boundary_nodes):
    # Modify rows corresponding to boundary nodes
    for i, is_boundary in enumerate(boundary_nodes):
        if not is_boundary:
            continue
        # Zero out the i-th row in A
        M.data[M.indptr[i] : M.indptr[i + 1]] = 0
        # Set the diagonal element to 1
        M[i, i] = 1
    return M
