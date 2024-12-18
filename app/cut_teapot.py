import discrete_exterior_calculus.DEC as DEC

mesh = DEC.Mesh.from_obj("./meshes/torus.obj", lazy=True)

import fast_simplification

points_out, faces_out = fast_simplification.simplify(
    mesh.vertices, mesh.faces, target_count=100
)

fast_simplification.simplify_mesh

new_mesh = DEC.Mesh(points_out, faces_out)
new_mesh.dump_to_JSON("cut_torus.json", profiles_dict={"s": [1] * len(points_out)})

new_mesh.save_to_OBJ("cut_torus.obj")


# from icosphere import icosphere

# # nu:       1   2   3   4    5    6    7    8    9    10
# # vertices: 12, 42, 92, 162, 252, 362, 492, 642, 812, 1002
# nu = 1
# vertices, faces = icosphere(nu=nu)
# n = len(vertices)
# mesh = DEC.Mesh(vertices, faces)
# laplacian = mesh.laplace_matrix

# mesh.save_to_OBJ("icosphereee.obj")
