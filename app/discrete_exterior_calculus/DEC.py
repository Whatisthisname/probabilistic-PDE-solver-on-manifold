import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from enum import Enum
import json
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import trimesh
import jax.numpy as jnp


class GridPattern(Enum):
    HEX = 1
    RECT = -1


def edge(a, b):
    return (min(a, b), max(a, b))


class Mesh:
    def __init__(self, vertices=None, faces=None, lazy=False):
        """
        Initialize the Mesh with given vertices and faces.
        If vertices and faces are not provided, it will generate a default mesh.
        """

        if vertices is None or faces is None:
            # Generate default mesh (e.g., unit square grid)
            self.vertices, self.faces = self.generate_default_mesh()
        else:
            if vertices.shape[1] == 2:
                # If it's a 2D mesh, add an extra coordinate between x and z
                self.vertices = np.column_stack(
                    (vertices[:, 0], np.zeros(vertices.shape[0]), vertices[:, 1])
                )
            else:
                self.vertices = np.array(vertices)
            self.faces = np.array(faces)

        self.boundary_mask = self.find_boundary_vertices()
        """Holds 1 for boundary vertices and 0 for interior vertices"""
        self.edges = self.compute_edges()
        """All pairs (a,b) of edges, ordered so a < b"""
        self.edge_map = self.compute_edge_map()
        """Mapping from each vertex to its connected vertices"""
        self.tri_map = self.compute_tri_map()
        """Mapping from each vertex to the triangles it belongs to, sort of the opposite of self.faces"""
        self.opposing_vertices = self.compute_opposing_vertices()
        """Map from edge to the (at most 2) vertices orthogonal to it"""
        self.d0 = self.compute_d0()
        self.d1 = self.compute_d1()

        if not lazy:
            self.recompute()

    def recompute(self):
        """
        Recomputes all mesh properties.
        """

        self.areas = self.compute_areas()
        self.lengths = self.compute_edge_lengths()
        self.star0 = self.compute_star0()
        self.star1_circ = self.compute_star1_circ()
        self.star1_bary = self.compute_star1_bary()
        self.laplace_matrix = -spsla.spsolve(
            self.star0, self.d0.T @ self.star1_circ @ self.d0
        )

    @staticmethod
    def generate_default_mesh(n=25, gridpattern=GridPattern.HEX):
        """
        Generates a default mesh (rectangular or hexagonal) for testing purposes.
        """
        if gridpattern == GridPattern.RECT:
            x = np.linspace(-1, 1, n)
            z = np.linspace(-1, 1, n)
            xv, zv = np.meshgrid(x, z)
            vertices = np.column_stack((xv.ravel(), np.zeros(n * n), zv.ravel()))
            triang = mtri.Triangulation(vertices[:, 0], vertices[:, 2])
            faces = triang.triangles
        elif gridpattern == GridPattern.HEX:
            # # Generate hexagonal grid
            # x = np.arange(-n, n)
            # z = np.arange(-n, n)
            # xv, zv = np.meshgrid(x, z)
            # xv = xv + (zv % 2) * 0.5
            # xv, zv = xv / n, zv / n
            # vertices = np.column_stack((xv.ravel(), np.zeros(xv.size), zv.ravel()))
            # triang = mtri.Triangulation(vertices[:, 0], vertices[:, 2])
            # faces = triang.triangles

            def get_hex_points(columns):
                long_row = np.linspace(-1, 1, columns)
                dist = 2 / (columns - 1) / 2
                small_row = np.linspace(-1 + dist, 1 - dist, columns - 1)
                height = dist * 2 * np.sqrt(3) / 2
                vertices = []
                borders = []
                rows = int(np.ceil(2 / (height)))
                for row in range(rows):
                    odd = row % 2 == 0
                    height_coord = (-1 + (height * row)) * np.ones_like(
                        small_row if odd else long_row
                    )
                    to_add = np.vstack((small_row if odd else long_row, height_coord)).T
                    vertices.extend(to_add)
                    if row != 0 and row != rows - 1:
                        borders.extend([True] + [False] * (len(to_add) - 2) + [True])
                    else:
                        borders.extend([True] * len(to_add))
                return np.array(vertices), np.array(borders)

            vertices, _border = get_hex_points(n)
            vertices = np.column_stack(
                (vertices[:, 0], np.zeros(vertices.shape[0]), vertices[:, 1])
            )
            triang = mtri.Triangulation(vertices[:, 0], vertices[:, 2])
            faces = triang.triangles

        else:
            raise ValueError("Unknown grid pattern.")
        return vertices, faces

    def compute_edges(self):
        """
        Computes unique edges from faces.
        """
        edges = set()
        for tri in self.faces:
            a, b, c = map(int, tri)
            edges.update(
                {
                    edge(a, b),
                    edge(b, c),
                    edge(c, a),
                }
            )
        return np.array(sorted(edges))

    def compute_edge_map(self):
        """
        Computes a mapping from each vertex to its connected vertices.
        """
        edge_map = [set() for _ in range(len(self.vertices))]
        for a, b in self.edges:
            edge_map[a].add(b)
            edge_map[b].add(a)
        return edge_map

    def compute_tri_map(self):
        """
        Computes a mapping from each vertex to the triangles it belongs to.
        """
        tri_map = [[] for _ in range(len(self.vertices))]
        for i, tri in enumerate(self.faces):
            a, b, c = map(int, tri)
            tri_map[a].append(i)
            tri_map[b].append(i)
            tri_map[c].append(i)
        return tri_map

    def compute_areas(self):
        """
        Computes the area of each triangle using NumPy.
        """
        # Get the coordinates of the triangle vertices
        vertex_coords = self.vertices[self.faces]  # shape (N, 3, D)
        a = vertex_coords[:, 0, :]  # shape (N, D)
        b = vertex_coords[:, 1, :]
        c = vertex_coords[:, 2, :]

        # Compute vectors
        ab = b - a  # shape (N, D)
        ac = c - a  # shape (N, D)

        # Compute cross product
        cross_prod = np.cross(ab, ac)  # shape (N, D)

        # Compute norms of cross product
        norms = np.linalg.norm(cross_prod, axis=1)  # shape (N,)

        # Compute areas
        areas = 0.5 * norms  # shape (N,)

        return areas

    def compute_edge_lengths(self):
        """
        Computes the length of each edge.
        """
        vertices = self.vertices
        edges = self.edges
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        return lengths

    def compute_d0(self):
        """
        Computes the exterior derivative d0.
        """
        num_edges = len(self.edges)
        num_vertices = len(self.vertices)
        row_indices = []
        col_indices = []
        data = []
        for i, (a, b) in enumerate(self.edges):
            row_indices.extend([i, i])
            col_indices.extend([a, b])
            data.extend([-1, 1])
        d0 = sps.coo_matrix(
            (data, (row_indices, col_indices)), shape=(num_edges, num_vertices)
        )
        return d0

    def compute_d1(self):
        """
        Computes the exterior derivative d1.
        """
        num_faces = len(self.faces)
        num_edges = len(self.edges)
        edge_dict = {e: idx for idx, e in enumerate(map(tuple, self.edges))}
        row_indices = []
        col_indices = []
        data = []
        for i, tri in enumerate(self.faces):
            a, b, c = map(int, tri)
            face_edges = [
                (a, b),
                (b, c),
                (c, a),
            ]
            for u, v in face_edges:
                e = edge(u, v)
                edge_idx = edge_dict[e]
                row_indices.append(i)
                col_indices.append(edge_idx)
                # Determine orientation
                orientation = 1 if (u, v) == e else -1
                data.append(orientation)
        d1 = sps.coo_matrix(
            (data, (row_indices, col_indices)), shape=(num_faces, num_edges)
        )
        return d1

    def compute_star0(self):
        """
        Computes the Hodge star operator on 0-forms, holds the dual areas of each vertex
        """
        areas = self.areas
        tri_map = self.tri_map
        star0_diagonal = np.array(
            [np.sum(areas[np.array(tris)]) / 3.0 for tris in tri_map]
        )
        return sps.diags(star0_diagonal).tocsc()

    def compute_star1_circ(self):
        A = np.zeros((len(self.edges), len(self.edges)), dtype=float)
        for i in range(len(self.edges)):
            (a, b) = self.edges[i]

            weights = []
            for oppo in self.opposing_vertices[i]:
                c = self.vertices[oppo]
                ca = self.vertices[a] - c
                ca /= np.linalg.norm(ca)
                cb = self.vertices[b] - c
                cb /= np.linalg.norm(cb)
                angle = np.arccos(np.dot(ca, cb))
                cot = np.cos(angle) / np.sin(angle)
                # print(f"cotangent of edge {a} {b} with vertex {oppo} is {cot}")
                weights.append(cot)

            A[i, i] = 0.5 * sum(weights)
        return A

    def compute_opposing_vertices(
        self,
    ):
        opposing_vertices = [[] for _ in range(len(self.edges))]
        for i, tri in enumerate(self.faces):
            for j in range(3):
                a, b, c = tri[j], tri[(j + 1) % 3], tri[(j + 2) % 3]
                index = np.where((self.edges == edge(a, b)).all(axis=1))[0][0]
                opposing_vertices[index].append(int(c))
        return opposing_vertices

    def compute_star1_bary(self):
        A = np.zeros((len(self.edges), len(self.edges)), dtype=float)
        for i in range(len(self.edges)):
            (a, b) = self.edges[i]

            barycenters = []
            for oppo in self.opposing_vertices[i]:
                c = self.vertices[oppo]
                barycenter = (self.vertices[a] + self.vertices[b] + c) / 3
                barycenters.append(barycenter)
            if len(barycenters) == 2:
                A[i, i] = (
                    np.linalg.norm(barycenters[0] - barycenters[1]) / self.lengths[i]
                )
            if len(barycenters) == 1:
                mp = (self.vertices[a] + self.vertices[b]) / 2
                A[i, i] = 2 * np.linalg.norm(barycenters[0] - mp) / self.lengths[i]
            else:
                A[i, i] = 1
        return A

    def find_boundary_vertices(self):
        """
        Finds the indices of boundary vertices in a mesh using trimesh.

        Parameters:
        - vertices: numpy array of shape (N, 3), where N is the number of vertices.
        - faces: numpy array of shape (M, 3), where M is the number of faces.

        Returns:
        - boundary_vertex_indices: numpy array of bool indicating whether each vertex is on the boundary.
        """
        # Create a mesh object
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

        unique_edges = mesh.edges[
            trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        ]
        # get all vertices in this mesh
        all_vertices = np.unique(unique_edges)
        # sort the vertices
        sorted_vertices = np.sort(all_vertices)

        mask = np.zeros(len(self.vertices), dtype=bool)
        mask[sorted_vertices] = True

        return mask

    def plot_mesh(self, ax=None):
        """
        Plots the mesh using matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots()
        triang = mtri.Triangulation(
            self.vertices[:, 0], self.vertices[:, 2], self.faces
        )
        ax.triplot(triang, color="black", linewidth=0.5)
        ax.set_aspect("equal")

    def dump_to_JSON(self, filename, profiles_dict, folder="produced_solutions"):
        """
        Dumps the mesh and profiles to a JSON file.
        """

        # convert to json serializable format

        serializable_profiles = {}
        for k, v in profiles_dict.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                serializable_profiles[k] = v.round(5).tolist()
            elif isinstance(v, dict):
                serializable_profiles[k] = {
                    "start": v["start"],
                    "end": v["end"],
                    "data": [color.round(5).tolist() for color in v["data"]],
                }

        mesh_data = {
            "vertices": self.vertices.round(5).tolist(),
            "faces": self.faces.tolist(),
            "colorProfiles": serializable_profiles,
        }
        print("here", os.path.join("."))
        with open(os.path.join(folder, filename), "w") as f:
            json.dump(mesh_data, f)

    @classmethod
    def from_obj(cls, filename, lazy=False):
        """
        Loads a mesh from an .obj file.
        """

        import trimesh
        import numpy as np

        # Load your mesh
        mesh = trimesh.load(filename)

        # Set a precision for rounding to identify duplicates (adjust decimals as needed)
        precision = 4

        # Round vertex coordinates to specified precision
        rounded_vertices = mesh.vertices.round(precision)

        # Find unique vertices and get indices for reindexing faces
        unique_vertices, inverse_indices = np.unique(
            rounded_vertices, axis=0, return_inverse=True
        )

        # Check if any vertices were removed
        if len(unique_vertices) != len(mesh.vertices):
            # Reassign vertices and faces only if duplicates were found
            mesh.vertices = unique_vertices
            mesh.faces = inverse_indices[mesh.faces]

            # Update faces by removing duplicates
            mesh.update_faces(mesh.unique_faces())

            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()

        # Prepare vertices array with 3 coordinates
        vertices = np.array(mesh.vertices)
        if (
            vertices.shape[1] == 2
        ):  # if it's a 2D mesh, add an extra coordinate between x and z
            vertices = np.column_stack(
                (vertices[:, 0], np.zeros(vertices.shape[0]), vertices[:, 1])
            )

        return cls(vertices=vertices, faces=np.array(mesh.faces), lazy=lazy)
