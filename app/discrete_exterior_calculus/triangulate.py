import numpy as np
from discrete_exterior_calculus import DEC
import networkx as nx


def should_subdivide(l1, l2, l3):
    longest = max(l1, l2, l3)
    shortest = min(l1, l2, l3)
    medium = l1 + l2 + l3 - longest - shortest

    is_triangle = longest < (medium + shortest)

    if not is_triangle:
        return True

    ## TO ENSURE MINIMUM AREA
    # s = (l1 + l2 + l3) / 2
    # area = (s * (s - l1) * (s - l2) * (s - l3)) ** 0.5
    # if area > 0.05:
    #     return

    if longest >= shortest * 2:
        return True

    return False


def tris_and_lengths_to_graph(tris: dict) -> nx.Graph:
    graph = nx.Graph()
    added_edges = set()
    for (a, b, c), (A, B, C) in tris.items():
        for (start, end), length in zip([(a, b), (b, c), (a, c)], [C, A, B]):
            start, end = edge(start, end)
            if (start, end) in added_edges:
                continue

            added_edges.add((start, end))
            graph.add_edge(start, end, weight=length.round(2))
    return graph


def triangulate_mesh_with_edge_distances(mesh: DEC.Mesh, distance_map: dict):
    """Returns new faces and vertices after subdividing the mesh according to the edge lengths in distance_map"""
    opposing_nodes_dict = {}

    for edge_, opposing in zip(mesh.edges, mesh.opposing_vertices):
        a, b = edge(*edge_)
        opposing_nodes_dict[(a, b)] = opposing

    faces = {tuple(face) for face in mesh.faces}
    vertices = list(mesh.vertices[:, [0, 2]])

    data = {}
    for face in faces:
        lengths = []
        for i, j in [(1, 2), (0, 2), (0, 1)]:
            a, b = edge(face[i], face[j])
            lengths.append(distance_map[edge(a, b)])

        add_triangle_with_opposing_sidelengths(data, face, tuple(lengths))

    triangles_to_visit = list(data.keys())
    i = 0

    while triangles_to_visit:
        faces = triangles_to_visit.pop(0)

        if not triangle_exists(data, faces):
            continue

        edgelengths = np.array(get_triangle_with_opposite_sidelengths(data, faces))
        faces = np.array(faces)
        SORT = np.argsort(edgelengths)
        faces = faces[SORT]
        edgelengths = edgelengths[SORT]

        medium_node, short_node, corner_node = faces
        short_side, medium_side, long_side = edgelengths

        # print("node:")
        # print(medium_node, short_node, corner_node)
        # print(short_side, medium_side, long_side)

        will_divide = should_subdivide(short_side, medium_side, long_side)
        # print("Triangle exists. Will repair:", will_divide)
        # print(", ".join(faces))
        # print(short_side, medium_side, long_side)
        if not will_divide:
            continue

        # insert new point
        new_node = len(vertices)

        corner_nodes_list = [
            x
            for x in opposing_nodes_dict[edge(medium_node, short_node)]
            if x != corner_node
        ]

        if len(corner_nodes_list) == 1:
            other_corner_node = corner_nodes_list[0]
        else:
            other_corner_node = None

        i += 1

        # FIX THE OPPOSING TRIANGLE:
        if other_corner_node is not None:
            other_tri = (medium_node, short_node, other_corner_node)
            o_short_side, o_medium_side, o_long_side = (
                get_triangle_with_opposite_sidelengths(data, other_tri)
            )

            assert o_long_side == long_side, f"{o_long_side} != {long_side}"

            cos_other_medium_angle = (
                o_medium_side**2 + long_side**2 - o_short_side**2
            ) / (2 * o_medium_side * long_side)

            other_new_length = np.sqrt(
                (long_side / 2) ** 2
                + o_medium_side**2
                - 2 * (long_side / 2) * o_medium_side * cos_other_medium_angle
            ).round(3)

            opposing_nodes_dict[edge(new_node, other_corner_node)] = [
                short_node,
                medium_node,
            ]
            opposing_nodes_dict[edge(medium_node, other_corner_node)].remove(short_node)
            opposing_nodes_dict[edge(medium_node, other_corner_node)].append(new_node)
            opposing_nodes_dict[edge(short_node, other_corner_node)].remove(medium_node)
            opposing_nodes_dict[edge(short_node, other_corner_node)].append(new_node)

            remove_triangle(data, other_tri)

            add_triangle_with_opposing_sidelengths(
                data,
                (medium_node, new_node, other_corner_node),
                (other_new_length, o_medium_side, long_side / 2),
            )

            add_triangle_with_opposing_sidelengths(
                data,
                (short_node, new_node, other_corner_node),
                (other_new_length, o_short_side, long_side / 2),
            )

            triangles_to_visit.append(
                tuple(np.sort([medium_node, new_node, other_corner_node]))
            )

            triangles_to_visit.append(
                tuple(np.sort([new_node, short_node, other_corner_node]))
            )

        # FIX THE CURRENT TRIANGLE:
        t = 0.5
        vertices.append(vertices[medium_node] * (1 - t) + vertices[short_node] * (t))

        opposing_nodes_dict[edge(new_node, medium_node)] = list.copy(
            opposing_nodes_dict[edge(medium_node, short_node)]
        )
        opposing_nodes_dict[edge(new_node, short_node)] = list.copy(
            opposing_nodes_dict[edge(medium_node, short_node)]
        )

        new_length = np.round(0.5 * (short_side + medium_side), 4)

        opposing_nodes_dict[edge(new_node, corner_node)] = [short_node, medium_node]
        opposing_nodes_dict[edge(medium_node, corner_node)].remove(short_node)
        opposing_nodes_dict[edge(medium_node, corner_node)].append(new_node)
        opposing_nodes_dict[edge(short_node, corner_node)].remove(medium_node)
        opposing_nodes_dict[edge(short_node, corner_node)].append(new_node)
        del opposing_nodes_dict[edge(medium_node, short_node)]

        triangles_to_visit.append(tuple(np.sort([new_node, medium_node, corner_node])))

        triangles_to_visit.append(tuple(np.sort([new_node, short_node, corner_node])))

        add_triangle_with_opposing_sidelengths(
            data,
            (medium_node, new_node, corner_node),
            (new_length, medium_side, long_side / 2),
        )

        add_triangle_with_opposing_sidelengths(
            data,
            (short_node, new_node, corner_node),
            (new_length, short_side, long_side / 2),
        )

        remove_triangle(data, faces)

    print("made", i, "new splits")

    return data, np.array(vertices)


def edge(a, b):
    return (a, b) if a < b else (b, a)


def add_triangle_with_opposing_sidelengths(data, faces: tuple, edgelengths: tuple):
    SORT = np.argsort(faces)
    faces = tuple(np.array(faces)[SORT])
    edgelengths = tuple(np.array(edgelengths)[SORT])
    data[faces] = edgelengths


def remove_triangle(data, faces: tuple):
    faces = tuple(np.sort(faces))
    del data[faces]


def get_triangle_with_opposite_sidelengths(data, faces: tuple):
    SORT = np.argsort(faces)
    UNSORT = np.argsort(SORT)
    faces = np.array(faces)[SORT]
    lengths = data.get(tuple(faces))
    return np.array(lengths)[UNSORT]


def triangle_exists(data, faces: tuple):
    faces = tuple(np.sort(faces))
    return data.get(faces) is not None


def tris_to_mesh(tris_lengths_sextuple: dict, vertices: np.ndarray):
    faces = []
    for (a, b, c), _lengths in tris_lengths_sextuple.items():
        faces.append([a, b, c])

    return DEC.Mesh(vertices=vertices, faces=np.array(faces), lazy=True)
