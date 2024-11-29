import collections
import itertools
import logging

import ipctk
import numpy as np
import pbatoolkit as pbat

logger = logging.getLogger(__name__)


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = [0] + list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C


def to_surface(x: np.ndarray, mesh: pbat.fem.Mesh, cmesh: ipctk.CollisionMesh):
    X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order="F").T
    XB = cmesh.map_displacements(X)
    return XB


def find_codim_vertices(mesh, boundary_edges):
    # Initialize a set with all vertex indices
    all_vertices = set(range(len(mesh.X)))

    # Find vertices connected to boundary edges
    surface_vertices = set()
    for edge in boundary_edges:
        surface_vertices.update(edge)

    # Codimensional vertices are those not connected to any boundary edge
    codim_vertices = list(all_vertices - surface_vertices)
    if len(codim_vertices) == 0:
        logging.warning("No codimensional vertices found.")
    return codim_vertices


def compute_tetrahedron_centroids(V, C):
    return np.mean(V[C], axis=1)


def compute_face_to_element_mapping(C: np.ndarray, F: np.ndarray) -> np.ndarray:
    face_to_element_dict = collections.defaultdict(list)
    # Iterate over each tetrahedron and its faces
    for elem_idx, tet in enumerate(C):
        # Define all four faces of the tetrahedron
        tet_faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for face in tet_faces:
            face_to_element_dict[face].append(elem_idx)

    # Map each boundary face to the corresponding element
    face_to_element = []
    for face in F:
        face_key = tuple(sorted(face))
        elems = face_to_element_dict.get(face_key, [])
        if len(elems) == 1:
            face_to_element.append(elems[0])
        elif len(elems) > 1:
            face_to_element.append(
                elems[0]
            )  # You might want to handle shared faces differently
        else:
            logger.warning(f"No element found for face: {face}")
            face_to_element.append(-1)

    return np.array(face_to_element)
