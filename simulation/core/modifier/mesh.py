import collections
import itertools
import logging

import ipctk
import numpy as np
import pbatoolkit as pbat

from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


def combine(V: list, C: list):
    """
    Combine multiple vertex and connectivity arrays into single arrays.

    Args:
        V (list): List of vertex arrays.
        C (list): List of connectivity arrays.

    Returns:
        tuple: Combined vertex and connectivity arrays.
    """
    try:
        Vsizes = [Vi.shape[0] for Vi in V]
        offsets = list(itertools.accumulate(Vsizes))
        C = [C[i] + offsets[i] - Vsizes[i] for i in range(len(C))]
        C = np.vstack(C)
        V = np.vstack(V)
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Combined vertex and connectivity arrays successfully."
            )
        )
        return V, C
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to combine vertex and connectivity arrays: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.MESH_SETUP,
            "Failed to combine vertex and connectivity arrays",
            details=str(e),
        )


def de_combine(V: np.ndarray, C: np.ndarray):
    """
    De-combine the vertex and connectivity arrays.

    Args:
        V (np.ndarray): Vertex array.
        C (np.ndarray): Connectivity array.

    Returns:
        tuple: List of vertex and connectivity arrays.
    """
    try:
        Vsizes = np.unique(C.flatten(), return_counts=True)[1]
        offsets = np.cumsum(Vsizes)
        offsets = np.insert(offsets, 0, 0)
        V = np.split(V, offsets[:-1])
        C = np.split(C, np.where(C[:, 0] == 0)[0])[1:]
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "De-combined vertex and connectivity arrays successfully."
            )
        )
        return V, C
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to de-combine vertex and connectivity arrays: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.MESH_SETUP,
            "Failed to de-combine vertex and connectivity arrays",
            details=str(e),
        )


def to_surface(x: np.ndarray, mesh: pbat.fem.Mesh, cmesh: ipctk.CollisionMesh):
    """
    Map displacements to the surface.

    Args:
        x (np.ndarray): Current positions.
        mesh (pbat.fem.Mesh): Mesh object.
        cmesh (ipctk.CollisionMesh): Collision mesh object.

    Returns:
        np.ndarray: Mapped displacements on the surface.
    """
    try:
        X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order="F").T
        XB = cmesh.map_displacements(X)
        logger.debug(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Mapped displacements to the surface successfully."
            )
        )
        return XB
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to map displacements to the surface: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.MESH_SETUP,
            "Failed to map displacements to the surface",
            details=str(e),
        )


def find_codim_vertices(mesh, boundary_edges):
    """
    Find codimensional vertices not connected to any boundary edge.

    Args:
        mesh: Mesh object.
        boundary_edges: List of boundary edges.

    Returns:
        list: List of codimensional vertices.
    """
    try:
        all_vertices = set(range(len(mesh.X)))
        surface_vertices = set()
        for edge in boundary_edges:
            surface_vertices.update(edge)
        codim_vertices = list(all_vertices - surface_vertices)
        if len(codim_vertices) == 0:
            logger.warning(
                SimulationLogMessageCode.COMMAND_FAILED.details("No codimensional vertices found.")
            )
        else:
            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    "Codimensional vertices found successfully."
                )
            )
        return codim_vertices
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to find codimensional vertices: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.MESH_SETUP, "Failed to find codimensional vertices", details=str(e)
        )


def compute_tetrahedron_centroids(V, C):
    """
    Compute the centroids of tetrahedrons.

    Args:
        V (np.ndarray): Vertex array.
        C (np.ndarray): Connectivity array.

    Returns:
        np.ndarray: Centroids of the tetrahedrons.
    """
    try:
        centroids = np.mean(V[C], axis=1)
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Computed tetrahedron centroids successfully."
            )
        )
        return centroids
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to compute tetrahedron centroids: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.MESH_SETUP,
            "Failed to compute tetrahedron centroids",
            details=str(e),
        )


def compute_face_to_element_mapping(C: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute the mapping from faces to elements.

    Args:
        C (np.ndarray): Connectivity array.
        F (np.ndarray): Face array.

    Returns:
        np.ndarray: Mapping from faces to elements.
    """
    try:
        face_to_element_dict = collections.defaultdict(list)
        for elem_idx, tet in enumerate(C):
            tet_faces = [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]])),
            ]
            for face in tet_faces:
                face_to_element_dict[face].append(elem_idx)

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
                logger.warning(
                    SimulationLogMessageCode.COMMAND_FAILED.details(
                        f"No element found for face: {face}"
                    )
                )
                face_to_element.append(-1)

        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Computed face to element mapping successfully."
            )
        )
        return np.array(face_to_element)
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to compute face to element mapping: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.MESH_SETUP,
            "Failed to compute face to element mapping",
            details=str(e),
        )
