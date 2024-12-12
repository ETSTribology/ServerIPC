import logging
from typing import Dict, List, Tuple

import meshio
import numpy as np
import pbatoolkit as pbat

from simulation.config.config import SimulationConfigManager
from simulation.core.modifier.transformation import apply_transformations, euler_to_quaternion

logger = logging.getLogger(__name__)


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a mesh from the given file path.

    Args:
        path (str): Path to the mesh file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Vertices and tetrahedral connectivity.

    Raises:
        ValueError: If no tetrahedral cells are found in the mesh.
    """
    try:
        imesh = meshio.read(path)
        V = imesh.points.astype(np.float64, order="C")

        # Check for tetrahedral cells, including different possible keys
        tetra_keys = ["tetra", "tetrahedron", "tetrahedral"]
        C = None
        for key in tetra_keys:
            C = imesh.cells_dict.get(key)
            if C is not None:
                break

        if C is None:
            logger.error(f"No tetrahedral cells found in the mesh file: {path}")
            raise ValueError(f"No tetrahedral cells found in the mesh file: {path}")

        logger.info(f"Loaded mesh from {path}.")
        logger.info(f"Mesh contains {V.shape[0]} vertices and {C.shape[0]} tetrahedral cells.")

        C = C.astype(np.int64, order="C")
        return V, C
    except Exception as e:
        logger.error(f"Error loading mesh from {path}: {e}")
        raise


def load_individual_meshes(
    inputs: List[Dict], config_manager: SimulationConfigManager
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Dict]]:
    all_meshes = []
    materials_list = []

    for idx, input_entry in enumerate(inputs):
        try:
            # Load mesh file
            path = input_entry.get("path")
            if not path:
                logger.error(f"Mesh path missing for entry {idx + 1}. Skipping.")
                continue

            percent_fixed = input_entry.get("percent_fixed", 0.0)
            material_name = input_entry.get("material")
            if not material_name:
                logger.error(f"Material name missing for entry {idx + 1}. Skipping.")
                continue

            # Load material properties from configuration
            materials = config_manager.get()["material"]
            material = next((m for m in materials if m["id"] == material_name), None)
            if material is None:
                logger.error(f"Material {material_name} not found in configuration. Skipping.")
                continue

            material["percent_fixed"] = percent_fixed

            # Load transformation properties
            transform = input_entry.get("transform", {})
            scale = transform.get("scale", [1.0, 1.0, 1.0])
            rotation = transform.get("rotation", {})
            if rotation.get("type") == "euler":
                rotation_values = rotation.get("values", [])
                if len(rotation_values) != 3:
                    raise ValueError(
                        f"Rotation must be a list of three elements representing Euler angles, got {len(rotation_values)} elements."
                    )
                rotation_values = euler_to_quaternion(rotation_values)
            elif len(rotation.get("values", [])) != 4:
                raise ValueError(
                    f"Rotation must be a list of four elements representing a quaternion, got {len(rotation.get('values', []))} elements."
                )
            else:
                rotation_values = rotation.get("values", [0.0, 0.0, 0.0, 1.0])

            translation = transform.get("translation", [0.0, 0.0, 0.0])

            # Load mesh data (vertices and connectivity)
            V, C = load_mesh(path)

            # Apply transformations to vertices
            V_transformed = apply_transformations(V, scale, rotation_values, translation)

            # Store transformed mesh and material
            all_meshes.append((V_transformed, C))
            materials_list.append(material)

            logger.info(f"Mesh {idx + 1} loaded, transformed, and processed from {path}.")

        except ValueError as ve:
            logger.error(f"Validation error for entry {idx + 1}: {ve}. Skipping this entry.")
        except Exception as e:
            logger.error(f"Unexpected error for entry {idx + 1}: {e}. Skipping this entry.")

    return all_meshes, materials_list


def combine_meshes(
    all_meshes: List[Tuple[np.ndarray, np.ndarray]], materials: List[Dict]
) -> Tuple[pbat.fem.Mesh, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Combines multiple meshes into a single mesh, optionally deduplicating vertices.

    Args:
        all_meshes (List[Tuple[np.ndarray, np.ndarray]]): List of meshes to combine.
        materials (List[Dict]): List of material properties corresponding to each mesh.

    Returns:
        Tuple[pbat.fem.Mesh, np.ndarray, np.ndarray, np.ndarray, List[int]]:
            - Combined mesh.
            - Combined vertices.
            - Combined connectivity.
            - Element materials.
            - List of node counts per mesh.
    """
    V_list, C_list, element_materials_list = [], [], []
    num_nodes_list = []
    vertex_offset = 0

    for idx, (V1, C1) in enumerate(all_meshes):
        num_elements = C1.shape[0]
        element_materials_list.append(np.full(num_elements, idx, dtype=int))

        V_list.append(V1)
        C_list.append(C1 + vertex_offset)
        num_nodes = V1.shape[0]
        num_nodes_list.append(num_nodes)

        # Update vertex offset for the next mesh
        vertex_offset += num_nodes

        logger.debug(f"Mesh {idx + 1}: {num_nodes} nodes, {num_elements} elements added.")

    # Stack all vertices and connectivity
    V = np.vstack(V_list)
    C = np.vstack(C_list)
    element_materials = np.concatenate(element_materials_list)

    # Create pbatoolkit mesh
    mesh = pbat.fem.Mesh(V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    V_combined, C_combined = mesh.X.T, mesh.E.T

    logger.info(f"Combined mesh contains {V_combined.shape[0]} nodes and {C_combined.shape[0]} elements.")
    logger.info("Meshes have been combined and deduplicated.")

    return mesh, V_combined, C_combined, element_materials, num_nodes_list
