import logging
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np
import pbatoolkit as pbat

from simulation.config.config import ConfigManager
from simulation.core.modifier.transformation import apply_transformations

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
    inputs: List[Dict],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Dict]]:
    """Loads and processes individual meshes based on input configurations.

    Args:
        inputs (List[Dict]): List of input configurations.

    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Dict]]:
            - List of transformed meshes (vertices and connectivity).
            - List of corresponding material properties.

    """
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

            # Load material properties from predefined materials
            material = ConfigManager().load_material_properties(material_name)
            material["percent_fixed"] = percent_fixed

            # Load transformation properties
            scale, rotation, translation = ConfigManager().load_transform_properties(input_entry)

            # Load mesh data (vertices and connectivity)
            V, C = load_mesh(path)

            # Apply transformations to vertices
            V_transformed = apply_transformations(V, scale, rotation, translation)

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
) -> Tuple[pbat.fem.Mesh, np.ndarray, np.ndarray, np.ndarray, List[int], Optional[List[Dict]]]:
    """Combines multiple meshes into a single mesh, optionally deduplicating vertices.

    Args:
        all_meshes (List[Tuple[np.ndarray, np.ndarray]]): List of meshes to combine.
        materials (List[Dict]): List of material properties corresponding to each mesh.
        instancing (bool, optional): Whether to use instancing. Defaults to False.

    Returns:
        Tuple[pbat.fem.Mesh, np.ndarray, np.ndarray, np.ndarray, List[int], Optional[List[Dict]]]:
            - Combined mesh.
            - Combined vertices.
            - Combined connectivity.
            - Element materials.
            - List of node counts per mesh.
            - List of instances if instancing is enabled.

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

    # Deduplicate vertices to optimize mesh
    decimals = 8  # Precision for deduplication
    V_rounded = np.round(V, decimals=decimals)
    unique_V, unique_indices, inverse_indices = np.unique(
        V_rounded, axis=0, return_index=True, return_inverse=True
    )
    C_remapped = inverse_indices[C]

    logger.info(
        f"Combined mesh has {V.shape[0]} vertices and {C.shape[0]} elements before deduplication."
    )
    logger.info(
        f"After deduplication: {unique_V.shape[0]} unique vertices out of {V.shape[0]} original vertices."
    )

    # Create pbatoolkit mesh
    mesh = pbat.fem.Mesh(unique_V.T, C_remapped.T, element=pbat.fem.Element.Tetrahedron, order=1)
    V_combined, C_combined = mesh.X.T, mesh.E.T

    # Validate remapped connectivity
    if not (np.all(C_remapped >= 0) and np.all(C_remapped < unique_V.shape[0])):
        logger.error("Remapped connectivity has invalid indices.")
        raise ValueError("Remapped connectivity contains invalid vertex indices.")

    logger.info("Instancing is disabled. Meshes have been combined and deduplicated.")

    return mesh, V_combined, C_combined, element_materials, num_nodes_list
