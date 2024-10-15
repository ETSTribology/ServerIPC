import meshio
import numpy as np
import logging
import pbatoolkit as pbat
from utils.transformation import apply_transformations

logger = logging.getLogger(__name__)

def load_mesh(path):
    imesh = meshio.read(path)
    V = imesh.points.astype(np.float64, order='C')
    C = imesh.cells_dict.get("tetra")
    logger.info(f"Loaded mesh from {path}.")
    logger.info(f"Mesh contains {V.shape[0]} vertices and {C.shape[0]} tetrahedral cells.")
    if C is None:
        raise ValueError(f"No tetrahedral cells found in the mesh file: {path}")
    C = C.astype(np.int64, order='C')
    return V, C


def load_material_properties(input_entry):
    material_props = input_entry.get('material', {})
    return {
        'density': material_props.get('density', 1000.0),
        'young_modulus': material_props.get('young_modulus', 1e6),
        'poisson_ratio': material_props.get('poisson_ratio', 0.45),
        'color': material_props.get('color', [255, 255, 255, 1])
    }

def load_transform_properties(input_entry):
    transform = input_entry.get('transform', {})
    return (
        transform.get('scale', [1.0, 1.0, 1.0]),
        transform.get('rotation', [0.0, 0.0, 0.0, 1.0]),
        transform.get('translation', [0.0, 0.0, 0.0])
    )

def load_individual_meshes(inputs):
    all_meshes = []
    materials = []

    for idx, input_entry in enumerate(inputs):
        # Load mesh file
        path = input_entry.get('path')
        if not path:
            logger.error(f"Mesh path missing for entry {idx + 1}. Skipping.")
            continue
        percent_fixed = input_entry.get('percent_fixed', 0.0)

        # Load material properties
        material = load_material_properties(input_entry)

        material['percent_fixed'] = percent_fixed

        # Load transformation properties
        scale, rotation, translation = load_transform_properties(input_entry)

        # Load mesh data (vertices and connectivity)
        V, C = load_mesh(path)

        # Apply transformations to vertices
        V_transformed = apply_transformations(V, scale, rotation, translation)

        # Store transformed mesh and material
        all_meshes.append((V_transformed, C))
        materials.append(material)

        logger.info(f"Mesh {idx + 1} loaded, transformed, and processed from {path}.")

    return all_meshes, materials

def combine_meshes(all_meshes, materials):
    V, C, element_materials = [], [], []
    num_nodes_list = []
    vertex_offset = 0
    for idx, (V1, C1) in enumerate(all_meshes):
        V.append(V1)
        C.append(C1 + vertex_offset)
        num_elements = C1.shape[0]
        num_nodes = V1.shape[0]
        num_nodes_list.append(num_nodes)
        element_materials.extend([idx] * num_elements)
        vertex_offset += num_nodes
    V = np.vstack(V)
    C = np.vstack(C)

    # Deduplicate vertices using np.unique with rounding
    decimals = 8  # Adjust based on precision
    V_rounded = np.round(V, decimals=decimals)
    unique_V, unique_indices, inverse_indices = np.unique(V_rounded, axis=0, return_index=True, return_inverse=True)
    logger.info(f"After Deduplication: {unique_V.shape[0]} unique vertices out of {V.shape[0]} original vertices.")

    # Remap connectivity
    C_remapped = inverse_indices[C]

    # Validate remapped connectivity
    assert np.all(C_remapped >= 0) and np.all(C_remapped < unique_V.shape[0]), "Remapped connectivity has invalid indices."

    # Create pbatoolkit mesh
    mesh = pbat.fem.Mesh(unique_V.T, C_remapped.T, element=pbat.fem.Element.Tetrahedron, order=1)
    V_combined, C_combined = mesh.X.T, mesh.E.T
    element_materials = np.array(element_materials, dtype=int)
    logger.info("Meshes combined and deduplicated successfully.")
    return mesh, V_combined, C_combined, element_materials, num_nodes_list