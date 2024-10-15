import numpy as np
import scipy.spatial.transform as spt

def apply_scaling(V1, scale):
    scaling_matrix = np.diag(scale)
    return V1 @ scaling_matrix

def apply_rotation(V1, rotation):
    R = spt.Rotation.from_quat(rotation).as_matrix()
    centroid = V1.mean(axis=0)
    V1_centered = V1 - centroid
    V1_rotated = V1_centered @ R.T
    return V1_rotated + centroid

def apply_translation(V1, translation):
    translation_vector = np.array(translation).reshape(1, 3)
    return V1 + translation_vector

def apply_transformations(vertices, scale, rotation, translation):
    # Convert rotation from quaternion to rotation matrix
    rotation_matrix = spt.Rotation.from_quat(rotation).as_matrix()

    # Apply scaling
    scaling_matrix = np.diag(scale)
    scaled_vertices = vertices @ scaling_matrix

    # Compute the mesh_centroid of the scaled mesh
    mesh_centroid = scaled_vertices.mean(axis=0)

    # Translate vertices to origin for rotation (mesh_centroid-based rotation)
    centered_vertices = scaled_vertices - mesh_centroid

    # Apply rotation
    rotated_vertices = centered_vertices @ rotation_matrix.T

    rotated_vertices += mesh_centroid

    # Apply translation
    translation_vector = np.array(translation).reshape(1, 3)
    transformed_vertices = rotated_vertices + translation_vector

    return transformed_vertices