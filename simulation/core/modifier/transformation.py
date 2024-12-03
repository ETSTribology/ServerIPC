from typing import List, Union

import numpy as np
import scipy.spatial.transform as spt


def apply_scaling(vertices: np.ndarray, scale: Union[float, List[float]]) -> np.ndarray:
    """Applies scaling to the mesh vertices.

    This function scales the input vertices along the X, Y, and Z axes using the provided scaling factors.

    Parameters
    ----------
    vertices : np.ndarray
        A 2D NumPy array of shape (N, 3) representing the coordinates of N vertices.
    scale : Union[float, List[float]]
        A scalar or list of three scaling factors [Sx, Sy, Sz] for the X, Y, and Z axes respectively.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, 3) representing the scaled vertices.

    Raises
    ------
    ValueError
        If the `scale` list does not contain exactly three elements.

    Examples
    --------
    >>> vertices = np.array([[1, 2, 3], [4, 5, 6]])
    >>> scale = [2, 2, 2]
    >>> apply_scaling(vertices, scale)
    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]])

    """
    # Convert scalar scale to list
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]

    if len(scale) != 3:
        raise ValueError(f"Scale must be a list of three elements, got {len(scale)} elements.")

    scaling_matrix = np.diag(scale)
    scaled_vertices = vertices @ scaling_matrix
    return scaled_vertices


def apply_rotation(vertices: np.ndarray, rotation: List[float]) -> np.ndarray:
    """Applies rotation to the mesh vertices around their centroid.

    This function rotates the input vertices using a quaternion-based rotation. The rotation is performed
    about the centroid of the mesh to ensure the mesh remains centered post-rotation.

    Parameters
    ----------
    vertices : np.ndarray
        A 2D NumPy array of shape (N, 3) representing the coordinates of N vertices.
    rotation : List[float]
        A list or array of four floats representing the rotation quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, 3) representing the rotated vertices.

    Raises
    ------
    ValueError
        If the `rotation` list does not contain exactly four elements or if the quaternion is invalid.

    Examples
    --------
    >>> vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> rotation = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # 90-degree rotation around Z-axis
    >>> apply_rotation(vertices, rotation)
    array([[ 0.,  1.,  0.],
           [-1.,  0.,  0.],
           [ 0.,  0.,  1.]])

    """
    if len(rotation) != 4:
        raise ValueError(
            f"Rotation must be a list of four elements representing a quaternion, got {len(rotation)} elements."
        )

    try:
        R = spt.Rotation.from_quat(rotation).as_matrix()
    except ValueError as e:
        raise ValueError(f"Invalid quaternion {rotation}: {e}")

    # Compute centroid with high precision
    centroid = vertices.mean(axis=0, dtype=np.float64)

    # Center vertices and rotate with high precision
    centered_vertices = vertices - centroid
    rotated_vertices = np.dot(centered_vertices, R.T)
    rotated_vertices += centroid

    # Round to avoid floating point precision issues
    rotated_vertices = np.round(rotated_vertices, decimals=10)

    return rotated_vertices


def apply_translation(vertices: np.ndarray, translation: List[float]) -> np.ndarray:
    """Applies translation to the mesh vertices.

    This function translates the input vertices by the specified offsets along the X, Y, and Z axes.

    Parameters
    ----------
    vertices : np.ndarray
        A 2D NumPy array of shape (N, 3) representing the coordinates of N vertices.
    translation : List[float]
        A list or array of three translation offsets [Tx, Ty, Tz] along the X, Y, and Z axes respectively.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, 3) representing the translated vertices.

    Raises
    ------
    ValueError
        If the `translation` list does not contain exactly three elements.

    Examples
    --------
    >>> vertices = np.array([[1, 2, 3], [4, 5, 6]])
    >>> translation = [10, 0, -5]
    >>> apply_translation(vertices, translation)
    array([[11,  2, -2],
           [14,  5,  1]])

    """
    if len(translation) != 3:
        raise ValueError(
            f"Translation must be a list of three elements, got {len(translation)} elements."
        )

    translation_vector = np.array(translation).reshape(1, 3)
    translated_vertices = vertices + translation_vector
    return translated_vertices


def apply_transformations(
    vertices: np.ndarray,
    scale: Union[float, List[float]],
    rotation: List[float],
    translation: List[float],
) -> np.ndarray:
    """Applies scaling, rotation, and translation transformations to the mesh vertices.

    This function performs a sequence of transformations on the input vertices:
    1. Scaling along the X, Y, and Z axes.
    2. Rotation around the centroid using a quaternion.
    3. Translation along the X, Y, and Z axes.

    Parameters
    ----------
    vertices : np.ndarray
        A 2D NumPy array of shape (N, 3) representing the original coordinates of N vertices.
    scale : Union[float, List[float]]
        A scalar or list of three scaling factors [Sx, Sy, Sz] for the X, Y, and Z axes respectively.
    rotation : List[float]
        A list or array of four floats representing the rotation quaternion [x, y, z, w].
    translation : List[float]
        A list or array of three translation offsets [Tx, Ty, Tz] along the X, Y, and Z axes respectively.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (N, 3) representing the transformed vertices.

    Raises
    ------
    ValueError
        If the `scale`, `rotation`, or `translation` lists do not contain the required number of elements.

    Examples
    --------
    >>> vertices = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    >>> scale = [2, 2, 2]
    >>> rotation = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]  # 90-degree rotation around Z-axis
    >>> translation = [5, 0, 0]
    >>> apply_transformations(vertices, scale, rotation, translation)
    array([[ 5.,  2.,  1.],
           [ 5.,  4.,  2.],
           [ 5.,  6.,  3.]])

    """
    # Convert scalar scale to list
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]

    # Validate input lengths
    if len(scale) != 3:
        raise ValueError(f"Scale must be a list of three elements, got {len(scale)} elements.")
    if len(rotation) != 4:
        raise ValueError(
            f"Rotation must be a list of four elements representing a quaternion, got {len(rotation)} elements."
        )
    if len(translation) != 3:
        raise ValueError(
            f"Translation must be a list of three elements, got {len(translation)} elements."
        )

    # Apply scaling
    scaled_vertices = apply_scaling(vertices, scale)

    # Apply rotation
    rotated_vertices = apply_rotation(scaled_vertices, rotation)

    # Apply translation
    transformed_vertices = apply_translation(rotated_vertices, translation)

    return transformed_vertices
