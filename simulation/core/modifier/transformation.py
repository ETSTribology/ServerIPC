import logging
from typing import List, Union

import numpy as np
import scipy.spatial.transform as spt

from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


def euler_to_quaternion(euler_angles: List[float]) -> List[float]:
    """Convert Euler angles to a quaternion.

    Args:
        euler_angles (List[float]): List of three Euler angles (in radians).

    Returns:
        List[float]: Quaternion [x, y, z, w].
    """
    if len(euler_angles) != 3:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Euler angles must be a list of three elements, got {len(euler_angles)}."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Euler angles must be a list of three elements, got {len(euler_angles)}.",
        )

    try:
        rotation = spt.Rotation.from_euler("xyz", euler_angles)
        quaternion = rotation.as_quat().tolist()
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Converted Euler angles to quaternion successfully."
            )
        )
        return quaternion
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to convert Euler angles to quaternion: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            "Failed to convert Euler angles to quaternion",
            details=str(e),
        )


def quaternion_to_euler(quaternion: List[float]) -> List[float]:
    """Convert a quaternion to Euler angles.

    Args:
        quaternion (List[float]): Quaternion [x, y, z, w].

    Returns:
        List[float]: Euler angles (in radians).
    """
    if len(quaternion) != 4:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Quaternion must be a list of four elements, got {len(quaternion)}."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Quaternion must be a list of four elements, got {len(quaternion)}.",
        )

    try:
        rotation = spt.Rotation.from_quat(quaternion)
        euler_angles = rotation.as_euler("xyz").tolist()
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Converted quaternion to Euler angles successfully."
            )
        )
        return euler_angles
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to convert quaternion to Euler angles: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            "Failed to convert quaternion to Euler angles",
            details=str(e),
        )


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
    SimulationError
        If the `scale` list does not contain exactly three elements.
    """
    # Convert scalar scale to list
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]

    if len(scale) != 3:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Scale must be a list of three elements, got {len(scale)} elements."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Scale must be a list of three elements, got {len(scale)} elements.",
        )

    try:
        scaling_matrix = np.diag(scale)
        scaled_vertices = vertices @ scaling_matrix
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Applied scaling to vertices successfully."
            )
        )
        return scaled_vertices
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to apply scaling to vertices: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            "Failed to apply scaling to vertices",
            details=str(e),
        )


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
    SimulationError
        If the `rotation` list does not contain exactly four elements or if the quaternion is invalid.
    """
    if len(rotation) != 4:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Rotation must be a quaternion with four elements, got {len(rotation)}."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Rotation must be a quaternion with four elements, got {len(rotation)}.",
        )

    try:
        rotation_matrix = spt.Rotation.from_quat(rotation).as_matrix()
    except ValueError as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(f"Invalid quaternion {rotation}: {e}")
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION, f"Invalid quaternion {rotation}", details=str(e)
        )

    try:
        # Compute centroid for rotation
        centroid = vertices.mean(axis=0, dtype=np.float64)

        # Center vertices, rotate, and uncenter
        centered_vertices = vertices - centroid
        rotated_vertices = centered_vertices @ rotation_matrix.T
        transformed_vertices = rotated_vertices + centroid
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Applied rotation to vertices successfully."
            )
        )
        return transformed_vertices
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to apply rotation to vertices: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            "Failed to apply rotation to vertices",
            details=str(e),
        )


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
    SimulationError
        If the `translation` list does not contain exactly three elements.
    """
    if len(translation) != 3:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Translation must be a list of three elements, got {len(translation)} elements."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Translation must be a list of three elements, got {len(translation)} elements.",
        )

    try:
        translation_vector = np.array(translation).reshape(1, 3)
        translated_vertices = vertices + translation_vector
        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Applied translation to vertices successfully."
            )
        )
        return translated_vertices
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to apply translation to vertices: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            "Failed to apply translation to vertices",
            details=str(e),
        )


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
    SimulationError
        If the `scale`, `rotation`, or `translation` lists do not contain the required number of elements.
    """
    # Convert scalar scale to list
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]

    # Validate input lengths
    if len(scale) != 3:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Scale must be a list of three elements, got {len(scale)} elements."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Scale must be a list of three elements, got {len(scale)} elements.",
        )
    if len(rotation) != 4:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Rotation must be a list of four elements representing a quaternion, got {len(rotation)} elements."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Rotation must be a list of four elements representing a quaternion, got {len(rotation)} elements.",
        )
    if len(translation) != 3:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Translation must be a list of three elements, got {len(translation)} elements."
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            f"Translation must be a list of three elements, got {len(translation)} elements.",
        )

    try:
        # Apply scaling
        scaled_vertices = apply_scaling(vertices, scale)

        # Apply rotation
        rotated_vertices = apply_rotation(scaled_vertices, rotation)

        # Apply translation
        transformed_vertices = apply_translation(rotated_vertices, translation)

        logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                "Applied transformations to vertices successfully."
            )
        )
        return transformed_vertices
    except Exception as e:
        logger.error(
            SimulationLogMessageCode.COMMAND_FAILED.details(
                f"Failed to apply transformations to vertices: {e}"
            )
        )
        raise SimulationError(
            SimulationErrorCode.INPUT_VALIDATION,
            "Failed to apply transformations to vertices",
            details=str(e),
        )
