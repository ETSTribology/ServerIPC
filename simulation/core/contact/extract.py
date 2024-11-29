import csv
import logging

import ipctk
import numpy as np

logger = logging.getLogger(__name__)


class CollisionInfoCollector:
    def __init__(self, filename="collision_info.csv"):
        """Initializes the CollisionInfoCollector with the filename for saving collision data.

        Args:
            filename (str): Name of the CSV file to save collision data.

        """
        self.filename = filename
        self.collision_data = []
        self.header = [
            "Collision Type",
            "Tangential Velocity",
            "Relative Velocity",
            "Weights",
            "Coefficients",
            "Normal Force Magnitude",
            "f0_SF",
            "Friction",
        ]
        self.logger = logging.getLogger(self.__class__.__name__)

    def collect_collision_info(self, fconstraints, cmesh, BX, BXdot, epsv, EF):
        """Collects collision information and stores it for later processing.

        Args:
            fconstraints: The friction constraints object.
            cmesh: The collision mesh object.
            BX: Current positions on the collision mesh.
            BXdot: Current velocities on the collision mesh.
            epsv: Threshold for tangential velocity.
            EF: Friction value for the current step.

        """
        self.logger.info("Collecting collision information...")

        # Process vertex-vertex collisions
        for i, collision in enumerate(fconstraints.vv_collisions):
            self._process_collision(
                collision_type="Vertex-Vertex",
                collision=collision,
                tangent_basis=collision.compute_tangent_basis(BX),
                relative_velocity=collision.relative_velocity(BXdot),
                weights=collision.weight[i],
                coefficients=collision.mu[i],
                normal_force_magnitude=collision.normal_force_magnitude[i],
                epsv=epsv,
                EF=EF,
            )

        # Process edge-vertex collisions
        for i, collision in enumerate(fconstraints.ev_collisions):
            self._process_collision(
                collision_type="Edge-Vertex",
                collision=collision,
                tangent_basis=collision.compute_tangent_basis(BX),
                relative_velocity=collision.relative_velocity(BXdot),
                weights=collision.weight[i],
                coefficients=collision.mu[i],
                normal_force_magnitude=collision.normal_force_magnitude[i],
                epsv=epsv,
                EF=EF,
            )

        # Process edge-edge collisions
        for i, collision in enumerate(fconstraints.ee_collisions):
            self.collision_data.append(
                [
                    "Edge-Edge",
                    None,
                    None,
                    collision.weight,
                    collision.mu,
                    collision.normal_force_magnitude,
                    None,
                    EF,
                ]
            )

        # Process face-vertex collisions
        for i, collision in enumerate(fconstraints.fv_collisions):
            self.collision_data.append(
                [
                    "Face-Vertex",
                    None,
                    None,
                    collision.weight,
                    collision.mu,
                    None,
                    None,
                    EF,
                ]
            )

        self.logger.info("Collision information collected.")

    def _process_collision(
        self,
        collision_type,
        collision,
        tangent_basis,
        relative_velocity,
        weights,
        coefficients,
        normal_force_magnitude,
        epsv,
        EF,
    ):
        """Helper method to process individual collision data.

        Args:
            collision_type (str): The type of collision (e.g., Vertex-Vertex).
            collision: The collision object.
            tangent_basis: Tangent basis for the collision.
            relative_velocity: Relative velocity at the collision.
            weights: Weights associated with the collision.
            coefficients: Coefficients of friction for the collision.
            normal_force_magnitude: Magnitude of the normal force.
            epsv: Threshold for tangential velocity.
            EF: Friction value for the current step.

        """
        tangent_rel_velocity = np.dot(tangent_basis.T, relative_velocity)
        f0_SF_value = ipctk.f0_SF(np.linalg.norm(tangent_rel_velocity), epsv)

        self.collision_data.append(
            [
                collision_type,
                tangent_rel_velocity.tolist(),
                relative_velocity.tolist(),
                weights,
                coefficients,
                normal_force_magnitude,
                f0_SF_value,
                EF,
            ]
        )
        self.logger.debug(f"{collision_type} collision processed: f0_SF={f0_SF_value}")

    def save_to_csv(self):
        """Saves the collected collision data to a CSV file."""
        self.logger.info(f"Saving collision data to {self.filename}")
        try:
            with open(self.filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.header)
                writer.writerows(self.collision_data)
            self.logger.info("Collision data saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save collision data: {e}")
