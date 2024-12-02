import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import logging
import queue
import os
from .screenshot_manager import ScreenshotManager
from visualization.backends.base import BaseCommunicationBackend
from visualization.backends.redis_backend import RedisCommunicationBackend

logger = logging.getLogger(__name__)

class PolyscopeVisualizer:
    def __init__(
        self,
        mesh_queue: queue.Queue,
        communication_backend: BaseCommunicationBackend,
        screenshot_manager: ScreenshotManager,
    ):
        ps.set_up_dir("z_up")
        ps.init()
        self.mesh_states = []
        self.step_names = []
        self.current_step_index = 0
        self.is_paused = False
        self.visual_mesh = None
        self.initialized = False
        self.mesh_queue = mesh_queue
        self.communication_backend = communication_backend
        self.screenshot_manager = screenshot_manager
        self.material_colors = {}
        self.faces = None

    def register_initial_mesh(
        self,
        BX: np.ndarray,
        faces: np.ndarray,
        face_materials: np.ndarray,
        materials: list,
    ):
        logger.info("Registering initial mesh with Polyscope.")
        if not self.initialized:
            self.faces = faces  # Store the faces for later use
            self.visual_mesh = ps.register_surface_mesh("Visual Mesh", BX, self.faces)
            # Build material colors mapping
            self.material_colors = self.build_material_colors(materials)
            # Map face materials to colors
            face_colors = np.array(
                [
                    self.material_colors.get(mat_idx, [0.5, 0.5, 0.5])
                    for mat_idx in face_materials
                ]
            )
            self.visual_mesh.add_color_quantity(
                "material_color", face_colors, defined_on="faces", enabled=True
            )
            self.mesh_states.append(
                {"BX": BX.copy(), "face_materials": face_materials.copy(), "step": "Initial_State"}
            )
            self.step_names.append("Initial_State")
            self.initialized = True
            logger.info("Initial mesh registered with Polyscope.")

    def build_material_colors(self, materials: list):
        material_colors = {}
        for idx, mat in enumerate(materials):
            color = mat.get("color", [128, 128, 128, 1])  # Default to gray
            # Normalize color to [0,1]
            color = np.array(color[:3]) / 255.0
            material_colors[idx] = color
        return material_colors

    def update_mesh(
        self, BX: np.ndarray, face_materials: np.ndarray, step_name: str
    ):
        if self.visual_mesh:
            # Check if the new vertex array has the same size as the current mesh
            if BX.shape[0] != self.visual_mesh.n_vertices():
                logger.info(
                    f"Vertex count changed, re-registering mesh for step: {step_name}"
                )
                self.visual_mesh = ps.register_surface_mesh(
                    "Visual Mesh", BX, self.faces
                )
            else:
                logger.info(f"Updating vertex positions for step: {step_name}")
                self.visual_mesh.update_vertex_positions(BX)

            # Update face colors based on materials
            face_colors = np.array(
                [
                    self.material_colors.get(mat_idx, [0.5, 0.5, 0.5])
                    for mat_idx in face_materials
                ]
            )
            self.visual_mesh.add_color_quantity(
                "material_color", face_colors, defined_on="faces", enabled=True
            )

            # Store the new state
            self.mesh_states.append(
                {"BX": BX.copy(), "face_materials": face_materials.copy(), "step": step_name}
            )
            self.step_names.append(step_name)
            self.current_step_index = len(self.mesh_states) - 1

            logger.info(f"Mesh updated to {step_name}.")

            # Save mesh using MeshManager
            mesh = meshio.Mesh(points=BX, cells=[("triangle", self.faces)])
            self.mesh_manager.save_mesh(mesh, step_name)

            # Capture screenshot
            self.screenshot_manager.save_screenshot(step_name)

    def reset_to_initial_state(self):
        if self.mesh_states:
            self.current_step_index = 0
            initial_state = self.mesh_states[0]
            self.visual_mesh.update_vertex_positions(initial_state["BX"])
            face_materials = initial_state["face_materials"]
            face_colors = np.array(
                [
                    self.material_colors.get(mat_idx, [0.5, 0.5, 0.5])
                    for mat_idx in face_materials
                ]
            )
            self.visual_mesh.add_color_quantity(
                "material_color", face_colors, defined_on="faces", enabled=True
            )
            logger.info("Mesh reset to initial state.")

    def process_queue(self):
        while not self.mesh_queue.empty():
            mesh_data = self.mesh_queue.get()
            try:
                # Deserialize the mesh data
                BX = np.frombuffer(
                    mesh_data["BX"], dtype=mesh_data["BX_dtype"]
                ).reshape(mesh_data["BX_shape"])
                faces = np.frombuffer(
                    mesh_data["faces"], dtype=mesh_data["faces_dtype"]
                ).reshape(mesh_data["faces_shape"])
                face_materials = np.frombuffer(
                    mesh_data["face_materials"],
                    dtype=mesh_data["face_materials_dtype"],
                ).reshape(mesh_data["face_materials_shape"])
                step_number = mesh_data["step"]  # Use the step number from the mesh data

                # Use the step number in the step name
                step_name = f"Step_{step_number}"

                # Update the mesh with the deserialized data
                self.update_mesh(BX, face_materials, step_name)

            except Exception as e:
                logger.error(f"Failed to process mesh update: {e}")

    def render_ui(self):
        # Handle Pause and Play buttons
        if imgui.Button("Pause"):
            self.is_paused = True
            self.communication_backend.send_command("pause")
            logger.info("Visualization Paused.")
        imgui.SameLine()
        if imgui.Button("Play"):
            self.is_paused = False
            self.communication_backend.send_command("play")
            logger.info("Visualization Resumed.")

        # Handle Stop and Start buttons
        if imgui.Button("Stop"):
            self.communication_backend.send_command("stop")
            logger.info("Sent stop command to simulation server.")
            self.reset_to_initial_state()
            self.mesh_states = []
            self.step_names = []
            self.is_paused = False
            logger.info("Stopped simulation and reset mesh to initial state.")
        imgui.SameLine()
        if imgui.Button("Start"):
            self.communication_backend.send_command("start")
            logger.info("Sent start command to simulation server.")

        imgui.Separator()

        # Handle Reset button
        if imgui.Button("Reset"):
            self.communication_backend.send_command("reset")
            logger.info("Sent reset command to simulation server.")

        # Handle Kill Simulation button
        if imgui.Button("Kill Simulation"):
            self.communication_backend.send_command("kill")
            logger.info("Sent kill command to simulation server.")

        imgui.Separator()

        # Handle Create GIF button
        if imgui.Button("Create GIF"):
            self.screenshot_manager.create_gif()
            logger.info("Requested GIF creation from screenshots.")

        imgui.Separator()

        # Process any pending mesh updates
        self.process_queue()

        # Add UI list to select steps
        if len(self.mesh_states) > 1:
            step_names = [state["step"] for state in self.mesh_states]
            changed, new_step_index = imgui.Combo(
                "Select Step", self.current_step_index, step_names
            )
            if changed:
                self.current_step_index = new_step_index
                selected_state = self.mesh_states[self.current_step_index]
                self.visual_mesh.update_vertex_positions(selected_state["BX"])
                face_materials = selected_state["face_materials"]
                face_colors = np.array(
                    [
                        self.material_colors.get(mat_idx, [0.5, 0.5, 0.5])
                        for mat_idx in face_materials
                    ]
                )
                self.visual_mesh.add_color_quantity(
                    "material_color", face_colors, defined_on="faces", enabled=True
                )
                logger.info(
                    f"Switched to {self.mesh_states[self.current_step_index]['step']}."
                )

    def show(self):
        ps.set_user_callback(self.render_ui)
        try:
            ps.show()
        except KeyboardInterrupt:
            logger.info("Shutting down visualizer.")
            import sys
            sys.exit(0)