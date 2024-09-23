import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
import logging
import queue

logger = logging.getLogger(__name__)

class PolyscopeVisualizer:
    def __init__(self, mesh_queue: queue.Queue, redis_client):
        ps.set_up_dir("z_up")
        ps.init()
        self.mesh_states = []
        self.step_names = []
        self.current_step_index = 0
        self.is_paused = False
        self.visual_mesh = None
        self.initialized = False
        self.mesh_queue = mesh_queue
        self.redis_client = redis_client

    def register_initial_mesh(self, BX: np.ndarray, faces: np.ndarray):
        if not self.initialized:
            self.visual_mesh = ps.register_surface_mesh("Visual Mesh", BX, faces)
            self.mesh_states.append(BX.copy())
            self.step_names.append("Initial State")
            self.initialized = True
            logger.info("Initial mesh registered with Polyscope.")

    def update_mesh(self, BX: np.ndarray, step_name: str):
        if self.visual_mesh:
            self.visual_mesh.update_vertex_positions(BX)
            self.mesh_states.append(BX.copy())
            self.step_names.append(step_name)
            self.current_step_index = len(self.mesh_states) - 1
            logger.info(f"Mesh updated to {step_name}.")

    def reset_to_initial_state(self):
        if self.mesh_states:
            self.current_step_index = 0
            self.visual_mesh.update_vertex_positions(self.mesh_states[0])
            logger.info("Mesh reset to initial state.")

    def process_queue(self):
        while not self.mesh_queue.empty():
            mesh_data = self.mesh_queue.get()
            try:
                BX = np.frombuffer(mesh_data['BX'], dtype=mesh_data['BX_dtype']).reshape(mesh_data['BX_shape'])
                step_name = f"Step {len(self.mesh_states)}"
                self.update_mesh(BX, step_name)
            except Exception as e:
                logger.error(f"Failed to process mesh update: {e}")

    def render_ui(self):
        # Handle Pause and Play buttons
        if imgui.Button("Pause"):
            self.is_paused = True
            self.redis_client.send_command("pause")
            logger.info("Visualization Paused.")
        imgui.SameLine()
        if imgui.Button("Play"):
            self.is_paused = False
            self.redis_client.send_command("play")
            logger.info("Visualization Resumed.")

        # Handle Stop and Start buttons
        if imgui.Button("Stop"):
            self.redis_client.send_command("stop")
            logger.info("Sent stop command to simulation server.")
            self.reset_to_initial_state()
            self.mesh_states = []
            self.step_names = []
            self.is_paused = False
            logger.info("Stopped simulation and reset mesh to initial state.")
        imgui.SameLine()
        if imgui.Button("Start"):
            self.redis_client.send_command("start")
            logger.info("Sent start command to simulation server.")
        # Add Reset button
        if imgui.Button("Reset"):           
            self.redis_client.send_command("reset")
            logger.info("Sent reset command to simulation server.")
        imgui.Separator()
        if imgui.Button("Kill Simulation"):
            self.redis_client.send_command("kill")
            logger.info("Sent kill command to simulation server.")

        # Handle Reset to Initial State button
        if imgui.Button("Reset to Initial State"):
            self.reset_to_initial_state()

        # Add UI list to select steps
        if len(self.mesh_states) > 1:
            changed, new_step_index = imgui.Combo("Select Step", self.current_step_index, self.step_names)
            if changed:
                self.current_step_index = new_step_index
                self.visual_mesh.update_vertex_positions(self.mesh_states[self.current_step_index])
                logger.info(f"Switched to {self.step_names[self.current_step_index]}.")

        # Process any pending mesh updates
        self.process_queue()

    def show(self):
        ps.set_user_callback(self.render_ui)
        ps.show()
