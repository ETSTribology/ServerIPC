import polyscope as ps
import polyscope.imgui as psim

import webbrowser

from visualization.config.config import VisualizationConfigManager
from visualization.storage.factory import StorageFactory
from visualization.backend.factory import BackendFactory
from visualization.extension.board.factory import BoardFactory

from visualization.storage.storage import Storage
from visualization.backend.backend import Backend
from visualization.extension.board.board import Board

import logging
logger = logging.getLogger(__name__)



class Polyscope:
    def __init__(self, config: VisualizationConfigManager, storage: Storage, backend: Backend, board: Board):
        self.config = config
        self.storage = storage
        self.backend = backend
        self.board = board
        self.performance_tracking = config.get().get("visualization", {}).get("performanceTracking", False)
        self.color_scheme = config.get().get("visualization", {}).get("colorScheme", "dark")
        self.interactive_mode = config.get().get("visualization", {}).get("interactiveMode", True)
        self.screen_width = config.get().get("visualization", {}).get("screenWidth", 1024)
        self.screen_height = config.get().get("visualization", {}).get("screenHeight", 768)

        self.steps = []
        self.selected_step = None
        self.screenshot_enabled = config.get().get("extensions", {}).get("screenshot", {}).get("enabled", False)
        self.mesh_save_enabled = config.get().get("extensions", {}).get("mesh", {}).get("enabled", False)
        self.board_url = f"http://{config.get().get('extensions', {}).get('board', {}).get('host', 'localhost')}:" \
                         f"{config.get().get('extensions', {}).get('board', {}).get('port', 6006)}"



    def initialize(self):
        ps.init()
        if self.color_scheme == "dark":
            ps.set_ground_plane_mode("tile_reflection")
        else:
            ps.set_ground_plane_mode("shadow_only")

        if self.performance_tracking:
            pass
        else:
            ps.set_verbosity(0)
            ps.set_use_prefs_file(False)

        ps.set_user_callback(self.callback)

    def callback(self):
        """
        This method will be executed every frame as the UI is updated.
        """
        # Display a menu with user-interactable settings
        self.menu()

    def menu(self):
        """
        Render a dynamic menu using ImGui.
        """

        # Color Scheme Toggle
        if psim.Button(f"Toggle Color Scheme ({self.color_scheme})"):
            self.color_scheme = "light" if self.color_scheme == "dark" else "dark"
            if self.color_scheme == "dark":
                ps.set_ground_plane_mode("tile_reflection")
            else:
                ps.set_ground_plane_mode("shadow_only")

        # Performance Tracking
        changed, self.performance_tracking = psim.Checkbox("Performance Tracking", self.performance_tracking)
        if changed:
            logger.info(f"Performance Tracking toggled: {self.performance_tracking}")

        psim.Separator()

        # Backend Configuration
        if psim.TreeNode("Backend Configuration"):
            backend_config = self.config.get().get("backend", {}).get("config", {})
            for key, value in backend_config.items():
                new_value = psim.InputText(key, str(value))[1]
                backend_config[key] = new_value
            psim.TreePop()

        psim.Separator()

        # Simulation Controls
        if psim.TreeNode("Simulation Controls"):
            if psim.Button("Start"):
                logger.info("Simulation started")
                self.backend.send_command("start")
            psim.SameLine()
            if psim.Button("Stop"):
                logger.info("Simulation stopped")
                self.backend.send_command("stop")
            psim.SameLine()
            if psim.Button("Pause"):
                logger.info("Simulation paused")
                self.backend.send_command("pause")
            psim.SameLine()
            if psim.Button("Continue"):
                logger.info("Simulation continued")
                self.backend.send_command("continue")

            # Dropdown for steps
            if self.steps:
                changed, self.selected_step = psim.Combo("Steps", self.selected_step or "", self.steps)
                if changed:
                    logger.info(f"Step selected: {self.steps[self.selected_step]}")

            psim.TreePop()

        psim.Separator()

        # Extensions Configuration
        if psim.TreeNode("Extensions"):
            # Screenshot Configuration
            if psim.TreeNode("Screenshot"):
                changed, self.screenshot_enabled = psim.Checkbox("Enable Screenshots", self.screenshot_enabled)
                if changed:
                    logger.info(f"Screenshots toggled: {self.screenshot_enabled}")

                screenshot_dir = self.config.get().get("extensions", {}).get("screenshot", {}).get("directory", "")
                new_dir = psim.InputText("Directory", screenshot_dir)[1]
                self.config.get()["extensions"]["screenshot"]["directory"] = new_dir
                psim.TreePop()

            # Mesh Saving Configuration
            if psim.TreeNode("Mesh Saving"):
                changed, self.mesh_save_enabled = psim.Checkbox("Enable Mesh Saving", self.mesh_save_enabled)
                if changed:
                    logger.info(f"Mesh saving toggled: {self.mesh_save_enabled}")

                mesh_dir = self.config.get().get("extensions", {}).get("mesh", {}).get("directory", "")
                new_dir = psim.InputText("Directory", mesh_dir)[1]
                self.config.get()["extensions"]["mesh"]["directory"] = new_dir
                psim.TreePop()

            # Board Configuration
            if psim.TreeNode("Board"):
                if psim.Button("Open Board Website"):
                    webbrowser.open(self.board_url)
                board_dir = self.config.get().get("extensions", {}).get("board", {}).get("directory", "")
                new_dir = psim.InputText("Directory", board_dir)[1]
                self.config.get()["extensions"]["board"]["directory"] = new_dir
                psim.TreePop()

            psim.TreePop()

        psim.End()

    def show(self):
        """
        Display the Polyscope visualization and clear the user callback after closing.
        """
        ps.show()
        ps.clear_user_callback()

    def main(self):
        """
        Main entry point for the visualization.
        """
        self.initialize()
        self.show()
