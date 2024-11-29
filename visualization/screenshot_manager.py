import os
import logging
import imageio.v3 as iio
import polyscope as ps

logger = logging.getLogger(__name__)

class ScreenshotManager:
    def __init__(self, directory="screenshots"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Screenshot directory set to '{self.directory}'.")

    def save_screenshot(self, step_name):
        filename = os.path.join(
            self.directory, f"screenshot_{step_name.replace(' ', '_')}.png")
        try:
            ps.screenshot(filename)
            logger.info(f"Screenshot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save screenshot '{filename}': {e}")

    def create_gif(self, output_filename="mesh_simulation.gif", duration=0.5):
        try:
            screenshots = sorted(
                [img for img in os.listdir(self.directory) if img.endswith(".png")]
            )
            images = [iio.imread(os.path.join(self.directory, img))
                      for img in screenshots]
            iio.imwrite(output_filename, images,
                        extension=".gif", duration=duration)
            logger.info(f"Created GIF: {output_filename}")
        except Exception as e:
            logger.error(f"Failed to create GIF '{output_filename}': {e}")