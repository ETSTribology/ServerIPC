import logging
import os
import typer
from visualization.config.config import VisualizationConfigManager

from visualization.storage.factory import StorageFactory
from visualization.backend.factory import BackendFactory
from visualization.extension.board.factory import BoardFactory

from visualization.polyscope import Polyscope

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

app = typer.Typer()


class VisualizationClient:
    def __init__(self, config: str):
        self.config_manager = None
        self.storage = None
        self.backend = None
        self.board = None
        self.polyscope = None
        self.config_path = config

    def load_configuration(self):
        logger.info("Loading configuration...")
        self.config_manager = VisualizationConfigManager(config_path=self.config_path)
        config = self.config_manager.get()
        logger.debug(f"Loaded configuration: {config}")

    def initialize_storage(self):
        logger.info("Initializing storage...")
        self.storage = StorageFactory.create(self.config_manager.get())

    def initialize_backend(self):
        logger.info("Initializing backend...")
        self.backend = BackendFactory.create(self.config_manager.get())

    def initialize_board(self):
        logger.info("Initializing board...")
        self.board = BoardFactory.create(self.config_manager.get())

    def initialize_polyscope(self):
        logger.info("Initializing Polyscope...")
        self.polyscope = Polyscope(config=self.config_manager, storage=self.storage, backend=self.backend, board=self.board)
        self.polyscope.main()

    def run(self):
        self.load_configuration()
        self.initialize_storage()
        self.initialize_backend()
        self.initialize_board()
        self.initialize_polyscope()
        logger.info("Visualization client started.")

@app.command()
def start(
    config: str = typer.Option(
        CONFIG_FILE,
        help="Path to the configuration file (JSON or YAML)"
    ),
    backend: str = typer.Option(
        None, 
        help="Specify the backend to override configuration settings."
    )
):
    """
    Start the Polyscope Visualization Client.
    """
    client = VisualizationClient(config=config)
    client.run()

if __name__ == "__main__":
    app()
