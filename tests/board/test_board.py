import os
import pytest
import torch
import tempfile
import matplotlib.pyplot as plt
import logging

from simulation.board.board import (
    BoardBase, 
    TensorBoardLogger, 
    BoardFactory
)
from simulation.core.registry.decorators import register
from simulation.core.registry.container import RegistryContainer
from simulation.core.config.config import ConfigManager

# Add setup method to ensure configuration is initialized before tests
@pytest.fixture(autouse=True)
def initialize_config():
    ConfigManager.reset()
    ConfigManager.quick_init()

class TestBoardBase:
    def test_board_base_abstract_methods(self):
        """Ensure BoardBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BoardBase()  # type: ignore

    def test_context_manager(self):
        """Test the context manager functionality."""
        class MockLogger(BoardBase):
            def __init__(self):
                self.closed = False

            def log_scalar(self, *args, **kwargs):
                pass
            
            def log_scalars(self, *args, **kwargs):
                pass
            
            def log_histogram(self, *args, **kwargs):
                pass
            
            def log_image(self, *args, **kwargs):
                pass
            
            def log_figure(self, *args, **kwargs):
                pass
            
            def log_text(self, *args, **kwargs):
                pass
            
            def flush(self):
                pass
            
            def close(self):
                self.closed = True

        with MockLogger() as logger:
            assert isinstance(logger, MockLogger)
            assert not logger.closed

        assert logger.closed


class TestTensorBoardLogger:
    @pytest.fixture
    def temp_logger(self):
        """Create a TensorBoard logger with a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TensorBoardLogger(log_dir=temp_dir)
            yield logger
            logger.close()

    def test_tensorboard_logger_initialization(self, temp_logger):
        """Test TensorBoard logger initialization."""
        assert isinstance(temp_logger, BoardBase)
        assert hasattr(temp_logger, 'writer')
        assert os.path.exists(temp_logger.tensor_data_dir)
        assert temp_logger._initialized is True

    def test_singleton_behavior(self):
        """Test that TensorBoardLogger is a singleton."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger1 = TensorBoardLogger(log_dir=temp_dir)
            logger2 = TensorBoardLogger(log_dir=temp_dir)
            assert logger1 is logger2

    def test_log_scalar(self, temp_logger, caplog):
        """Test logging a scalar value."""
        caplog.set_level(logging.DEBUG)
        temp_logger.log_scalar('test_scalar', 42.0, global_step=1)
        assert any('Logged scalar' in record.message for record in caplog.records)

    def test_log_scalars(self, temp_logger, caplog):
        """Test logging multiple scalar values."""
        caplog.set_level(logging.DEBUG)
        scalar_dict = {
            'loss': 0.1,
            'accuracy': 0.95
        }
        temp_logger.log_scalars('metrics', scalar_dict, global_step=1)
        assert any('Logged scalars' in record.message for record in caplog.records)

    def test_log_histogram(self, temp_logger, caplog):
        """Test logging a histogram."""
        caplog.set_level(logging.DEBUG)
        data = torch.randn(100)
        temp_logger.log_histogram('test_histogram', data, global_step=1)
        
        # Check logging
        assert any('Logged histogram' in record.message for record in caplog.records)
        
        # Check tensor data saved
        expected_path = os.path.join(temp_logger.tensor_data_dir, 'test_histogram_1.pt')
        assert os.path.exists(expected_path)

    def test_log_image(self, temp_logger, caplog):
        """Test logging an image."""
        caplog.set_level(logging.DEBUG)
        image = torch.rand(3, 64, 64)  # RGB image
        temp_logger.log_image('test_image', image, global_step=1)
        
        # Check logging
        assert any('Logged image' in record.message for record in caplog.records)
        
        # Check tensor data saved
        expected_path = os.path.join(temp_logger.tensor_data_dir, 'test_image_1.pt')
        assert os.path.exists(expected_path)

    def test_log_figure(self, temp_logger, caplog):
        """Test logging a matplotlib figure."""
        caplog.set_level(logging.DEBUG)
        plt.figure()
        plt.plot([1, 2, 3], [4, 5, 6])
        temp_logger.log_figure('test_figure', plt.gcf(), global_step=1)
        plt.close()  # Clean up
        
        # Check logging
        assert any('Logged figure' in record.message for record in caplog.records)

    def test_log_text(self, temp_logger, caplog):
        """Test logging a text string."""
        caplog.set_level(logging.DEBUG)
        temp_logger.log_text('test_text', 'Hello, TensorBoard!', global_step=1)
        assert any('Logged text' in record.message for record in caplog.records)

    def test_flush_and_close(self, temp_logger, caplog):
        """Test flushing and closing the logger."""
        caplog.set_level(logging.INFO)
        temp_logger.log_scalar('test', 42.0)
        temp_logger.flush()
        temp_logger.close()
        
        # Check logging messages
        assert any('Flushed' in record.message for record in caplog.records)
        assert any('TensorBoard logger closed' in record.message for record in caplog.records)


class TestBoardFactory:
    def test_board_factory_singleton(self):
        """Test that BoardFactory is a singleton."""
        factory1 = BoardFactory()
        factory2 = BoardFactory()
        assert factory1 is factory2

    def test_board_factory_create_tensorboard(self):
        """Test creating a TensorBoard logger through the factory."""
        factory = BoardFactory()
        
        # Create a temporary directory for logs
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = factory.create('tensorboard', log_dir=temp_dir)
            
            assert isinstance(logger, TensorBoardLogger)
            assert logger.log_dir == temp_dir

    def test_board_factory_invalid_type(self):
        """Test creating a board with an invalid type raises an exception."""
        factory = BoardFactory()
        
        with pytest.raises(Exception):
            factory.create('non_existent_logger')

    def test_board_factory_registry(self):
        """Test that the board is correctly registered in the registry."""
        from simulation.core.registry.container import RegistryContainer

        registry = RegistryContainer()
        board_registry = registry.board

        # Check that 'tensorboard' is registered
        assert 'tensorboard' in board_registry._registry
        assert board_registry.get('tensorboard') is TensorBoardLogger


class TestCustomBoardRegistration:
    def test_custom_board_registration(self):
        """Test registering and using a custom board logger."""
        from simulation.core.registry.decorators import register
        from simulation.core.registry.container import RegistryContainer

        @register(type="board", name="custom_logger")
        class CustomBoardLogger(BoardBase):
            def __init__(self, cfg=None, **kwargs):
                super().__init__(cfg or {}, **kwargs)

            def log_scalar(self, *args, **kwargs):
                pass
            
            def log_scalars(self, *args, **kwargs):
                pass
            
            def log_histogram(self, *args, **kwargs):
                pass
            
            def log_image(self, *args, **kwargs):
                pass
            
            def log_figure(self, *args, **kwargs):
                pass
            
            def log_text(self, *args, **kwargs):
                pass
            
            def flush(self):
                pass
            
            def close(self):
                pass

        # Verify the custom logger is registered
        registry = RegistryContainer()
        board_registry = registry.board
        
        assert 'custom_logger' in board_registry._registry
        assert board_registry.get('custom_logger') is CustomBoardLogger

        # Test creating the custom logger through the factory
        factory = BoardFactory()
        custom_logger = factory.create('custom_logger')
        assert isinstance(custom_logger, CustomBoardLogger)
