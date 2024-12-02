"""Unit tests for the manage.py script."""

import os
import signal
import subprocess
from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner

from manage import SimulationManager, app

# Initialize test runner
runner = CliRunner()

@pytest.fixture
def sim_manager():
    """Create a SimulationManager instance for testing."""
    return SimulationManager()

@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing process management."""
    with patch('subprocess.Popen') as mock_popen:
        # Configure mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.stdout = []
        mock_process.stderr = []
        mock_popen.return_value = mock_process
        yield mock_popen

@pytest.fixture
def mock_signal():
    """Mock signal handling."""
    with patch('signal.signal') as mock_sig:
        yield mock_sig

class TestSimulationManager:
    """Test cases for SimulationManager class."""

    def test_check_dependency_success(self, sim_manager):
        """Test successful dependency check."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            assert sim_manager._check_dependency('pytest')

    def test_check_dependency_failure(self, sim_manager):
        """Test failed dependency check."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd')
            assert not sim_manager._check_dependency('nonexistent')

    def test_start_simulation_success(self, sim_manager, mock_subprocess, tmp_path):
        """Test successful simulation start."""
        # Create temporary test files
        scenario_file = tmp_path / "test_scenario.json"
        config_file = tmp_path / "test_config.json"
        scenario_file.write_text("{}")
        config_file.write_text("{}")

        server_process, client_process = sim_manager.start_simulation(
            str(scenario_file),
            str(config_file)
        )
        
        assert server_process is not None
        assert client_process is not None
        assert mock_subprocess.call_count == 2

    def test_start_simulation_missing_files(self, sim_manager):
        """Test simulation start with missing files."""
        with pytest.raises(FileNotFoundError):
            sim_manager.start_simulation(
                "nonexistent_scenario.json",
                "nonexistent_config.json"
            )

    def test_monitor_simulation_normal_exit(self, sim_manager, mock_subprocess, mock_signal):
        """Test normal simulation termination."""
        server_process = Mock()
        client_process = Mock()
        
        # Simulate normal exit
        server_process.poll.side_effect = [None, 0]
        client_process.poll.return_value = None
        
        sim_manager.monitor_simulation(server_process, client_process)
        
        assert mock_signal.call_count == 2
        server_process.terminate.assert_not_called()
        client_process.terminate.assert_called_once()

    def test_monitor_simulation_error_exit(self, sim_manager, mock_subprocess, mock_signal):
        """Test simulation termination with error."""
        server_process = Mock()
        client_process = Mock()
        
        # Simulate error exit
        server_process.poll.return_value = 1
        client_process.poll.return_value = None
        
        with pytest.raises(SystemExit) as exc_info:
            sim_manager.monitor_simulation(server_process, client_process)
        
        assert exc_info.value.code == 1
        client_process.terminate.assert_called_once()

    @patch('logging.getLogger')
    def test_stream_output(self, mock_logger, sim_manager):
        """Test process output streaming."""
        process = Mock()
        process.stdout = ["Test output\n"]
        process.stderr = ["Test error\n"]
        
        sim_manager._stream_output(process, "test_process")
        
        mock_logger.return_value.info.assert_called_with("test_process stdout: Test output")
        mock_logger.return_value.error.assert_called_with("test_process stderr: Test error")

class TestCLI:
    """Test cases for CLI commands."""

    def test_run_command_success(self, tmp_path):
        """Test successful run command."""
        # Create temporary test files
        scenario_file = tmp_path / "test_scenario.json"
        config_file = tmp_path / "test_config.json"
        scenario_file.write_text("{}")
        config_file.write_text("{}")

        with patch('manage.SimulationManager.start_simulation') as mock_start:
            with patch('manage.SimulationManager.monitor_simulation'):
                result = runner.invoke(app, [
                    "run",
                    "--scenario", str(scenario_file),
                    "--config", str(config_file)
                ])
                assert result.exit_code == 0

    def test_run_command_missing_files(self):
        """Test run command with missing files."""
        result = runner.invoke(app, [
            "run",
            "--scenario", "nonexistent.json",
            "--config", "nonexistent.json"
        ])
        assert result.exit_code == 1

    def test_install_command(self):
        """Test install command."""
        with patch('manage.SimulationManager.install_dependencies') as mock_install:
            mock_install.return_value = True
            result = runner.invoke(app, ["install"])
            assert result.exit_code == 0

    def test_validate_command(self):
        """Test validate command."""
        with patch('manage.SimulationManager.validate_code_quality') as mock_validate:
            mock_validate.return_value = {'black_formatting': True, 'isort_imports': True}
            result = runner.invoke(app, ["validate", "--code"])
            assert result.exit_code == 0

    def test_benchmark_command(self):
        """Test benchmark command."""
        with patch('manage.SimulationManager.run_benchmarks') as mock_benchmark:
            mock_benchmark.return_value = True
            result = runner.invoke(app, ["benchmark", "--all-benchmarks"])
            assert result.exit_code == 0

    def test_test_command(self):
        """Test test command."""
        with patch('manage.SimulationManager.run_tests') as mock_test:
            mock_test.return_value = True
            result = runner.invoke(app, ["test", "--coverage"])
            assert result.exit_code == 0

    def test_help_output(self):
        """Test help command output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ServerIPC Simulation Management CLI" in result.stdout
