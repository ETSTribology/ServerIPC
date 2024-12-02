#!/usr/bin/env python3
"""
ServerIPC Project Management CLI

This script provides utilities for managing simulation configurations,
running benchmarks, and performing code quality checks.
"""

import logging

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os
import sys
import subprocess
import signal
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
console = Console()

# Create Typer app with metadata
app = typer.Typer(
    help="ServerIPC Simulation Management CLI",
    add_completion=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]}
)

class SimulationManager:
    """Manages simulation processes and project configuration."""
    
    def __init__(
        self, 
        project_root: Optional[str] = None,
        python_executable: Optional[str] = None,
        conda_env_name: Optional[str] = None
    ):
        """
        Initialize the Simulation Manager.
        
        Args:
            project_root: Root directory of the project. Defaults to current working directory.
            python_executable: Path to Python executable. Defaults to sys.executable.
            conda_env_name: Name of Conda environment to activate.
        """
        self.project_root = project_root or os.getcwd()
        
        # Determine Python executable
        if python_executable:
            self.python_executable = python_executable
        elif conda_env_name:
            # Try to find Python in Conda environment
            conda_base = os.path.expanduser("~/anaconda3")
            self.python_executable = os.path.join(
                conda_base, 
                "envs", 
                conda_env_name, 
                "bin", 
                "python"
            )
        else:
            # Default to system Python
            self.python_executable = sys.executable
        
        # Configurable tool paths
        self.tool_paths = {
            'pytest': self._find_tool_path('pytest'),
            'black': self._find_tool_path('black'),
            'isort': self._find_tool_path('isort'),
            'ruff': self._find_tool_path('ruff'),
            'pylint': self._find_tool_path('pylint'),
            'mypy': self._find_tool_path('mypy'),
            'bandit': self._find_tool_path('bandit'),
            'pre-commit': self._find_tool_path('pre-commit')
        }
        
        self.dependencies = [
            'pre-commit',
            'black',
            'ruff',
            'isort',
            'pylint',
            'pytest',
            'pytest-cov',
            'pytest-mock',
            'pytest-sugar',
            'pytest-xdist',
            'hypothesis',
            'faker',
            'freezegun',
            'typer',
            'rich'
        ]

    def _find_tool_path(self, tool_name: str) -> str:
        """
        Find the full path of a tool, checking multiple locations.
        
        Args:
            tool_name: Name of the tool to find
        
        Returns:
            Full path to the tool or just the tool name if not found
        """
        # Possible locations to check
        possible_locations = [
            os.path.join(os.path.dirname(self.python_executable), tool_name),  # Same directory as Python
            os.path.join(sys.prefix, 'bin', tool_name),  # System prefix
            os.path.join(os.path.expanduser('~'), '.local', 'bin', tool_name),  # User local bin
            os.path.join('/usr', 'local', 'bin', tool_name),  # System local bin
            os.path.join('/usr', 'bin', tool_name),  # System bin
            tool_name  # Fallback to just the tool name
        ]
        
        for location in possible_locations:
            if os.path.exists(location) or self._is_executable(location):
                return location
        
        return tool_name  # Fallback to tool name if not found

    def _is_executable(self, path: str) -> bool:
        """
        Check if a path is an executable file.
        
        Args:
            path: Path to check
        
        Returns:
            True if the path is an executable file, False otherwise
        """
        return os.path.isfile(path) and os.access(path, os.X_OK)

    def _check_dependency(self, dependency: str) -> bool:
        """Check if a Python package is installed."""
        try:
            subprocess.run(
                [self.python_executable, '-m', 'pip', 'show', dependency],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def validate_code_quality(self) -> Dict[str, bool]:
        """Run comprehensive code quality checks."""
        validation_results = {
            'black_formatting': False,
            'isort_imports': False,
            'ruff_linting': False,
            'pylint_checks': False,
            'type_checking': False,
            'security_checks': False
        }

        try:
            # Black formatting check
            try:
                subprocess.run([self.tool_paths['black'], '--check', '.'], check=True)
                validation_results['black_formatting'] = True
            except subprocess.CalledProcessError:
                logger.error("Black formatting check failed")

            # isort import sorting check
            try:
                subprocess.run([self.tool_paths['isort'], '--check', '.'], check=True)
                validation_results['isort_imports'] = True
            except subprocess.CalledProcessError:
                logger.error("isort import check failed")

            # Ruff linting check
            try:
                subprocess.run([self.tool_paths['ruff'], 'check', '.'], check=True)
                validation_results['ruff_linting'] = True
            except subprocess.CalledProcessError:
                logger.error("Ruff linting check failed")

            # Pylint check
            try:
                subprocess.run([self.tool_paths['pylint'], 'simulation'], check=True)
                validation_results['pylint_checks'] = True
            except subprocess.CalledProcessError:
                logger.error("Pylint check failed")

            # Type checking with mypy
            try:
                subprocess.run([self.tool_paths['mypy'], 'simulation'], check=True)
                validation_results['type_checking'] = True
            except subprocess.CalledProcessError:
                logger.error("Type checking failed")

            # Security checks with bandit
            try:
                subprocess.run([self.tool_paths['bandit'], '-r', 'simulation'], check=True)
                validation_results['security_checks'] = True
            except subprocess.CalledProcessError:
                logger.error("Security check failed")

            return validation_results

        except Exception as e:
            logger.error(f"Unexpected error during code quality validation: {e}")
            return validation_results

    def run_benchmarks(self, 
                      solver: bool = False, 
                      optimizer: bool = False,
                      memory: bool = False,
                      all_benchmarks: bool = False) -> bool:
        """
        Run specified benchmarks.
        
        Args:
            solver: Run solver benchmarks
            optimizer: Run optimizer benchmarks
            memory: Run memory benchmarks
            all_benchmarks: Run all benchmarks
        """
        try:
            from benchmarks.solver_benchmarks import (
                LinearSolverBenchmark,
                OptimizerBenchmark,
                LineSearchBenchmark,
                MemoryBenchmark
            )

            if all_benchmarks or solver:
                logger.info("Running solver benchmarks...")
                LinearSolverBenchmark().run()

            if all_benchmarks or optimizer:
                logger.info("Running optimizer benchmarks...")
                OptimizerBenchmark().run()
                LineSearchBenchmark().run()

            if all_benchmarks or memory:
                logger.info("Running memory benchmarks...")
                MemoryBenchmark().run()

            return True

        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            return False

    def run_tests(
        self, 
        test_path: Optional[str] = None, 
        coverage: bool = False, 
        verbose: bool = False
    ):
        """
        Run tests using pytest with optional coverage and verbosity.
        
        Args:
            test_path (Optional[str]): Specific test path or file to run
            coverage (bool): Enable coverage reporting
            verbose (bool): Enable verbose output
        """
        # Construct pytest command
        pytest_cmd = [self.tool_paths['pytest']]
        
        # Add coverage if requested
        if coverage:
            pytest_cmd.extend([
                "--cov=simulation", 
                "--cov=visualization", 
                "--cov-report=term", 
                "--cov-report=html"
            ])
        
        # Add verbosity
        if verbose:
            pytest_cmd.append("-v")
        
        # Add specific test path if provided
        if test_path:
            pytest_cmd.append(test_path)
        else:
            pytest_cmd.append("tests")
        
        try:
            # Run pytest
            result = subprocess.run(
                pytest_cmd, 
                check=True, 
                capture_output=False, 
                text=True,
                # Ensure the subprocess uses the same environment
                env=os.environ.copy()
            )
            
            # Generate coverage badge if coverage is enabled
            if coverage:
                coverage_cmd = [self.tool_paths['pytest'], "-m", "genbadge", "coverage"]
                subprocess.run(
                    coverage_cmd, 
                    check=True,
                    env=os.environ.copy()
                )
            
            logger.info("Tests completed successfully.")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Test execution failed: {e}")
            return False
        except FileNotFoundError:
            logger.error(
                f"pytest not found. Current Python: {self.python_executable}. "
                "Please ensure pytest is installed in this environment."
            )
            return False

    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                for dep in self.dependencies:
                    task = progress.add_task(f"Installing {dep}...", total=None)
                    if not self._check_dependency(dep):
                        try:
                            subprocess.run(
                                [self.python_executable, '-m', 'pip', 'install', dep],
                                check=True,
                                capture_output=True
                            )
                        except subprocess.CalledProcessError:
                            logger.error(f"Failed to install {dep}")
                            return False
                    progress.update(task, completed=True)

            # Install pre-commit hooks
            if self._check_dependency('pre-commit'):
                subprocess.run([self.tool_paths['pre-commit'], 'install'], check=True)
                logger.info("Successfully installed pre-commit hooks")

            return True

        except Exception as e:
            logger.error(f"Unexpected error during dependency installation: {e}")
            return False

    def validate_config(self, config_file: str) -> bool:
        """
        Validate simulation configuration file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            from simulation.core.utils.config.config import ConfigManager
            
            config = ConfigManager()
            config.load_simulation_parameters()
            config.load_optimizer_settings()
            config.load_collision_settings()
            config.load_solver_settings()
            config.load_output_settings()
            
            logger.info("Configuration validation successful")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def start_simulation(self, 
                        scenario: str = "scenario/rectangle.json",
                        config: str = "visualization/config.json") -> Tuple[subprocess.Popen, subprocess.Popen]:
        """
        Start both simulation server and visualization client.
        
        Args:
            scenario: Path to the scenario JSON file
            config: Path to the visualization config file
            
        Returns:
            Tuple of server and client processes
        """
        # Validate file paths
        if not os.path.exists(scenario):
            raise FileNotFoundError(f"Scenario file not found: {scenario}")
        
        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file not found: {config}")

        # Define commands
        server_cmd = [self.python_executable, "simulation/server.py", "--json", scenario]
        client_cmd = [self.python_executable, "simulation/visualization/client.py", "--json", config]

        try:
            # Start processes
            server_process = self._start_process(server_cmd, "server")
            client_process = self._start_process(client_cmd, "client")
            
            return server_process, client_process

        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            raise

    def _start_process(self, cmd: List[str], process_name: str) -> subprocess.Popen:
        """Start a subprocess with error handling."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            logger.info(f"Started {process_name} (PID: {process.pid})")
            return process
        except FileNotFoundError:
            logger.error(f"Error: {process_name} script not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to start {process_name}: {e}")
            raise

    def _stream_output(self, process: subprocess.Popen, process_name: str) -> None:
        """Stream output from a process in real-time."""
        for line in process.stdout:
            logger.info(f"{process_name} stdout: {line.strip()}")
        for line in process.stderr:
            logger.error(f"{process_name} stderr: {line.strip()}")

    def monitor_simulation(self, server_process: subprocess.Popen, 
                         client_process: subprocess.Popen) -> None:
        """Monitor and manage simulation processes."""
        def terminate_processes(signum=None, frame=None):
            logger.info("Termination signal received. Shutting down processes...")
            server_process.terminate()
            client_process.terminate()
            sys.exit(0)

        # Set up signal handlers
        signal.signal(signal.SIGINT, terminate_processes)
        signal.signal(signal.SIGTERM, terminate_processes)

        try:
            while True:
                server_retcode = server_process.poll()
                client_retcode = client_process.poll()

                if server_retcode is not None:
                    logger.warning(f"Server terminated with return code {server_retcode}")
                    if server_retcode != 0:
                        client_process.terminate()
                        sys.exit(server_retcode)
                    break

                if client_retcode is not None:
                    logger.warning(f"Client terminated with return code {client_retcode}")
                    if client_retcode != 0:
                        server_process.terminate()
                        sys.exit(client_retcode)
                    break

        except KeyboardInterrupt:
            terminate_processes()

        finally:
            # Ensure both processes are terminated
            if server_process.poll() is None:
                server_process.terminate()
            if client_process.poll() is None:
                client_process.terminate()

    def generate_proto(self):
        """
        Generate protobuf files for the project.
        """
        try:
            subprocess.run(
                [self.python_executable, 'generate_proto.py'], 
                check=True, 
                capture_output=True, 
                text=True
            )
            logger.info("Protobuf files generated successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Protobuf generation failed: {e.stderr}")
            return False

sim_manager = SimulationManager(
    project_root=os.getcwd(),
    python_executable=sys.executable,
    conda_env_name=os.environ.get('CONDA_DEFAULT_ENV')
)

@app.command()
def install():
    """Install project dependencies."""
    if sim_manager.install_dependencies():
        logger.info("Dependencies installed successfully!")
    else:
        logger.error("Failed to install dependencies")
        raise typer.Abort()

@app.command()
def validate(
    code: bool = typer.Option(False, help="Run code quality checks"),
    config: bool = typer.Option(False, help="Validate configuration files"),
    all_checks: bool = typer.Option(False, help="Run all validation checks")
):
    """Validate project code and configuration."""
    if all_checks or code:
        results = sim_manager.validate_code_quality()
        if not all(results.values()):
            logger.error("Code quality validation failed")
            raise typer.Abort()

    if all_checks or config:
        if not sim_manager.validate_config("config.yaml"):
            logger.error("Configuration validation failed")
            raise typer.Abort()

    logger.info("All validations passed successfully!")

@app.command()
def benchmark(
    solver: bool = typer.Option(False, help="Run solver benchmarks"),
    optimizer: bool = typer.Option(False, help="Run optimizer benchmarks"),
    memory: bool = typer.Option(False, help="Run memory benchmarks"),
    all_benchmarks: bool = typer.Option(False, help="Run all benchmarks")
):
    """Run performance benchmarks."""
    if not sim_manager.run_benchmarks(solver, optimizer, memory, all_benchmarks):
        raise typer.Abort()

@app.command()
def test(
    test_path: str = typer.Option(None, help="Specific test path or file to run"),
    coverage: bool = typer.Option(False, help="Enable coverage reporting"),
    verbose: bool = typer.Option(False, help="Enable verbose output")
):
    """Run project tests."""
    if not sim_manager.run_tests(test_path, coverage, verbose):
        raise typer.Abort()

@app.command()
def run(
    scenario: Path = typer.Option(
        Path("scenario/rectangle.json"),
        help="Path to the scenario JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    ),
    config: Path = typer.Option(
        Path("visualization/config.json"),
        help="Path to the visualization config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    ),
    log_level: str = typer.Option(
        "INFO",
        help="Logging level",
        case_sensitive=False
    )
):
    """
    Run the simulation server and visualization client.
    
    This command starts both the simulation server and the visualization client
    processes, monitors their execution, and handles graceful termination.
    """
    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_log_levels:
        console.print(f"[red]Invalid log level. Choose from: {', '.join(valid_log_levels)}[/red]")
        raise typer.Exit(1)

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    try:
        with console.status("[bold green]Starting simulation...") as status:
            # Start simulation processes
            server_process, client_process = sim_manager.start_simulation(
                str(scenario),
                str(config)
            )
            
            status.update("[bold green]Simulation running...")
            
            # Monitor processes
            sim_manager.monitor_simulation(server_process, client_process)
            
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)

@app.command()
def generate_proto():
    """Generate protobuf files for the project."""
    if sim_manager.generate_proto():
        logger.info("Protobuf files generated successfully!")
    else:
        logger.error("Failed to generate protobuf files")
        raise typer.Abort()

def main():
    app()

if __name__ == "__main__":
    main()
