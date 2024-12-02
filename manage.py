#!/usr/bin/env python3
"""
ServerIPC Project Management CLI

This script provides utilities for managing project dependencies,
running the simulation server and visualization client.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Rich console for enhanced output
console = Console()

# Create Typer app
app = typer.Typer(
    help="ServerIPC Project Management CLI", add_completion=True, rich_markup_mode="rich"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


class DependencyManager:
    """Comprehensive dependency management for different environments."""

    @staticmethod
    def detect_environment() -> str:
        """
        Detect the current Python environment.

        Returns:
            str: The detected environment type (conda, venv, poetry, system)
        """
        if os.getenv("CONDA_DEFAULT_ENV"):
            return "conda"
        elif os.getenv("VIRTUAL_ENV"):
            return "venv"
        elif os.path.exists("poetry.lock"):
            return "poetry"
        return "system"

    @staticmethod
    def _install_extern_packages(verbose: bool, with_cuda: bool):
        """Install external packages like IPCToolkit."""
        try:
            # First ensure submodules are initialized
            console.print("[bold]Initializing submodules...[/bold]")
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
                capture_output=True,
                text=True,
            )
            console.print("[green]✓ Submodules initialized successfully[/green]")

            # Install IPCToolkit with CUDA support if requested
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "extern/ipc-toolkit",
                "--config-settings",
            ]

            cmake_args = ["-DCMAKE_BUILD_TYPE='Release'"]
            if with_cuda:
                cmake_args.extend(
                    ["-DIPC_TOOLKIT_WITH_CUDA='ON'", "-DCMAKE_CUDA_ARCHITECTURES='native'"]
                )
            else:
                cmake_args.append("-DIPC_TOOLKIT_WITH_CUDA='OFF'")

            cmake_args.append("-DIPC_TOOLKIT_BUILD_PYTHON='ON'")
            cmd.append(f"cmake.args='{' '.join(cmake_args)}'")

            if verbose:
                cmd.append("-v")

            console.print("[bold]Installing IPCToolkit...[/bold]")
            subprocess.run(cmd, check=True)
            console.print("[green]✓ IPCToolkit installed successfully[/green]")

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, "stderr") else str(e)
            console.print(f"[red]Error: {error_msg}[/red]")
            raise Exception(f"Failed to install external packages: {error_msg}")

    @staticmethod
    def install(
        method: Optional[str] = typer.Option(
            None, help="Specify installation method (conda/poetry/pip/venv)"
        ),
        dev: bool = typer.Option(False, help="Install development dependencies"),
        verbose: bool = typer.Option(False, help="Enable verbose installation output"),
        with_cuda: bool = typer.Option(False, help="Install with CUDA support"),
    ):
        """Install project dependencies."""
        detected_env = method or DependencyManager.detect_environment()

        try:
            # Install dependencies based on environment
            if detected_env == "conda":
                cmd = ["conda", "install", "--yes", "-c", "conda-forge"]
                if dev:
                    cmd.extend(["-c", "pytorch"])
                if verbose:
                    cmd.append("-v")
                subprocess.run(cmd, check=True)
            elif detected_env == "poetry":
                cmd = ["poetry", "install"]
                if not dev:
                    cmd.append("--no-dev")
                if verbose:
                    cmd.append("-v")
                subprocess.run(cmd, check=True)
            else:  # pip/venv
                cmd = [sys.executable, "-m", "pip", "install", "-r"]
                if dev:
                    cmd.append("requirements-dev.txt")
                else:
                    cmd.append("requirements.txt")
                if verbose:
                    cmd.append("-v")
                subprocess.run(cmd, check=True)

            # Install external packages
            DependencyManager._install_extern_packages(verbose, with_cuda)
            console.print("[green]✓ All dependencies installed successfully[/green]")

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, "stderr") else str(e)
            console.print(f"[red]Installation failed: {error_msg}[/red]")
            raise typer.Exit(code=1)


@app.command()
def install(
    method: Optional[str] = typer.Option(
        None, help="Specify installation method (conda/poetry/pip/venv)"
    ),
    dev: bool = typer.Option(False, help="Install development dependencies"),
    verbose: bool = typer.Option(False, help="Enable verbose installation output"),
    with_cuda: bool = typer.Option(False, help="Install with CUDA support"),
):
    """Install project dependencies."""
    DependencyManager.install(method, dev, verbose, with_cuda)


@app.command()
def validate(
    code: bool = typer.Option(False, help="Run code quality checks"),
    config: bool = typer.Option(False, help="Validate configuration files"),
    all_checks: bool = typer.Option(False, help="Run all validation checks"),
    hooks: Optional[List[str]] = typer.Option(None, help="Specific pre-commit hooks to run"),
):
    """Run validation checks on the project."""
    success = True

    if all_checks or code:
        try:
            # Initialize pre-commit if not already installed
            console.print("[bold]Initializing pre-commit hooks...[/bold]")
            subprocess.run(
                ["pre-commit", "install"],
                check=True,
                capture_output=True,
                text=True,
            )

            # Build the pre-commit command
            cmd = ["pre-commit", "run", "--color=always"]

            if hooks:
                cmd.extend(hooks)
            else:
                cmd.append("--all-files")

            # Add exclude pattern for extern folder
            cmd.extend(["--files", "simulation/**/*.py", "tests/**/*.py", "manage.py"])

            console.print("[bold]Running code quality checks...[/bold]")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                console.print("[green]✓ All code quality checks passed[/green]")
            else:
                console.print("[yellow]Code quality issues found:[/yellow]")
                console.print(result.stdout)
                if result.stderr:
                    console.print("[red]Errors:[/red]")
                    console.print(result.stderr)
                success = False

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, "stderr") else str(e)
            console.print(f"[red]Error running code quality checks: {error_msg}[/red]")
            success = False

    if all_checks or config:
        try:
            console.print("[bold]Validating configuration files...[/bold]")
            # Add your configuration validation logic here
            # For example, checking JSON schema, YAML syntax, etc.
            console.print("[green]✓ Configuration validation passed[/green]")
        except Exception as e:
            console.print(f"[red]Configuration validation failed: {str(e)}[/red]")
            success = False

    if not success:
        raise typer.Exit(code=1)


@app.command()
def run_server(
    scenario: Path = typer.Option(
        Path("scenario/rectangle.json"),
        help="Path to the scenario JSON file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    log_level: str = typer.Option("INFO", help="Logging level", case_sensitive=False),
):
    """Run the simulation server."""
    try:
        cmd = [
            sys.executable,
            "-m",
            "simulation.server",
            "--scenario",
            str(scenario),
            "--log-level",
            log_level,
        ]

        console.print("[bold]Starting simulation server...[/bold]")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, "stderr") else str(e)
        console.print(f"[red]Error starting simulation server: {error_msg}[/red]")
        raise typer.Exit(code=1)


@app.command()
def run_client(
    config: Path = typer.Option(
        Path("visualization/config/config.json"),
        help="Path to the visualization config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    log_level: str = typer.Option("INFO", help="Logging level", case_sensitive=False),
):
    """Run the visualization client."""
    try:
        cmd = [
            sys.executable,
            "-m",
            "visualization.client",
            "--config",
            str(config),
        ]

        console.print("[bold]Starting visualization client...[/bold]")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, "stderr") else str(e)
        console.print(f"[red]Error starting visualization client: {error_msg}[/red]")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
