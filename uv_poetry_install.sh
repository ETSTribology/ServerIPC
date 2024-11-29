#!/bin/bash

# uv_poetry_install.sh
# A script to replace 'poetry install' using uv for faster dependency installation.

set -e  # Exit immediately if a command exits with a non-zero status.

# Function to display usage instructions
usage() {
    echo "Usage: $0 [--keyring]"
    echo ""
    echo "Options:"
    echo "  --keyring    Handle keyring authentication."
    exit 1
}

# Parse command-line arguments
KEYRING=false
if [ "$1" == "--keyring" ]; then
    KEYRING=true
elif [ -n "$1" ]; then
    usage
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Verify that Poetry is installed
if ! command_exists poetry; then
    echo "Error: Poetry is not installed. Please install Poetry before running this script."
    exit 1
fi

# Verify that uv is installed
if ! command_exists uv; then
    echo "Error: uv is not installed. Please install uv before running this script."
    exit 1
fi

# Step 1: Export dependencies using Poetry
echo "Step 1: Exporting dependencies using 'poetry export'..."
# Suppress Poetry warnings during export
export POETRY_WARNINGS_EXPORT=false
# Export to requirements.txt format without hashes and including dev dependencies
requirements=$(poetry export --without-hashes --with dev -f requirements.txt)

# Step 2: Handle Keyring Authentication if requested
if [ "$KEYRING" = true ]; then
    echo "Step 2: Handling keyring authentication..."
    # Install the keyring provider required for authentication
    uv pip install keyrings.google-artifactregistry-auth

    # Modify the --extra-index-url to include the oauth2accesstoken prefix
    # This is necessary for authenticated access to private repositories
    requirements=$(echo "$requirements" | sed 's|--extra-index-url https://us-central1-python.pkg.dev/|--extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/|g')
fi

# Step 3: Install dependencies using uv pip
echo "Step 3: Installing dependencies using 'uv pip install'..."
# Use process substitution to pass the requirements to uv pip install
uv pip install --no-deps -r <(echo "$requirements")

# Step 4: Install the package itself without dependencies
echo "Step 4: Installing the package itself without dependencies using 'poetry install --only-root'..."
poetry install --only-root

echo "✅ Installation completed successfully."
