# tests/units/core/registry/test_decorators.py

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Type

from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.registry import Registry
from simulation.core.registry.decorators import register

import logging

# Suppress logging during testing
logging.disable(logging.CRITICAL)

# Define a base class for testing
class Base:
    pass

# Define subclasses
class Optimizer(Base):
    pass

class NotSub:
    pass

# Mock RegistryContainer with necessary registries
class MockRegistryContainer:
    def __init__(self):
        self.optimizer = Registry(Base)
        self.line_search = Registry(Base)
        self.linear_solver = Registry(Base)
        # Add other registries as needed

# Fixture to provide a mock RegistryContainer
@pytest.fixture
def mock_registry_container():
    return MockRegistryContainer()

# Patch the _instance attribute of RegistryContainer to return the mock
@pytest.fixture(autouse=True)
def patch_registry_container(mock_registry_container):
    with patch.object(
        RegistryContainer, '_instance', mock_registry_container
    ):
        yield

def test_register_decorator_success(mock_registry_container):
    """Test that the register decorator successfully registers a class."""
    @register(type="optimizer", name="SGD")
    class SGDOptimizer(Optimizer):
        pass

    registry = mock_registry_container.optimizer
    assert "sgd" in registry._registry
    assert registry.get("sgd") is SGDOptimizer

def test_register_decorator_with_invalid_type(mock_registry_container):
    """Test that using an invalid component type raises AttributeError."""
    with pytest.raises(AttributeError) as exc_info:
        @register(type="invalid_type", name="Invalid")
        class InvalidClass(Base):
            pass

    assert "Component type 'invalid_type' does not have an associated registry." in str(exc_info.value)

def test_register_decorator_with_non_subclass(mock_registry_container):
    """Test that registering a non-subclass raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        @register(type="optimizer", name="NotSub")
        class NotSubClass(NotSub):
            pass

    assert "Cannot register NotSubClass as it is not a subclass of Base." in str(exc_info.value)

def test_register_decorator_with_empty_name(mock_registry_container):
    """Test that registering with an empty name raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        @register(type="optimizer", name="")
        class EmptyNameOptimizer(Optimizer):
            pass

    assert "Registration name must be a non-empty string." in str(exc_info.value)

def test_register_decorator_duplicate_name(mock_registry_container):
    """Test that registering with a duplicate name raises KeyError."""
    @register(type="optimizer", name="SGD")
    class SGDOptimizer(Optimizer):
        pass

    with pytest.raises(KeyError) as exc_info:
        @register(type="optimizer", name="SGD")
        class AnotherSGDOptimizer(Optimizer):
            pass

    assert "Class 'AnotherSGDOptimizer' is already registered under name 'sgd'." in str(exc_info.value)
