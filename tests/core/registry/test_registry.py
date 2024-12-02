import logging

import pytest

from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.registry.registry import Registry


# Base classes for testing
class BaseComponent:
    """A base class for testing registry functionality."""

    def __init__(self, name):
        self.name = name


class BaseService:
    """Another base class for testing registry functionality."""

    def __init__(self, config):
        self.config = config


class TestRegistry:
    def test_registry_initialization(self):
        """Test initializing a registry with a base class."""
        registry = Registry(BaseComponent)
        assert registry.base_class is BaseComponent

    def test_registry_initialization_invalid_base_class(self):
        """Test that initialization fails with an invalid base class."""
        with pytest.raises(TypeError):
            Registry("not a class")

    def test_registry_register_decorator(self):
        """Test registering a class using the decorator."""
        registry = Registry(BaseComponent)

        @registry.register("test_component")
        class TestComponent(BaseComponent):
            pass

        # Verify registration
        assert registry.get("test_component") is TestComponent

    def test_registry_register_duplicate_name(self):
        """Test that registering a class with a duplicate name raises an error."""
        registry = Registry(BaseComponent)

        @registry.register("test_component")
        class TestComponent1(BaseComponent):
            pass

        # Attempt to register another class with the same name
        with pytest.raises(KeyError):

            @registry.register("test_component")
            class TestComponent2(BaseComponent):
                pass

    def test_registry_register_invalid_subclass(self):
        """Test that registering a non-subclass raises a TypeError."""
        registry = Registry(BaseComponent)

        with pytest.raises(TypeError):

            @registry.register("invalid_component")
            class InvalidComponent:
                pass

    def test_registry_get_invalid_name(self):
        """Test getting a class with an invalid name raises a ValueError."""
        registry = Registry(BaseComponent)

        with pytest.raises(ValueError):
            registry.get("non_existent_component")

    def test_registry_list(self):
        """Test listing registered components."""
        registry = Registry(BaseComponent)

        @registry.register("component1")
        class TestComponent1(BaseComponent):
            pass

        @registry.register("component2")
        class TestComponent2(BaseComponent):
            pass

        registered_list = registry.list()
        assert "component1" in registered_list
        assert "component2" in registered_list

    def test_registry_get_with_invalid_type(self):
        """Test getting a class with an invalid type raises a TypeError."""
        registry = Registry(BaseComponent)

        with pytest.raises(TypeError):
            registry.get(123)  # Non-string input


class TestRegistryContainer:
    def test_registry_container_singleton(self):
        """Test that RegistryContainer is a singleton."""
        container1 = RegistryContainer()
        container2 = RegistryContainer()
        assert container1 is container2

    def test_registry_container_add_registry(self):
        """Test adding a registry to the container."""
        container = RegistryContainer()
        container.add_registry("test_registry", "tests.core.registry.test_registry.BaseComponent")

        # Verify registry was added
        assert hasattr(container, "test_registry")
        assert container.get_registry("test_registry") is not None

    def test_registry_container_add_duplicate_registry(self):
        """Test adding a duplicate registry does not create a new one."""
        container = RegistryContainer()
        container.add_registry("test_registry", "tests.core.registry.test_registry.BaseComponent")

        # Adding the same registry again should not raise an error
        container.add_registry("test_registry", "tests.core.registry.test_registry.BaseComponent")

    def test_registry_container_invalid_base_class(self):
        """Test adding a registry with an invalid base class raises an ImportError."""
        container = RegistryContainer()

        with pytest.raises(ImportError):
            container.add_registry("invalid_registry", "non.existent.module.NonExistentClass")

    def test_registry_container_get_registry(self):
        """Test getting a registry from the container."""
        container = RegistryContainer()
        container.add_registry("test_registry", "tests.core.registry.test_registry.BaseComponent")

        registry = container.get_registry("test_registry")
        assert isinstance(registry, Registry)
        assert registry.base_class is BaseComponent

    def test_registry_container_get_nonexistent_registry(self):
        """Test getting a non-existent registry raises a KeyError."""
        container = RegistryContainer()

        with pytest.raises(KeyError):
            container.get_registry("non_existent_registry")

    def test_registry_container_list_all(self):
        """Test listing all registered components."""
        container = RegistryContainer()
        container.add_registry("test_registry1", "tests.core.registry.test_registry.BaseComponent")
        container.add_registry("test_registry2", "tests.core.registry.test_registry.BaseService")

        # Capture logging output
        with self._caplog.at_level(logging.INFO):
            container.list_all()

        # Check that logging occurred
        assert any("Registered" in record.message for record in self._caplog.records)

    @pytest.fixture
    def _caplog(self, caplog):
        """Fixture to provide caplog with logging level set."""
        caplog.set_level(logging.INFO)
        return caplog


class TestRegistryDecorator:
    def test_register_decorator(self):
        """Test the register decorator."""
        # Create a registry first
        container = RegistryContainer()
        container.add_registry("test_service", "tests.core.registry.test_registry.BaseService")

        @register(type="test_service", name="test_service_impl")
        class TestServiceImpl(BaseService):
            pass

        # Verify registration
        registry = container.get_registry("test_service")
        assert registry.get("test_service_impl") is TestServiceImpl

    def test_register_decorator_invalid_type(self):
        """Test registering with an invalid registry type raises an AttributeError."""
        with pytest.raises(AttributeError):

            @register(type="non_existent_type", name="test_component")
            class InvalidComponent:
                pass

    def test_register_decorator_invalid_name(self):
        """Test registering with an invalid name."""
        # Create a registry first
        container = RegistryContainer()
        container.add_registry("test_service", "tests.core.registry.test_registry.BaseService")

        with pytest.raises(ValueError):

            @register(type="test_service", name="")
            class InvalidServiceImpl(BaseService):
                pass

    def test_register_multiple_components(self):
        """Test registering multiple components in the same registry."""
        # Create a registry first
        container = RegistryContainer()
        container.add_registry("test_service", "tests.core.registry.test_registry.BaseService")

        @register(type="test_service", name="service1")
        class TestService1(BaseService):
            pass

        @register(type="test_service", name="service2")
        class TestService2(BaseService):
            pass

        registry = container.get_registry("test_service")
        assert registry.get("service1") is TestService1
        assert registry.get("service2") is TestService2

    def test_register_decorator_exception_handling(self):
        """Test exception handling in the register decorator."""
        # Simulate a scenario where registry creation fails
        container = RegistryContainer()

        with pytest.raises(AttributeError):

            @register(type="test_service", name="test_service_impl")
            class TestServiceImpl:
                pass  # No base registry exists
