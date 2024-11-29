import pytest
from simulation.core.registry.registry import Registry
import logging

# Suppress logging during testing to keep test output clean
logging.disable(logging.CRITICAL)

# Define a base class for testing
class Base:
    pass

# Define subclasses
class SubA(Base):
    pass

class SubB(Base):
    pass

# Define a non-subclass
class NotSub:
    pass

@pytest.fixture
def registry_instance():
    """Fixture to create a new Registry instance for each test."""
    return Registry(Base)

def test_successful_registration(registry_instance):
    """Test that a valid subclass can be registered successfully."""
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    assert "suba" in registry_instance._registry
    assert registry_instance.get("suba") is SubARegistered

def test_registration_of_non_subclass(registry_instance):
    """Test that registering a non-subclass raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        @registry_instance.register(name="NotSub")
        class NotSubRegistered:
            pass

    assert "Cannot register NotSubRegistered as it is not a subclass of Base." in str(exc_info.value)

def test_duplicate_registration(registry_instance):
    """Test that registering a class with an already registered name raises KeyError."""
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    with pytest.raises(KeyError) as exc_info:
        @registry_instance.register(name="SubA")
        class AnotherSubARegistered(Base):
            pass

    assert "Class 'AnotherSubARegistered' is already registered under name 'suba'." in str(exc_info.value)

def test_retrieval_of_registered_class(registry_instance):
    """Test that a registered class can be retrieved by name."""
    @registry_instance.register(name="SubB")
    class SubBRegistered(Base):
        pass

    retrieved_class = registry_instance.get("SubB")
    assert retrieved_class is SubBRegistered

def test_case_insensitive_retrieval(registry_instance):
    """Test that retrieval is case-insensitive."""
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    retrieved_class_lower = registry_instance.get("suba")
    retrieved_class_upper = registry_instance.get("SUBA")
    retrieved_class_mixed = registry_instance.get("SuBa")

    assert retrieved_class_lower is SubARegistered
    assert retrieved_class_upper is SubARegistered
    assert retrieved_class_mixed is SubARegistered

def test_retrieval_of_non_registered_class(registry_instance):
    """Test that retrieving a non-registered class raises ValueError."""
    # Register some classes first
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    @registry_instance.register(name="SubB")
    class SubBRegistered(Base):
        pass

    with pytest.raises(ValueError) as exc_info:
        registry_instance.get("NonExistent")

    # Adjust the expected available types based on previous registrations
    assert "Unsupported type: 'nonexistent'. Available types: suba, subb" in str(exc_info.value)

def test_listing_registered_names(registry_instance):
    """Test that all registered names are listed correctly."""
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    @registry_instance.register(name="SubB")
    class SubBRegistered(Base):
        pass

    registered_list = registry_instance.list()
    expected = "suba, subb"
    # The order may vary since dictionaries are insertion ordered from Python 3.7+
    assert registered_list == expected or registered_list == "subb, suba"

def test_multiple_registrations(registry_instance):
    """Test registering multiple classes and retrieving them."""
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    @registry_instance.register(name="SubB")
    class SubBRegistered(Base):
        pass

    retrieved_a = registry_instance.get("SubA")
    retrieved_b = registry_instance.get("SubB")

    assert retrieved_a is SubARegistered
    assert retrieved_b is SubBRegistered

def test_registry_initialization_with_invalid_base_class():
    """Test that initializing Registry with non-class raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        Registry("NotAClass")  # Passing a string instead of a class

    assert "base_class must be a class type." in str(exc_info.value)

def test_register_decorator_returns_class(registry_instance):
    """Test that the register decorator returns the class unmodified."""
    @registry_instance.register(name="SubA")
    class SubARegistered(Base):
        pass

    assert SubARegistered.__name__ == "SubARegistered"

def test_register_with_empty_name(registry_instance):
    """Test that registering with an empty name raises an error."""
    with pytest.raises(ValueError) as exc_info:
        @registry_instance.register(name="")
        class EmptyNameRegistered(Base):
            pass

    assert "Registration name must be a non-empty string." in str(exc_info.value)
