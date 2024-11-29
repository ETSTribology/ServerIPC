import pytest
from simulation.core.utils.singleton import SingletonMeta


class TestSingletonMeta:
    def test_singleton_instance_creation(self):
        """Test that only one instance of a class is created."""
        class SingletonClass(metaclass=SingletonMeta):
            def __init__(self, value=None):
                self.value = value

        # Create multiple instances
        instance1 = SingletonClass(42)
        instance2 = SingletonClass(99)
        instance3 = SingletonClass()

        # Verify that all instances are the same object
        assert instance1 is instance2
        assert instance2 is instance3

        # Verify that the first value is preserved
        assert instance1.value == 42
        assert instance2.value == 42
        assert instance3.value == 42

    def test_different_classes_have_separate_instances(self):
        """Test that different classes using SingletonMeta have separate instances."""
        class SingletonClassA(metaclass=SingletonMeta):
            def __init__(self, value=None):
                self.value_a = value

        class SingletonClassB(metaclass=SingletonMeta):
            def __init__(self, value=None):
                self.value_b = value

        # Create instances of different classes
        instance_a1 = SingletonClassA(10)
        instance_a2 = SingletonClassA(20)
        instance_b1 = SingletonClassB(30)
        instance_b2 = SingletonClassB(40)

        # Verify that instances of the same class are the same
        assert instance_a1 is instance_a2
        assert instance_b1 is instance_b2

        # Verify that instances of different classes are different
        assert instance_a1 is not instance_b1

        # Verify that the first values are preserved
        assert instance_a1.value_a == 10
        assert instance_b1.value_b == 30

    def test_thread_safety(self):
        """Basic test to ensure thread safety of singleton creation.
        Note: A comprehensive thread safety test would require more complex multi-threading setup."""
        import threading

        class ThreadSafeSingleton(metaclass=SingletonMeta):
            def __init__(self):
                self.initialized = True

        # Create multiple threads to instantiate the singleton
        threads = []
        instances = []

        def create_instance():
            instances.append(ThreadSafeSingleton())

        # Create multiple threads trying to create the singleton
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all instances are the same object
        assert len(set(instances)) == 1
